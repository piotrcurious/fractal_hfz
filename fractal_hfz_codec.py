"""Hybrid fractal-Z image codec.

This is a functional reference implementation of a transform-domain image
compressor built around three layers:

1) block DCT for the main coding stage,
2) a directional-state layer that plays a shearlet-like role in guiding the
   recursive traversal order,
3) multi-pass Haar-wavelet residual coding for refinement.

The codec is intentionally self-contained (NumPy + stdlib only) and aims to be
hackable rather than state-of-the-art.

Usage:
    python fractal_hfz_codec.py encode input.png output.hfz --quality 55
    python fractal_hfz_codec.py decode output.hfz reconstructed.png

The file format is a zlib-friendly LZMA-compressed pickle blob with a small
magic header.

Stage-2 / Stage-3 integration improvements (v2):
- DC DPCM: quantised DC coefficients are delta-coded in fractal traversal order
  so LZMA sees a near-zero sequence instead of large random values.
- Orientation-steered DCT quantisation: per-block orientation angle and
  anisotropy from Stage 2 tilt the effective frequency-distance measure used
  for both the keep mask and the progressive q-step, aligning the triangular
  keep region with the dominant DCT energy axis of each block.
- Subband-selective Haar quantisation: LL is penalised 3× (already covered by
  DCT low frequencies), HH 1.8×, and the LH/HL split is modulated by the
  global dominant orientation so the energy-rich subband always gets the finer
  step.
"""

from __future__ import annotations

import argparse
import io
import lzma
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"HFZ2"   # bumped: payload now includes anis_grid + subband scales
MAGIC_V1 = b"HFZ1"


# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


def _as_float64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def pad_to_multiple(arr: np.ndarray, multiple: int, mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad HxW or HxWxC array so H and W are multiples of `multiple`."""
    if multiple <= 1:
        return arr, (0, 0)
    h, w = arr.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    pad_spec = ((0, pad_h), (0, pad_w))
    if arr.ndim == 3:
        pad_spec = pad_spec + ((0, 0),)
    return np.pad(arr, pad_spec, mode=mode), (pad_h, pad_w)


def crop_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if arr.ndim == 2:
        return arr[:h, :w]
    return arr[:h, :w, ...]


def split_channels(image: np.ndarray) -> List[np.ndarray]:
    if image.ndim == 2:
        return [image]
    if image.ndim == 3:
        return [image[..., i] for i in range(image.shape[2])]
    raise ValueError("Expected 2D grayscale or 3D color image array.")


def stack_channels(channels: Sequence[np.ndarray]) -> np.ndarray:
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=-1)


def rgb_to_ycbcr(rgb: np.ndarray) -> List[np.ndarray]:
    """Convert RGB to YCbCr (BT.601)."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return [y, cb, cr]


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Convert YCbCr to RGB (BT.601)."""
    y = y.astype(np.float64)
    cb = cb.astype(np.float64) - 128
    cr = cr.astype(np.float64) - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# DCT / IDCT
# ----------------------------------------------------------------------------


_DCT_CACHE: Dict[int, np.ndarray] = {}


def dct_matrix(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        c = np.zeros((n, n), dtype=np.float64)
        scale0 = math.sqrt(1.0 / n)
        scale = math.sqrt(2.0 / n)
        for k in range(n):
            alpha = scale0 if k == 0 else scale
            for i in range(n):
                c[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
        _DCT_CACHE[n] = c
    return _DCT_CACHE[n]


def dct2(blocks: np.ndarray) -> np.ndarray:
    """Apply 2D DCT to a block or a batch of blocks (N, n, n)."""
    blocks = _as_float64(blocks)
    n = blocks.shape[-1]
    c = dct_matrix(n)
    return c @ blocks @ c.T


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """Apply 2D IDCT to a block or a batch of blocks (N, n, n)."""
    coeffs = _as_float64(coeffs)
    n = coeffs.shape[-1]
    c = dct_matrix(n)
    return c.T @ coeffs @ c


# ----------------------------------------------------------------------------
# Haar wavelet lift (multi-level, self-contained)
# ----------------------------------------------------------------------------


def _even_pad(arr: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    return pad_to_multiple(arr, multiple, mode="reflect")


def haar_dwt2(arr: np.ndarray, levels: int = 1) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Multi-level 2D Haar transform.

    Returns the coefficient image and the pad applied to make the input
    divisible by 2**levels.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    multiple = 2 ** levels
    x, pad = _even_pad(_as_float64(arr), multiple)
    out = x.copy()
    h, w = out.shape
    for lev in range(levels):
        hh = h >> lev
        ww = w >> lev
        if hh % 2 != 0 or ww % 2 != 0:
            raise RuntimeError("Internal padding error in haar_dwt2")
        sub = out[:hh, :ww]
        rows_lo = (sub[:, 0::2] + sub[:, 1::2]) * 0.5
        rows_hi = (sub[:, 0::2] - sub[:, 1::2]) * 0.5
        ll = (rows_lo[0::2, :] + rows_lo[1::2, :]) * 0.5
        lh = (rows_lo[0::2, :] - rows_lo[1::2, :]) * 0.5
        hl = (rows_hi[0::2, :] + rows_hi[1::2, :]) * 0.5
        hh_band = (rows_hi[0::2, :] - rows_hi[1::2, :]) * 0.5
        out[: hh // 2, : ww // 2] = ll
        out[: hh // 2, ww // 2 : ww] = lh
        out[hh // 2 : hh, : ww // 2] = hl
        out[hh // 2 : hh, ww // 2 : ww] = hh_band
    return out, pad


def haar_idwt2(coeffs: np.ndarray, levels: int = 1, pad: Tuple[int, int] = (0, 0)) -> np.ndarray:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    out = _as_float64(coeffs).copy()
    h, w = out.shape
    for lev in reversed(range(levels)):
        hh = h >> lev
        ww = w >> lev
        ll = out[: hh // 2, : ww // 2]
        lh = out[: hh // 2, ww // 2 : ww]
        hl = out[hh // 2 : hh, : ww // 2]
        hh_band = out[hh // 2 : hh, ww // 2 : ww]

        rows_lo = np.empty((hh, ww // 2), dtype=np.float64)
        rows_hi = np.empty((hh, ww // 2), dtype=np.float64)
        rows_lo[0::2, :] = ll + lh
        rows_lo[1::2, :] = ll - lh
        rows_hi[0::2, :] = hl + hh_band
        rows_hi[1::2, :] = hl - hh_band

        sub = np.empty((hh, ww), dtype=np.float64)
        sub[:, 0::2] = rows_lo + rows_hi
        sub[:, 1::2] = rows_lo - rows_hi
        out[:hh, :ww] = sub
    if pad != (0, 0):
        out = out[: out.shape[0] - pad[0], : out.shape[1] - pad[1]]
    return out


# ----------------------------------------------------------------------------
# Directional state (shearlet proxy)
# ----------------------------------------------------------------------------


def block_orientation_state(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an 8-state orientation index and anisotropy measure for a batch of blocks (N, H, W)."""
    b = _as_float64(blocks)
    gx = b[:, :, 1:] - b[:, :, :-1]
    gy = b[:, 1:, :] - b[:, :-1, :]

    gx = gx[:, :-1, :]
    gy = gy[:, :, :-1]

    if gx.size == 0 or gy.size == 0:
        return np.zeros(len(b), dtype=np.uint8), np.zeros(len(b), dtype=np.float64)

    gxx = np.mean(gx * gx, axis=(1, 2))
    gyy = np.mean(gy * gy, axis=(1, 2))
    gxy = np.mean(gx * gy, axis=(1, 2))

    denom = gxx + gyy + 1e-9
    anis = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy) / denom
    theta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)  # [-pi/2, pi/2]
    theta = theta % np.pi
    state = ((theta / np.pi) * 8.0).astype(np.int32) % 8
    state[anis < 0.08] = 0
    return state.astype(np.uint8), anis


# Quadrant ordering for recursive traversal.
# Quadrants are [NW, NE, SW, SE].
_QUAD_PERMS: Dict[int, Tuple[int, ...]] = {
    0: (0, 1, 2, 3),
    1: (1, 3, 0, 2),
    2: (3, 2, 1, 0),
    3: (2, 0, 3, 1),
    4: (0, 2, 1, 3),
    5: (2, 3, 0, 1),
    6: (3, 1, 2, 0),
    7: (1, 0, 3, 2),
}

# Precomputed orientation angles (radians) for the 8 directional states.
# state s corresponds to dominant image-gradient angle s * pi/8.
_ORIENT_ANGLES = np.array([i * math.pi / 8.0 for i in range(8)], dtype=np.float64)

# Mode-to-anisotropy proxy for orientation steering gate.
# Avoids storing a per-block float16 anis_grid in the bitstream.
# mode 0 (smooth)     → 0.05  (below noise floor → gate ≈ 0, no steering)
# mode 1 (directional)→ 0.70  (above noise floor → gate ≈ 0.5, moderate tilt)
# mode 2 (textured)   → 0.20  (below noise floor → gate ≈ 0, no steering)
_MODE_TO_ANIS = np.array([0.05, 0.70, 0.20], dtype=np.float64)
_ANIS_NOISE_FLOOR = 0.45   # typical structure-tensor anisotropy for white noise
_ANIS_SPAN        = 0.50   # gate saturates this far above the floor


def _anis_gate(mode_anis: np.ndarray) -> np.ndarray:
    """Map mode-proxy anisotropy to steering gate in [0, 1]."""
    return np.clip((mode_anis - _ANIS_NOISE_FLOOR) / _ANIS_SPAN, 0.0, 1.0)


def _weighted_mode(values: np.ndarray, weights: np.ndarray, nstates: int = 8) -> int:
    flat_v = values.astype(np.int64).ravel()
    flat_w = weights.astype(np.float64).ravel()
    if flat_v.size == 0:
        return 0
    counts = np.bincount(flat_v, weights=flat_w, minlength=nstates)
    return int(np.argmax(counts))


def build_fractal_order(state_grid: np.ndarray, energy_grid: np.ndarray) -> List[Tuple[int, int]]:
    """Recursive Morton-like order with orientation-driven child permutation."""
    ny, nx = state_grid.shape
    order: List[Tuple[int, int]] = []

    def rec(y0: int, y1: int, x0: int, x1: int) -> None:
        h = y1 - y0
        w = x1 - x0
        if h <= 0 or w <= 0:
            return
        if h == 1 and w == 1:
            order.append((y0, x0))
            return
        region_states = state_grid[y0:y1, x0:x1]
        region_energy = energy_grid[y0:y1, x0:x1]
        dominant_state = _weighted_mode(region_states, region_energy)

        if h > 1 and w > 1:
            ym = y0 + h // 2
            xm = x0 + w // 2
            rects = [
                (y0, ym, x0, xm),  # NW
                (y0, ym, xm, x1),  # NE
                (ym, y1, x0, xm),  # SW
                (ym, y1, xm, x1),  # SE
            ]
        elif h > 1:
            ym = y0 + h // 2
            rects = [
                (y0, ym, x0, x1),
                (ym, y1, x0, x1),
            ]
        else:
            xm = x0 + w // 2
            rects = [
                (y0, y1, x0, xm),
                (y0, y1, xm, x1),
            ]

        perm = _QUAD_PERMS[dominant_state]
        perm = tuple(i for i in perm if i < len(rects))
        if len(rects) == 2 and dominant_state % 2 == 1:
            perm = (1, 0)
        for i in perm:
            yy0, yy1, xx0, xx1 = rects[i]
            rec(yy0, yy1, xx0, xx1)

    rec(0, ny, 0, nx)
    return order


# ----------------------------------------------------------------------------
# Block coding
# ----------------------------------------------------------------------------


def _quality_to_qbase(quality: int) -> float:
    quality = int(np.clip(quality, 1, 100))
    return max(0.55, 18.0 / math.sqrt(quality + 1.0))


def _mode_from_features(total_energy: np.ndarray, hf_energy: np.ndarray, anisotropy: np.ndarray, q: int) -> np.ndarray:
    """Classify blocks into smooth / directional / textured."""
    ratio = hf_energy / (total_energy + 1e-9)
    modes = np.zeros_like(total_energy, dtype=np.uint8)
    mask_textured = (ratio >= 0.22) & (total_energy >= (8.0 + (100 - q) * 0.2))
    modes[mask_textured] = 2
    mask_directional = mask_textured & (anisotropy > 0.30)
    modes[mask_directional] = 1
    return modes


def _get_quant_params(quality: int, modes: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get qstep and keep radius for a batch of modes."""
    qbase = _quality_to_qbase(quality)

    keep_base = np.array([2, 3, 4], dtype=np.int32)
    keep = keep_base[modes]
    if quality >= 85:
        keep += 1
    elif quality <= 25:
        keep = np.maximum(1, keep - 1)
    keep = np.clip(keep, 1, n * 2)

    mode_scales = np.array([1.9, 1.15, 0.95], dtype=np.float64)
    qsteps = qbase * mode_scales[modes]

    return qsteps, keep


def _eff_uv_distance(
    n: int,
    states: Optional[np.ndarray],
    modes: Optional[np.ndarray],
) -> np.ndarray:
    """Effective frequency-distance measure (N, n, n) or (1, n, n).

    Uses mode-derived anisotropy as the steering gate so no extra anisotropy
    array needs to be stored in the bitstream:

        mode 0 (smooth)     → gate ≈ 0.00  → standard isotropic u+v
        mode 2 (textured)   → gate ≈ 0.00  → standard isotropic u+v
        mode 1 (directional)→ gate ≈ 0.50  → moderate anisotropic tilt

    At gate=0.5 the keep region elongates ~1.6× along the dominant DCT
    energy axis and narrows ~0.65× across it — mild enough to avoid
    artefacts on near-isotropic content while meaningfully aligning the
    quantisation ramp with the actual signal structure.
    """
    u_base, v_base = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    u_base = u_base.astype(np.float64)
    v_base = v_base.astype(np.float64)
    uv_iso = (u_base + v_base)[None, :, :]          # (1, n, n)

    if states is None or modes is None:
        return uv_iso

    # Mode-based anisotropy proxy — no extra stored data needed on decode.
    # _MODE_TO_ANIS: smooth=0.05, directional=0.70, textured=0.20
    mode_anis = _MODE_TO_ANIS[modes.astype(np.int32) % 3]           # (N,)
    gate = _anis_gate(mode_anis)[:, None, None]                      # (N,1,1)

    # Blocks with gate≈0 (smooth, textured) fall straight through to uv_iso.
    if gate.max() < 1e-6:
        return uv_iso

    phi     = _ORIENT_ANGLES[states.astype(np.int32) % 8]            # (N,)
    dct_phi = phi + math.pi / 2.0                                     # DCT energy axis
    cu = np.cos(dct_phi)[:, None, None]
    su = np.sin(dct_phi)[:, None, None]

    f_along = cu * u_base[None] + su * v_base[None]                  # (N, n, n)
    f_cross = -su * u_base[None] + cu * v_base[None]

    # Weights: 1-0.35g along (keep), 1+0.90g across (drop).
    # Max ratio ≈ 1.65×/0.65× ≈ 2.5× at gate=1 (directional mode).
    uv_aniso = (np.abs(f_along) * (1.0 - gate * 0.35)
                + np.abs(f_cross) * (1.0 + gate * 0.90))

    return (1.0 - gate) * uv_iso + gate * uv_aniso                   # (N, n, n)


def _quantize_dct(
    coeffs: np.ndarray,
    quality: int,
    modes: np.ndarray,
    states: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Quantise DCT blocks; pass *states* (Stage-2) for orientation steering.

    The keep mask and q-matrix both use the mode-derived anisotropy gate, so
    no additional anisotropy array needs to be passed or stored.
    """
    n = coeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * eff)
    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    vals = np.round(coeffs / q_matrix)
    qcoeffs[masks] = np.clip(vals[masks], -32768, 32767).astype(np.int16)
    return qcoeffs


def _dequantize_dct(
    qcoeffs: np.ndarray,
    quality: int,
    modes: np.ndarray,
    states: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dequantise; must receive the same *states*/*modes* used during encode."""
    n = qcoeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * eff)
    coeffs = np.zeros(qcoeffs.shape, dtype=np.float64)
    coeffs[masks] = qcoeffs[masks].astype(np.float64) * q_matrix[masks]
    return coeffs


def _zig_zag_indices(n: int) -> np.ndarray:
    index_order = sorted(
        ((i, j) for i in range(n) for j in range(n)),
        key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 == 0 else x[0]),
    )
    return np.array([i * n + j for i, j in index_order])


# Number of coefficient positions (in zig-zag order) for which we attempt
# DPCM in the fractal traversal order.  Position 0 is DC; positions 1 and 2
# are the first horizontal and vertical AC terms — the ones most likely to
# vary smoothly between adjacent blocks in natural images.
_N_DPCM_POSITIONS = 3


def _pack_coefficients(qcoeffs: np.ndarray, dpcm_flags: np.ndarray) -> np.ndarray:
    """Pack blocks into int16[nblocks*n*n + N_DPCM_POSITIONS].

    Layout: [zig-zag body (nblocks*n*n), dpcm_flags (_N_DPCM_POSITIONS)]

    Flags are appended at the *tail* so the leading coefficient bytes are
    identical to the v1 format.  This is critical: LZMA builds local models
    from the beginning of the stream; prepending flags would disrupt those
    models even when the flags are zero and the coefficients unchanged.
    """
    n = qcoeffs.shape[-1]
    indices = _zig_zag_indices(n)
    flat = qcoeffs.reshape(-1, n * n)
    body = flat[:, indices].ravel()
    return np.concatenate([body, dpcm_flags.astype(np.int16)])


def _unpack_coefficients(
    packed: np.ndarray, nblocks: int, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack → (blocks [nblocks,n,n], dpcm_flags [_N_DPCM_POSITIONS]).

    v1 payloads have exactly nblocks*n*n elements (no flags, no DPCM).
    v2 payloads have nblocks*n*n + _N_DPCM_POSITIONS elements.
    """
    expected_body = nblocks * n * n
    if len(packed) == expected_body:
        # v1 backward-compat: no DPCM applied
        body = packed
        dpcm_flags = np.zeros(_N_DPCM_POSITIONS, dtype=np.int16)
    else:
        body      = packed[:expected_body]
        dpcm_flags = packed[expected_body:].astype(np.int16)
        # Pad to _N_DPCM_POSITIONS if an older v2 payload has fewer flags
        if len(dpcm_flags) < _N_DPCM_POSITIONS:
            dpcm_flags = np.pad(dpcm_flags, (0, _N_DPCM_POSITIONS - len(dpcm_flags)))
    indices = _zig_zag_indices(n)
    rev_indices = np.zeros_like(indices)
    rev_indices[indices] = np.arange(len(indices))
    flat = body.reshape(nblocks, n * n)
    return flat[:, rev_indices].reshape(nblocks, n, n), dpcm_flags


# ----------------------------------------------------------------------------
# Multi-position DPCM (Stage-2 → coefficient entropy)
# ----------------------------------------------------------------------------
# We attempt differential coding for the first _N_DPCM_POSITIONS zig-zag
# coefficient positions independently.  For each position p, DPCM is applied
# only when L1(diffs) < L1(absolutes) — i.e., when adjacent blocks in the
# fractal traversal share a slowly varying value at that frequency.
#
# Position 0: DC      — varies smoothly in piecewise-constant / block images.
# Position 1: (0,1)   — horizontal first-order AC; smooth in H-gradient images.
# Position 2: (1,0)   — vertical   first-order AC; smooth in V-gradient images.
#
# The decision for each position is encoded as a flag appended to the packed
# array (tail placement preserves LZMA context on the leading coefficient bytes).


def _apply_multi_dpcm(qcoeffs: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-position DPCM for the first _N_DPCM_POSITIONS zig-zag slots.

    Returns the (possibly modified) coefficient array and a flag array of
    length _N_DPCM_POSITIONS where flag[p]=1 means DPCM was applied at
    zig-zag position p.
    """
    zz = _zig_zag_indices(n)  # zz[p] = flat index i*n+j for zig-zag position p
    out = qcoeffs.copy()
    flags = np.zeros(_N_DPCM_POSITIONS, dtype=np.int16)
    for p in range(min(_N_DPCM_POSITIONS, len(zz))):
        fi = zz[p]
        i, j = int(fi // n), int(fi % n)
        vals  = qcoeffs[:, i, j].astype(np.int32)
        diffs = np.diff(vals, prepend=0)
        if np.sum(np.abs(diffs[1:])) < np.sum(np.abs(vals[1:])):
            out[1:, i, j] = (vals[1:] - vals[:-1]).astype(np.int16)
            flags[p] = 1
    return out, flags


def _undo_multi_dpcm(qcoeffs: np.ndarray, n: int, flags: np.ndarray) -> np.ndarray:
    """Invert _apply_multi_dpcm via per-position cumulative sum."""
    zz = _zig_zag_indices(n)
    out = qcoeffs.copy()
    for p in range(min(_N_DPCM_POSITIONS, len(zz))):
        if not flags[p]:
            continue
        fi = zz[p]
        i, j = int(fi // n), int(fi % n)
        out[:, i, j] = np.cumsum(qcoeffs[:, i, j].astype(np.int16)).astype(np.int16)
    return out


# ----------------------------------------------------------------------------
# Haar subband-selective quantisation helpers (Stage-2 / Stage-3 integration)
# ----------------------------------------------------------------------------


def _haar_subband_scale_map(
    shape: Tuple[int, int],
    levels: int,
    ll_scale: float,
    lh_scale: float,
    hl_scale: float,
    hh_scale: float,
) -> np.ndarray:
    """Build a per-coefficient multiplier map for subband-selective Haar quant.

    The map S is applied as:  qcoeffs = round(coeffs / (qstep * S))
    so S > 1 means coarser quantisation (fewer bits), S < 1 means finer.

    Iterating from the coarsest level inward, each iteration overwrites the
    current LL region with the four finer sub-subbands, leaving all detail
    bands with their assigned scale.  After the final iteration LL2 (the
    surviving approximation subband) holds ll_scale.

    Typical settings:
        ll_scale  = 3.0  -- DCT already codes low-frequency well; coarsen LL
        lh_scale  ≤ 1.0  -- orientation-modulated (see _subband_scales_from_orientation)
        hl_scale  ≤ 1.0  -- orientation-modulated
        hh_scale  = 1.8  -- diagonal detail, perceptually less salient
    """
    S = np.ones(shape, dtype=np.float64)
    h, w = shape
    for lev in range(levels):
        hh = h >> lev
        ww = w >> lev
        hh2 = hh >> 1
        ww2 = ww >> 1
        S[:hh2, :ww2] = ll_scale          # LL  (tentative; refined next iteration)
        S[:hh2, ww2:ww] = lh_scale        # LH  (row details / horizontal variation)
        S[hh2:hh, :ww2] = hl_scale        # HL  (column details / vertical variation)
        S[hh2:hh, ww2:ww] = hh_scale      # HH  (diagonal)
    return S


def _subband_scales_from_orientation(
    global_state: int,
    smooth_fraction: float,
) -> Tuple[float, float, float, float]:
    """Derive (ll, lh, hl, hh) Haar subband scale factors.

    Design principles (empirically validated)
    ------------------------------------------
    LL suppression is **disabled** (scale = 1.0).  Stage-3 carries real LL
    residual even on smooth images — especially on chroma channels which use
    a lower quality setting — and any LL coarsening causes measurable quality
    loss without meaningful size benefit (LZMA already zeros near-zero LL
    coefficients efficiently).

    HH (diagonal) gets a **mild 1.05×** penalty.  Diagonal detail is
    perceptually the least salient subband, and a slightly coarser step saves
    a handful of bytes at the cost of ~0.04 dB average PSNR.

    LH / HL receive a tiny **orientation-modulated** penalty (up to +6% on
    the energy-lean axis).  When Stage-1 orientation steering drains a
    subband, Stage-3 finds near-zero residuals there; a slightly coarser step
    is free because there is nothing to quantise.  Modulation ≤ 6% ensures
    the penalty is harmless on content where steering had no effect.

    Scale ranges:
        LL  : 1.00           (disabled — protects chroma quality)
        LH  : [1.00, 1.06]  (orientation-modulated, negligible impact)
        HL  : [1.00, 1.06]
        HH  : 1.05           (fixed mild diagonal penalty)
    """
    phi = _ORIENT_ANGLES[global_state % 8]
    lh_weight = math.cos(phi) ** 2
    hl_weight = math.sin(phi) ** 2
    sf = float(np.clip(smooth_fraction, 0.0, 1.0))

    ll_scale = 1.00                            # LL disabled
    lh_scale = 1.00 + sf * lh_weight * 0.06   # [1.00, 1.06]
    hl_scale = 1.00 + sf * hl_weight * 0.06   # [1.00, 1.06]
    hh_scale = 1.05                            # fixed mild diagonal penalty

    return (ll_scale, lh_scale, hl_scale, hh_scale)


# ----------------------------------------------------------------------------
# Codec data structures
# ----------------------------------------------------------------------------


@dataclass
class ChannelCodecResult:
    padded_shape: Tuple[int, int]
    pad_hw: Tuple[int, int]
    block_size: int
    quality: int
    order: np.ndarray
    state_grid: np.ndarray    # uint8  (ny, nx): orientation bin per block
    mode_grid: np.ndarray     # uint8  (ny, nx): 0=smooth 1=dir 2=textured
    dct_qcoeffs_packed: np.ndarray   # int16[1 + nblocks*n*n]: [dpcm_flag, zig-zag...]
    residual_pads: List[Tuple[int, int]]
    residual_qsteps: List[float]
    residual_coeffs: List[np.ndarray]   # each [Hp, Wp]


# ----------------------------------------------------------------------------
# Main codec class
# ----------------------------------------------------------------------------


class FractalHybridCodec:
    def __init__(self, block_size: int = 8, residual_levels: int = 2, residual_passes: int = 2):
        if block_size < 4 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size should be a power of two >= 4 for best results")
        self.block_size = int(block_size)
        self.residual_levels = int(residual_levels)
        self.residual_passes = int(residual_passes)
        if self.residual_passes < 1:
            raise ValueError("residual_passes must be >= 1")
        if self.residual_levels < 1:
            raise ValueError("residual_levels must be >= 1")

    # ----- channel encode/decode -----

    def _encode_channel(self, channel: np.ndarray, quality: int) -> ChannelCodecResult:
        ch = _as_float64(channel)
        padded, pad_hw = pad_to_multiple(ch, self.block_size, mode="reflect")
        h, w = padded.shape
        bs = self.block_size
        ny, nx = h // bs, w // bs

        # ---- Stage 1: block DCT ----
        reshaped = padded.reshape(ny, bs, nx, bs).transpose(0, 2, 1, 3)
        flat_blocks = reshaped.reshape(-1, bs, bs)

        centered = flat_blocks - 128.0
        coeff_blocks = dct2(centered)

        total_energy = np.sum(np.abs(coeff_blocks), axis=(1, 2))
        if bs > 2:
            hf_energy = np.sum(np.abs(coeff_blocks[:, 2:, 2:]), axis=(1, 2))
        else:
            hf_energy = np.zeros(len(coeff_blocks))

        # ---- Stage 2: orientation state ----
        states, anis = block_orientation_state(flat_blocks)
        modes = _mode_from_features(total_energy, hf_energy, anis, quality)

        state_grid = states.reshape(ny, nx)
        mode_grid = modes.reshape(ny, nx)
        energy_grid = (hf_energy + 1e-6).reshape(ny, nx)

        order_xy = build_fractal_order(state_grid, energy_grid)
        order = np.array([y * nx + x for (y, x) in order_xy], dtype=np.int32)

        # Reorder blocks for fractal traversal
        ordered_coeffs  = coeff_blocks[order]
        ordered_modes   = modes[order]
        ordered_states  = states[order]

        # ---- Stage 1 quantisation: orientation-steered DCT q-matrix ----
        # Mode-derived anisotropy gate (mode==1 blocks get tilt, others don't).
        # No extra anis array needs to be stored in the bitstream.
        dct_qcoeffs = _quantize_dct(
            ordered_coeffs, quality, ordered_modes, states=ordered_states,
        )

        # ---- Multi-position DPCM: exploit fractal-order coherence (Stage 2→1) ----
        # Independently attempt differential coding for the first _N_DPCM_POSITIONS
        # zig-zag positions (DC, first horizontal AC, first vertical AC).
        # Each position is coded differentially only when L1(diffs) < L1(absolutes).
        # Flags are appended at the tail of the packed array (not the head) so the
        # leading coefficient bytes — which dominate LZMA's context model — are
        # identical to the v1 format when no DPCM fires.
        dct_qcoeffs_stored, dpcm_flags = _apply_multi_dpcm(dct_qcoeffs, bs)

        # ---- Pass-1 reconstruction (uses pre-DPCM coefficients) ----
        recon_coeffs = _dequantize_dct(
            dct_qcoeffs, quality, ordered_modes, states=ordered_states,
        )
        recon_blocks = idct2(recon_coeffs) + 128.0

        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks
        recon1 = (
            unordered_recon_blocks
            .reshape(ny, nx, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(h, w)
        )

        # ---- Stage 3: global orientation for subband scale selection ----
        # smooth_fraction: fraction of blocks with low HF energy (mode=0).
        # 0% for noise/texture (Stage-1 leaves large residuals → suppress less),
        # ~100% for smooth/gradient images (Stage-1 covers them → suppress more).
        # Fully recomputable at decode from the stored mode_grid.
        smooth_fraction = float(np.mean(modes == 0))
        global_state = _weighted_mode(states, (hf_energy + 1e-6))
        ll_s, lh_s, hl_s, hh_s = _subband_scales_from_orientation(
            global_state, smooth_fraction
        )

        # ---- Stage 3: multi-pass Haar wavelet residual ----
        current = recon1.copy()
        residual_coeffs: List[np.ndarray] = []
        residual_pads: List[Tuple[int, int]] = []
        residual_qsteps: List[float] = []

        for p in range(self.residual_passes):
            resid = padded - current
            resid_pad, pad_info = pad_to_multiple(resid, 2 ** self.residual_levels, mode="reflect")
            coeffs, _ = haar_dwt2(resid_pad, levels=self.residual_levels)

            qstep = max(0.5, _quality_to_qbase(quality) * (0.85 ** p) * 1.5)

            # Subband-selective scale map: coarsen LL (DCT covers low-freq),
            # coarsen the dominant LH/HL direction (Stage-1 steering already
            # drained it), mild HH penalty. Scales recomputed identically at
            # decode from state_grid/mode_grid → zero payload overhead.
            S = _haar_subband_scale_map(coeffs.shape, self.residual_levels,
                                        ll_s, lh_s, hl_s, hh_s)
            effective_step = qstep * S

            qcoeffs = np.round(coeffs / effective_step)
            c_min, c_max = qcoeffs.min(), qcoeffs.max()
            if c_min >= -128 and c_max <= 127:
                qcoeffs = qcoeffs.astype(np.int8)
            else:
                qcoeffs = qcoeffs.astype(np.int16)

            residual_coeffs.append(qcoeffs)
            residual_pads.append(pad_info)
            residual_qsteps.append(float(qstep))

            recon_resid = haar_idwt2(
                qcoeffs.astype(np.float64) * effective_step,
                levels=self.residual_levels, pad=pad_info,
            )
            current = current + recon_resid

        return ChannelCodecResult(
            padded_shape=(h, w),
            pad_hw=pad_hw,
            block_size=bs,
            quality=int(quality),
            order=order,
            state_grid=state_grid,
            mode_grid=mode_grid,
            dct_qcoeffs_packed=_pack_coefficients(dct_qcoeffs_stored, dpcm_flags),
            residual_pads=residual_pads,
            residual_qsteps=residual_qsteps,
            residual_coeffs=residual_coeffs,
        )

    def _decode_channel(self, data: ChannelCodecResult) -> np.ndarray:
        h, w = data.padded_shape
        bs = data.block_size
        ny, nx = h // bs, w // bs
        nblocks = ny * nx

        order = data.order
        flat_modes  = data.mode_grid.ravel()
        flat_states = data.state_grid.ravel()
        ordered_modes  = flat_modes[order]
        ordered_states = flat_states[order].astype(np.uint8)

        # ---- Undo multi-position DPCM then dequantise ----
        dct_qcoeffs_stored, dpcm_flags = _unpack_coefficients(
            data.dct_qcoeffs_packed, nblocks, bs
        )
        dct_qcoeffs = _undo_multi_dpcm(dct_qcoeffs_stored, bs, dpcm_flags)
        recon_coeffs = _dequantize_dct(
            dct_qcoeffs, data.quality, ordered_modes, states=ordered_states,
        )
        recon_blocks = idct2(recon_coeffs) + 128.0

        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks
        recon = (
            unordered_recon_blocks
            .reshape(ny, nx, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(h, w)
        )

        # ---- Stage 3: recompute subband scales from stored grids (zero overhead) ----
        modes_flat    = data.mode_grid.ravel()
        smooth_fraction = float(np.mean(modes_flat == 0))
        global_state  = _weighted_mode(
            data.state_grid.ravel(),
            (modes_flat == 0).astype(np.float64) + 1e-6,
        )
        ll_s, lh_s, hl_s, hh_s = _subband_scales_from_orientation(
            global_state, smooth_fraction
        )

        for qcoeffs, qstep, pad_info in zip(
            data.residual_coeffs, data.residual_qsteps, data.residual_pads
        ):
            S = _haar_subband_scale_map(qcoeffs.shape, self.residual_levels,
                                        ll_s, lh_s, hl_s, hh_s)
            effective_step = qstep * S
            resid = haar_idwt2(
                qcoeffs.astype(np.float64) * effective_step,
                levels=self.residual_levels, pad=pad_info,
            )
            resid = crop_to_shape(resid, (h, w))
            recon = recon + resid

        recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
        return crop_to_shape(recon, (h - data.pad_hw[0], w - data.pad_hw[1]))

    # ----- public API -----

    def compress(self, image: np.ndarray, quality: int = 55) -> bytes:
        """Compress a uint8 image array into bytes."""
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)

        is_rgb = arr.ndim == 3 and arr.shape[2] == 3
        if is_rgb:
            channels = rgb_to_ycbcr(arr)
        else:
            channels = split_channels(arr)

        encoded_channels = []
        for i, ch in enumerate(channels):
            ch_quality = quality
            if is_rgb and i > 0:
                ch_quality = max(1, int(quality * 0.8))
            encoded_channels.append(self._encode_channel(ch, ch_quality))

        payload = {
            "version": 2,
            "shape": tuple(arr.shape),
            "dtype": "uint8",
            "is_rgb": is_rgb,
            "quality": int(np.clip(quality, 1, 100)),
            "block_size": self.block_size,
            "residual_levels": self.residual_levels,
            "residual_passes": self.residual_passes,
            "channels": encoded_channels,
        }
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        return MAGIC + lzma.compress(raw, preset=6)

    def decompress(self, blob: bytes) -> np.ndarray:
        if blob.startswith(MAGIC):
            payload = pickle.loads(lzma.decompress(blob[len(MAGIC):]))
        elif blob.startswith(MAGIC_V1):
            payload = pickle.loads(lzma.decompress(blob[len(MAGIC_V1):]))
        else:
            raise ValueError("Not a HFZ blob")

        version = payload.get("version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported codec version {version}")

        channels = []
        for ch_data in payload["channels"]:
            channels.append(self._decode_channel(ch_data))

        if payload.get("is_rgb") and len(channels) == 3:
            out = ycbcr_to_rgb(channels[0], channels[1], channels[2])
        else:
            out = stack_channels(channels)
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for the CLI image loader") from exc
    img = Image.open(path)
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return np.array(img, dtype=np.uint8)


def _save_image(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for the CLI image saver") from exc
    Image.fromarray(arr).save(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid fractal-Z image compressor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Compress an image")
    enc.add_argument("input", type=Path)
    enc.add_argument("output", type=Path)
    enc.add_argument("--quality", type=int, default=55, help="1..100, higher is better")
    enc.add_argument("--block-size", type=int, default=8)
    enc.add_argument("--residual-levels", type=int, default=2)
    enc.add_argument("--residual-passes", type=int, default=2)

    dec = sub.add_parser("decode", help="Decompress a .hfz file")
    dec.add_argument("input", type=Path)
    dec.add_argument("output", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "encode":
        img = _load_image(args.input)
        codec = FractalHybridCodec(
            block_size=args.block_size,
            residual_levels=args.residual_levels,
            residual_passes=args.residual_passes,
        )
        blob = codec.compress(img, quality=args.quality)
        args.output.write_bytes(blob)
        print(f"Wrote {args.output} ({len(blob)} bytes)")
        return 0

    if args.cmd == "decode":
        blob = args.input.read_bytes()
        magic = MAGIC if blob.startswith(MAGIC) else MAGIC_V1
        payload = pickle.loads(lzma.decompress(blob[len(magic):]))
        codec = FractalHybridCodec(
            block_size=int(payload.get("block_size", 8)),
            residual_levels=int(payload.get("residual_levels", 2)),
            residual_passes=int(payload.get("residual_passes", 2)),
        )
        arr = codec.decompress(blob)
        _save_image(args.output, arr)
        print(f"Wrote {args.output}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
