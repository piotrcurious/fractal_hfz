"""Hybrid fractal-Z image codec  (v3.4).

A transform-domain image compressor built around three layers:

1) Block DCT — orientation-steered quantisation aligned with each block's
   dominant frequency axis (Stage 2 → Stage 1 q-matrix).
2) Directional-state layer — orientation-guided Morton fractal traversal and
   multi-position DPCM (DC + first two AC zig-zag positions delta-coded along
   the traversal for entropy reduction).
3) Haar-wavelet residual — subband-selective scale map; optionally operates on
   the tighter fractal-prediction residual when Stage 2.5 fires.

v3.4 additions:
- Extended-Basis Overlap-Add (EBOA): DCT basis functions are evaluated on an
  extended grid (e.g., 10x10 or 12x12 instead of 8x8) and blended using a
  partition-of-unity window. This eliminates block artifacts in Stage 1 and
  allows the multi-pass Haar stage to focus on true image detail rather than
  correcting block boundaries, significantly improving compression ratio.

The codec is self-contained (NumPy + stdlib only) and hackable.

Usage:
    python fractal_hfz_codec.py encode input.png output.hfz --quality 55
    python fractal_hfz_codec.py decode output.hfz reconstructed.png

File format: LZMA-compressed pickle blob with 4-byte magic header (HFZ4).
Backward-compatible with HFZ3, HFZ2, and HFZ1 blobs.
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

MAGIC    = b"HFZ4"   # v3.4: Extended-Basis Overlap-Add (EBOA)
MAGIC_V2 = b"HFZ2"   # v2: multi-position DPCM + orientation steering
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


_IDCT_EXT_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def idct_matrix_extended(n_coeffs: int, n_out: int) -> np.ndarray:
    """Matrix for evaluating n_coeffs DCT-II basis functions at n_out points."""
    key = (n_coeffs, n_out)
    if key not in _IDCT_EXT_CACHE:
        # basis i is evaluation of cos(pi * (2*j + 1) * i / (2*n_coeffs))
        # but we evaluate it at j in range [0, n_out)
        c = np.zeros((n_out, n_coeffs), dtype=np.float64)
        scale0 = math.sqrt(1.0 / n_coeffs)
        scale = math.sqrt(2.0 / n_coeffs)
        for i in range(n_coeffs):
            alpha = scale0 if i == 0 else scale
            for j in range(n_out):
                c[j, i] = alpha * math.cos(math.pi * (2 * (j - (n_out - n_coeffs) / 2.0) + 1) * i / (2 * n_coeffs))
        _IDCT_EXT_CACHE[key] = c
    return _IDCT_EXT_CACHE[key]


def idct2_extended(coeffs: np.ndarray, n_out: int) -> np.ndarray:
    """Apply 2D IDCT with extended basis evaluation.

    Takes (N, n, n) coefficients and returns (N, n_out, n_out) blocks.
    The output is centered on the original (n, n) region.
    """
    coeffs = _as_float64(coeffs)
    n_coeffs = coeffs.shape[-1]
    c = idct_matrix_extended(n_coeffs, n_out)
    return c @ coeffs @ c.T


def get_eboa_window_2d(n_out: int, n_bs: int) -> np.ndarray:
    """Create a 2D partition-of-unity window for EBOA.
    Uses a matched sin^2/cos^2 ramp that sums to 1.0 when overlapped at stride n_bs.
    """
    pad = (n_out - n_bs) // 2
    if pad <= 0:
        return np.ones((n_out, n_out))

    w1 = np.zeros(n_out)
    # The total width is n_bs + 2*pad.
    # Stride is n_bs.
    # Overlap regions are [0, 2*pad] and [n_bs, n_bs + 2*pad].
    # Middle region [2*pad, n_bs] is exclusive.

    ov = 2 * pad
    for i in range(ov):
        # Quadratic-smooth ramp (sin^2)
        val = math.sin(math.pi / 2.0 * (i + 0.5) / ov) ** 2
        w1[i] = val
        w1[n_out - 1 - i] = val

    if n_out > 2 * ov:
        w1[ov : n_out - ov] = 1.0

    return w1[:, None] * w1[None, :]


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




# ============================================================================
# Hilbert-curve traversal
# ============================================================================
#
# Two complementary traversals are provided and adaptively combined:
#
# 1. build_hilbert_order_flat — vectorised O(N log N) flat Hilbert curve.
#    Uses the standard d2xy bit-manipulation; provably achieves L∞=1 adjacency
#    between every consecutive block pair.  A D4 orientation transform aligns
#    the traversal direction with the dominant gradient.
#
# 2. build_fractal_order — the original orientation-steered Morton recursion.
#    Creates spatially clustered groups that LZMA can model efficiently for
#    regular / structured content.
#
# build_hilbert_order runs both traversals, computes the DC coherence of each
# (L1 norm of first-differences of the DC sequence), and returns whichever
# ordering produces the more predictable (lower L1) DC stream.  This adaptive
# choice is cost-free: no bitstream signalling is needed because the decoder
# runs the same metric on the same data and makes the identical decision.


def _hilbert_all_d2xy(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Hilbert d → (row, col) for all d in [0, n²), n = power of 2.

    Applies the standard d2xy bit-manipulation algorithm to every Hilbert
    index simultaneously via NumPy — O(N·log₂N) total work with only log₂(n)
    Python-level loop iterations.

    Correctness guarantee: every consecutive pair in the resulting order has
    L∞(Δrow, Δcol) = 1 (proven by the Hilbert construction).
    """
    N = n * n
    t = np.arange(N, dtype=np.int64)
    x = np.zeros(N, dtype=np.int64)
    y = np.zeros(N, dtype=np.int64)
    s = 1
    while s < n:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        m0 = ry == 0
        m1 = m0 & (rx == 1)
        x[m1] = s - 1 - x[m1]
        y[m1] = s - 1 - y[m1]
        x[m0], y[m0] = y[m0].copy(), x[m0].copy()
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return y, x   # (row, col)


def build_hilbert_order_flat(ny: int, nx: int, init_state: int) -> List[Tuple[int, int]]:
    """Pure flat Hilbert order for ny×nx grid with D4 orientation (state 0–7).

    States 0-3 are 90° rotations; states 4-7 add a horizontal reflection.
    Non-power-of-2 grids are handled by padding to the next power-of-2 square
    and filtering out-of-range cells.
    """
    n = 1
    while n < max(ny, nx):
        n <<= 1

    rows, cols = _hilbert_all_d2xy(n)
    m = n - 1
    s8 = init_state % 8
    if   s8 == 0: tr, tc = rows,         cols
    elif s8 == 1: tr, tc = cols.copy(),  m - rows
    elif s8 == 2: tr, tc = m - rows,     m - cols
    elif s8 == 3: tr, tc = m - cols,     rows.copy()
    elif s8 == 4: tr, tc = rows.copy(),  m - cols
    elif s8 == 5: tr, tc = m - cols,     m - rows
    elif s8 == 6: tr, tc = m - rows,     cols.copy()
    else:         tr, tc = cols.copy(),  rows.copy()

    valid = (tr >= 0) & (tr < ny) & (tc >= 0) & (tc < nx)
    tr_v  = tr[valid].astype(int)
    tc_v  = tc[valid].astype(int)

    seen  = np.zeros((ny, nx), dtype=bool)
    order: List[Tuple[int, int]] = []
    for r, c in zip(tr_v, tc_v):
        if not seen[r, c]:
            order.append((r, c))
            seen[r, c] = True
    for r in range(ny):
        for c in range(nx):
            if not seen[r, c]:
                order.append((r, c))
    return order


def build_hilbert_order(
    state_grid: np.ndarray,
    energy_grid: np.ndarray,
    coeff_blocks: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Adaptive traversal: best of flat Hilbert and Morton-like recursion.

    Both orders are evaluated on the DC coherence metric
        L1 = Σ |DC[i+1] − DC[i]|
    along the traversal.  The order with lower L1 is returned, ensuring the
    multi-position DPCM stage always operates on the most predictable sequence.

    *coeff_blocks* (N, bs, bs) provides exact DC values.  When not supplied,
    energy_grid is used as a proxy (same decision in most cases).
    """
    ny, nx = state_grid.shape
    dom_state = int(_weighted_mode(state_grid.ravel(), energy_grid.ravel() + 1e-6))

    order_h = build_hilbert_order_flat(ny, nx, dom_state)
    order_m = build_fractal_order(state_grid, energy_grid)

    if coeff_blocks is not None:
        dc = coeff_blocks[:, 0, 0]
    else:
        dc = energy_grid.ravel()

    idx_h = np.array([r * nx + c for r, c in order_h], dtype=np.int32)
    idx_m = np.array([r * nx + c for r, c in order_m], dtype=np.int32)
    l1_h  = float(np.sum(np.abs(np.diff(dc.ravel()[idx_h]))))
    l1_m  = float(np.sum(np.abs(np.diff(dc.ravel()[idx_m]))))

    return order_h if l1_h <= l1_m else order_m


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




# ============================================================================
# Fractal Engine v3 — Z-order affine prediction with progressive decomposition
# ============================================================================
#
# DESIGN
# ------
# 1. DOMAIN POOL IN MORTON Z-ORDER
#    Domain blocks are enumerated by Morton (Z-curve) index, so consecutive pool
#    entries are spatially adjacent.  The fractal_domain_idx stream along the
#    Morton-ordered traversal of range blocks therefore has slowly-varying values
#    → strong DPCM compression without any extra spatial bookkeeping.
#
# 2. FULL D4 AFFINE TRANSFORMS (8 variants)
#    Each domain block is pre-processed with all 8 elements of the dihedral group
#    D4 (4 rotations × 2 reflections) applied in pixel space before the DCT.
#    Doubling from 4 to 8 variants costs nothing extra in storage and substantially
#    improves prediction quality for directional textures and edges.
#
# 3. AFFINE PREDICTION: R ≈ α·T(D) + β  (contrast + brightness)
#    The previous engine only fitted α (contrast scale), leaving the DC residual
#    large when domain and range blocks have different means.  Adding β
#    (brightness offset, applied only to the DC coefficient in DCT space) gives
#    exact DC match after prediction → Haar LL subband residual collapses to near
#    zero for smooth content.
#
#    Optimal α = (R_ac · D_ac) / ‖D_ac‖²     (AC patch, excludes DC)
#    Optimal β = R_dc − α · D_dc              (closes DC gap exactly)
#
# 4. PROGRESSIVE FRACTAL RESIDUAL DECOMPOSITION
#    Pass 1: full affine (α, β, domain, D4) on original ordered DCT blocks
#            → fractal_residual_1
#    Pass 2: contrast-only (α, domain, D4) on fractal_residual_1
#            (β ≈ 0 since DC was zeroed by pass 1)
#            → fractal_residual_2
#    Stage 3: Haar wavelet on fractal_residual_2  (much tighter than raw residual)
#
#    Each pass engages only when its measured gain exceeds the respective threshold.
#
# 5. Z-ORDER NEIGHBOURHOOD SEARCH
#    For each range block at traversal step i, first evaluate domain candidates in
#    window [i//4 − W, i//4 + W] (all 8 D4 variants).  Full-grid search is used
#    only for blocks where the neighbourhood gain is insufficient.
#
# PAYLOAD OVERHEAD (when both passes engage)
#    Pass 1: domain_idx int16 + alpha_q int8 + beta_q int8  = 4 bytes/block
#    Pass 2: domain_idx int16 + alpha_q int8                = 3 bytes/block
#    Both streams are DPCM-coded.  Domain pool is deterministic → no stored state.

_FRACTAL_K          = 4       # DCT patch dimension for matching (k×k)
_FRACTAL_TRANSFORMS = 8       # |D4| = 8 affine variants per domain block
_FRACTAL_ALPHA_MAX  = 2.0
_FRACTAL_ALPHA_HALF = 127
_FRACTAL_BETA_MAX   = 64.0    # DC-only offset: max 64 DCT units ≈ 64 pixel-mean
_FRACTAL_BETA_HALF  = 127
_FRACTAL_MIN_POOL   = 8       # minimum domain blocks to engage
_FRACTAL_MIN_BLOCKS = 256     # minimum range blocks to justify overhead
_FRACTAL_MAX_OPS    = 8_000_000  # N × D cap for batch matrix multiply
_FRACTAL_NBHD_W     = 20      # z-order neighbourhood half-width (base blocks)
_FRACTAL_GAIN_P1    = 1.60    # pass-1 must reduce mean|coeff| by ≥ 37.5 %
_FRACTAL_GAIN_P2    = 1.20    # pass-2 threshold (residuals are already smaller)


# ── Morton Z-order helpers ────────────────────────────────────────────────────

def _morton_encode_2d(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorised 2D Morton (Z-curve) encode for integer arrays rows, cols.

    Returns int64 array of same shape via bit-interleaving:
        bit 2i   ← bit i of col
        bit 2i+1 ← bit i of row
    """
    r = rows.astype(np.int64)
    c = cols.astype(np.int64)
    max_bits = int(max(int(r.max()), int(c.max()), 1)).bit_length()
    d = np.zeros_like(r, dtype=np.int64)
    for i in range(max_bits):
        d |= ((c >> i) & 1) << (2 * i)
        d |= ((r >> i) & 1) << (2 * i + 1)
    return d


def _morton_order(ny: int, nx: int) -> np.ndarray:
    """Return flat block indices in Morton Z-curve order for an ny×nx grid."""
    row_idx, col_idx = np.mgrid[0:ny, 0:nx]
    morton = _morton_encode_2d(row_idx.ravel(), col_idx.ravel())
    return np.argsort(morton).astype(np.int32)   # (ny*nx,) sorted flat indices


# ── D4 pixel-domain transforms ────────────────────────────────────────────────

def _d4_apply(px: np.ndarray, t: int) -> np.ndarray:
    """Apply D4 element t ∈ {0..7} to batch of square blocks (N, n, n).

    0 = identity           4 = horizontal flip
    1 = 90° CW rotation    5 = vertical flip
    2 = 180° rotation      6 = main-diagonal transpose
    3 = 270° CW rotation   7 = anti-diagonal transpose
    """
    if   t == 0: return px.copy()
    elif t == 1: return np.rot90(px, k=1, axes=(1, 2))
    elif t == 2: return np.rot90(px, k=2, axes=(1, 2))
    elif t == 3: return np.rot90(px, k=3, axes=(1, 2))
    elif t == 4: return px[:, :, ::-1].copy()
    elif t == 5: return px[:, ::-1, :].copy()
    elif t == 6: return np.swapaxes(px, 1, 2).copy()
    else:        return np.swapaxes(px[:, ::-1, ::-1], 1, 2).copy()


# ── Domain pool construction ──────────────────────────────────────────────────

def _build_fractal_domain_pool(
    recon: np.ndarray,
    bs: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Build the affine D4 fractal domain pool in Morton Z-order.

    Each domain block is a 2×-downsampled (average-pooled) 2bs×2bs region of the
    Stage-1 reconstruction.  The base blocks are sorted by Morton index so that
    consecutive pool entries are spatially adjacent — this makes the domain_idx
    stream DPCM-compressible when range blocks are also visited in spatially
    coherent order.

    All 8 D4 transforms are appended along axis 0, giving D_base*8 total entries.
    D4 blocks for transform t occupy pool[t * D_base : (t+1) * D_base].

    Returns
    -------
    pool_dct   : float64 (D, bs, bs)  — D = D_base * 8
    pool_dc    : float64 (D,)          — DC coefficient of each pool entry
    D_base     : int                   — number of base domain blocks
    """
    h, w = recon.shape
    dbs = bs * 2
    dny, dnx = h // dbs, w // dbs
    D_base = dny * dnx
    if D_base < _FRACTAL_MIN_POOL:
        return None

    # Enumerate domain blocks in Morton Z-order
    morton_order = _morton_order(dny, dnx)          # (D_base,) flat indices in Z-order
    dy_all = (morton_order // dnx).astype(int)
    dx_all = (morton_order %  dnx).astype(int)

    base_px = np.empty((D_base, bs, bs), dtype=np.float64)
    for k in range(D_base):
        dy, dx = dy_all[k], dx_all[k]
        r = recon[dy * dbs:(dy + 1) * dbs, dx * dbs:(dx + 1) * dbs]
        base_px[k] = (
            r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2] + r[1::2, 1::2]
        ) * 0.25 - 128.0

    # All 8 D4 transforms, then batch-DCT the full pool
    pool_px  = np.concatenate([_d4_apply(base_px, t) for t in range(_FRACTAL_TRANSFORMS)], axis=0)
    pool_dct = dct2(pool_px)                # (D_base*8, bs, bs)
    pool_dc  = pool_dct[:, 0, 0].copy()    # (D_base*8,)

    return pool_dct, pool_dc, D_base


# ── Z-order neighbourhood map ─────────────────────────────────────────────────

def _make_nbhd_map(N: int, D_base: int, nbhd_w: int = _FRACTAL_NBHD_W) -> np.ndarray:
    """For each of N range blocks (visited in traversal order), return the
    Z-order neighbourhood of candidate pool indices (all 8 D4 variants).

    Range block i ↔ domain base block near i // 4 in Z-order (because domain
    blocks are 2× the size of range blocks, one domain covers ≈4 range cells).

    Returns int32 (N, W_total) where W_total = 2*nbhd_w * _FRACTAL_TRANSFORMS.
    """
    W = nbhd_w
    W_total = 2 * W * _FRACTAL_TRANSFORMS
    nbhd = np.empty((N, W_total), dtype=np.int32)
    D_total = D_base * _FRACTAL_TRANSFORMS
    for i in range(N):
        d_ctr = i // 4
        lo = max(0, d_ctr - W)
        hi = min(D_base - 1, d_ctr + W - 1)
        base = np.arange(lo, hi + 1, dtype=np.int32)
        cands = np.concatenate([base + t * D_base for t in range(_FRACTAL_TRANSFORMS)])
        cands = np.clip(cands, 0, D_total - 1)
        if len(cands) >= W_total:
            nbhd[i] = cands[:W_total]
        else:
            # Tile to fill
            reps = (W_total + len(cands) - 1) // len(cands)
            nbhd[i] = np.tile(cands, reps)[:W_total]
    return nbhd


# ── Affine prediction ─────────────────────────────────────────────────────────

def _affine_predict(
    range_dct:  np.ndarray,      # (N, bs, bs)
    pool_dct:   np.ndarray,      # (D, bs, bs)
    pool_dc:    np.ndarray,      # (D,)
    D_base:     int,
    nbhd:       np.ndarray,      # (N, W) neighbourhood candidate indices
    with_beta:  bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """Find best (domain, D4 transform, α, β) for each range block.

    Affine model in DCT space:
        P[0,0]   = α · D[0,0] + β      (DC: exact if β stored)
        P[u,v]   = α · D[u,v]           (AC: scaled domain)

    Optimal α (from AC patch, k×k, excludes DC):
        α* = (R_ac · D_ac) / ‖D_ac‖²
    Optimal β (closes DC gap):
        β* = R[0,0] − α* · D[0,0]

    Two-stage search:
      1. Neighbourhood (fast): evaluate only spatially nearby candidates.
      2. Full-grid: applied to blocks whose neighbourhood best-fit is poor
         (residual > 60 % of original) AND N*D ≤ _FRACTAL_MAX_OPS.

    Returns
    -------
    best_idx : (N,) int16  — combined pool index
    alpha_q  : (N,) int8
    beta_q   : (N,) int8 or None (when with_beta=False)
    residual : (N, bs, bs)
    gain     : float
    """
    N, bs = range_dct.shape[0], range_dct.shape[-1]
    D = pool_dct.shape[0]
    k = min(_FRACTAL_K, bs)

    # AC patch (zero out DC before matching)
    r_ac = range_dct.copy();  r_ac[:, 0, 0] = 0.0
    d_ac = pool_dct.copy();   d_ac[:, 0, 0] = 0.0
    r_flat = r_ac[:, :k, :k].reshape(N, k * k)     # (N, k²)
    d_flat = d_ac[:, :k, :k].reshape(D, k * k)     # (D, k²)
    d2     = np.einsum("ij,ij->i", d_flat, d_flat) + 1e-9  # (D,)
    r2     = np.einsum("ij,ij->i", r_flat, r_flat)          # (N,)
    r_dc   = range_dct[:, 0, 0]                    # (N,)

    best_idx   = np.zeros(N, dtype=np.int32)
    best_alpha = np.zeros(N, dtype=np.float64)
    best_energy = np.full(N, np.inf)

    # ── Pass 1: neighbourhood search ─────────────────────────────────────────
    W = nbhd.shape[1]
    if N * W <= _FRACTAL_MAX_OPS:
        for i in range(N):
            cands  = nbhd[i]                                # (W,)
            d_c    = d_flat[cands]                          # (W, k²)
            d2_c   = d2[cands]                              # (W,)
            rd     = (r_flat[i] * d_c).sum(axis=1)         # (W,)
            alpha  = rd / d2_c                              # (W,)
            e      = r2[i] - alpha * rd                     # (W,) residual energy
            j      = int(np.argmin(e))
            best_idx[i]    = int(cands[j])
            best_alpha[i]  = alpha[j]
            best_energy[i] = e[j]

    # ── Pass 2: full-grid search for poorly-matched blocks ───────────────────
    poor = best_energy > 0.6 * r2        # (N,) mask
    n_poor = int(poor.sum())
    if n_poor > 0 and n_poor * D <= _FRACTAL_MAX_OPS:
        gi     = np.where(poor)[0]
        rg     = r_flat[gi]              # (Ng, k²)
        rd_g   = rg @ d_flat.T           # (Ng, D)
        alp_g  = rd_g / d2               # (Ng, D)
        e_g    = r2[gi, None] - alp_g * rd_g  # (Ng, D)
        j_g    = np.argmin(e_g, axis=1)  # (Ng,)
        best_idx[gi]   = j_g
        best_alpha[gi] = alp_g[np.arange(n_poor), j_g]

    # ── Quantise α and β ──────────────────────────────────────────────────────
    alpha_q = np.clip(
        np.round(best_alpha * _FRACTAL_ALPHA_HALF / _FRACTAL_ALPHA_MAX),
        -_FRACTAL_ALPHA_HALF, _FRACTAL_ALPHA_HALF,
    ).astype(np.int8)
    alpha_r = alpha_q.astype(np.float64) * _FRACTAL_ALPHA_MAX / _FRACTAL_ALPHA_HALF

    if with_beta:
        beta_raw = r_dc - alpha_r * pool_dc[best_idx]   # (N,)
        beta_q   = np.clip(
            np.round(beta_raw * _FRACTAL_BETA_HALF / _FRACTAL_BETA_MAX),
            -_FRACTAL_BETA_HALF, _FRACTAL_BETA_HALF,
        ).astype(np.int8)
        beta_r   = beta_q.astype(np.float64) * _FRACTAL_BETA_MAX / _FRACTAL_BETA_HALF
    else:
        beta_q = None
        beta_r = np.zeros(N, dtype=np.float64)

    # ── Build residual ────────────────────────────────────────────────────────
    idx = best_idx.astype(np.int64)
    pred  = pool_dct[idx] * alpha_r[:, None, None]
    pred[:, 0, 0] += beta_r          # DC adjustment (β is in DCT units = pixel-mean)
    residual = range_dct - pred

    orig_mag = float(np.mean(np.abs(range_dct)))
    res_mag  = float(np.mean(np.abs(residual)))
    gain     = orig_mag / max(res_mag, 1e-9)

    return best_idx.astype(np.int16), alpha_q, beta_q, residual, gain


# ── Reconstruction ────────────────────────────────────────────────────────────

def _fractal_reconstruct_affine(
    pool_dct:   np.ndarray,             # (D, bs, bs)
    domain_idx: np.ndarray,             # (N,) int16
    alpha_q:    np.ndarray,             # (N,) int8
    beta_q:     Optional[np.ndarray],   # (N,) int8 or None
) -> np.ndarray:
    """Reconstruct predicted DCT blocks from stored (domain_idx, α, β)."""
    alpha = alpha_q.astype(np.float64) * _FRACTAL_ALPHA_MAX / _FRACTAL_ALPHA_HALF
    idx   = np.clip(domain_idx.astype(np.int64), 0, pool_dct.shape[0] - 1)
    pred  = pool_dct[idx] * alpha[:, None, None]
    if beta_q is not None:
        beta      = beta_q.astype(np.float64) * _FRACTAL_BETA_MAX / _FRACTAL_BETA_HALF
        pred[:, 0, 0] += beta
    return pred


def _eboa_reconstruct_image(
    coeff_blocks: np.ndarray,
    order: np.ndarray,
    ny: int,
    nx: int,
    bs: int,
    pad: int = 1,
) -> np.ndarray:
    """Reconstruct image using Extended-Basis Overlap-Add (EBOA).

    Evaluates DCT basis functions on an extended grid and blends them using a
    partition-of-unity window. This eliminates block artifacts and creates
    a smoother base for residual coding.
    """
    N = len(coeff_blocks)
    n_out = bs + 2 * pad

    # Eval basis functions on extended grid
    ext_blocks = idct2_extended(coeff_blocks, n_out)

    # Apply partition-of-unity window
    win = get_eboa_window_2d(n_out, bs)
    ext_blocks *= win[None, :, :]

    # Unorder blocks
    unordered_ext = np.zeros_like(ext_blocks)
    unordered_ext[order] = ext_blocks

    # Accumulate into image
    h, w = ny * bs, nx * bs
    img = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            block = unordered_ext[i * nx + j]
            img[i * bs : i * bs + n_out, j * bs : j * bs + n_out] += block

    # Crop to original size (centered)
    return img[pad : h + pad, pad : w + pad]


def _dpcm_encode(vals: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Conditionally delta-code an integer sequence.

    Stores vals[0] verbatim; vals[1:] are stored as first-differences.
    DPCM is applied only when L1(diffs[1:]) < L1(vals[1:]).

    Layout: [vals[0], vals[1]-vals[0], vals[2]-vals[1], ...]
    Decode: cumsum of layout → original values.
    """
    v = vals.astype(np.int32)
    coded = np.empty_like(v)
    coded[0]  = v[0]
    if len(v) > 1:
        coded[1:] = v[1:] - v[:-1]
    if len(v) > 1 and np.sum(np.abs(coded[1:])) < np.sum(np.abs(v[1:])):
        return coded.astype(np.int16), True
    return v.astype(np.int16), False


def _dpcm_decode(vals: np.ndarray, applied: bool) -> np.ndarray:
    """Invert _dpcm_encode: cumsum([v0, d1, d2, ...]) → [v0, v0+d1, ...]."""
    if not applied:
        return vals.astype(np.int16)
    return np.cumsum(vals.astype(np.int16), dtype=np.int16)


# ----------------------------------------------------------------------------
# Codec data structures
# ----------------------------------------------------------------------------


@dataclass
class ChannelCodecResult:
    """Per-channel codec result stored in the compressed payload.

    The two fractal fields are excluded from pickle when None (the common case)
    via __getstate__/__setstate__, preventing any overhead to non-fractal channels.
    """
    padded_shape: Tuple[int, int]
    pad_hw: Tuple[int, int]
    block_size: int
    quality: int
    order: np.ndarray
    state_grid: np.ndarray    # uint8  (ny, nx): orientation bin per block
    mode_grid: np.ndarray     # uint8  (ny, nx): 0=smooth 1=dir 2=textured
    dct_qcoeffs_packed: np.ndarray   # int16[nblocks*n*n + N_DPCM_POSITIONS]
    residual_pads: List[Tuple[int, int]]
    residual_qsteps: List[float]
    residual_coeffs: List[np.ndarray]   # each [Hp, Wp]
    # ── Fractal prediction pass 1 (affine: α, β, domain, D4) ────────────────
    frac1_idx:   Optional[np.ndarray] = None   # (N,) int16 DPCM-coded domain index
    frac1_alpha: Optional[np.ndarray] = None   # (N,) int8
    frac1_beta:  Optional[np.ndarray] = None   # (N,) int8  DC offset
    frac1_idx_dpcm:   bool = False
    frac1_alpha_dpcm: bool = False
    frac1_beta_dpcm:  bool = False

    # ── Fractal prediction pass 2 (contrast-only: α, domain, D4) ─────────────
    frac2_idx:   Optional[np.ndarray] = None   # (N,) int16 DPCM-coded
    frac2_alpha: Optional[np.ndarray] = None   # (N,) int8
    frac2_idx_dpcm:   bool = False
    frac2_alpha_dpcm: bool = False

    _FRAC_NONE_KEYS = (
        "frac1_idx","frac1_alpha","frac1_beta",
        "frac2_idx","frac2_alpha",
    )
    _FRAC_BOOL_KEYS = (
        "frac1_idx_dpcm","frac1_alpha_dpcm","frac1_beta_dpcm",
        "frac2_idx_dpcm","frac2_alpha_dpcm",
    )

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        for k in self._FRAC_NONE_KEYS:
            if d.get(k) is None: d.pop(k, None)
        for k in self._FRAC_BOOL_KEYS:
            if not d.get(k, False): d.pop(k, None)
        return d

    def __setstate__(self, state: dict) -> None:
        for k in ("frac1_idx","frac1_alpha","frac1_beta","frac2_idx","frac2_alpha"):
            state.setdefault(k, None)
        for k in ("frac1_idx_dpcm","frac1_alpha_dpcm","frac1_beta_dpcm",
                  "frac2_idx_dpcm","frac2_alpha_dpcm",
                  # compat with v3.0 field names
                  "fractal_domain_idx","fractal_alpha_q",
                  "fractal_idx_dpcm","fractal_aq_dpcm"):
            state.setdefault(k, False if "dpcm" in k else None)
        self.__dict__.update(state)


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
        hf_energy = (
            np.sum(np.abs(coeff_blocks[:, 2:, 2:]), axis=(1, 2))
            if bs > 2 else np.zeros(len(coeff_blocks))
        )

        # ---- Stage 2: orientation state + orientation-steered Morton traversal ----
        # Main coefficient block ordering uses the Morton-like recursive traversal
        # (build_fractal_order) which consistently gives better LZMA compression than
        # the Hilbert curve for structured images: it clusters blocks into spatial
        # quadrants, enabling long LZ-match runs for regions with repeated patterns.
        # The Hilbert curve is used in Stage 2.5 for domain pool spatial ordering.
        states, anis = block_orientation_state(flat_blocks)
        modes = _mode_from_features(total_energy, hf_energy, anis, quality)

        state_grid  = states.reshape(ny, nx)
        mode_grid   = modes.reshape(ny, nx)
        energy_grid = (hf_energy + 1e-6).reshape(ny, nx)

        order_xy = build_fractal_order(state_grid, energy_grid)
        order    = np.array([y * nx + x for (y, x) in order_xy], dtype=np.int32)

        ordered_coeffs = coeff_blocks[order]
        ordered_modes  = modes[order]
        ordered_states = states[order]

        # ---- Stage 1 quantisation: orientation-steered DCT q-matrix ----
        dct_qcoeffs = _quantize_dct(
            ordered_coeffs, quality, ordered_modes, states=ordered_states,
        )

        # ---- Multi-position DPCM along Hilbert order (Stage 2 → entropy) ----
        dct_qcoeffs_stored, dpcm_flags = _apply_multi_dpcm(dct_qcoeffs, bs)

        # ---- Stage-1 reconstruction (needed for fractal pool + Haar base) ----
        recon_coeffs = _dequantize_dct(
            dct_qcoeffs, quality, ordered_modes, states=ordered_states,
        )

        # EBOA-based Stage-1 reconstruction (masking function to extend beyond boundaries)
        # By blending basis functions across boundaries, we create a much smoother base
        # for Stage-3 refinement, allowing the Haar passes to compress more efficiently.
        recon1 = _eboa_reconstruct_image(recon_coeffs, order, ny, nx, bs, pad=1) + 128.0

        # ---- Stage 2.5: Progressive fractal DCT prediction ----
        # Pass 1: Affine (α, β, D4 transform, domain) on ordered DCT blocks.
        # Pass 2: Contrast-only (α, D4, domain) on Pass-1 residual.
        # Haar (Stage 3) then refines the remaining fractal residual.
        # Domain pool is Morton-ordered and reproduced deterministically at decode.
        _f1_idx = _f1_alpha = _f1_beta = None
        _f1_idx_dpcm = _f1_alpha_dpcm = _f1_beta_dpcm = False
        _f2_idx = _f2_alpha = None
        _f2_idx_dpcm = _f2_alpha_dpcm = False
        current = recon1.copy()

        N_blocks = ny * nx
        pool_result = _build_fractal_domain_pool(recon1, bs)
        if pool_result is not None and N_blocks >= _FRACTAL_MIN_BLOCKS:
            pool_dct, pool_dc, D_base = pool_result
            nbhd = _make_nbhd_map(N_blocks, D_base)

            # ── Pass 1: affine prediction (with brightness offset β) ──────────
            p1_idx, p1_alpha, p1_beta, p1_resid, _gain1 = _affine_predict(
                ordered_coeffs, pool_dct, pool_dc, D_base, nbhd, with_beta=True
            )

            # Engage only when the Haar residual after fractal prediction is
            # smaller than before — measured directly in coefficient L1 norm.
            # Also deduct estimated payload cost (4 bytes/block for pass 1)
            # scaled by the LZMA factor (~2× compression on DPCM-coded integers)
            # so the test reflects actual bitstream savings.
            #
            # This check avoids the common failure mode where fractal prediction
            # looks good by DCT gain but increases the Haar residual (e.g. linear
            # gradients whose domain blocks cancel each other badly at DC).
            pred_dct_p1 = _fractal_reconstruct_affine(pool_dct, p1_idx, p1_alpha, p1_beta)
            # Use EBOA for fractal prediction too
            recon_frac1 = _eboa_reconstruct_image(pred_dct_p1, order, ny, nx, bs, pad=1) + 128.0

            resid_pre  = padded - recon1
            resid_post = padded - recon_frac1
            haar_pre, _ = haar_dwt2(pad_to_multiple(resid_pre, 2**self.residual_levels)[0],
                                     self.residual_levels)
            haar_post,_ = haar_dwt2(pad_to_multiple(resid_post,2**self.residual_levels)[0],
                                     self.residual_levels)
            haar_l1_pre  = float(np.sum(np.abs(haar_pre)))
            haar_l1_post = float(np.sum(np.abs(haar_post)))
            # Each Haar coefficient is ~2 bytes; payload is 4 bytes/block for pass 1
            payload_equiv = N_blocks * 4 * (haar_l1_pre / max(np.sum(np.abs(p1_resid)) * 2.0, 1e-9))
            haar_saving  = haar_l1_pre - haar_l1_post

            if haar_saving > haar_l1_pre * 0.20:   # need ≥20% Haar L1 reduction
                _f1_idx,   _f1_idx_dpcm   = _dpcm_encode(p1_idx)
                _f1_alpha, _f1_alpha_dpcm = _dpcm_encode(p1_alpha)
                _f1_beta,  _f1_beta_dpcm  = _dpcm_encode(p1_beta)
                current = recon_frac1

                # ── Pass 2: contrast-only on pass-1 Haar residual ─────────────
                p2_idx, p2_alpha, _, p2_resid, _gain2 = _affine_predict(
                    p1_resid, pool_dct, pool_dc, D_base, nbhd, with_beta=False
                )
                pred_dct_p2 = _fractal_reconstruct_affine(pool_dct, p2_idx, p2_alpha, None)
                # Use EBOA for combined fractal prediction
                recon_frac12 = _eboa_reconstruct_image(pred_dct_p1 + pred_dct_p2, order, ny, nx, bs, pad=1) + 128.0
                resid_post2  = padded - recon_frac12
                haar_post2,_ = haar_dwt2(pad_to_multiple(resid_post2,2**self.residual_levels)[0],
                                          self.residual_levels)
                haar_l1_post2 = float(np.sum(np.abs(haar_post2)))

                if haar_l1_post2 < haar_l1_post * 0.90:   # pass 2 gives ≥10% extra
                    _f2_idx,   _f2_idx_dpcm   = _dpcm_encode(p2_idx)
                    _f2_alpha, _f2_alpha_dpcm = _dpcm_encode(p2_alpha)
                    current = recon_frac12

        # ---- Stage 3: Haar wavelet residual coding ----
        smooth_fraction = float(np.mean(modes == 0))
        global_state    = _weighted_mode(states, (hf_energy + 1e-6))
        ll_s, lh_s, hl_s, hh_s = _subband_scales_from_orientation(
            global_state, smooth_fraction
        )

        residual_coeffs_list: List[np.ndarray] = []
        residual_pads:        List[Tuple[int, int]] = []
        residual_qsteps:      List[float] = []

        for p in range(self.residual_passes):
            resid = padded - current
            resid_pad, pad_info = pad_to_multiple(
                resid, 2 ** self.residual_levels, mode="reflect"
            )
            coeffs, _ = haar_dwt2(resid_pad, levels=self.residual_levels)

            # With EBOA, the Stage-1 reconstruction is already smoother,
            # reducing the high-frequency energy needed in the Haar residuals.
            # We use a slightly coarser step (1.6x) to achieve higher compression
            # while maintaining visual quality.
            qstep = max(0.5, _quality_to_qbase(quality) * (0.85 ** p) * 1.60)
            S = _haar_subband_scale_map(
                coeffs.shape, self.residual_levels, ll_s, lh_s, hl_s, hh_s
            )
            effective_step = qstep * S

            qc = np.round(coeffs / effective_step)
            if qc.min() >= -128 and qc.max() <= 127:
                qc = qc.astype(np.int8)
            else:
                qc = qc.astype(np.int16)

            residual_coeffs_list.append(qc)
            residual_pads.append(pad_info)
            residual_qsteps.append(float(qstep))
            current = current + haar_idwt2(
                qc.astype(np.float64) * effective_step,
                levels=self.residual_levels, pad=pad_info,
            )

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
            residual_coeffs=residual_coeffs_list,
            frac1_idx=_f1_idx,   frac1_alpha=_f1_alpha, frac1_beta=_f1_beta,
            frac1_idx_dpcm=_f1_idx_dpcm,
            frac1_alpha_dpcm=_f1_alpha_dpcm,
            frac1_beta_dpcm=_f1_beta_dpcm,
            frac2_idx=_f2_idx,   frac2_alpha=_f2_alpha,
            frac2_idx_dpcm=_f2_idx_dpcm,
            frac2_alpha_dpcm=_f2_alpha_dpcm,
        )

    def _decode_channel(self, data: ChannelCodecResult) -> np.ndarray:
        h, w = data.padded_shape
        bs = data.block_size
        ny, nx = h // bs, w // bs
        nblocks = ny * nx

        order         = data.order
        ordered_modes  = data.mode_grid.ravel()[order]
        ordered_states = data.state_grid.ravel()[order].astype(np.uint8)

        # ---- Stage 1: undo DPCM → dequantise → pixel reconstruction ----
        dct_stored, dpcm_flags = _unpack_coefficients(
            data.dct_qcoeffs_packed, nblocks, bs
        )
        dct_q = _undo_multi_dpcm(dct_stored, bs, dpcm_flags)
        recon_c = _dequantize_dct(dct_q, data.quality, ordered_modes,
                                   states=ordered_states)

        # Identical EBOA-based Stage-1 reconstruction
        recon1 = _eboa_reconstruct_image(recon_c, order, ny, nx, bs, pad=1) + 128.0

        # ---- Stage 2.5: Progressive fractal reconstruction ----
        # Rebuild domain pool from recon1 (identical procedure as encoder).
        # Undo DPCM on each stream, reconstruct affine predictions, sum.
        recon = recon1
        if getattr(data, "frac1_idx", None) is not None:
            pool_result = _build_fractal_domain_pool(recon1, bs)
            if pool_result is not None:
                pool_dct, pool_dc, D_base = pool_result
                D_total = pool_dct.shape[0]

                def _decode_idx(arr, flag):
                    raw = _dpcm_decode(arr.astype(np.int16), flag)
                    return np.clip(raw.astype(np.int64), 0, D_total - 1).astype(np.int16)
                def _decode_int8(arr, flag):
                    return _dpcm_decode(arr.astype(np.int16), flag).astype(np.int8)

                idx1  = _decode_idx(data.frac1_idx,
                                    getattr(data, "frac1_idx_dpcm", False))
                alp1  = _decode_int8(data.frac1_alpha,
                                     getattr(data, "frac1_alpha_dpcm", False))
                bet1  = _decode_int8(data.frac1_beta,
                                     getattr(data, "frac1_beta_dpcm", False))
                pred_dct = _fractal_reconstruct_affine(pool_dct, idx1, alp1, bet1)

                if getattr(data, "frac2_idx", None) is not None:
                    idx2  = _decode_idx(data.frac2_idx,
                                        getattr(data, "frac2_idx_dpcm", False))
                    alp2  = _decode_int8(data.frac2_alpha,
                                         getattr(data, "frac2_alpha_dpcm", False))
                    pred_dct = pred_dct + _fractal_reconstruct_affine(
                        pool_dct, idx2, alp2, None
                    )

                # Use EBOA for fractal reconstruction
                recon = _eboa_reconstruct_image(pred_dct, order, ny, nx, bs, pad=1) + 128.0

        # ---- Stage 3: Haar wavelet residual (operates on fractal base) ----
        modes_flat      = data.mode_grid.ravel()
        smooth_fraction = float(np.mean(modes_flat == 0))
        global_state    = _weighted_mode(
            data.state_grid.ravel(),
            (modes_flat == 0).astype(np.float64) + 1e-6,
        )
        ll_s, lh_s, hl_s, hh_s = _subband_scales_from_orientation(
            global_state, smooth_fraction
        )

        for qc, qstep, pad_info in zip(
            data.residual_coeffs, data.residual_qsteps, data.residual_pads
        ):
            # The encoder uses the stored qstep which now defaults to the 1.65x base
            S = _haar_subband_scale_map(
                qc.shape, self.residual_levels, ll_s, lh_s, hl_s, hh_s
            )
            resid = haar_idwt2(
                qc.astype(np.float64) * (qstep * S),
                levels=self.residual_levels, pad=pad_info,
            )
            recon = recon + crop_to_shape(resid, (h, w))

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
            "version": 4,
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
        for magic in (MAGIC, MAGIC_V2, MAGIC_V1):
            if blob.startswith(magic):
                payload = pickle.loads(lzma.decompress(blob[len(magic):]))
                break
        else:
            raise ValueError("Not a HFZ blob (unrecognised magic header)")

        version = payload.get("version", 1)
        if version not in (1, 2, 3, 4):
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
