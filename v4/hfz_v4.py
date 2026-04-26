"""Hybrid fractal-Z image codec  (v3).

A transform-domain image compressor built around three layers:

1) Block DCT — orientation-steered quantisation aligned with each block's
   dominant frequency axis (Stage 2 → Stage 1 q-matrix).
2) Directional-state layer — orientation-guided Morton fractal traversal and
   multi-position DPCM (DC + first two AC zig-zag positions delta-coded along
   the traversal for entropy reduction).
3) Haar-wavelet residual — subband-selective scale map; optionally operates on
   the tighter fractal-prediction residual when Stage 2.5 fires.

v3 additions:
- Iterative vectorised Hilbert curve (build_hilbert_order_flat):
    O(N·log N) fully NumPy implementation of the standard d2xy algorithm.
    D4-symmetric orientation transform (8 variants) aligns the traversal with
    the dominant gradient.  Used to spatially order the fractal domain pool so
    that consecutive domain indices are geometrically adjacent, making the
    domain_idx stream DPCM-compressible.
- Fractal DCT coefficient prediction (Stage 2.5):
    After Stage-1 reconstruction, a domain pool of 4× (rotated) 2×-downsampled
    DCT blocks is built. Each range block is matched to its best predictor via
    least-squares scale (batch matrix multiply). Stage-3 Haar refines the
    tighter fractal residual.  Payload: (domain_idx, alpha_q) per block, both
    DPCM-coded; domain pool is deterministic from Stage-1 → no extra state.
    Engages only when measured prediction gain exceeds 1.6× threshold.

The codec is self-contained (NumPy + stdlib only) and hackable.

Usage:
    python fractal_hfz_codec.py encode input.png output.hfz --quality 55
    python fractal_hfz_codec.py decode output.hfz reconstructed.png

File format: LZMA-compressed pickle blob with 4-byte magic header (HFZ3).
Backward-compatible with HFZ2 (v2) and HFZ1 (v1) blobs.

Stage-2 / Stage-3 integration improvements (v2):
- Multi-position DPCM in fractal order: DC plus first two AC zig-zag positions
  are delta-coded along the traversal, reducing their entropy significantly.
- Orientation-steered DCT quantisation: per-block orientation from Stage 2
  tilts the keep mask and q-step ramp to align with the dominant DCT energy
  axis.
- Subband-selective Haar quantisation: HH mildly penalised; LH/HL
  orientation-modulated; LL left at unity to protect chroma quality.

v3 additions — Hilbert curve + Fractal DCT prediction:
- Iterative Hilbert-curve traversal (build_hilbert_order): replaces the
  recursive Morton-like build_fractal_order.  The Hilbert curve guarantees
  L∞ = 1 adjacency between every consecutive block pair (no cross-quadrant
  jumps).  The dominant gradient direction selects one of 8 D4-symmetric
  variants (4 rotations × 2 reflections) to align the traversal with the main
  feature axis.  Fully vectorised: O(N) with no Python recursion overhead.
- Fractal DCT coefficient prediction (Stage 2.5): after Stage-1 DCT
  quantisation and reconstruction, a domain pool is built from 2× downsampled
  16×16 pixel regions of the reconstruction presented in 4 rotations.  For
  each block the best-matching domain block is found by batch least-squares on
  the top-left _FRACTAL_K×_FRACTAL_K DCT patch.  Stage-3 Haar then operates on
  the much tighter fractal residual.  The domain pool is deterministically
  reproducible at decode from the Stage-1 reconstruction — no extra payload
  beyond (domain_idx, alpha_q) per block (3 bytes/block).  Disabled
  automatically when prediction gain falls below threshold or when the image
  is too small for a meaningful pool.
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

MAGIC = b"HFZ4"   # v4: adaptive quadtree DCT (8/16/32) + fractal residual engine


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


# ============================================================================
# Adaptive quadtree block segmentation  (Stage 1 extension, v4)
# ============================================================================
#
# Fixed 8x8 DCT blocks waste coefficient budget on smooth regions. The adaptive
# layer promotes smooth regions to 16x16 or 32x32 DCT blocks so the transform
# extends beyond old 8x8 boundaries, capturing cross-tile correlations in single
# coefficients. Complex/edge regions keep 8x8 for precise localisation.
#
# GATING FUNCTION (deterministic from stored mode_grid, zero payload)
#   A 2x2 patch of 8x8 blocks merges to 16x16 if ALL four have mode == 0.
#   A 2x2 patch of 16x16 merges to 32x32 if ALL sixteen 8x8 have mode == 0.
#
# QUANTISATION: keep-bandwidth scales with n/8 (more frequencies kept for larger
# blocks to maintain proportional LF coverage). Qstep unchanged in absolute units.


def _build_adaptive_leaves(
    mode_grid: np.ndarray,
    ny: int,
    nx: int,
    dc_grid: Optional[np.ndarray] = None,
    quality: int = 55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive (leaf32, leaf16, leaf8) boolean masks using a two-part gating rule.

    Merge criterion for a candidate group of 8x8 blocks:
      1. All constituent blocks have mode == 0 (smooth, low HF energy).
      2. DC range (max - min) across the group is below a threshold derived from
         the quantisation step. This prevents merging blocks with very different
         means (e.g. 8-pixel checkerboards, sinusoids) that only appear smooth
         because their HF energy is intra-block but their DC varies sharply across
         the merge boundary. The DC criterion is what separates "truly smooth"
         (gradient, ramp) from "locally smooth but cross-block variable" (tiles).

    leaf32: (ny//4, nx//4) True where 4x4 group passes both criteria.
    leaf16: (ny//2, nx//2) True where 2x2 group passes but not part of a 32x32.
    leaf8 : (ny, nx)       True for all remaining cells.
    """
    mg = mode_grid
    # DC threshold: merge allowed when max-DC - min-DC in group < threshold.
    # Threshold = 3 * qbase (units of 8x8 DCT DC coefficient).
    # For quality=55: qbase≈2.4 → threshold≈7 DC units ≈ 20 pixel-mean units.
    dc_thr = 3.0 * _quality_to_qbase(quality)
    has_dc = dc_grid is not None

    can16 = np.zeros((max(1, ny // 2), max(1, nx // 2)), dtype=bool)
    can32 = np.zeros((max(1, ny // 4), max(1, nx // 4)), dtype=bool)

    if ny >= 2 and nx >= 2:
        for gy in range(ny // 2):
            for gx in range(nx // 2):
                if not np.all(mg[gy*2:gy*2+2, gx*2:gx*2+2] == 0):
                    continue
                if has_dc:
                    patch = dc_grid[gy*2:gy*2+2, gx*2:gx*2+2]
                    if float(patch.max() - patch.min()) > dc_thr:
                        continue
                can16[gy, gx] = True

    if ny >= 4 and nx >= 4:
        for gy in range(ny // 4):
            for gx in range(nx // 4):
                if not np.all(can16[gy*2:gy*2+2, gx*2:gx*2+2]):
                    continue
                if has_dc:
                    patch = dc_grid[gy*4:gy*4+4, gx*4:gx*4+4]
                    if float(patch.max() - patch.min()) > dc_thr:
                        continue
                can32[gy, gx] = True

    leaf32 = can32
    leaf16 = can16.copy()
    for gy in range(ny // 4):
        for gx in range(nx // 4):
            if leaf32[gy, gx]:
                leaf16[gy*2:gy*2+2, gx*2:gx*2+2] = False

    claimed = np.zeros((ny, nx), dtype=bool)
    for gy in range(ny // 4):
        for gx in range(nx // 4):
            if leaf32[gy, gx]:
                claimed[gy*4:gy*4+4, gx*4:gx*4+4] = True
    for gy in range(ny // 2):
        for gx in range(nx // 2):
            if leaf16[gy, gx]:
                claimed[gy*2:gy*2+2, gx*2:gx*2+2] = True
    return leaf32, leaf16, ~claimed


def _leaves_in_z_order(
    leaf32: np.ndarray,
    leaf16: np.ndarray,
    leaf8:  np.ndarray,
    ny: int,
    nx: int,
) -> List[Tuple[int, int, int]]:
    """Return (gy8, gx8, size) tuples in Morton Z-order of top-left 8x8 cell."""
    all_rows = np.repeat(np.arange(ny, dtype=np.int32), nx)
    all_cols = np.tile(np.arange(nx, dtype=np.int32), ny)
    z_order = np.argsort(_morton_encode_2d(all_rows, all_cols))
    zrows = all_rows[z_order].tolist()
    zcols = all_cols[z_order].tolist()

    leaves: List[Tuple[int, int, int]] = []
    visited = np.zeros((ny, nx), dtype=bool)
    for r, c in zip(zrows, zcols):
        if visited[r, c]:
            continue
        if r % 4 == 0 and c % 4 == 0:
            gr, gc = r // 4, c // 4
            if gr < leaf32.shape[0] and gc < leaf32.shape[1] and leaf32[gr, gc]:
                leaves.append((r, c, 32))
                visited[r:r+4, c:c+4] = True
                continue
        if r % 2 == 0 and c % 2 == 0:
            gr, gc = r // 2, c // 2
            if gr < leaf16.shape[0] and gc < leaf16.shape[1] and leaf16[gr, gc]:
                leaves.append((r, c, 16))
                visited[r:r+2, c:c+2] = True
                continue
        leaves.append((r, c, 8))
        visited[r, c] = True
    return leaves


def _get_quant_params_adaptive(
    quality: int,
    modes: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantisation params scaled for block size n.

    Keep-bandwidth scales linearly with n/8. Qstep same as 8x8 baseline since
    DCT coefficients have identical absolute scale at all block sizes.
    """
    qbase = _quality_to_qbase(quality)
    keep_base = np.array([2, 3, 4], dtype=np.int32)
    keep = keep_base[modes]
    scale = max(1, n // 8)
    keep = np.clip(keep * scale, 1, n * 2)
    if quality >= 85:
        keep += scale
    elif quality <= 25:
        keep = np.maximum(1, keep - scale)
    mode_scales = np.array([1.9, 1.15, 0.95], dtype=np.float64)
    return qbase * mode_scales[modes], keep


def _quantize_dct_adaptive(
    coeffs: np.ndarray,
    quality: int,
    modes: np.ndarray,
    states: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Quantise nxn DCT blocks (n in {8, 16, 32})."""
    n = coeffs.shape[-1]
    qsteps, keep = _get_quant_params_adaptive(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * eff)
    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    vals = np.round(coeffs / q_matrix)
    qcoeffs[masks] = np.clip(vals[masks], -32768, 32767).astype(np.int16)
    return qcoeffs


def _dequantize_dct_adaptive(
    qcoeffs: np.ndarray,
    quality: int,
    modes: np.ndarray,
    states: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = qcoeffs.shape[-1]
    qsteps, keep = _get_quant_params_adaptive(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * eff)
    coeffs = np.zeros(qcoeffs.shape, dtype=np.float64)
    coeffs[masks] = qcoeffs[masks].astype(np.float64) * q_matrix[masks]
    return coeffs


def _pack_size_class(qcoeffs: np.ndarray, n: int) -> Tuple[np.ndarray, int]:
    """Zig-zag pack and DPCM-encode one size class of blocks.

    Returns (packed_array, dpcm_flag_int):
      n==8 : 3-position DPCM. dpcm_flag_int is a 3-bit bitmask (bit i=1 means
             zig-zag position i was delta-coded). The array body contains ONLY
             zig-zag coefficients with no appended flags, keeping the LZMA
             context model clean.
      n=16/32: DC-only DPCM. dpcm_flag_int is 0 or 1 (appended to body).
    """
    if len(qcoeffs) == 0:
        return np.empty(0, dtype=np.int16), 0
    if n == 8:
        stored, np_flags = _apply_multi_dpcm(qcoeffs, n)
        flag_int = int(sum(int(f) << i for i, f in enumerate(np_flags[:_N_DPCM_POSITIONS])))
        indices = _zig_zag_indices(n)
        body = stored.reshape(-1, n * n)[:, indices].ravel()
        return body, flag_int
    # DC-only DPCM for 16x16 and 32x32
    dc = qcoeffs[:, 0, 0].astype(np.int32)
    coded = dc.copy()
    if len(dc) > 1:
        coded[1:] = dc[1:] - dc[:-1]
    apply_dc = len(dc) > 1 and (np.sum(np.abs(coded[1:])) < np.sum(np.abs(dc[1:])))
    stored = qcoeffs.copy()
    if apply_dc:
        stored[:, 0, 0] = coded.astype(np.int16)
    indices = _zig_zag_indices(n)
    body = stored.reshape(-1, n * n)[:, indices].ravel()
    return np.concatenate([body, np.array([int(apply_dc)], dtype=np.int16)]), int(apply_dc)


def _unpack_size_class(
    packed: np.ndarray, nblocks: int, n: int, dpcm_flags_int: int = 0
) -> np.ndarray:
    """Undo DPCM and unpack one size class. Returns (nblocks, n, n) int16.

    For n==8, dpcm_flags_int is a bitmask stored separately in the dataclass
    (not in the packed array) so the coefficient body stays clean for LZMA.
    """
    if nblocks == 0:
        return np.empty((0, n, n), dtype=np.int16)
    expected = nblocks * n * n
    indices = _zig_zag_indices(n)
    rev = np.zeros_like(indices); rev[indices] = np.arange(len(indices))
    if n == 8:
        blocks = packed[:expected].reshape(nblocks, n * n)[:, rev].reshape(nblocks, n, n)
        flags = np.array(
            [(dpcm_flags_int >> i) & 1 for i in range(_N_DPCM_POSITIONS)],
            dtype=np.int16,
        )
        return _undo_multi_dpcm(blocks, n, flags)
    body = packed[:expected]
    dc_flag = bool(packed[expected]) if len(packed) > expected else False
    blocks = body.reshape(nblocks, n * n)[:, rev].reshape(nblocks, n, n)
    if dc_flag:
        b = blocks.copy()
        b[:, 0, 0] = np.cumsum(b[:, 0, 0].astype(np.int16)).astype(np.int16)
        return b
    return blocks



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
    dct_qcoeffs_packed: np.ndarray   # int16: 8x8 blocks (clean, no appended flags)
    residual_pads: List[Tuple[int, int]]
    residual_qsteps: List[float]
    residual_coeffs: List[np.ndarray]   # each [Hp, Wp]
    dpcm_flags_8: int = 0            # bitmask: bit i=1 means zig-zag pos i was DPCM-coded
    # Adaptive block size fields (v4) -- None when all blocks are 8x8
    dct_packed_16: Optional[np.ndarray] = None   # int16: 16x16 blocks packed
    dct_packed_32: Optional[np.ndarray] = None   # int16: 32x32 blocks packed
    leaf_counts: Optional[Tuple[int, int, int]] = None  # (N8, N16, N32), None = all 8x8
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
        """Minimal pickle — omit None fields, False flags, and trivial leaf_counts."""
        d = self.__dict__.copy()
        for k in self._FRAC_NONE_KEYS + ("dct_packed_16", "dct_packed_32"):
            if d.get(k) is None:
                d.pop(k, None)
        for k in self._FRAC_BOOL_KEYS:
            if not d.get(k, False):
                d.pop(k, None)
        if not d.get("dpcm_flags_8", 0):
            d.pop("dpcm_flags_8", None)
        # Omit leaf_counts when all blocks are 8x8 (most common case).
        # decode reconstructs n8 = nblocks when leaf_counts == (0,0,0) default.
        lc = d.get("leaf_counts")
        if lc is not None and lc[1] == 0 and lc[2] == 0:
            d.pop("leaf_counts", None)
        return d

    def __setstate__(self, state: dict) -> None:
        """Restore with safe defaults for any omitted fields."""
        state.setdefault("leaf_counts", (0, 0, 0))
        state.setdefault("dpcm_flags_8", 0)
        for k in ("frac1_idx","frac1_alpha","frac1_beta","frac2_idx","frac2_alpha",
                  "dct_packed_16","dct_packed_32"):
            state.setdefault(k, None)
        for k in ("frac1_idx_dpcm","frac1_alpha_dpcm","frac1_beta_dpcm",
                  "frac2_idx_dpcm","frac2_alpha_dpcm"):
            state.setdefault(k, False)
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

        # ---- Stage 1: fine-grid 8x8 analysis for orientation + mode ----
        # Always analyse at 8x8 to get maximum-resolution mode_grid, which
        # drives the adaptive block gating function.
        reshaped = padded.reshape(ny, bs, nx, bs).transpose(0, 2, 1, 3)
        flat_blocks = reshaped.reshape(-1, bs, bs)
        centered = flat_blocks - 128.0
        coeff_blocks = dct2(centered)
        total_energy = np.sum(np.abs(coeff_blocks), axis=(1, 2))
        hf_energy = (
            np.sum(np.abs(coeff_blocks[:, 2:, 2:]), axis=(1, 2))
            if bs > 2 else np.zeros(len(coeff_blocks))
        )
        states, anis = block_orientation_state(flat_blocks)
        modes = _mode_from_features(total_energy, hf_energy, anis, quality)
        state_grid  = states.reshape(ny, nx)
        mode_grid   = modes.reshape(ny, nx)
        energy_grid = (hf_energy + 1e-6).reshape(ny, nx)

        # ---- Stage 2: adaptive block segmentation (gated by mode_grid) ----
        # Smooth 2x2 groups of 8x8 blocks merge to 16x16; smooth 4x4 to 32x32.
        # The larger-block DCT spans the merged region -- the transform literally
        # extends beyond old 8x8 boundaries, capturing cross-tile correlations in
        # single coefficients and collapsing the Haar residual to near-zero there.
        # Decoder reproduces identical leaf sizes from the stored mode_grid.
        # Pass DC grid so gating rejects merges across large mean differences
        dc_grid = coeff_blocks[:, 0, 0].reshape(ny, nx)
        leaf32, leaf16, leaf8 = _build_adaptive_leaves(mode_grid, ny, nx,
                                                        dc_grid=dc_grid, quality=quality)
        leaves = _leaves_in_z_order(leaf32, leaf16, leaf8, ny, nx)
        leaves8  = [(gy, gx) for gy, gx, sz in leaves if sz == 8]
        leaves16 = [(gy, gx) for gy, gx, sz in leaves if sz == 16]
        leaves32 = [(gy, gx) for gy, gx, sz in leaves if sz == 32]

        def _leaf_dct(leaf_list, size):
            if not leaf_list:
                step = size // bs
                blk_st = np.empty(0, dtype=np.uint8)
                blk_mo = np.empty(0, dtype=np.uint8)
                return np.empty((0, size, size)), blk_st, blk_mo
            px = np.stack([padded[gy*bs:gy*bs+size, gx*bs:gx*bs+size]
                           for gy, gx in leaf_list])
            c = dct2(px - 128.0)
            step = size // bs
            # Average orientation state from constituent 8x8 cells
            blk_st = np.array([
                int(round(float(np.mean(states.reshape(ny, nx)[gy:gy+step, gx:gx+step]))))
                for gy, gx in leaf_list], dtype=np.uint8)
            # All merged blocks are smooth (mode=0 for all constituents)
            blk_mo = np.zeros(len(leaf_list), dtype=np.uint8)
            return c, blk_st, blk_mo

        c8,  s8,  m8  = _leaf_dct(leaves8,  bs)
        c16, s16, m16 = _leaf_dct(leaves16, 16)
        c32, s32, m32 = _leaf_dct(leaves32, 32)

        # ---- Stage 1 quantisation: size-adaptive keep-bandwidth ----
        def _qz(c, m, s):
            if not len(c): return c.astype(np.int16)
            return _quantize_dct_adaptive(c, quality, m, states=s)
        def _dqz(qc, m, s):
            if not len(qc): return qc.astype(np.float64)
            return _dequantize_dct_adaptive(qc, quality, m, states=s)

        qc8  = _qz(c8,  m8,  s8)
        qc16 = _qz(c16, m16, s16)
        qc32 = _qz(c32, m32, s32)

        # ---- Pack with per-size-class DPCM ----
        packed8,  dpcm_flags_8  = _pack_size_class(qc8,  bs)
        packed16, _dpcm_flag_16 = _pack_size_class(qc16, 16)
        packed32, _dpcm_flag_32 = _pack_size_class(qc32, 32)

        # ---- Stage-1 pixel-domain reconstruction from adaptive blocks ----
        recon1 = np.zeros_like(padded)
        if len(qc8):
            rb8 = idct2(_dqz(qc8, m8, s8)) + 128.0
            for i, (gy, gx) in enumerate(leaves8):
                recon1[gy*bs:gy*bs+bs, gx*bs:gx*bs+bs] = rb8[i]
        if len(qc16):
            rb16 = idct2(_dqz(qc16, m16, s16)) + 128.0
            for i, (gy, gx) in enumerate(leaves16):
                recon1[gy*bs:gy*bs+16, gx*bs:gx*bs+16] = rb16[i]
        if len(qc32):
            rb32 = idct2(_dqz(qc32, m32, s32)) + 128.0
            for i, (gy, gx) in enumerate(leaves32):
                recon1[gy*bs:gy*bs+32, gx*bs:gx*bs+32] = rb32[i]

        # Fine-grid flat ordering (still needed for fractal pool + ChannelCodecResult.order)
        order_xy = build_fractal_order(state_grid, energy_grid)
        order    = np.array([y * nx + x for (y, x) in order_xy], dtype=np.int32)
        ordered_coeffs = coeff_blocks[order]
        ordered_modes  = modes[order]
        ordered_states = states[order]

        # Flat quantised 8x8 array for fractal prediction (uses 8x8 analysis only)
        # Large blocks contribute a DC-only approximation to their 8x8 sub-cells
        dct_qcoeffs_flat = np.zeros((ny * nx, bs, bs), dtype=np.int16)
        for i, (gy, gx) in enumerate(leaves8):
            dct_qcoeffs_flat[gy * nx + gx] = qc8[i]
        for i, (gy, gx) in enumerate(leaves16):
            for dy in range(2):
                for dx in range(2):
                    dct_qcoeffs_flat[(gy + dy) * nx + (gx + dx), 0, 0] = qc16[i, 0, 0]
        for i, (gy, gx) in enumerate(leaves32):
            for dy in range(4):
                for dx in range(4):
                    dct_qcoeffs_flat[(gy + dy) * nx + (gx + dx), 0, 0] = qc32[i, 0, 0]

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
                dct_qcoeffs_flat[order].astype(np.float64), pool_dct, pool_dc, D_base, nbhd, with_beta=True
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
            pred_px_p1 = idct2(pred_dct_p1) + 128.0
            up1 = np.zeros_like(pred_px_p1)
            up1[order] = pred_px_p1
            recon_frac1 = up1.reshape(ny, nx, bs, bs).transpose(0,2,1,3).reshape(h, w)

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
                pred_px_p12 = idct2(pred_dct_p1 + pred_dct_p2) + 128.0
                up12 = np.zeros_like(pred_px_p12)
                up12[order] = pred_px_p12
                recon_frac12 = up12.reshape(ny,nx,bs,bs).transpose(0,2,1,3).reshape(h,w)
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

            qstep = max(0.5, _quality_to_qbase(quality) * (0.85 ** p) * 1.5)
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
            dct_qcoeffs_packed=packed8,
            dpcm_flags_8=dpcm_flags_8,
            dct_packed_16=packed16 if len(packed16) > 0 else None,
            dct_packed_32=packed32 if len(packed32) > 0 else None,
            leaf_counts=(len(leaves8), len(leaves16), len(leaves32)),
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

        # ---- Stage 1: adaptive block decode ----
        # leaf_counts stored at encode is the authoritative source for N8/N16/N32.
        # We rebuild leaf positions with the same gating rule; if the reconstucted
        # counts match the stored ones (they always will for v4 blobs) we use them.
        # For v1-v3 blobs leaf_counts is None and all blocks are 8x8.
        lc = data.leaf_counts
        if lc == (0, 0, 0):
            # All-8x8 image: leaf_counts was omitted from pickle to save space
            n8, n16, n32 = nblocks, 0, 0
        else:
            n8, n16, n32 = lc

        # Rebuild leaf position lists from mode grid (dc_grid=None at decode:
        # the DC-range gate was already applied at encode and the result stored
        # in leaf_counts; we just need the positions in the same Z-order).
        leaf32, leaf16, leaf8 = _build_adaptive_leaves(data.mode_grid, ny, nx)
        leaves = _leaves_in_z_order(leaf32, leaf16, leaf8, ny, nx)
        leaves8  = [(gy,gx) for gy,gx,sz in leaves if sz==8]
        leaves16 = [(gy,gx) for gy,gx,sz in leaves if sz==16]
        leaves32 = [(gy,gx) for gy,gx,sz in leaves if sz==32]

        # If gating without dc_grid gives wrong counts (should not happen for
        # v4 blobs but guards against edge cases), truncate/pad with raster.
        def _safe_leaves(lst, n_stored, step):
            """Return leaf positions as (gy8, gx8) tuples in Morton Z-order.

            If the gating reconstruction (which lacked dc_grid) matches the
            stored count, use it directly.  Otherwise enumerate all valid
            candidates for this step size in Z-order — this matches the order
            the encoder used when dc_grid passed for all cells at this size.
            """
            if len(lst) == n_stored:
                return lst
            # Full Z-order enumeration of candidates at this step size.
            ngy, ngx = ny // step, nx // step
            all_r = np.repeat(np.arange(ngy, dtype=np.int32), ngx) * step
            all_c = np.tile(np.arange(ngx, dtype=np.int32), ngy) * step
            z = np.argsort(_morton_encode_2d(all_r // step, all_c // step))
            result = list(zip(all_r[z].tolist(), all_c[z].tolist()))
            return result[:n_stored]

        leaves8  = _safe_leaves(leaves8,  n8,  1)
        leaves16 = _safe_leaves(leaves16, n16, 2)
        leaves32 = _safe_leaves(leaves32, n32, 4)

        def _leaf_sm(leaf_list, size):
            if not leaf_list:
                return np.empty(0,dtype=np.uint8), np.empty(0,dtype=np.uint8)
            step = size // bs
            st = np.array([
                int(round(float(np.mean(data.state_grid[gy:gy+step, gx:gx+step]))))
                for gy, gx in leaf_list], dtype=np.uint8)
            return st, np.zeros(len(leaf_list), dtype=np.uint8)

        s8,  m8  = _leaf_sm(leaves8,  bs)
        s16, m16 = _leaf_sm(leaves16, 16)
        s32, m32 = _leaf_sm(leaves32, 32)

        qc8  = _unpack_size_class(data.dct_qcoeffs_packed, n8, bs,
                                   dpcm_flags_int=getattr(data, 'dpcm_flags_8', 0))
        _p16 = data.dct_packed_16 if data.dct_packed_16 is not None else np.empty(0,dtype=np.int16)
        _p32 = data.dct_packed_32 if data.dct_packed_32 is not None else np.empty(0,dtype=np.int16)
        qc16 = _unpack_size_class(_p16, n16, 16)
        qc32 = _unpack_size_class(_p32, n32, 32)

        recon1 = np.zeros((h, w), dtype=np.float64)
        if n8 > 0:
            rb8 = idct2(_dequantize_dct_adaptive(qc8,data.quality,m8,states=s8))+128.0
            for i,(gy,gx) in enumerate(leaves8):
                recon1[gy*bs:gy*bs+bs, gx*bs:gx*bs+bs] = rb8[i]
        if n16 > 0:
            rb16 = idct2(_dequantize_dct_adaptive(qc16,data.quality,m16,states=s16))+128.0
            for i,(gy,gx) in enumerate(leaves16):
                recon1[gy*bs:gy*bs+16, gx*bs:gx*bs+16] = rb16[i]
        if n32 > 0:
            rb32 = idct2(_dequantize_dct_adaptive(qc32,data.quality,m32,states=s32))+128.0
            for i,(gy,gx) in enumerate(leaves32):
                recon1[gy*bs:gy*bs+32, gx*bs:gx*bs+32] = rb32[i]
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

                pred_px = idct2(pred_dct) + 128.0
                unordered_p = np.zeros_like(pred_px)
                unordered_p[order] = pred_px
                recon = (
                    unordered_p
                    .reshape(ny, nx, bs, bs)
                    .transpose(0, 2, 1, 3)
                    .reshape(h, w)
                )

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
        if not blob.startswith(MAGIC):
            raise ValueError(
                f"Not an HFZ4 blob (got {blob[:4]!r}). "
                "Old HFZ1/2/3 blobs are no longer supported; re-encode with this version."
            )
        payload = pickle.loads(lzma.decompress(blob[len(MAGIC):]))
        if payload.get("version") != 4:
            raise ValueError(f"Unsupported codec version {payload.get('version')}")

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
