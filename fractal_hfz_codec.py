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
"""

from __future__ import annotations

import argparse
import io
import lzma
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

MAGIC = b"HFZ1"


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


def rgb_to_ycbcr(rgb: np.ndarray, subsample: bool = True) -> List[np.ndarray]:
    """Convert RGB to YCbCr (BT.601)."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    if subsample:
        # 4:2:0 subsampling: simple 2x2 average pooling
        cb = (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2]) * 0.25
        cr = (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2]) * 0.25

    return [y, cb, cr]


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Convert YCbCr to RGB (BT.601)."""
    # Upsample cb, cr if needed (4:2:0)
    if cb.shape != y.shape:
        # Simple nearest-neighbor upsampling
        cb = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]
        cr = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)[:y.shape[0], :y.shape[1]]

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
    # Using matrix multiplication that handles batches: (N, n, n)
    # Equivalent to c @ blocks @ c.T for each block
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
    # b: (N, H, W)
    gx = b[:, :, 1:] - b[:, :, :-1]
    gy = b[:, 1:, :] - b[:, :-1, :]

    # Matching the behavior of the original for edge handling
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
    # Map undirected orientation to 8 bins.
    theta = theta % np.pi
    state = ( (theta / np.pi) * 8.0 ).astype(np.int32) % 8
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
    # Larger quality -> smaller quantization step.
    return max(0.55, 18.0 / math.sqrt(quality + 1.0))


def _mode_from_features(total_energy: np.ndarray, hf_energy: np.ndarray, anisotropy: np.ndarray, q: int) -> np.ndarray:
    """Classify blocks into smooth / directional / textured."""
    ratio = hf_energy / (total_energy + 1e-9)
    modes = np.zeros_like(total_energy, dtype=np.uint8)
    # Default is 0 (smooth)
    mask_textured = (ratio >= 0.18) & (total_energy >= (5.0 + (100 - q) * 0.1))
    modes[mask_textured] = 2
    mask_directional = mask_textured & (anisotropy > 0.25)
    modes[mask_directional] = 1
    return modes


def _get_quant_params(quality: int, modes: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get qstep and keep radius for a batch of modes."""
    qbase = _quality_to_qbase(quality)

    # keep radius based on mode
    keep_base = np.array([2, 3, 4], dtype=np.int32)
    keep = keep_base[modes]
    if quality >= 85:
        keep += 1
    elif quality <= 25:
        keep = np.maximum(1, keep - 1)
    keep = np.clip(keep, 1, n * 2)

    # qstep based on mode
    mode_scales = np.array([1.9, 1.15, 0.95], dtype=np.float64)
    qsteps = qbase * mode_scales[modes]

    return qsteps, keep


def _quantize_dct(coeffs: np.ndarray, quality: int, modes: np.ndarray, states: np.ndarray) -> np.ndarray:
    n = coeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)

    # Create a mask for (u+v) < keep
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    uv_sum = u + v

    # masks: (N, n, n)
    masks = uv_sum[None, :, :] < keep[:, None, None]

    # Directional pruning for mode 1 (directional)
    # states: 0=smooth/none, 1-7=orientations
    # If state > 0 and mode == 1, we can prune coefficients perpendicular to the dominant direction
    # For simplicity, we'll use a rough heuristic: if u > v and mostly horizontal, or v > u and mostly vertical
    directional_mask = np.ones(coeffs.shape, dtype=bool)
    for s in range(1, 8):
        s_mask = (modes == 1) & (states == s)
        if not np.any(s_mask): continue

        # angle = (s / 8) * pi
        angle = (s / 8.0) * np.pi
        # Perpendicular direction in DCT space
        if 0.375 * np.pi <= angle <= 0.625 * np.pi: # Vertical-ish
            directional_mask[s_mask, :, 1:] = False
        elif angle <= 0.125 * np.pi or angle >= 0.875 * np.pi: # Horizontal-ish
            directional_mask[s_mask, 1:, :] = False

    masks &= directional_mask

    # qstep matrix: (N, n, n)
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * uv_sum[None, :, :])

    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    vals = np.round(coeffs / q_matrix)
    qcoeffs[masks] = np.clip(vals[masks], -32768, 32767).astype(np.int16)
    return qcoeffs


def _dequantize_dct(qcoeffs: np.ndarray, quality: int, modes: np.ndarray, states: np.ndarray) -> np.ndarray:
    n = qcoeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)

    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    uv_sum = u + v
    masks = uv_sum[None, :, :] < keep[:, None, None]

    # Must use same directional pruning logic
    directional_mask = np.ones(qcoeffs.shape, dtype=bool)
    for s in range(1, 8):
        s_mask = (modes == 1) & (states == s)
        if not np.any(s_mask): continue
        angle = (s / 8.0) * np.pi
        if 0.375 * np.pi <= angle <= 0.625 * np.pi:
            directional_mask[s_mask, :, 1:] = False
        elif angle <= 0.125 * np.pi or angle >= 0.875 * np.pi:
            directional_mask[s_mask, 1:, :] = False
    masks &= directional_mask

    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * uv_sum[None, :, :])

    coeffs = np.zeros(qcoeffs.shape, dtype=np.float64)
    coeffs[masks] = qcoeffs[masks].astype(np.float64) * q_matrix[masks]
    return coeffs


def _zig_zag_indices(n: int) -> np.ndarray:
    index_order = sorted(((i, j) for i in range(n) for j in range(n)),
                         key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 == 0 else x[0]))
    return np.array([i * n + j for i, j in index_order])


def _pack_coefficients(qcoeffs: np.ndarray) -> np.ndarray:
    """Pack a batch of blocks into a 1D array using zig-zag scan."""
    n = qcoeffs.shape[-1]
    indices = _zig_zag_indices(n)
    flat = qcoeffs.reshape(-1, n * n)
    return flat[:, indices].ravel()


def _unpack_coefficients(packed: np.ndarray, nblocks: int, n: int) -> np.ndarray:
    """Unpack a 1D array into a batch of blocks using zig-zag scan."""
    indices = _zig_zag_indices(n)
    rev_indices = np.zeros_like(indices)
    rev_indices[indices] = np.arange(len(indices))

    flat = packed.reshape(nblocks, n * n)
    return flat[:, rev_indices].reshape(nblocks, n, n)


@dataclass
class ChannelCodecResult:
    padded_shape: Tuple[int, int]
    pad_hw: Tuple[int, int]
    block_size: int
    quality: int
    order: np.ndarray
    state_grid: np.ndarray
    mode_grid: np.ndarray
    dct_qcoeffs_packed: np.ndarray  # 1D zig-zag
    residual_pads: List[Tuple[int, int]]
    residual_qsteps: List[float]
    residual_coeffs: List[np.ndarray]  # each [Hp, Wp]


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

        # Reshape into blocks: (ny, bs, nx, bs) -> (ny, nx, bs, bs)
        reshaped = padded.reshape(ny, bs, nx, bs).transpose(0, 2, 1, 3)
        flat_blocks = reshaped.reshape(-1, bs, bs)

        centered = flat_blocks - 128.0
        coeff_blocks = dct2(centered)

        total_energy = np.sum(np.abs(coeff_blocks), axis=(1, 2))
        if bs > 2:
            hf_energy = np.sum(np.abs(coeff_blocks[:, 2:, 2:]), axis=(1, 2))
        else:
            hf_energy = np.zeros(len(coeff_blocks))

        states, anis = block_orientation_state(flat_blocks)
        modes = _mode_from_features(total_energy, hf_energy, anis, quality)

        state_grid = states.reshape(ny, nx)
        mode_grid = modes.reshape(ny, nx)
        energy_grid = (hf_energy + 1e-6).reshape(ny, nx)

        order_xy = build_fractal_order(state_grid, energy_grid)
        order = np.array([y * nx + x for (y, x) in order_xy], dtype=np.int32)

        # Reorder blocks for fractal traversal
        ordered_coeffs = coeff_blocks[order]
        ordered_modes = modes[order]
        ordered_states = states[order]

        dct_qcoeffs = _quantize_dct(ordered_coeffs, quality, ordered_modes, ordered_states)

        # Reconstruct pass 1.
        recon_coeffs = _dequantize_dct(dct_qcoeffs, quality, ordered_modes, ordered_states)
        recon_blocks = idct2(recon_coeffs) + 128.0

        # Put blocks back into grid
        # First undo the fractal order
        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks

        recon1 = unordered_recon_blocks.reshape(ny, nx, bs, bs).transpose(0, 2, 1, 3).reshape(h, w)

        residual = padded - recon1
        residual_coeffs: List[np.ndarray] = []
        residual_pads: List[Tuple[int, int]] = []
        residual_qsteps: List[float] = []
        current = recon1.copy()

        for p in range(self.residual_passes):
            resid = padded - current

            # Mask residual based on mode_grid
            # mode 0 (smooth) blocks get their residuals attenuated
            mask = np.ones_like(mode_grid, dtype=np.float64)
            mask[mode_grid == 0] = 0.50
            # Upsample mask to pixel resolution
            pixel_mask = np.repeat(np.repeat(mask, bs, axis=0), bs, axis=1)
            resid = resid * pixel_mask

            resid_pad, pad_info = pad_to_multiple(resid, 2 ** self.residual_levels, mode="reflect")
            coeffs, _ = haar_dwt2(resid_pad, levels=self.residual_levels)
            # Quantize residual.
            qstep = max(0.5, _quality_to_qbase(quality) * (0.85 ** p) * 1.5)
            qcoeffs = np.round(coeffs / qstep)
            # Find appropriate integer type for residuals
            c_min, c_max = qcoeffs.min(), qcoeffs.max()
            if c_min >= -128 and c_max <= 127:
                qcoeffs = qcoeffs.astype(np.int8)
            else:
                qcoeffs = qcoeffs.astype(np.int16)
            residual_coeffs.append(qcoeffs)
            residual_pads.append(pad_info)
            residual_qsteps.append(float(qstep))
            recon_resid = haar_idwt2(qcoeffs.astype(np.float64) * qstep, levels=self.residual_levels, pad=pad_info)
            current = current + recon_resid

        return ChannelCodecResult(
            padded_shape=(h, w),
            pad_hw=pad_hw,
            block_size=bs,
            quality=int(quality),
            order=order,
            state_grid=state_grid,
            mode_grid=mode_grid,
            dct_qcoeffs_packed=_pack_coefficients(dct_qcoeffs),
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
        # We need modes and states in fractal order
        flat_modes = data.mode_grid.ravel()
        flat_states = data.state_grid.ravel()
        ordered_modes = flat_modes[order]
        ordered_states = flat_states[order]

        dct_qcoeffs = _unpack_coefficients(data.dct_qcoeffs_packed, nblocks, bs)
        recon_coeffs = _dequantize_dct(dct_qcoeffs, data.quality, ordered_modes, ordered_states)
        recon_blocks = idct2(recon_coeffs) + 128.0

        # Put blocks back into grid
        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks
        recon = unordered_recon_blocks.reshape(ny, nx, bs, bs).transpose(0, 2, 1, 3).reshape(h, w)

        for qcoeffs, qstep, pad_info in zip(data.residual_coeffs, data.residual_qsteps, data.residual_pads):
            resid = haar_idwt2(qcoeffs.astype(np.float64) * qstep, levels=self.residual_levels, pad=pad_info)
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
            # Pad to even dimensions for 4:2:0 subsampling
            h, w = arr.shape[:2]
            if h % 2 != 0 or w % 2 != 0:
                arr, _ = pad_to_multiple(arr, 2, mode="reflect")
            channels = rgb_to_ycbcr(arr, subsample=True)
        else:
            channels = split_channels(arr)

        encoded_channels = []
        for i, ch in enumerate(channels):
            # Chroma channels (Cb, Cr) can be quantized more aggressively
            ch_quality = quality
            if is_rgb and i > 0:
                # Subsampled chroma needs higher quality to maintain same perceptual level
                ch_quality = max(1, int(quality * 1.0))
            encoded_channels.append(self._encode_channel(ch, ch_quality))

        payload = {
            "version": 1,
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
            raise ValueError("Not a HFZ blob")
        payload = pickle.loads(lzma.decompress(blob[len(MAGIC) :]))
        if payload.get("version") != 1:
            raise ValueError("Unsupported codec version")

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
        # We can decode with defaults because the payload contains its own parameters.
        payload = pickle.loads(lzma.decompress(blob[len(MAGIC) :]))
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
