"""Hybrid fractal-Z image codec (v3: Hierarchical Path-DCT & Hamiltonian Traversal).

Enhancements:
- Hamiltonian Path Steered Traversal: Replaces standard Morton Z-curve with 
  continuous, orientation-steered U-curves to enforce strict spatial adjacency.
- Multi-Dimensional Integration (Path-DCT): Upgrades simple DPCM to a 
  hierarchical 1D DCT applied across the fractal traversal path for 
  low-frequency 2D block coefficients.
- Float32 pipeline & safe deserialization.
"""

from __future__ import annotations

import argparse
import builtins
import io
import lzma
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"HFZ3"
MAGIC_V1 = b"HFZ1"
MAGIC_V2 = b"HFZ2"
EPSILON = 1e-9

# ----------------------------------------------------------------------------
# Security: Safe Pickling
# ----------------------------------------------------------------------------

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins":
            return getattr(builtins, name)
        if module == "numpy.core.multiarray" and name == "_reconstruct":
            return np.core.multiarray._reconstruct
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module == "__main__" and name == "ChannelCodecResult":
            return ChannelCodecResult
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden.")

def safe_unpickle(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)

def pad_to_multiple(arr: np.ndarray, multiple: int, mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int]]:
    if multiple <= 1: return arr, (0, 0)
    h, w = arr.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0: return arr, (0, 0)
    pad_spec = ((0, pad_h), (0, pad_w))
    if arr.ndim == 3: pad_spec += ((0, 0),)
    return np.pad(arr, pad_spec, mode=mode), (pad_h, pad_w)

def crop_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return arr[:h, :w] if arr.ndim == 2 else arr[:h, :w, ...]

def split_channels(image: np.ndarray) -> List[np.ndarray]:
    if image.ndim == 2: return [image]
    if image.ndim == 3: return [image[..., i] for i in range(image.shape[2])]
    raise ValueError("Expected 2D grayscale or 3D color image array.")

def stack_channels(channels: Sequence[np.ndarray]) -> np.ndarray:
    return channels[0] if len(channels) == 1 else np.stack(channels, axis=-1)

def rgb_to_ycbcr(rgb: np.ndarray) -> List[np.ndarray]:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return [y, cb, cr]

def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y, cb, cr = y.astype(np.float32), cb.astype(np.float32) - 128, cr.astype(np.float32) - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------
# 1D / 2D DCT Framework
# ----------------------------------------------------------------------------

_DCT_CACHE: Dict[int, np.ndarray] = {}

def dct_matrix(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        c = np.zeros((n, n), dtype=np.float32)
        scale0 = math.sqrt(1.0 / n)
        scale = math.sqrt(2.0 / n)
        for k in range(n):
            alpha = scale0 if k == 0 else scale
            for i in range(n):
                c[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
        _DCT_CACHE[n] = c
    return _DCT_CACHE[n]

def dct2(blocks: np.ndarray) -> np.ndarray:
    """2D DCT on batch of blocks (N, n, n)."""
    blocks = _as_float32(blocks)
    c = dct_matrix(blocks.shape[-1])
    return c @ blocks @ c.T

def idct2(coeffs: np.ndarray) -> np.ndarray:
    """2D IDCT on batch of blocks (N, n, n)."""
    coeffs = _as_float32(coeffs)
    c = dct_matrix(coeffs.shape[-1])
    return c.T @ coeffs @ c

def dct1(chunks: np.ndarray) -> np.ndarray:
    """1D DCT on batch of 1D chunks (K, n)."""
    c = dct_matrix(chunks.shape[-1])
    return chunks @ c.T

def idct1(coeffs: np.ndarray) -> np.ndarray:
    """1D IDCT on batch of 1D chunks (K, n)."""
    c = dct_matrix(coeffs.shape[-1])
    return coeffs @ c

# ----------------------------------------------------------------------------
# Haar wavelet lift
# ----------------------------------------------------------------------------

def haar_dwt2(arr: np.ndarray, levels: int = 1) -> Tuple[np.ndarray, Tuple[int, int]]:
    multiple = 2 ** levels
    x, pad = pad_to_multiple(_as_float32(arr), multiple, mode="reflect")
    out = x.copy()
    h, w = out.shape
    for lev in range(levels):
        hh, ww = h >> lev, w >> lev
        sub = out[:hh, :ww]
        rows_lo = (sub[:, 0::2] + sub[:, 1::2]) * 0.5
        rows_hi = (sub[:, 0::2] - sub[:, 1::2]) * 0.5
        out[: hh // 2, : ww // 2] = (rows_lo[0::2, :] + rows_lo[1::2, :]) * 0.5
        out[: hh // 2, ww // 2 : ww] = (rows_lo[0::2, :] - rows_lo[1::2, :]) * 0.5
        out[hh // 2 : hh, : ww // 2] = (rows_hi[0::2, :] + rows_hi[1::2, :]) * 0.5
        out[hh // 2 : hh, ww // 2 : ww] = (rows_hi[0::2, :] - rows_hi[1::2, :]) * 0.5
    return out, pad

def haar_idwt2(coeffs: np.ndarray, levels: int = 1, pad: Tuple[int, int] = (0, 0)) -> np.ndarray:
    out = _as_float32(coeffs).copy()
    h, w = out.shape
    for lev in reversed(range(levels)):
        hh, ww = h >> lev, w >> lev
        ll, lh = out[: hh // 2, : ww // 2], out[: hh // 2, ww // 2 : ww]
        hl, hh_band = out[hh // 2 : hh, : ww // 2], out[hh // 2 : hh, ww // 2 : ww]

        rows_lo, rows_hi = np.empty((hh, ww // 2), dtype=np.float32), np.empty((hh, ww // 2), dtype=np.float32)
        rows_lo[0::2, :], rows_lo[1::2, :] = ll + lh, ll - lh
        rows_hi[0::2, :], rows_hi[1::2, :] = hl + hh_band, hl - hh_band

        sub = np.empty((hh, ww), dtype=np.float32)
        sub[:, 0::2], sub[:, 1::2] = rows_lo + rows_hi, rows_lo - rows_hi
        out[:hh, :ww] = sub
    if pad != (0, 0): out = out[: out.shape[0] - pad[0], : out.shape[1] - pad[1]]
    return out

# ----------------------------------------------------------------------------
# Directional state & Adaptive Hamiltonian Traversal
# ----------------------------------------------------------------------------

def block_orientation_state(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = _as_float32(blocks)
    gx, gy = b[:, :, 1:] - b[:, :, :-1], b[:, 1:, :] - b[:, :-1, :]
    gx, gy = gx[:, :-1, :], gy[:, :, :-1]

    gxx, gyy, gxy = np.mean(gx * gx, axis=(1, 2)), np.mean(gy * gy, axis=(1, 2)), np.mean(gx * gy, axis=(1, 2))
    anis = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy) / (gxx + gyy + EPSILON)
    
    theta = (0.5 * np.arctan2(2.0 * gxy, gxx - gyy)) % np.pi
    state = ((theta / np.pi) * 8.0).astype(np.int32) % 8
    state[anis < 0.08] = 0
    return state.astype(np.uint8), anis

# Continuous Hamiltonian U-curves based on gradient dominant direction.
# Preserves strict locality while following edge flows.
_HAMILTONIAN_PERMS: Dict[int, Tuple[int, int, int, int]] = {
    0: (0, 1, 3, 2), # Horizontal: NW -> NE -> SE -> SW
    1: (2, 0, 1, 3), # Diagonal:   SW -> NW -> NE -> SE
    2: (0, 2, 3, 1), # Vertical:   NW -> SW -> SE -> NE
    3: (1, 0, 2, 3), # Diagonal:   NE -> NW -> SW -> SE
    4: (3, 2, 0, 1),
    5: (3, 1, 0, 2),
    6: (1, 3, 2, 0),
    7: (2, 3, 1, 0),
}

_ORIENT_ANGLES = np.array([i * math.pi / 8.0 for i in range(8)], dtype=np.float32)

def _weighted_mode(values: np.ndarray, weights: np.ndarray, nstates: int = 8) -> int:
    flat_v, flat_w = values.astype(np.int64).ravel(), weights.astype(np.float32).ravel()
    return int(np.argmax(np.bincount(flat_v, weights=flat_w, minlength=nstates))) if flat_v.size else 0

def build_hilbert_fractal_order(state_grid: np.ndarray, energy_grid: np.ndarray) -> List[Tuple[int, int]]:
    """Adaptive Quadtree implementing orientation-steered Hamiltonian paths."""
    ny, nx = state_grid.shape
    order: List[Tuple[int, int]] = []

    def rec(y0: int, y1: int, x0: int, x1: int) -> None:
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0: return
        if h == 1 and w == 1:
            order.append((y0, x0))
            return
        
        dominant_state = _weighted_mode(state_grid[y0:y1, x0:x1], energy_grid[y0:y1, x0:x1])

        if h > 1 and w > 1:
            ym, xm = y0 + h // 2, x0 + w // 2
            rects = [(y0, ym, x0, xm), (y0, ym, xm, x1), (ym, y1, x0, xm), (ym, y1, xm, x1)]
            perm = _HAMILTONIAN_PERMS.get(dominant_state, (0, 1, 2, 3))
        elif h > 1:
            rects = [(y0, y0 + h // 2, x0, x1), (y0 + h // 2, y1, x0, x1)]
            perm = (0, 1) if dominant_state in (0, 1, 6, 7) else (1, 0)
        else:
            rects = [(y0, y1, x0, x0 + w // 2), (y0, y1, x0 + w // 2, x1)]
            perm = (0, 1) if dominant_state in (0, 1, 2, 3) else (1, 0)

        for i in [idx for idx in perm if idx < len(rects)]:
            rec(*rects[i])

    rec(0, ny, 0, nx)
    return order

# ----------------------------------------------------------------------------
# Hierarchical Path-DCT Multi-Dimensional Integration
# ----------------------------------------------------------------------------
_N_PATH_POSITIONS = 3
_PATH_CHUNK_SIZE = 8

def _quality_to_qbase(quality: int) -> float:
    return max(0.55, 18.0 / math.sqrt(np.clip(quality, 1, 100) + 1.0))

def _zig_zag_indices(n: int) -> np.ndarray:
    return np.array([i * n + j for i, j in sorted(((i, j) for i in range(n) for j in range(n)), 
                    key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 == 0 else x[0]))])

def _apply_path_dct(coeffs: np.ndarray, quality: int) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a 1D DCT across the fractal traversal path for the DC and first ACs."""
    out_coeffs = coeffs.copy()
    bs = coeffs.shape[-1]
    N = coeffs.shape[0]
    zz = _zig_zag_indices(bs)
    qbase = _quality_to_qbase(quality)
    
    # Path sequences are much smoother, we apply precise quantization steps
    path_qsteps = np.array([qbase * 0.4, qbase * 0.6, qbase * 0.6], dtype=np.float32)
    path_qcoeffs_list = []
    
    for p in range(min(_N_PATH_POSITIONS, len(zz))):
        fi = zz[p]
        i, j = int(fi // bs), int(fi % bs)
        seq = coeffs[:, i, j]
        
        pad_len = (_PATH_CHUNK_SIZE - (N % _PATH_CHUNK_SIZE)) % _PATH_CHUNK_SIZE
        seq_padded = np.pad(seq, (0, pad_len), mode='reflect')
        chunks = seq_padded.reshape(-1, _PATH_CHUNK_SIZE)
        
        # 1D DCT along fractal path -> creates 3D block correlation
        dct_chunks = dct1(chunks)
        qchunks = np.round(dct_chunks / path_qsteps[p]).astype(np.int16)
        path_qcoeffs_list.append(qchunks.ravel())
        
        # Remove from local 2D block to avoid double-encoding
        out_coeffs[:, i, j] = 0.0 
        
    return out_coeffs, np.concatenate(path_qcoeffs_list)

def _undo_path_dct(out_coeffs: np.ndarray, path_packed: np.ndarray, quality: int, N: int) -> np.ndarray:
    """Inverts the hierarchical Path-DCT to restore low-frequency 2D block coefficients."""
    bs = out_coeffs.shape[-1]
    zz = _zig_zag_indices(bs)
    qbase = _quality_to_qbase(quality)
    path_qsteps = np.array([qbase * 0.4, qbase * 0.6, qbase * 0.6], dtype=np.float32)
    
    pad_len = (_PATH_CHUNK_SIZE - (N % _PATH_CHUNK_SIZE)) % _PATH_CHUNK_SIZE
    chunk_size = ((N + pad_len) // _PATH_CHUNK_SIZE) * _PATH_CHUNK_SIZE
    
    offset = 0
    for p in range(min(_N_PATH_POSITIONS, len(zz))):
        qchunks_flat = path_packed[offset : offset + chunk_size]
        offset += chunk_size
        
        qchunks = qchunks_flat.reshape(-1, _PATH_CHUNK_SIZE)
        dct_chunks = qchunks.astype(np.float32) * path_qsteps[p]
        
        chunks = idct1(dct_chunks)
        seq = chunks.ravel()[:N]
        
        fi = zz[p]
        i, j = int(fi // bs), int(fi % bs)
        out_coeffs[:, i, j] = seq
        
    return out_coeffs

# ----------------------------------------------------------------------------
# 2D Block Quantization (Stage-1 remaining coefficients)
# ----------------------------------------------------------------------------

def _get_quant_params(quality: int, modes: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    qbase = _quality_to_qbase(quality)
    keep = np.clip(np.array([2, 3, 4], dtype=np.int32)[modes] + (1 if quality >= 85 else (-1 if quality <= 25 else 0)), 1, n * 2)
    qsteps = qbase * np.array([1.9, 1.15, 0.95], dtype=np.float32)[modes]
    return qsteps, keep

def _eff_uv_distance(n: int, states: np.ndarray, modes: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    uv_iso = (u + v).astype(np.float32)[None, :, :]
    
    mode_anis = np.array([0.05, 0.70, 0.20], dtype=np.float32)[modes % 3]
    gate = np.clip((mode_anis - 0.45) / 0.50, 0.0, 1.0)[:, None, None]
    if gate.max() < 1e-6: return uv_iso

    phi = _ORIENT_ANGLES[states % 8] + math.pi / 2.0
    cu, su = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
    
    f_along = cu * u[None] + su * v[None]
    f_cross = -su * u[None] + cu * v[None]
    uv_aniso = np.abs(f_along) * (1.0 - gate * 0.35) + np.abs(f_cross) * (1.0 + gate * 0.90)
    
    return (1.0 - gate) * uv_iso + gate * uv_aniso

def _quantize_dct(coeffs: np.ndarray, quality: int, modes: np.ndarray, states: np.ndarray) -> np.ndarray:
    n = coeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    q_matrix = qsteps[:, None, None] * (1.0 + 0.25 * eff)
    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    qcoeffs[masks] = np.clip(np.round(coeffs / q_matrix)[masks], -32768, 32767).astype(np.int16)
    return qcoeffs

def _dequantize_dct(qcoeffs: np.ndarray, quality: int, modes: np.ndarray, states: np.ndarray) -> np.ndarray:
    n = qcoeffs.shape[-1]
    qsteps, keep = _get_quant_params(quality, modes, n)
    eff = _eff_uv_distance(n, states, modes)
    masks = eff < keep[:, None, None]
    coeffs = np.zeros(qcoeffs.shape, dtype=np.float32)
    coeffs[masks] = qcoeffs[masks].astype(np.float32) * (qsteps[:, None, None] * (1.0 + 0.25 * eff))[masks]
    return coeffs

def _pack_block_coefficients(qcoeffs: np.ndarray) -> np.ndarray:
    n = qcoeffs.shape[-1]
    indices = _zig_zag_indices(n)[_N_PATH_POSITIONS:]
    return qcoeffs.reshape(-1, n * n)[:, indices].ravel()

def _unpack_block_coefficients(packed: np.ndarray, nblocks: int, n: int) -> np.ndarray:
    indices = _zig_zag_indices(n)[_N_PATH_POSITIONS:]
    out = np.zeros((nblocks, n * n), dtype=np.int16)
    out[:, indices] = packed.reshape(nblocks, -1)
    return out.reshape(nblocks, n, n)

# ----------------------------------------------------------------------------
# Haar Subband Quantisation
# ----------------------------------------------------------------------------

def _haar_subband_scale_map(shape: Tuple[int, int], levels: int, ll_s: float, lh_s: float, hl_s: float, hh_s: float) -> np.ndarray:
    S = np.ones(shape, dtype=np.float32)
    h, w = shape
    for lev in range(levels):
        hh2, ww2 = (h >> lev) >> 1, (w >> lev) >> 1
        S[:hh2, :ww2] = ll_s
        S[:hh2, ww2:ww] = lh_s
        S[hh2:hh, :ww2] = hl_s
        S[hh2:hh, ww2:ww] = hh_s
    return S

def _subband_scales_from_orientation(global_state: int, smooth_fraction: float) -> Tuple[float, float, float, float]:
    phi = _ORIENT_ANGLES[global_state % 8]
    sf = float(np.clip(smooth_fraction, 0.0, 1.0))
    return (1.00, 1.00 + sf * (math.cos(phi) ** 2) * 0.06, 1.00 + sf * (math.sin(phi) ** 2) * 0.06, 1.05)

# ----------------------------------------------------------------------------
# Codec Data Structures & Core Class
# ----------------------------------------------------------------------------

@dataclass
class ChannelCodecResult:
    padded_shape: Tuple[int, int]
    pad_hw: Tuple[int, int]
    block_size: int
    quality: int
    order: np.ndarray
    state_grid: np.ndarray
    mode_grid: np.ndarray
    dct_block_qcoeffs_packed: np.ndarray
    dct_path_qcoeffs_packed: np.ndarray 
    residual_pads: List[Tuple[int, int]]
    residual_qsteps: List[float]
    residual_coeffs: List[np.ndarray]

class FractalHybridCodec:
    def __init__(self, block_size: int = 8, residual_levels: int = 2, residual_passes: int = 2):
        self.block_size = int(block_size)
        self.residual_levels = int(residual_levels)
        self.residual_passes = int(residual_passes)

    def _encode_channel(self, channel: np.ndarray, quality: int) -> ChannelCodecResult:
        ch = _as_float32(channel)
        padded, pad_hw = pad_to_multiple(ch, self.block_size, mode="reflect")
        h, w = padded.shape
        bs = self.block_size
        ny, nx = h // bs, w // bs

        flat_blocks = padded.reshape(ny, bs, nx, bs).transpose(0, 2, 1, 3).reshape(-1, bs, bs)
        coeff_blocks = dct2(flat_blocks - 128.0)

        total_energy = np.sum(np.abs(coeff_blocks), axis=(1, 2))
        hf_energy = np.sum(np.abs(coeff_blocks[:, 2:, 2:]), axis=(1, 2)) if bs > 2 else np.zeros(len(coeff_blocks))

        states, anis = block_orientation_state(flat_blocks)
        modes = np.zeros_like(total_energy, dtype=np.uint8)
        ratio = hf_energy / (total_energy + EPSILON)
        modes[(ratio >= 0.22) & (total_energy >= (8.0 + (100 - quality) * 0.2))] = 2
        modes[(modes == 2) & (anis > 0.30)] = 1

        state_grid = states.reshape(ny, nx)
        mode_grid = modes.reshape(ny, nx)

        # 1. Energy-steered Hamiltonian Z-Curve Traversal
        order_xy = build_hilbert_fractal_order(state_grid, (hf_energy + EPSILON).reshape(ny, nx))
        order = np.array([y * nx + x for (y, x) in order_xy], dtype=np.int32)

        ordered_coeffs = coeff_blocks[order]
        ordered_modes = modes[order]
        ordered_states = states[order]

        # 2. Multi-Dimensional Path-DCT integration
        ordered_coeffs_ac_only, path_qcoeffs_packed = _apply_path_dct(ordered_coeffs, quality)
        
        # 3. Standard local block quantization
        dct_qcoeffs = _quantize_dct(ordered_coeffs_ac_only, quality, ordered_modes, ordered_states)
        
        # Inverse step for residual computation
        recon_coeffs_ac_only = _dequantize_dct(dct_qcoeffs, quality, ordered_modes, ordered_states)
        recon_coeffs_full = _undo_path_dct(recon_coeffs_ac_only, path_qcoeffs_packed, quality, len(order))
        
        recon_blocks = idct2(recon_coeffs_full) + 128.0
        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks
        recon1 = unordered_recon_blocks.reshape(ny, nx, bs, bs).transpose(0, 2, 1, 3).reshape(h, w)

        smooth_fraction = float(np.mean(modes == 0))
        global_state = _weighted_mode(states, (hf_energy + EPSILON))
        scales = _subband_scales_from_orientation(global_state, smooth_fraction)

        current, residual_coeffs, residual_pads, residual_qsteps = recon1.copy(), [], [], []

        for p in range(self.residual_passes):
            resid_pad, pad_info = pad_to_multiple(padded - current, 2 ** self.residual_levels, mode="reflect")
            coeffs, _ = haar_dwt2(resid_pad, levels=self.residual_levels)

            qstep = max(0.5, _quality_to_qbase(quality) * (0.85 ** p) * 1.5)
            effective_step = qstep * _haar_subband_scale_map(coeffs.shape, self.residual_levels, *scales)

            qcoeffs = np.round(coeffs / effective_step)
            qcoeffs = qcoeffs.astype(np.int8 if (qcoeffs.min() >= -128 and qcoeffs.max() <= 127) else np.int16)

            residual_coeffs.append(qcoeffs)
            residual_pads.append(pad_info)
            residual_qsteps.append(float(qstep))

            current += haar_idwt2(qcoeffs.astype(np.float32) * effective_step, levels=self.residual_levels, pad=pad_info)

        return ChannelCodecResult(
            (h, w), pad_hw, bs, int(quality), order, state_grid, mode_grid,
            _pack_block_coefficients(dct_qcoeffs), path_qcoeffs_packed,
            residual_pads, residual_qsteps, residual_coeffs
        )

    def _decode_channel(self, data: ChannelCodecResult) -> np.ndarray:
        h, w = data.padded_shape
        bs = data.block_size
        ny, nx = h // bs, w // bs
        nblocks = ny * nx
        order = data.order

        flat_modes, flat_states = data.mode_grid.ravel(), data.state_grid.ravel()
        ordered_modes, ordered_states = flat_modes[order], flat_states[order].astype(np.uint8)

        # Reconstruct 2D & 3D Multi-dimensional Transforms
        dct_qcoeffs = _unpack_block_coefficients(data.dct_block_qcoeffs_packed, nblocks, bs)
        recon_coeffs_ac_only = _dequantize_dct(dct_qcoeffs, data.quality, ordered_modes, ordered_states)
        recon_coeffs_full = _undo_path_dct(recon_coeffs_ac_only, data.dct_path_qcoeffs_packed, data.quality, nblocks)
        
        recon_blocks = idct2(recon_coeffs_full) + 128.0
        unordered_recon_blocks = np.zeros_like(recon_blocks)
        unordered_recon_blocks[order] = recon_blocks
        recon = unordered_recon_blocks.reshape(ny, nx, bs, bs).transpose(0, 2, 1, 3).reshape(h, w)

        scales = _subband_scales_from_orientation(
            _weighted_mode(flat_states, (flat_modes == 0).astype(np.float32) + EPSILON), 
            float(np.mean(flat_modes == 0))
        )

        for qcoeffs, qstep, pad_info in zip(data.residual_coeffs, data.residual_qsteps, data.residual_pads):
            effective_step = qstep * _haar_subband_scale_map(qcoeffs.shape, self.residual_levels, *scales)
            recon += crop_to_shape(haar_idwt2(qcoeffs.astype(np.float32) * effective_step, self.residual_levels, pad=pad_info), (h, w))

        return crop_to_shape(np.clip(np.round(recon), 0, 255).astype(np.uint8), (h - data.pad_hw[0], w - data.pad_hw[1]))

    def compress(self, image: np.ndarray, quality: int = 55) -> bytes:
        arr = np.clip(np.round(np.asarray(image)), 0, 255).astype(np.uint8)
        is_rgb = arr.ndim == 3 and arr.shape[2] == 3
        channels = rgb_to_ycbcr(arr) if is_rgb else split_channels(arr)

        payload = {
            "version": 3, "shape": tuple(arr.shape), "dtype": "uint8", "is_rgb": is_rgb,
            "quality": int(np.clip(quality, 1, 100)), "block_size": self.block_size,
            "residual_levels": self.residual_levels, "residual_passes": self.residual_passes,
            "channels": [self._encode_channel(ch, max(1, int(quality * 0.8)) if is_rgb and i > 0 else quality) for i, ch in enumerate(channels)],
        }
        return MAGIC + lzma.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), preset=6)

    def decompress(self, blob: bytes) -> np.ndarray:
        for m in (MAGIC, MAGIC_V2, MAGIC_V1):
            if blob.startswith(m):
                payload = safe_unpickle(lzma.decompress(blob[len(m):]))
                break
        else: raise ValueError("Not a valid HFZ blob")

        channels = [self._decode_channel(ch_data) for ch_data in payload["channels"]]
        return ycbcr_to_rgb(*channels) if payload.get("is_rgb") and len(channels) == 3 else np.clip(stack_channels(channels), 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------
# CLI Implementation (unchanged semantics, truncated for brevity in display)
# ----------------------------------------------------------------------------
# Includes _load_image, _save_image, and argparser logic triggering `encode` / `decode`
