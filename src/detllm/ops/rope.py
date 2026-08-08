"""Integer rotary embeddings (§6.6 — not in the paper; our design).

Offline (prepare.py): cos/sin tables for all positions are precomputed at
fixed dyadic precision — int16 values, single global k = ROPE_FRAC_BITS
(values in [-2^14, 2^14]) — using HF Qwen3's convention: for head_dim d,
inv_freq_j = base^(-2j/d) for j < d/2, and the angle vector is tiled twice
so cos/sin have length d (rotate_half pairing: (x_j, x_{j+d/2})).

Runtime: pure elementwise integer multiply-add against the position-indexed
table rows, one rshift_round at the end (single rounding after the add).
The input's dyadic scale is unchanged (|cos|,|sin| ≤ 1 → no overflow growth:
|out| ≤ |x|·2^ROPE_FRAC_BITS·√2 in int64 before the shift-back).
"""

from __future__ import annotations

import torch

from ..intmath import rshift_round

I64 = torch.int64
ROPE_FRAC_BITS = 14


def build_rope_tables(head_dim: int, max_pos: int, base: float):
    """OFFLINE ONLY (float allowed): int16 cos/sin tables [max_pos, head_dim]."""
    j = torch.arange(0, head_dim // 2, dtype=torch.float64)
    inv_freq = base ** (-2.0 * j / head_dim)
    pos = torch.arange(max_pos, dtype=torch.float64)
    ang = torch.outer(pos, inv_freq)  # [max_pos, d/2]
    ang = torch.cat([ang, ang], dim=-1)  # HF convention: tile, not interleave
    scale = float(1 << ROPE_FRAC_BITS)
    cos_t = torch.round(torch.cos(ang) * scale).to(torch.int16)
    sin_t = torch.round(torch.sin(ang) * scale).to(torch.int16)
    return cos_t, sin_t


def apply_int_rope(x: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor,
                   positions: torch.Tensor) -> torch.Tensor:
    """x: int tensor [..., T, d] (per-head Q or K rows, any dyadic scale —
    unchanged by rotation). cos_t/sin_t: int16 [max_pos, d].
    positions: int64 [T] or [..., T] absolute position indices.

    Returns int64 tensor, same shape/scale as x (one rshift_round by
    ROPE_FRAC_BITS after the multiply-add).
    """
    d = x.shape[-1]
    half = d // 2
    c = cos_t[positions].to(I64)  # [..., T, d]
    s = sin_t[positions].to(I64)
    x64 = x.to(I64)
    x1 = x64[..., :half]
    x2 = x64[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)  # rotate_half
    return rshift_round(x64 * c + rot * s, ROPE_FRAC_BITS)
