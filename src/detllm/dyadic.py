"""Dyadic-scale bookkeeping on the integer path.

A quantized tensor is carried as ``(data, m, k)`` with value ≈ data · m / 2^k.
``m``/``k`` are int64: per-token column vectors (shape [..., 1]) for
activations, per-channel vectors + scalar k for weights. All arithmetic here
is integer-only and uses the `intmath` primitives exclusively.
"""

from __future__ import annotations

import torch

from .intmath import (
    clamp_i8,
    floor_div,
    ilog2_floor,
    lshift_t,
    round_half_away_div,
    rshift_round,
    rshift_round_t,
)

I64 = torch.int64


def scaled_round_div(num: torch.Tensor, den: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """round(num · 2^e / den) with per-element (possibly negative) e. int64."""
    epos = torch.clamp(e, min=0)
    eneg = torch.clamp(-e, min=0)
    return round_half_away_div(lshift_t(num, epos), lshift_t(den, eneg))


def fit_dyadic_ratio(num: torch.Tensor, den, bits: int = 16):
    """Approximate num/den by m/2^e with m in [2^(bits-1), 2^bits).

    num, den: int64 tensors (or python int den), all >= 1.
    Returns (m, e) int64 tensors; e may be negative (ratio > 2^(bits-1)).
    Worst-case relative error 2^-bits.
    """
    num = num.to(I64)
    if not isinstance(den, torch.Tensor):
        den = torch.tensor(den, dtype=I64, device=num.device)
    den = den.to(I64).broadcast_to(num.shape)
    L = ilog2_floor(num) - ilog2_floor(den)
    e = (bits - 1) - L
    m = scaled_round_div(num, den, e)
    adj = torch.where(m >= (1 << bits), -1, 0) + torch.where(m < (1 << (bits - 1)), 1, 0)
    e = e + adj
    m = torch.where(adj != 0, scaled_round_div(num, den, e), m)
    over = m >= (1 << bits)
    m = torch.where(over, rshift_round(m, 1), m)
    e = e - over.to(I64)
    return m, e


def norm_scale(m: torch.Tensor, k: torch.Tensor, bits: int = 16):
    """Renormalize a dyadic scale so m lands in [2^(bits-1), 2^bits).

    Exact when m grows (left shift of k), rounded (rshift_round) when m
    shrinks. Keeps k drift bounded across long op chains.
    """
    m = m.to(I64)
    k = k.to(I64)
    sh = (bits - 1) - ilog2_floor(torch.clamp(m, min=1))
    up = torch.clamp(sh, min=0)
    down = torch.clamp(-sh, min=0)
    m2 = rshift_round_t(lshift_t(m, up), down)
    over = m2 >= (1 << bits)
    m2 = torch.where(over, rshift_round(m2, 1), m2)
    sh = sh - over.to(I64)
    return m2, k + sh


def quant_i8_pertoken(x: torch.Tensor, m: torch.Tensor, k: torch.Tensor):
    """Symmetric per-token int8 quantization of an int32/int64 activation.

    x: [..., C] integer tensor with per-token dyadic scale (m, k) [..., 1].
    Returns (x8 int8, m_out, k_out) with value ≈ x8 · m_out / 2^k_out.
    """
    x64 = x.to(I64)
    a = x64.abs().amax(dim=-1, keepdim=True)
    a = torch.clamp(a, min=1)
    x8 = clamp_i8(round_half_away_div(x64 * 127, a))
    m_out, e = fit_dyadic_ratio(a * m.to(I64), 127)
    return x8, m_out, k.to(I64) + e


def requant_i32_common(P: torch.Tensor, row_m, row_k, col_m, col_k: int, target_bits: int = 12):
    """Requantize a GEMM output (int32, per-row (row_m, row_k) × per-col
    (col_m, col_k) scales) to a common per-row scale at ~target_bits of
    magnitude, returned as int32.

    Shift-only (single rounding). Returns (y int32, m, k) per-row dyadic.
    """
    scaled = P.to(I64) * col_m.to(I64)  # per-row scale now (row_m, row_k+col_k)
    a = scaled.abs().amax(dim=-1, keepdim=True)
    a = torch.clamp(a, min=1)
    sh = torch.clamp(ilog2_floor(a) - (target_bits - 1), min=0)
    y = rshift_round_t(scaled, sh.broadcast_to(scaled.shape)).to(torch.int32)
    return y, row_m.to(I64), row_k.to(I64) + col_k - sh


def requant_i8_rowcol(P: torch.Tensor, row_m, row_k, col_m, col_k: int):
    """Requantize a GEMM output (per-row × per-col scales) straight to
    symmetric per-token int8 (division-based, uses the full ±127 range).

    Returns (y8 int8, m_out, k_out).
    """
    scaled = P.to(I64) * col_m.to(I64)
    a = scaled.abs().amax(dim=-1, keepdim=True)
    a = torch.clamp(a, min=1)
    y8 = clamp_i8(round_half_away_div(scaled * 127, a))
    m_out, e = fit_dyadic_ratio(a * row_m.to(I64), 127)
    return y8, m_out, row_k.to(I64) + col_k + e


def requant_i8_static(P: torch.Tensor, row_m, row_k, col_m, col_k: int,
                      s_m: int, s_k: int):
    """Requantize a GEMM output to int8 with a STATIC dyadic target scale
    s_m/2^s_k (used for K and V heading into the cache).

    y = round(P · col_m · row_m / 2^(row_k+col_k) · 2^s_k / s_m), clamped.
    """
    num = P.to(I64) * col_m.to(I64) * row_m.to(I64)
    shift = row_k.to(I64) + col_k - s_k  # may be negative
    up = torch.clamp(-shift, min=0)
    down = torch.clamp(shift, min=0)
    num = lshift_t(num, up.broadcast_to(num.shape))
    # single fused round: divide by (s_m << down)
    den = lshift_t(torch.tensor(s_m, dtype=I64, device=P.device).broadcast_to(num.shape),
                   down.broadcast_to(num.shape))
    return clamp_i8(round_half_away_div(num, den))


# ---------------------------------------------------------------------------
# Residual stream: int32 data, per-token power-of-two scale (m=1, k_res).
# ---------------------------------------------------------------------------

RES_MAX_BITS = 30  # keep max|h| < 2^30
RES_MIN_BITS = 24  # upshift (exact) if max|h| < 2^24


def residual_renorm(h64: torch.Tensor, k_res: torch.Tensor):
    """Bring per-row max|h| into [2^RES_MIN_BITS, 2^RES_MAX_BITS), adjusting
    k_res. Left shifts are exact; right shifts use rshift_round."""
    g = h64.abs().amax(dim=-1, keepdim=True)
    lg = ilog2_floor(torch.clamp(g, min=1))
    sh_down = torch.clamp(lg - (RES_MAX_BITS - 1), min=0)
    sh_up = torch.where(g > 0, torch.clamp((RES_MIN_BITS + 2) - lg, min=0), 0)
    sh_up = torch.where(sh_down > 0, torch.zeros_like(sh_up), sh_up)
    h64 = rshift_round_t(h64, sh_down.broadcast_to(h64.shape))
    h64 = lshift_t(h64, sh_up.broadcast_to(h64.shape))
    return h64.to(torch.int32), k_res.to(I64) - sh_down + sh_up


def residual_from_delta(d: torch.Tensor, dm: torch.Tensor, dk: torch.Tensor):
    """Turn a dyadic-scaled delta into residual-grid form (h, k_res)."""
    h64 = d.to(I64) * dm.to(I64)
    return residual_renorm(h64, dk.to(I64))


def residual_add(h: torch.Tensor, k_res: torch.Tensor, d: torch.Tensor,
                 dm: torch.Tensor, dk: torch.Tensor):
    """h (int32, scale 2^-k_res) += d·dm/2^dk. Returns (h', k_res').

    Alignment rule (spec §6.8): shift the finer operand down to the coarser k
    with rshift_round; shifting up is exact. Then renormalize.
    """
    d64 = d.to(I64) * dm.to(I64)
    rel = dk.to(I64) - k_res.to(I64)  # >0: delta finer -> shift delta down
    down = torch.clamp(rel, min=0)
    up = torch.clamp(-rel, min=0)
    d_aligned = lshift_t(rshift_round_t(d64, down.broadcast_to(d64.shape)),
                         up.broadcast_to(d64.shape))
    return residual_renorm(h.to(I64) + d_aligned, k_res)
