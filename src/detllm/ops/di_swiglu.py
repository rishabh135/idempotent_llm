"""DI-SwiGLU — paper Algorithm 3, without FSBR smoothing (α_smooth ≡ 1),
with one deliberate fix (see NOTES.md): the sigmoid is computed with the
numerically-stable two-branch form instead of the paper's shared row-max
frame.

The paper computes σ(x_i) = e^(x_i-M)/(e^(x_i-M) + e^(-M)) with M = row max.
When M is large (our un-smoothed gates reach |x| ≈ 8+), BOTH exponentials
underflow the DI-Exp output resolution (~1/t ≈ 2^-8) and σ collapses to 0
for moderately positive inputs where the true σ ≈ 0.9. The stable form

    σ(x) = tpos/(tpos + E)  if x ≥ 0        E = DI-Exp(-|x|) · scale tpos
    σ(x) = E/(tpos + E)     if x < 0

is elementwise (no row coupling), uses the same DI-Exp machinery, keeps
everything integer, and its argument -|x| never underflows in the region
where σ has curvature. Determinism/invariance unaffected (per-element op).

σ is quantized to [0, 128] (7 fractional bits) and multiplied by both
operands: y = x_gate · σ · x_up with dyadic bookkeeping per Algorithm 3
(m_out = m_gate·m_up, k_out = k_gate + k_up + p_out - 1).

Inputs must each carry a common per-row dyadic scale; the caller
requantizes GEMM outputs to ≤ ~12-bit magnitudes first
(dyadic.requant_i32_common) so gate·σ·up stays comfortably in int64.
"""

from __future__ import annotations

import torch

from ..intmath import round_half_away_div
from .di_exp import di_exp

I64 = torch.int64
SIG_BITS = 8
SIG_ONE = 1 << (SIG_BITS - 1)  # 128


def di_swiglu(gate: torch.Tensor, gm: torch.Tensor, gk: torch.Tensor,
              up: torch.Tensor, um: torch.Tensor, uk: torch.Tensor):
    """gate, up: int32 [..., C] with per-row dyadic scales (gm, gk), (um, uk).

    Returns (y int64 [..., C], m_out, k_out):
        value ≈ gate·σ(gate)·up · m_out / 2^k_out,
        m_out = gm·um,  k_out = gk + uk + (SIG_BITS - 1).
    """
    g64 = gate.to(I64)
    gm = gm.to(I64)
    gk = gk.to(I64)
    E, tpos = di_exp(-g64.abs(), gm, gk)  # e^(-|x|) · tpos, per-row tpos
    tpos_b = tpos.broadcast_to(E.shape)
    denom = torch.clamp(tpos_b + E, min=1)
    num = torch.where(g64 >= 0, tpos_b, E)
    sig = round_half_away_div(num << (SIG_BITS - 1), denom)  # [0, 128]
    y = g64 * sig * up.to(I64)
    m_out = gm * um.to(I64)
    k_out = gk + uk.to(I64) + (SIG_BITS - 1)
    return y, m_out, k_out
