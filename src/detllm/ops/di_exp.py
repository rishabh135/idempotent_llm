"""DI-Exp — paper Algorithm 1: integer exponential via shift decomposition.

Given x ≤ 0 (already max-subtracted) with per-row dyadic scale m/2^k, computes
result ≈ e^(x·m/2^k) · (-t) where t = -round(2^k / m_f) and
m_f = m + (m>>1) - (m>>4) ≈ 1.4375·m ≈ m·log2(e)  (the paper's constant;
log2(e) = 1.4427, so the approximation is 0.36% low — kept verbatim per spec).

The implicit output scale 1/(-t) is NOT dyadic; it is always consumed by a
ratio (softmax / sigmoid), where it cancels. Returns (result, tpos = -t).
"""

from __future__ import annotations

import torch

from ..intmath import ashr, ashr_t, floor_div, lshift_t, round_half_away_div

I64 = torch.int64


def di_exp(x: torch.Tensor, m: torch.Tensor, k: torch.Tensor):
    """x: int tensor [..., C], all values ≤ 0. m, k: int64 [..., 1] (k ≤ 62).

    Returns (result int64 [..., C] in [0, tpos], tpos int64 [..., 1]).
    """
    x = x.to(I64)
    m = m.to(I64)
    k = k.to(I64)
    m_f = m + ashr(m, 1) - ashr(m, 4)
    one = torch.ones_like(k)
    tpos = torch.clamp(round_half_away_div(lshift_t(one, k), m_f), min=1)
    t = -tpos
    q = floor_div(x, t.broadcast_to(x.shape))  # ≥ 0 since x ≤ 0, t < 0
    r = x - q * t  # in (t, 0]
    unshifted = ashr(r, 1) - t  # in (tpos/2, tpos]
    result = ashr_t(unshifted, torch.clamp(q, max=62))
    return result, tpos
