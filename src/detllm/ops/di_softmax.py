"""DI-Softmax — paper Algorithm 2 (+ optional Eq. 10 clipping for ablation).

Input: integer scores with per-row dyadic scale (m, k) and a validity mask
(causal + padding). Masked positions are excluded from the row max and the
sum via integer sentinel handling — no float -inf anywhere.

Output: probabilities in u8 range [0, 128] (stored int32) with the fixed
dyadic scale m=1, k=7 (value = p / 2^7), per the algorithm's convention
(IntDiv(·,·,p_out) with p_out = 8 → m_out = 1, k_out = p_out - 1 = 7).

Default clip c = None (∞): DI-Exp on full-precision scores underflows to 0
below ≈ -21 by itself, which is strictly more accurate than clipping (see
NOTES.md §4). c, when given for ablation, is applied per Eq. 10 as an
integer clamp at -round(c · 2^k / m) in raw score units.
"""

from __future__ import annotations

import torch

from ..intmath import lshift_t, round_half_away_div
from .di_exp import di_exp

I64 = torch.int64
P_OUT_BITS = 8
PROB_ONE = 1 << (P_OUT_BITS - 1)  # 128: fixed-point 1.0 of the output


def di_softmax(scores: torch.Tensor, m: torch.Tensor, k: torch.Tensor,
               valid: torch.Tensor, clip_c: int | None = None):
    """scores: int tensor [..., T]; m, k: int64 [..., 1]; valid: bool [..., T].

    Returns probs int32 [..., T] in [0, 128], fixed scale (1, 7).
    Rows with no valid positions return all zeros.
    """
    x = scores.to(I64)
    m = m.to(I64)
    k = k.to(I64)
    neg_inf = torch.tensor(torch.iinfo(I64).min + 1, dtype=I64, device=x.device)
    xm = torch.where(valid, x, neg_inf)
    row_max = xm.amax(dim=-1, keepdim=True)
    any_valid = valid.any(dim=-1, keepdim=True)
    row_max = torch.where(any_valid, row_max, torch.zeros_like(row_max))
    xd = x - row_max
    xd = torch.where(valid, xd, torch.zeros_like(xd))  # dummy 0 for masked
    if clip_c is not None:
        # Eq. 10: confine the range to (max - c, max): clamp at -round(c·2^k/m)
        one = torch.ones_like(k)
        c_int = round_half_away_div(clip_c * lshift_t(one, k), m)
        xd = torch.maximum(xd, -c_int.broadcast_to(xd.shape))
    exps, _tpos = di_exp(xd, m, k)
    exps = torch.where(valid, exps, torch.zeros_like(exps))
    denom = torch.clamp(exps.sum(dim=-1, keepdim=True), min=1)
    probs = round_half_away_div(exps << (P_OUT_BITS - 1), denom)
    return probs.to(torch.int32)
