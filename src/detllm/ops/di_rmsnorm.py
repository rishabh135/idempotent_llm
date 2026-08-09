"""DI-RMSNorm — paper Algorithm 4, adapted to a per-token-scaled input.

Our norm inputs carry a single per-token dyadic scale (not the paper's
per-channel scales, which stem from its FSBR stage), so the scale cancels
in x/rms(x) entirely and the core is scale-free:

    y ≈ x / sqrt(mean(x²))       (integer, via I-SQRT = intmath.isqrt)

γ handling (see docs/NOTES.md): hidden-dim norms have γ folded into the following
linear's weights offline, so `di_rmsnorm` returns the bare normalized value
with fixed scale (1, OUT_FRAC_BITS). The QK-Norm variant
(`di_rmsnorm_gamma`) multiplies a per-channel dyadic γ into the data at
runtime — the row scale stays uniform because the *data* absorbs γ.
eps is dropped (rms clamped ≥ 1); see docs/NOTES.md §7.

Overflow discipline: rows are pre-shifted so max|x'| has MSB ≤ bit 25, hence
Σx'² ≤ n·2^52 ≤ 2^62 for n ≤ 1024. Up-shifts are exact; down-shifts round.
The pre-shift cancels exactly (numerator and rms use the same x').
"""

from __future__ import annotations

import torch

from ..intmath import ilog2_floor, isqrt, lshift_t, round_half_away_div, rshift_round_t

I64 = torch.int64

OUT_FRAC_BITS = 20  # output scale (m=1, k=20); |y| ≤ sqrt(n)·2^20 ≤ 2^25 for n≤1024
_TARGET_HI = 25     # pre-shift rows so max|x'| has its MSB at bit 25


def di_rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """x: int32/int64 [..., n] with any per-token scale (it cancels).

    Returns y int32 [..., n]; value ≈ x/rms(x) with fixed dyadic scale
    (m=1, k=OUT_FRAC_BITS). All-zero rows return zeros.
    """
    x64 = x.to(I64)
    n = x64.shape[-1]
    g = x64.abs().amax(dim=-1, keepdim=True)
    lg = ilog2_floor(torch.clamp(g, min=1))
    sh = _TARGET_HI - lg  # >0: exact up-shift; <0: rounded down-shift
    up = torch.clamp(sh, min=0)
    down = torch.clamp(-sh, min=0)
    xp = lshift_t(rshift_round_t(x64, down.broadcast_to(x64.shape)),
                  up.broadcast_to(x64.shape))
    ssum = (xp * xp).sum(dim=-1, keepdim=True)
    mean = round_half_away_div(ssum, n)
    rms = torch.clamp(isqrt(mean), min=1)
    y = round_half_away_div(xp << OUT_FRAC_BITS, rms.broadcast_to(xp.shape))
    return y.to(torch.int32)


def di_rmsnorm_gamma(x: torch.Tensor, gamma_m: torch.Tensor, gamma_k: int):
    """QK-Norm variant (reduction over head_dim): (x/rms)·γ with γ given as
    signed per-channel dyadic mantissas gamma_m (int64 [n]) / 2^gamma_k.

    Returns (y int64 [..., n], k_out): value ≈ (x/rms)·γ_c / 2^k_out with
    uniform row scale (m=1, k_out = OUT_FRAC_BITS + gamma_k).
    |y| ≤ 2^25 · max|γ_m|; the caller rescales/quantizes next.
    """
    y = di_rmsnorm(x).to(I64) * gamma_m.to(I64)
    return y, OUT_FRAC_BITS + gamma_k
