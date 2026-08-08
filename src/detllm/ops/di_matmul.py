"""DI-MatMul — paper Eqs. 2–8, symmetric variant (zp = 0 throughout).

`di_linear` is the building block used by the model: int8 activation
(per-token dyadic scale) × int8 weight (per-output-channel dyadic scale,
pre-transposed to [K, N]) → int32 accumulator plus its row/col scale pair.
Requantization to the consumer's preferred form (int8 per-token, int32
common-row-scale, or static-scale int8) lives in `dyadic` and is chosen at
each call site (§10 fusion notes).

`di_matmul` is the paper's end-to-end M(·) — quantize int32 input, GEMM,
requantize output to int8 — used by the operator golden tests.
"""

from __future__ import annotations

import torch

from ..backends import int_gemm
from ..dyadic import quant_i8_pertoken, requant_i8_rowcol


def di_linear(x8: torch.Tensor, xm: torch.Tensor, xk: torch.Tensor,
              w8_t: torch.Tensor, wm: torch.Tensor, wk: int, backend: str):
    """x8 [..., T, K] int8, (xm, xk) [..., T, 1]; w8_t [K, N] int8,
    wm int64 [N], wk int. Returns (P int32 [..., T, N], row (xm, xk),
    col (wm, wk)) — value ≈ P · xm·wm / 2^(xk+wk)."""
    P = int_gemm(x8, w8_t.expand(*x8.shape[:-2], *w8_t.shape) if x8.dim() > 2 else w8_t,
                 backend)
    return P, xm, xk, wm, wk


def di_matmul(x: torch.Tensor, xm: torch.Tensor, xk: torch.Tensor,
              w8_t: torch.Tensor, wm: torch.Tensor, wk: int, backend: str):
    """Full dynamic-quantized matmul: int32 activation in, int8 out.

    Returns (y8 int8, m_out, k_out) per-token.
    """
    x8, qm, qk = quant_i8_pertoken(x, xm, xk)
    P, rm, rk, cm, ck = di_linear(x8, qm, qk, w8_t, wm, wk, backend)
    return requant_i8_rowcol(P, rm, rk, cm, ck)
