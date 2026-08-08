"""Integer GEMM dispatch — the ONLY place where backend divergence exists.

Both backends compute the mathematically identical exact int8×int8→int32
product-sum; integer accumulation is order-independent, so results are
bit-identical by construction (verified by §9.3 tests, not assumed).

- reference: CPU, cast to int32 and use torch.matmul (exact, slow).
- cuda: torch._int_mm (cuBLASLt int8 tensor-core GEMM, exact int32 accum),
  with deterministic zero padding to satisfy shape constraints probed on
  this torch build: M ≡ 0 (mod 32) and ≥ 32, K ≡ 0 (mod 8), N ≡ 0 (mod 8).
"""

from __future__ import annotations

import torch

BACKENDS = ("reference", "cuda")


def _pad_to(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if x.shape[-2] == rows and x.shape[-1] == cols:
        return x
    out = x.new_zeros(*x.shape[:-2], rows, cols)
    out[..., : x.shape[-2], : x.shape[-1]] = x
    return out


def _ceil_to(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


def int_gemm(a: torch.Tensor, b: torch.Tensor, backend: str) -> torch.Tensor:
    """Exact integer GEMM: a [*, M, K] int8 × b [*, K, N] int8 -> int32.

    Batched dims (if any) must match exactly (no broadcasting).
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8, (a.dtype, b.dtype)
    assert a.shape[-1] == b.shape[-2]
    if backend == "reference":
        return torch.matmul(a.to(torch.int32), b.to(torch.int32))
    if backend == "cuda":
        M, K = a.shape[-2], a.shape[-1]
        N = b.shape[-1]
        Mp, Kp, Np = max(32, _ceil_to(M, 32)), _ceil_to(K, 8), _ceil_to(N, 8)
        ap = _pad_to(a, Mp, Kp)
        bp = _pad_to(b, Kp, Np)
        if a.dim() == 2:
            return torch._int_mm(ap, bp)[:M, :N]
        flat_a = ap.reshape(-1, Mp, Kp)
        flat_b = bp.reshape(-1, Kp, Np)
        out = torch.empty(flat_a.shape[0], Mp, Np, dtype=torch.int32, device=a.device)
        for i in range(flat_a.shape[0]):
            out[i] = torch._int_mm(flat_a[i], flat_b[i])
        return out.reshape(*a.shape[:-2], Mp, Np)[..., :M, :N]
    raise ValueError(f"unknown backend {backend!r}")


def int_gemm_u8i8(p_u8: torch.Tensor, b: torch.Tensor, backend: str) -> torch.Tensor:
    """Exact GEMM for unsigned-range left operand (values in [0, 128], stored
    int32) against int8 b: computes (p-64)@b + 64·colsum(b), which equals
    p@b exactly in integer arithmetic. Shared by both backends so divergence
    stays inside int_gemm."""
    assert int(p_u8.min()) >= 0 and int(p_u8.max()) <= 128
    shifted = (p_u8 - 64).to(torch.int8)
    core = int_gemm(shifted, b, backend)
    corr = b.to(torch.int32).sum(dim=-2, keepdim=True) * 64
    return core + corr
