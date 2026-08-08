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


# cuBLASLt's int8 CUTLASS kernels issue speculative reads past the end of
# their operands (observed up to 16KB by compute-sanitizer; the values are
# discarded, so results are unaffected). If an operand ends exactly at a
# mapped-region boundary this is an illegal access — so every operand handed
# to _int_mm must be backed by an allocation with guard rows after it.
GUARD_ROWS = 64


def guard_rows(x: torch.Tensor) -> torch.Tensor:
    """Copy a 2-D tensor into a taller backing buffer, return the prefix
    view (same shape/values; over-reads land in our allocation). Used once
    at weight-load time."""
    buf = x.new_zeros(x.shape[0] + GUARD_ROWS, x.shape[1])
    buf[: x.shape[0]] = x
    return buf[: x.shape[0]]


def _pad_to(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    # ALWAYS copies into a guarded backing buffer (see note above), even
    # when no shape padding is needed.
    out = x.new_zeros(*x.shape[:-2], rows + GUARD_ROWS, cols)
    out[..., : x.shape[-2], : x.shape[-1]] = x
    return out[..., :rows, :]


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
        if b.dim() == 2 and a.dim() > 2:
            # shared weight: fold batch dims into M (one big exact GEMM)
            lead = a.shape[:-1]
            out = int_gemm(a.reshape(-1, a.shape[-1]), b, backend)
            return out.reshape(*lead, b.shape[-1])
        M, K = a.shape[-2], a.shape[-1]
        N = b.shape[-1]
        Mp, Kp, Np = max(32, _ceil_to(M, 32)), _ceil_to(K, 8), _ceil_to(N, 8)
        if a.dim() == 2 and Mp * Np >= (1 << 31):
            # cuBLASLt indexes the output with 32-bit math; slab rows so each
            # call stays below ~2^30 elements (row slabs are bit-exact).
            # Write into one preallocated output to halve peak memory.
            slab = max(32, ((1 << 30) // Np) // 32 * 32)
            out = torch.empty(M, N, dtype=torch.int32, device=a.device)
            for i in range(0, M, slab):
                out[i:i + slab] = int_gemm(a[i:i + slab], b, backend)
            return out
        if a.dim() == 2:
            # b in the 2-D path is always a weight matrix, guarded at load —
            # skip the copy when its shape already satisfies the constraints
            bp = b if (K == Kp and N == Np) else _pad_to(b, Kp, Np)
            return torch._int_mm(_pad_to(a, Mp, Kp), bp)[:M, :N]
        assert a.shape[:-2] == b.shape[:-2], (a.shape, b.shape)
        if M <= 4:
            # decode fast path: integer multiply-sum. Exact and
            # order-independent (integer addition), so bit-identical to the
            # GEMM; avoids 32x M-padding waste and per-slice kernel launches.
            prod = a.to(torch.int32).unsqueeze(-2) * b.to(torch.int32).transpose(-1, -2).unsqueeze(-3)
            return prod.sum(dim=-1, dtype=torch.int32)
        flat_a = a.reshape(-1, M, K)
        flat_b = b.reshape(-1, K, N)
        out = torch.empty(flat_a.shape[0], Mp, Np, dtype=torch.int32, device=a.device)
        for i in range(flat_a.shape[0]):
            # per-slice guarded padding (2-D prefix views stay contiguous)
            out[i] = torch._int_mm(_pad_to(flat_a[i], Mp, Kp),
                                   _pad_to(flat_b[i], Kp, Np))
        return out.reshape(*a.shape[:-2], Mp, Np)[..., :M, :N]
    raise ValueError(f"unknown backend {backend!r}")


def int_gemm_u8i8(p_u8: torch.Tensor, b: torch.Tensor, backend: str) -> torch.Tensor:
    """Exact GEMM for unsigned-range left operand (values in [0, 128], stored
    int32) against int8 b: computes (p-64)@b + 64·colsum(b), which equals
    p@b exactly in integer arithmetic. Shared by both backends so divergence
    stays inside int_gemm."""
    from .intmath import DEBUG_CHECKS
    if DEBUG_CHECKS:
        assert int(p_u8.min()) >= 0 and int(p_u8.max()) <= 128
    shifted = (p_u8 - 64).to(torch.int8)
    core = int_gemm(shifted, b, backend)
    corr = b.to(torch.int32).sum(dim=-2, keepdim=True) * 64
    return core + corr


def int_gemm_u16i8(p_u16: torch.Tensor, b: torch.Tensor, backend: str) -> torch.Tensor:
    """Exact GEMM for 15-bit probabilities (values in [0, 2^14], stored int32)
    against int8 b, using two int8 GEMMs: p = hi·2^7 + lo with hi in [0, 128]
    (−64 trick) and lo in [0, 127] (plain int8). Returns int64 (the shifted
    hi part can exceed int32 for long sequences)."""
    from .intmath import DEBUG_CHECKS
    if DEBUG_CHECKS:
        assert int(p_u16.min()) >= 0 and int(p_u16.max()) <= (1 << 14)
    hi = p_u16 >> 7
    lo = p_u16 - (hi << 7)
    hi_part = int_gemm_u8i8(hi, b, backend).to(torch.int64)
    lo_part = int_gemm(lo.to(torch.int8), b, backend).to(torch.int64)
    return (hi_part << 7) + lo_part
