"""Loop-heavy integer primitives as CUSTOM OPS backed by tiny Triton kernels.

Motivation (§10, compile-time AND runtime): `isqrt` is a 32-iteration
bit-loop and `ilog2_floor` a 6-step binary reduction. Traced by dynamo they
unroll into ~200 and ~24 graph nodes per call site, inflating every
compilation (inductor scheduling is superlinear in graph size). As custom
ops, each is ONE opaque node; the loops live inside a single Triton kernel.

Bit-exactness: the kernels implement the identical bitwise algorithms as
`intmath.isqrt` / `intmath.ilog2_floor`; the CPU registration IS the
original tensor-loop code, so the reference backend is unchanged by
construction. CUDA-vs-CPU equality is enforced by tests
(tests/test_intmath.py::TestCudaPrims) over boundaries (0, 1, 2^62-1,
INT-MIN-adjacent ranges) and random int64 sweeps.
"""

from __future__ import annotations

import torch

I64 = torch.int64

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def _isqrt_loop(a: torch.Tensor) -> torch.Tensor:
    """Original tensor-loop isqrt (floor sqrt, bitwise method), int64."""
    n = a.to(I64)
    rem = n.clone()
    c = torch.zeros_like(n)
    d = 1 << 62
    for _ in range(32):
        t = c + d
        ge = rem >= t
        rem = torch.where(ge, rem - t, rem)
        c = torch.bitwise_right_shift(c, 1) + torch.where(
            ge, torch.full_like(c, d), torch.zeros_like(c))
        d >>= 2
    return c


def _ilog2_loop(a: torch.Tensor) -> torch.Tensor:
    """Original tensor-loop floor(log2(a)) for a >= 1, int64."""
    v = a.to(I64)
    r = torch.zeros_like(v)
    for s in (32, 16, 8, 4, 2, 1):
        hit = v >= (1 << s)
        r = r + hit.to(I64) * s
        v = torch.bitwise_right_shift(v, hit.to(I64) * s)
    return r


if HAVE_TRITON:

    @triton.jit
    def _isqrt_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(x_ptr + offs, mask=m, other=0)
        rem = a
        c = tl.zeros_like(a)
        for i in tl.static_range(32):  # unrolled; dd = 4^(31-i) constant
            dd = (1 << 62) >> (2 * i)
            t = c + dd
            ge = rem >= t
            rem = tl.where(ge, rem - t, rem)
            c = (c >> 1) + tl.where(ge, dd, 0)
        tl.store(out_ptr + offs, c, mask=m)

    @triton.jit
    def _ilog2_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m, other=1)
        r = tl.zeros_like(v)
        # manually unrolled: this triton build MISCOMPILES the equivalent
        # static_range loop with a shifted loop variable (stored pointer
        # values); constant-amount shifts + where-selection are bit-identical
        # to the tensor-loop formulation
        hit = v >= (1 << 32); r = tl.where(hit, r + 32, r); v = tl.where(hit, v >> 32, v)
        hit = v >= (1 << 16); r = tl.where(hit, r + 16, r); v = tl.where(hit, v >> 16, v)
        hit = v >= (1 << 8);  r = tl.where(hit, r + 8, r);  v = tl.where(hit, v >> 8, v)
        hit = v >= (1 << 4);  r = tl.where(hit, r + 4, r);  v = tl.where(hit, v >> 4, v)
        hit = v >= (1 << 2);  r = tl.where(hit, r + 2, r);  v = tl.where(hit, v >> 2, v)
        hit = v >= (1 << 1);  r = tl.where(hit, r + 1, r);  v = tl.where(hit, v >> 1, v)
        tl.store(out_ptr + offs, r, mask=m)


@torch.library.custom_op("detllm::isqrt_i64", mutates_args=())
def isqrt_i64(a: torch.Tensor) -> torch.Tensor:
    return _isqrt_loop(a)


@isqrt_i64.register_fake
def _(a):
    return torch.empty_like(a, dtype=I64)


@torch.library.custom_op("detllm::ilog2_i64", mutates_args=())
def ilog2_i64(a: torch.Tensor) -> torch.Tensor:
    return _ilog2_loop(a)


@ilog2_i64.register_fake
def _(a):
    return torch.empty_like(a, dtype=I64)


if HAVE_TRITON:

    @isqrt_i64.register_kernel("cuda")
    def _(a):
        x = a.to(I64).contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        if n:
            grid = ((n + 1023) // 1024,)
            _isqrt_kernel[grid](x.view(-1), out.view(-1), n, BLOCK=1024)
        return out

    @ilog2_i64.register_kernel("cuda")
    def _(a):
        x = a.to(I64).contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        if n:
            grid = ((n + 1023) // 1024,)
            _ilog2_kernel[grid](x.view(-1), out.view(-1), n, BLOCK=1024)
        return out
