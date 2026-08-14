"""§10 Phase 2: operator-level torch.compile.

Compiling the whole 28-layer forward produces an enormous graph (the isqrt
bit-loop alone unrolls ×32) and takes tens of minutes; compiling the LEAF
operators instead gives one small fused graph per op, each reused by every
layer. GEMMs stay on the eager `_int_mm` path (already single kernels).

Bit-exactness contract: fusion must not change results — everything is
integer arithmetic, so any mismatch is a miscompilation. Verified
empirically rather than by a dedicated unit test: the §9.3 invariances were
re-run at every perf stage (RESULTS.md §10), and scripts/demo.py's
compiled+graphed CUDA rows reproduce the eager CPU reference hash exactly.

Usage: call `compile_ops()` once (idempotent), before or after model
construction. `dynamic=True` keeps one graph across growing cache lengths.
"""

from __future__ import annotations

import torch

_COMPILED = False


def compile_ops(mode: str | None = None) -> None:
    global _COMPILED
    if _COMPILED:
        return
    import importlib

    from . import dyadic, model as model_mod
    # ops/__init__ re-exports functions under the submodule names, so reach
    # the actual modules via importlib
    rn = importlib.import_module("detllm.ops.di_rmsnorm")
    sm = importlib.import_module("detllm.ops.di_softmax")
    sg = importlib.import_module("detllm.ops.di_swiglu")
    rope = importlib.import_module("detllm.ops.rope")

    def c(fn):
        return torch.compile(fn, dynamic=True, mode=mode)

    # leaf operators (each internally loops/chains many elementwise int ops)
    rmsnorm = c(rn.di_rmsnorm)
    rmsnorm_gamma = c(rn.di_rmsnorm_gamma)
    softmax = c(sm.di_softmax)
    swiglu = c(sg.di_swiglu)
    int_rope = c(rope.apply_int_rope)
    quant_i8 = c(dyadic.quant_i8_pertoken)
    requant_common = c(dyadic.requant_i32_common)
    requant_rowcol = c(dyadic.requant_i8_rowcol)
    requant_static = c(dyadic.requant_i8_static)
    res_add = c(dyadic.residual_add)
    res_from_delta = c(dyadic.residual_from_delta)

    # rebind in the defining modules AND in model.py's imported names
    rn.di_rmsnorm = rmsnorm
    rn.di_rmsnorm_gamma = rmsnorm_gamma
    sm.di_softmax = softmax
    sg.di_swiglu = swiglu
    rope.apply_int_rope = int_rope
    dyadic.quant_i8_pertoken = quant_i8
    dyadic.requant_i32_common = requant_common
    dyadic.requant_i8_rowcol = requant_rowcol
    dyadic.requant_i8_static = requant_static
    dyadic.residual_add = res_add
    dyadic.residual_from_delta = res_from_delta
    model_mod.di_rmsnorm = rmsnorm
    model_mod.di_rmsnorm_gamma = rmsnorm_gamma
    model_mod.di_softmax = softmax
    model_mod.di_swiglu = swiglu
    model_mod.apply_int_rope = int_rope
    _COMPILED = True
