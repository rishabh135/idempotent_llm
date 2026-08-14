"""Fused integer decode attention (Triton) — §10 step 2.

One kernel per (batch, q-head) computes, for a single query row, the exact
integer chain scores → DI-Exp → normalize → probs·V in three streaming
passes over the KV cache, with NO materialized [B, H, S, hd] intermediates:

  pass 1: row max of valid scores      (int64 max — order-independent)
  pass 2: DI-Exp per slot, denom sum   (int64 sum — order-independent)
  pass 3: p = round((exp<<14)/denom), out_d = Σ p·v   (int64 sum)

Bit-exactness contract: every arithmetic step reproduces the eager path
(di_softmax/di_exp + the broadcast mul-sums) EXACTLY:
- scores: Σ_d q8·k8 in int32 (≤ 128·127² < 2^21).
- DI-Exp with the same per-row tpos (computed OUTSIDE the kernel by the
  same intmath ops): q = floor(xd/t), r = xd − q·t, ((r>>1) − t) >> min(q,62).
  All divisions are rewritten in the POSITIVE domain (neg = −xd ≥ 0;
  q = neg//tpos; unshifted = tpos − ((rem+1)>>1) ≡ (r>>1) + tpos), so
  floor-vs-truncation ambiguity cannot arise.
- p = (ex<<14 + denom>>1) // denom ≡ round_half_away_div(ex<<14, denom)
  for non-negative operands, denom clamped ≥ 1.
- Invalid slots are excluded from the max (integer sentinel) and contribute
  exactly 0 to denom and output.

Reduction order inside tl.sum/tl.max varies with launch config, but every
reduction here is an integer sum/max — associative and commutative — so any
order yields identical bits (the same §5 argument as the GEMMs).

Verified by tests/test_determinism.py (prefill/decode invariance pits this
kernel against the eager GEMM prefill path) and TestGraphedDecode.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

I64 = torch.int64

if HAVE_TRITON:

    @triton.jit
    def _fused_int_attn_kernel(
        q_ptr, k_ptr, v_ptr, valid_ptr, tpos_ptr, out_ptr,
        S: tl.constexpr, HD: tl.constexpr, NQ: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_S: tl.constexpr,
        sq_b, sq_h,            # q strides (batch, head); row is contiguous
        sk_b, sk_h, sk_s,      # k/v strides (batch, kv-head, slot)
        sv_b, sv_h, sv_s,
        so_b, so_h,
    ):
        pid = tl.program_id(0)
        b = pid // NQ
        qh = pid % NQ
        kvh = qh // GROUP
        d = tl.arange(0, HD)
        soff = tl.arange(0, BLOCK_S)

        q = tl.load(q_ptr + b * sq_b + qh * sq_h + d).to(tl.int32)   # [HD]
        tpos = tl.load(tpos_ptr + b * NQ + qh)                        # int64 ≥ 1

        NEG: tl.constexpr = -(1 << 62)
        # ---- pass 1: row max over valid slots ----
        m_i = tl.full((), NEG, tl.int64)
        for s0 in range(0, S, BLOCK_S):
            offs = s0 + soff
            kt = tl.load(k_ptr + b * sk_b + kvh * sk_h
                         + offs[:, None] * sk_s + d[None, :]).to(tl.int32)
            sc = tl.sum(kt * q[None, :], axis=1).to(tl.int64)        # [BLOCK_S]
            vm = tl.load(valid_ptr + b * S + offs) != 0
            m_i = tl.maximum(m_i, tl.max(tl.where(vm, sc, NEG), axis=0))

        # ---- pass 2: DI-Exp per slot, sum -> denom ----
        den = tl.zeros((), dtype=tl.int64)
        for s0 in range(0, S, BLOCK_S):
            offs = s0 + soff
            kt = tl.load(k_ptr + b * sk_b + kvh * sk_h
                         + offs[:, None] * sk_s + d[None, :]).to(tl.int32)
            sc = tl.sum(kt * q[None, :], axis=1).to(tl.int64)
            vm = tl.load(valid_ptr + b * S + offs) != 0
            neg = m_i - sc                       # = -xd ≥ 0 (valid slots)
            neg = tl.where(vm, neg, 0)
            qq = neg // tpos                     # ≥0 // >0: floor == trunc
            rem = neg - qq * tpos                # in [0, tpos)
            unshifted = tpos - ((rem + 1) >> 1)  # == (r>>1) + tpos, r = -rem
            qc = tl.minimum(qq, 62)
            ex = tl.where(vm, unshifted >> qc, 0)
            den += tl.sum(ex, axis=0)
        den = tl.maximum(den, 1)

        # ---- pass 3: p = round((ex<<14)/den); out_d = Σ p·v ----
        acc = tl.zeros((HD,), dtype=tl.int64)
        half = den >> 1
        for s0 in range(0, S, BLOCK_S):
            offs = s0 + soff
            kt = tl.load(k_ptr + b * sk_b + kvh * sk_h
                         + offs[:, None] * sk_s + d[None, :]).to(tl.int32)
            sc = tl.sum(kt * q[None, :], axis=1).to(tl.int64)
            vm = tl.load(valid_ptr + b * S + offs) != 0
            neg = tl.where(vm, m_i - sc, 0)
            qq = neg // tpos
            rem = neg - qq * tpos
            unshifted = tpos - ((rem + 1) >> 1)
            qc = tl.minimum(qq, 62)
            ex = tl.where(vm, unshifted >> qc, 0)
            p = ((ex << 14) + half) // den       # ≥0 // >0: exact round-half-up
            vt = tl.load(v_ptr + b * sv_b + kvh * sv_h
                         + offs[:, None] * sv_s + d[None, :]).to(tl.int64)
            acc += tl.sum(p[:, None] * vt, axis=0)
        tl.store(out_ptr + b * so_b + qh * so_h + d, acc)


def fused_int_attn(q8: torch.Tensor, K8: torch.Tensor, V8: torch.Tensor,
                   valid: torch.Tensor, tpos: torch.Tensor,
                   n_q: int, group: int) -> torch.Tensor:
    """q8 int8 [B, n_q, 1, hd]; K8/V8 int8 [B, n_kv, S, hd] (S = static cap,
    multiple of the block size); valid bool [B, S]; tpos int64 [B, n_q].

    Returns attn int64 [B, n_q, hd] — identical bits to the eager
    scores→di_softmax→context chain.
    """
    B, _, S, hd = K8.shape
    assert hd == 128 and S % 128 == 0
    out = torch.empty(B, n_q, hd, dtype=I64, device=q8.device)
    qv = q8.view(B, n_q, hd)
    valid_i8 = valid.view(torch.int8)
    grid = (B * n_q,)
    _fused_int_attn_kernel[grid](
        qv, K8, V8, valid_i8, tpos, out,
        S=S, HD=hd, NQ=n_q, GROUP=group, BLOCK_S=128,
        sq_b=qv.stride(0), sq_h=qv.stride(1),
        sk_b=K8.stride(0), sk_h=K8.stride(1), sk_s=K8.stride(2),
        sv_b=V8.stride(0), sv_h=V8.stride(1), sv_s=V8.stride(2),
        so_b=out.stride(0), so_h=out.stride(1),
        num_warps=4,
    )
    return out
