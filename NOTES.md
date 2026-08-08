# Design notes & deviation log

Working notes for the deterministic integer-only Qwen3-0.6B pipeline.
Spec references are to the project spec; paper references are to I-LLM
(arXiv:2405.17849v2), whose Eqs. 2–8 / 10 and Appendix Algorithms 1–4 were
read and transcribed before implementation.

## Frozen global decisions (spec §5 / §12)

- **Rounding rules**: `round_half_away_div` (ties away from zero) for all
  round-to-nearest divisions; `rshift_round(a,k) = (a + (1<<(k-1))) >> k`
  (ties toward +inf, spec-fixed formula) for all power-of-two rescales.
  Both frozen in `intmath.py` after milestone 1.
- **Greedy only**, tie-break = lowest token id (implemented as an explicit
  min-index-over-argmax-ties, not relying on backend argmax order).
- **Symmetric quantization everywhere** (zp = 0): activations per-token int8,
  weights per-output-channel int8. Attention probabilities are the one
  unsigned tensor (u8 range [0,128], fixed scale 1/2^7).
- Dyadic scale mantissas: 16-bit for activations & weights (worst-case fit
  error 2^-16 ≈ 0.0015% ≪ the 0.5% spec bound), 14-bit for QK-Norm γ,
  int16 RoPE tables at k = 14.

## torch facts probed on this machine (torch 2.13.0+cu130, A100)

- `torch._int_mm` requires M ≥ 32 with M ≡ 0 (mod 32) safe (M=17, 20 fail;
  M=32 works), K ≡ 0 (mod 8), N ≡ 0 (mod 8). We pad M up to a multiple of
  32 (min 32), K and N up to multiples of 8, with zeros (exact in integer
  GEMM), then slice.
- CPU `torch.matmul` supports int32 and int64 tensors → reference backend
  uses `a.to(int32) @ b.to(int32)` (exact, slow).

## Key deviations from spec text (with rationale)

1. **K and V caches use STATIC per-(layer, head) dyadic scales calibrated
   offline**, not per-token dynamic scales (spec §6.7 said per-token).
   Reason: per-token scales on K sit on the *row-compare* dim of the softmax
   (row max over keys j needs a common scale across j), and per-token scales
   on V sit on the *reduction* dim of probs·V (scale can't factor out of the
   GEMM). The paper's Eqs. 6–8 treat the operand scales as scalars per
   matmul, i.e. it has the same constraint. Static scales keep every GEMM a
   pure int8 GEMM, and trivially satisfy "quantized once, never
   re-quantized" (prefill/decode invariance). Qwen3's QK-Norm bounds K
   tightly, making static K scales well-conditioned; V ranges are calibrated
   with margin. Q stays dynamic per-token (its scale is on the row dim and
   factors out of the score row).
2. **1/√d is folded into the static K scale offline** (spec suggested a
   runtime shift; folding is exact for any d, not just powers of two).
3. **Hidden-dim RMSNorm γ is folded into the *following* linear's weight
   columns offline** (attn-norm → q/k/v, mlp-norm → gate/up, final-norm →
   LM head), not into the norm output scale: γ is per-channel and the norm
   output scale is per-token, so folding into the consumer weights is the
   exact version of the spec's intent. Runtime hidden RMSNorm is therefore
   the pure scale-free x/rms(x). QK-Norm γ cannot be folded through RoPE
   (RoPE mixes channel pairs (d, d+64) which have different γ), so it stays
   a runtime per-channel dyadic integer multiply.
4. **Softmax input is NOT pre-quantized to 8 bits**, so the Eq. 10 clipping
   constant c is unnecessary in the default path: DI-Exp runs directly on
   the int32 scores after integer max-subtraction, and values below ≈ -21
   underflow to 0 via the `>> q` shift, exactly as e^x → 0. This is
   strictly more accurate than any clip. A clip mode (c ∈ {12, 15, 20})
   is implemented for the §9.2 ablation only. (Spec pre-approved c=15 vs ∞
   as an 8-bit-insensitive choice; we default to ∞ and verify.)
5. **Embedding table quantized per-row (per vocab id)**, not per-channel:
   after lookup a row IS the token, so this is per-token quantization, and
   the same int8 tensor + per-row scales serves as the per-output-channel
   LM head weight (tied). Final-norm γ folded into a separate LM-head copy
   of the embedding (γ-fold would corrupt lookups if applied to the shared
   tensor).
6. **Attention probs u8 [0,128] through int8 GEMM**: `_int_mm` takes int8
   only, so the dispatch layer computes P@V as (P-64)@V + 64·colsum(V),
   which is exact in integer arithmetic; the reference backend computes the
   same formula. Backend divergence remains confined to the single
   `int_gemm` call.
7. **RMSNorm eps is dropped** (guard `rms >= 1` instead): after scale
   cancellation an integer eps is ~0; it only matters for near-zero rows.
   Verified acceptable via perplexity (§9.2).

## DI-Exp semantics (paper Alg. 1, as implemented)

Input (x ≤ 0, per-row dyadic scale m/2^k, m normalized to [2^15, 2^16)):
- m_f = m + (m>>1) - (m>>4)  (≈ 1.4375·m ≈ m·log2 e; kept per spec)
- t = -round(2^k / m_f)  (per-row int64, ≤ -1)
- q = floor(x / t) ≥ 0;  r = x - q·t ∈ (t, 0]
- result = ((r >> 1) - t) >> min(q, 62)
Output value ≈ e^(x·m/2^k) · (-t): implicit scale 1/(-t), range [0, -t].
The non-dyadic 1/(-t) scale is always consumed by a ratio (softmax /
sigmoid denominators share it), so it cancels and never propagates.

## Residual stream convention

int32 data, per-token power-of-two scale (m = 1, k_res per token).
Deltas (m_d, k_d) align by d·m_d (int64) then shift to k_res
(right = rshift_round, left = exact); after each add the row renormalizes
max|h| into [2^24, 2^30) by exact left shifts / rounded right shifts,
adjusting k_res. Everything is per-token → batch and prefill/decode
invariant by construction.

## Overflow budget (asserted from config at load)

- Scores GEMM: K=128 → |P| ≤ 128·127² < 2^21.
- Hidden GEMMs: K=1024 → < 2^24; MLP down: K=3072 → < 2^25.6. All < 2^31. ✓
- probs·V: T·128·127 < 2^31 → T ≤ 131k ≥ 32k context. ✓
- Requant epilogues multiply int32 P by ≤16-bit col mantissas in int64.
- RMSNorm: rows pre-shifted so max|x| < 2^26 → Σx² ≤ 1024·2^52 = 2^62 < 2^63. ✓
