# Spec: Deterministic Integer-Only Inference for Qwen3-0.6B

> Original project specification, reproduced verbatim for completeness.
> Implementation status: all acceptance criteria passed — see
> [RESULTS.md](RESULTS.md); deviations documented in [NOTES.md](NOTES.md).

## 1. Goal

Build an inference pipeline for **Qwen/Qwen3-0.6B** in PyTorch where the entire forward pass — every matmul, normalization, activation, softmax, rotary embedding, and residual add — is computed in **exact integer arithmetic**. No floating-point operations anywhere in the numerical path after one-time offline preparation.

**The purpose is determinism, not compression.** The pipeline must produce **bit-identical token sequences and logits** regardless of:

- Run (same machine, repeated runs)
- Batch size and batch composition (a sequence's output must not change based on what it's batched with)
- Prefill vs. incremental decode (processing a prompt in one shot vs. token-by-token must give identical results)
- Hardware backend (NVIDIA GPU vs. CPU reference must match bit-for-bit)
- Parallelization / reduction order in kernels

Accuracy target: W8A8-class quality. WikiText2 perplexity within ~5% of the fp16 baseline (see §9).

Speed target (NVIDIA, after optimization phase): tokens/sec ≥ the fp16 eager baseline for the same model on the same GPU. See §10 for the phased approach — correctness first, speed second.

## 2. Non-Goals

- No 4-bit / 6-bit quantization. int8 for matmul inputs is chosen for tensor-core support, not memory savings.
- No FSBR / learned reconstruction / QAT. Calibration is analytic only (per-channel smoothing scales + weight quantization computed from statistics; optionally zero training).
- No MPS/Metal fast path. Apple silicon runs the slow reference backend only.
- No support for other models in v1 (design cleanly, but only Qwen3-0.6B must work).
- Training / backward pass: out of scope entirely.

## 3. Background (context for implementer)

This is a simplification of the I-LLM paper (arXiv:2405.17849, "I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models"). Read it before starting; its Appendix Algorithms 1–4 (DI-Exp, DI-Softmax, DI-SwiGLU, DI-RMSNorm) and its DI-MatMul equations (Eqs. 2–8) are the basis for the operators below. We drop the paper's FSBR training stage (only needed for ≤6-bit) and keep:

- **Dynamic per-token quantization** with all scale arithmetic done in integers via **dyadic numbers** (scale = mᴵ / 2^kᴵ, both integers).
- **Integer-only non-linear operators** built on bit shifts.

Key deviation from the paper: at 8 bits we may relax or skip the Softmax clipping constant tuning (use c = 15 or larger; verify empirically it doesn't matter at 8-bit).

## 4. Model Architecture Facts (Qwen3-0.6B)

Load config from HuggingFace `Qwen/Qwen3-0.6B` and verify these at startup rather than hardcoding, but for planning:

- 28 layers, hidden 1024, GQA: 16 query heads / 8 KV heads, head_dim 128
- RMSNorm (pre-attn, pre-MLP, final) — **plus QK-Norm**: per-head RMSNorm applied to Q and K projections before RoPE. Do not miss this; it does not exist in LLaMA.
- SwiGLU MLP (gate/up/down)
- RoPE, context up to 32,768
- Tied input embedding / LM head weights
- Weights ship as BF16

## 5. Numerics Foundation (build this first — everything depends on it)

Create a module `intmath.py` defining the **single canonical semantics** for all integer arithmetic. Every operator must use these helpers; no raw `//`, `>>`, or `torch.round` on the numerical path elsewhere.

Required primitives (all pure integer, all with explicit unit tests including negative inputs and boundary values):

- `floor_div(a, b)` — floor division (Python semantics, NOT C truncation). Document and test behavior for negative a.
- `round_half_to_even_div(a, b)` or `round_half_away_div(a, b)` — pick ONE rounding-to-nearest rule for the ⌊·⌉ operations in the paper, document it, use it everywhere.
- `ashr(a, k)` — arithmetic right shift, defined for k ≥ 0, with the floor-rounding convention stated.
- `rshift_round(a, k)` — right shift with round-to-nearest (a + (1 << (k-1))) >> k, with the k = 0 case handled.
- `clamp_i8(a)`, `clamp_u8(a)` — saturating casts.
- `ilog2_floor(a)` — ⌊log₂ a⌋ for a ≥ 1 (MSB position).
- `isqrt(a)` — integer square root, bit-wise check method (paper Algorithm 4's I-SQRT), for int64 inputs.

Dtype policy:

- **int8**: inputs to every tensor-core matmul; KV cache storage.
- **int32**: matmul accumulators, activations between ops, all elementwise math, softmax/norm internals.
- **int64**: scalar/vector rescale arithmetic where products of dyadic multipliers could exceed int32 (e.g., m₁·m₂·p terms), and isqrt inputs. Keep int64 off large tensors (scalars and per-token vectors only).
- **Forbidden**: any float dtype on the numerical path; any float→int or int→float cast after offline prep. Enforce mechanically (see §9.4).
- **Forbidden**: emulating integer matmul via fp32 GEMM (fp32 is exact only to 2²⁴; our accumulations approach/exceed that). The reference backend must use true integer matmul even though it is slow.

Overflow analysis: document, per matmul site, the worst-case accumulator magnitude (K · 127 · 127 for K = reduction dim; largest K in this model is the MLP intermediate — verify < 2³¹, which it is, but write the check as an assertion computed from config).

## 6. Operator Specifications

All operators take/return **(tensor_int, m, k)** triples — the integer tensor plus its dyadic scale (value ≈ tensor · m / 2^k) — and optionally a zero-point. m, k are per-token vectors where the paper specifies dynamic per-token quantization; per-channel constants for weights.

### 6.1 `di_matmul` — dynamic integer matmul (paper Eqs. 2–8)

- Inputs: int8 X₁ (per-token dyadic scale + zero-point), int8 X₂ (weights: per-output-channel dyadic scale, symmetric so zp = 0 preferred; verify accuracy — if per-channel asymmetric is needed, handle the zero-point correction term in integer).
- Core: int8×int8→int32 GEMM. On CUDA use `torch._int_mm` (2-D; reshape batched inputs). On reference backend use int32 matmul (cast int8→int32, `torch.matmul` on CPU int tensors, or an explicit einsum) — must be exact, slowness accepted.
- Output requantization to int8 (or handoff as int32 + scale where the consumer prefers it — see fusion notes §10): compute p_min, p_max per token (integer min/max reductions — order-independent by nature), derive output m, k, zp per paper Eqs. 6–8 using `ilog2_floor` and the §5 rounding helpers.
- Per-token vs per-tensor: per-token for activations always (this is what makes batch-invariance and prefill/decode-invariance achievable — a token's scale depends only on that token's values).

### 6.2 `di_exp` — integer exponential (paper Algorithm 1)

- Input already max-subtracted (all values ≤ 0). Shift-based 2^x decomposition with linear interpolation on the fractional part, exactly per the algorithm. Note the constant `m + (m>>1) − (m>>4)` approximates m·log₂e ≈ 1.4375·m; keep it, document it.
- Test: compare against float exp over the full representable input range; assert max relative error bound (establish empirically, then freeze as a regression bound).

### 6.3 `di_softmax` — integer softmax (paper Algorithm 2, + clipping per Eq. 10)

- Per-token (per-row) max subtraction, DI-Exp, integer division normalize to the output bit-width (u8 probabilities with fixed m = 1, k = 7 per the algorithm's convention — verify the convention against the algorithm and document).
- Causal masking: masked positions must be excluded from row max and from the sum — implement as integer sentinel handling, NOT as float −inf.
- Clipping constant c: default 15 as a dyadic number; run the §9.2 ablation to confirm 8-bit insensitivity.

### 6.4 `di_rmsnorm` — integer RMSNorm (paper Algorithm 4)

- Uses `isqrt` for the RMS. Weight γ folded into the output scale offline (γ is a per-channel fp constant at prep time → convert to per-channel dyadic and fold; no runtime float).
- Two variants needed: (a) standard over hidden dim 1024; (b) **QK-Norm variant** over head_dim 128, applied per head to Q and K after projection, before RoPE. Same code path, different reduction dim; test both.

### 6.5 `di_swiglu` — integer SwiGLU (paper Algorithm 3, minus smoothing)

- σ(x_gate) via DI-Exp composition per the algorithm; output y = x_gate · σ(x_gate) · x_up with scale bookkeeping in integer. We are NOT doing FSBR smoothing, so drop α_smooth (or fix it to 1) — but keep the hook so an analytic SmoothQuant-style per-channel factor can be folded into the gate/up projection weights offline if §9.2 accuracy demands it.

### 6.6 `int_rope` — integer rotary embeddings (not in the paper; design required)

- Offline: precompute cos/sin tables for all positions up to max context at a fixed dyadic precision (suggest int16 values with a single global k, e.g. values in [−2^14, 2^14] for k = 14; justify precision by measuring end-to-end perplexity sensitivity).
- Runtime: rotation = integer multiply-add of int32 activations by table entries, then `rshift_round` by k. Pure elementwise; per-position table lookup by integer index.
- Test: compare rotated Q/K against float RoPE; assert error bound; verify long-position (e.g. position 30k) precision doesn't degrade perplexity on long-context eval.

### 6.7 Attention assembly

- Q·Kᵀ: int8 GEMM (quantize Q, K per-token to int8 after QK-Norm + RoPE). KV cache stores **int8 K and V with their per-token dyadic scales** — cached values are quantized ONCE when produced and never re-quantized (this is required for prefill/decode invariance).
- Scale 1/√d: fold into the dyadic scale bookkeeping (d = 128 = 2^7, so it's a pure shift — convenient; but compute it from config, don't assume).
- probs·V: probabilities are u8, V is int8 → GEMM to int32, requantize.
- GQA: KV head broadcasting is indexing only; no numerics involved.

### 6.8 Residual adds, embeddings, LM head

- Residual add: both operands int32; align scales by shifting to the coarser k (rounding via `rshift_round`), add in int32. Define alignment rule once, test associativity of your chosen scheme across the two adds per layer.
- Embedding lookup: table pre-quantized offline (per-channel int8 + dyadic scales); lookup is indexing, no math.
- LM head (tied weights): int8 GEMM producing int32 logits + a dyadic scale. **Greedy decode = integer argmax on int32 logits** (define tie-break: lowest token id). If sampling support is added later it must be integer (e.g. Gumbel via integer fixed-point) — v1 is greedy only.

## 7. Offline Preparation (floats allowed here only)

One-time script `prepare.py`:

1. Load BF16 checkpoint.
2. (Optional, gated by §9.2 results) Compute SmoothQuant-style analytic per-channel smoothing factors from ~128 calibration samples (WikiText2 train slices); fold into weights. Alpha ≈ 0.5 default. This is closed-form, no training.
3. Quantize all weights: per-output-channel symmetric int8 + dyadic scales (fit m/2^k to the float scale; document the fitting rule and its worst-case relative error, target < 0.5%).
4. Quantize embedding table, fold RMSNorm γ's, precompute RoPE tables, precompute any per-layer constant shifts.
5. Serialize everything to a single artifact (safetensors of int tensors + a JSON of scalar constants). The runtime loads ONLY this artifact and never touches floats.

## 8. Backends

- `backend="cuda"`: `torch._int_mm` for all GEMMs; everything else in eager int tensor ops (Phase 2 optimizes this — §10).
- `backend="reference"`: CPU, int32 matmuls, same code for everything else. This is the correctness oracle and the cross-device determinism witness.
- Both backends share ALL operator code except the GEMM call, which goes through one dispatch function. There must be exactly one place where backend divergence is possible, so bit-exactness between backends reduces to: does cuBLASLt int8 GEMM produce exact int32 results? (It does — integer accumulation is exact — but §9.3 verifies it anyway.)

## 9. Testing & Acceptance Criteria (this is most of the project)

### 9.1 Operator unit tests (write BEFORE assembly)

For each operator in §6: (a) golden tests against float reference implementations with frozen error bounds; (b) negative-value and boundary tests on all §5 primitives; (c) property tests — e.g., di_softmax rows sum to the expected fixed-point total, di_matmul matches a naive int loop implementation exactly on random small cases.

### 9.2 Accuracy

- WikiText2 perplexity, full test set, via the standard sliding-window eval. Baseline: fp16 Qwen3-0.6B, same eval code. **Accept: int8 PPL ≤ 1.05 × fp16 PPL.** Report C4 as secondary.
- Ablations to run once: smoothing on/off; softmax clip c ∈ {12, 15, 20, ∞}. Keep whatever config meets the accuracy bar with least machinery.
- Qualitative: 20 fixed prompts, greedy, 200 tokens; eyeball for degeneration/repetition.

### 9.3 Determinism (the actual point — must all pass exactly, no tolerance)

1. **Run-to-run**: same prompt, 10 runs, CUDA → identical token ids AND identical int32 logits at every step.
2. **Batch invariance**: prompt evaluated alone vs. inside batches of size 2, 8, 32 with random co-prompts (padded) → bit-identical logits for that prompt. Test both left- and whatever padding scheme is used; masking must be integer-clean.
3. **Prefill/decode invariance**: full-prompt prefill vs. token-by-token decode of the same prompt → identical logits at every position.
4. **Cross-device**: CUDA backend vs. reference CPU backend, same artifact → bit-identical logits on ≥ 3 prompts × 100 tokens. If a second GPU model is available, CUDA-vs-CUDA across GPUs too.
5. **Long context**: one 8k-token prompt through checks 1–3.

### 9.4 Float-leak guard

A test that runs a forward pass under a mode that (a) walks the module tree asserting no parameter/buffer on the numerical path has a float dtype, and (b) monkeypatches/hooks to fail on any float-dtype tensor produced between the input ids and the logits. This prevents silent regressions where someone "fixes" a bug with a convenient `.float()`.

### 9.5 Regression harness

One command (`pytest` + a `make eval`) that runs everything above except full perplexity; full PPL as a separate command. CI-style: any numerics change must re-freeze golden outputs deliberately.

## 10. Performance Plan (Phase 2 — only after §9.3 passes)

- Measure first: tokens/sec decode + prefill throughput vs. fp16 eager baseline, batch 1 and 8, on the target GPU. Profile with `torch.profiler`; expect the GEMMs to be fast and the requantize/rescale/norm glue to dominate via kernel-launch overhead.
- Step 1: `torch.compile` (mode="max-autotune") on the block. Integer elementwise chains fuse well; this alone may reach parity.
- Step 2: Triton kernels for the two worst offenders, likely (a) the fused dequant-scale-clamp-requant epilogue after GEMMs and (b) di_softmax. **Constraint: Triton kernels must reproduce §5 semantics bit-exactly** — every kernel gets a bit-exactness test against the eager op before it's enabled, and §9.3 re-runs after each kernel lands.
- Step 3 (optional): int8 KV cache already halves cache bandwidth vs fp16 — verify decode benefits show up; consider CUDA graphs for decode.
- **Accept: decode tokens/sec ≥ fp16 eager baseline at batch 1 and batch 8.** Stretch: ≥ 1.3×.
- Rule: no perf change may alter any golden output. Determinism outranks speed at every decision.

## 11. Milestones (suggested order)

1. `intmath.py` + full unit tests. (Foundation; ~half a day of care here saves days later.)
2. Operators §6.1–6.6 each with golden tests, reference backend only.
3. `prepare.py`; single transformer block assembled; block output vs. float block diff measured and documented.
4. Full model, reference backend; greedy generation works; first PPL number.
5. CUDA backend (`_int_mm` dispatch); §9.3 checks 1–4 pass.
6. Accuracy pass (§9.2, ablations, smoothing decision).
7. Phase 2 performance (§10).

Deliverables: the repo, the prepared artifact, a RESULTS.md with the §9.2 table, §9.3 pass evidence, and §10 benchmark table.

## 12. Known Risks & Pre-Made Decisions

- **Silent numerical bugs are the #1 risk.** They present as bad perplexity, not crashes. Mitigation is the test-first structure above and layer-by-layer float-diffing when PPL is off: run float and int models side by side, hook every sublayer, find the first divergence exceeding quantization noise.
- **`torch._int_mm` constraints**: 2-D only, shape/alignment requirements (dims typically must be ≥ 16 / multiples of 8 depending on version); wrap it with reshape + padding as needed, and pad deterministically (zeros, then slice — zeros are exact in integer GEMM).
- **QK-Norm is the easiest thing to get subtly wrong** (per-head reduction, applied before RoPE). Diff against HF's Qwen3 implementation specifically.
- **Padding/masking** must never introduce data-dependent behavior across batch layouts; masked positions must contribute exactly nothing to max, sum, and GEMM results.
- Pre-decided: rounding rule chosen in §5 is global and frozen after milestone 1; greedy-only decoding in v1; tie-break = lowest token id; c = 15 unless ablation says otherwise; symmetric per-channel weights unless accuracy forces asymmetric.
