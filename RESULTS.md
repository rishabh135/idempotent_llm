# Results

Deterministic integer-only inference for Qwen/Qwen3-0.6B. Every operation
between input ids and int32 logits is exact integer arithmetic (int8 GEMMs
with int32 accumulation; int32/int64 elementwise; dyadic scales). Floats
exist only in offline preparation (`prepare.py`) and in analysis scripts.

Design notes and the deviation log are in [NOTES.md](NOTES.md).

## §9.2 Accuracy

WikiText2 (full test set, 146 non-overlapping segments of 2048, standard
concatenated eval; same protocol for all rows):

| model                         | PPL     | ratio vs fp16 |
|-------------------------------|---------|---------------|
| fp32 HF baseline              | 20.9542 | 1.000         |
| fp16 HF baseline              | 20.9549 | 1.000         |
| **int8 deterministic (ours)** | **20.7170** | **0.989** |

Acceptance was int ≤ 1.05 × fp16 → **passed** (the int pipeline is slightly
*below* the float baseline; per-token dynamic quantization plus analytic
smoothing acts as mild regularization at this scale).

C4 (validation, 40 segments of 2048, secondary):

| model | C4 PPL | ratio |
|-------|--------|-------|
| fp16  | 31.5696 | 1.000 |
| int8  | 31.6966 | 1.004 |

### Ablations (5-segment PPL, fp16 = 22.42)

| config | PPL |
|--------|-----|
| final (QK smoothing α=0.3, act smoothing, 15-bit probs, calib 48×1024) | 22.34 |
| − QK smoothing (the original faithful pipeline) | 44.94 |
| − activation smoothing (QK smoothing only) | 25.19 |
| 7-bit probs (paper's p_out = 8) instead of 15-bit | 24.71 |
| QK smoothing α=0.5 instead of 0.3 | 24.71 |
| QK smoothing α=0.7 | 25.19 |
| calibration 16×512 instead of 48×1024 | ~24.7 |
| softmax clip c ∈ {12, 15, 20, ∞} | 22.3421 (all four IDENTICAL) |

Clip conclusion: with DI-Exp running on full-precision int32 scores (no
8-bit pre-quantization of softmax input), the Eq. 10 clip is a no-op —
kept disabled (c = ∞), matching the spec's expectation.

Smoothing decision: analytic QK smoothing and activation smoothing are both
REQUIRED for the accuracy bar; both are closed-form calibration folds (no
training), within the spec's §7.2 allowance.

### Qualitative

20 fixed prompts × 200 greedy tokens (run as one batch — valid because
batch invariance is bit-exact): fluent, grammatical, on-topic continuations;
factual recall intact (e.g. "The capital of France is Paris"). Some prompts
drift into repetition loops late in the continuation, which is
characteristic of a 0.6B base model under pure greedy decoding (fp16 greedy
shows the same tendency) — not a quantization artifact.
Transcript: `scripts/eval_extras.py --what qual`.

## §9.3 Determinism (all exact, zero tolerance)

Checks run via `tests/test_determinism.py` (fast set) and `-m slow` (4k long-context):

| check | status |
|-------|--------|
| 1. run-to-run: 10 CUDA runs, 20 steps — identical ids AND int32 logits | **PASS** |
| 2. batch invariance: alone vs batch 2/8/32, random co-prompts, right-pad | **PASS** |
| 3. prefill vs token-by-token decode — identical logits at every position | **PASS** |
| 4. cross-device: CUDA (cuBLASLt int8) vs CPU reference — 3 prompts × 100 tok | **PASS** |
| 5. long context: 4k prompt through checks 1–3 (exercises the query-chunked attention path + long RoPE positions) | **PASS** |
| §9.4 float-leak guard: TorchDispatchMode forbids float tensors in forward | **PASS** |

Fast set: `9 passed in 1562s` (the cross-device check runs 600 full CPU
reference forwards — slow by design, exact by construction).

## §10 Performance

A100, measured with `scripts/bench.py`. Three int8 configurations, each
verified **bit-identical** to the one before it (tokens AND int32 logits;
§9.3 invariances re-verified at every stage):

- *compiled*: operator-level torch.compile (`detllm.compile.compile_ops()`)
- *graphed*: + CUDA-graphed static-shape decode (`use_graph=True`;
  steady-state) + fused integer decode-attention Triton kernel
  (`ops/fused_attn.py`: scores → DI-Exp → normalize → probs·V in one
  kernel, three streaming passes, zero materialized intermediates)

| decode tok/s | fp16 eager | int8 eager | int8 compiled | int8 graphed+fused | vs fp16 |
|--------------|-----------|------------|---------------|--------------------|---------|
| batch 1 | 29.3 | 0.8 | 5.6 | **85.1** | **2.9×** |
| batch 8 | 233.8 | 5.5 | 42.9 | 166.2 | 0.71× |

Prefill tok/s (2048, batch 1): fp16 48,898; int8 eager 1,057; compiled 6,427.

**Acceptance (§10: decode ≥ fp16 eager): PASSED at batch 1** (2.9×, beyond
the 1.3× stretch goal). Batch 8 stands at 0.71× — the step is bounded by
per-kernel overhead across the ~2,500 remaining small kernels inside the
replay (GPU busy time is only ~18 ms of the 48 ms step); the identified
next lever is layer-level compilation (fusing each layer's glue into one
inductor graph under the whole-step CUDA graph).

The Triton kernel reproduces the §5 integer semantics exactly (per the
§10 constraint): DI-Exp's shift decomposition is computed per cache slot
with divisions rewritten in the positive domain, reductions are integer
sums/maxes (order-independent), and the kernel is validated bit-for-bit
against the independent eager formulation at both batch sizes, across
cache-bucket growth, plus prefill/decode invariance and CUDA-vs-CPU
cross-device checks.

How the batch-1 win happened (details in NOTES.md):
- eager decode launched ~96,000 CUDA kernels/step (isqrt/ilog2 bit-loops ×
  28 layers); op-level compile cut this to ~4,100 (GPU busy 20.7 ms/step)
  but ~200 ms/step of Python orchestration remained;
- a manual whole-step CUDA graph replays the entire step in one launch;
  static shapes come from bucketed cache capacity + validity masking
  (invalid slots contribute exact zeros — the same §9.3 masking argument),
  cache writes via `index_copy_` at a device-tensor offset;
- decode attention avoids GQA `repeat_interleave` copies via broadcasting
  and computes 15-bit probs·V in one int32-multiply/int64-accumulate pass
  (bit-identical to the hi/lo GEMM split by associativity).

A day of flaky illegal-memory-access crashes turned out to be an upstream
bug, not ours: cuBLASLt's int8 CUTLASS kernels speculatively read up to
~16KB past operand ends (compute-sanitizer evidence; values discarded, so
results were always correct). All `_int_mm` operands are now backed by
guard rows. See NOTES.md.

## Reproduce

```bash
make prepare    # build artifact from HF checkpoint (float, offline, once)
make test       # unit + golden + fast determinism checks
make test-all   # + 4k-long-context slow checks
make ppl        # full WikiText2 perplexity (int8 + fp16)
```
