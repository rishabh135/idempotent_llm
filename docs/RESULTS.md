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

Re-run on an H100 / Xeon Platinum 8480+ box (torch 2.13.0+cu130, same
artifact): int8 **20.7170** — identical to all four decimals, as it must
be. The fp16 baseline came back **20.9552** vs 20.9549 in the table above.
The headline
determinism claim, showing up unprompted in the accuracy table: the integer
number is a property of the arithmetic, the float number is a property of
the machine.

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

Re-run on the H100 / Xeon box: the whole non-slow suite is `78 passed,
1 deselected in 506s` (`make test`), determinism checks included. Note the
fast set is 11 tests there, not the 9 recorded above — `test_determinism.py`
gained two since that line was written.

## §10 Performance

A100, measured with `scripts/bench.py`. Three int8 configurations, each
verified **bit-identical** to the one before it (tokens AND int32 logits;
§9.3 invariances re-verified at every stage):

- *compiled*: operator-level torch.compile (`detllm.compile.compile_ops()`)
- *graphed*: + CUDA-graphed static-shape decode (`use_graph=True`;
  steady-state) + fused integer decode-attention Triton kernel
  (`ops/fused_attn.py`: scores → DI-Exp → normalize → probs·V in one
  kernel, three streaming passes, zero materialized intermediates)

| decode tok/s | fp16 eager | int8 eager | int8 compiled | int8 final | vs fp16 |
|--------------|-----------|------------|---------------|------------|---------|
| batch 1 | 29.3 | 0.8 | 5.6 | **106.3** | **3.6×** |
| batch 8 | 233.8 | 5.5 | 42.9 | **783.3** | **3.4×** |

Prefill tok/s (2048, batch 1): fp16 48,898; int8 eager 1,057; compiled 6,427.

*final* = layer-level compilation (ONE generic inductor graph reused by all
28 layers) under the whole-step CUDA graph, with the fused attention kernel.
One-time setup ≈ 55 s cold (inductor-disk-cached afterwards).

**Acceptance (§10: decode ≥ fp16 eager): PASSED at BOTH batch sizes**
(3.6× / 3.4×, far beyond the 1.3× stretch goal).

### Same grid on an H100 / Xeon Platinum 8480+

| decode tok/s | fp16 eager | int8 eager | int8 compiled | int8 final | vs fp16 |
|--------------|-----------|------------|---------------|------------|---------|
| batch 1 | 19.5 | 2.8 | 5.8 | **73.8** | **3.8×** |
| batch 8 | 157.9 | 22.3 | 43.4 | **566.7** | **3.6×** |

Prefill tok/s (2048, batch 1): fp16 87,334; int8 eager 3,194; compiled
7,839; graphed 7,878. Batch 8: fp16 152,137; int8 eager 2,502; compiled
4,661; graphed 4,663.

Acceptance passes here too, by a wider ratio (3.8× / 3.6×) — but the ratio
improves partly because fp16 eager is *slower* on this box, so read the
columns, not the multiplier. The absolute numbers split by regime:

- **compute-bound work is much faster on Hopper**, as expected: fp16
  prefill 1.8× the A100 (87,334 vs 48,898), and the eager int8 path — many
  small kernels, throughput-bound — is 3–4× faster at both batch sizes.
- **the graphed decode path is ~30% slower** (73.8 vs 106.3 at batch 1;
  566.7 vs 783.3 at batch 8). For a 0.6B model under a whole-step CUDA
  graph, per-step cost is dominated by fixed replay/launch overhead rather
  than FLOPs, so it tracks host CPU and driver more than GPU class. Not
  investigated further; flagged so the A100 numbers above are not read as a
  floor.

None of this affects any determinism claim — the §9.3 fast checks (1–4 plus
the float-leak guard; the 4k long-context check is `-m slow` and was
deselected) and the demo's four int8 configurations are all bit-exact on
this box (see the README's H100 table).

Two structural changes made layer-level compilation affordable (it was
30–60 min of codegen before; 55 s after):
- **loop-heavy primitives as custom ops** (`ops/int_prims.py`): isqrt's
  32-step and ilog2's 6-step bit-loops each became ONE opaque graph node
  backed by a tiny Triton kernel instead of ~200/~24 traced nodes per call
  site — inductor scheduling is superlinear in graph size, so every
  compilation got several times faster (CUDA kernels verified bit-identical
  to the CPU loop implementations, which are the original algorithms);
- **per-layer scalar constants tensorized** (0-dim int64 tensors instead of
  Python ints) and the layer function takes its cache buffers as tensor
  arguments instead of a Python layer index — dynamo specializes on Python
  scalars, so these two changes collapse 28 per-layer compilations into
  one generic graph.

The Triton kernel reproduces the §5 integer semantics exactly (per the
§10 constraint): DI-Exp's shift decomposition is computed per cache slot
with divisions rewritten in the positive domain, reductions are integer
sums/maxes (order-independent), and the kernel is validated bit-for-bit
against the independent eager formulation at both batch sizes, across
cache-bucket growth, plus prefill/decode invariance and CUDA-vs-CPU
cross-device checks.

How the CUDA-graph decode works (details in NOTES.md):
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
