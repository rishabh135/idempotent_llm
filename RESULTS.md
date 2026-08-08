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

Checks run via `tests/test_determinism.py` (fast set) and `-m slow` (8k):

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

A100, measured with `scripts/bench.py`. Compiled = operator-level
torch.compile (`detllm.compile.compile_ops()`), verified **bit-identical**
to eager on tokens and logits, and §9.3 invariances re-verified compiled.

| metric | fp16 eager | int8 eager | int8 compiled |
|--------|-----------|------------|---------------|
| decode tok/s, batch 1 | 29.3 | 0.8 | 5.6 |
| decode tok/s, batch 8 | 233.8 | 5.5 | 42.9 |
| prefill tok/s (2048), batch 1 | 48,898 | 1,057 | 6,427 |

Profiling (batch-1 decode step): eager launches ~96,000 CUDA kernels
(the isqrt/ilog2 bit-loops unroll into hundreds of tiny int kernels per
call × 28 layers); compiled reduces this to ~4,100 with GPU busy time
**20.7 ms/step** — under the 34 ms/step needed for fp16 parity — but
~200 ms/step of Python/dynamo orchestration still dominates wall clock.

Next step (per §10 step 3): CUDA-graph the decode step — static-shape
decode with bucketed cache capacity, tensor-indexed cache writes, and the
existing exact masking (invalid slots already contribute exact zeros), so
graph replay eliminates the CPU wall without touching numerics. The
decode acceptance bar (≥ fp16 eager) is NOT yet met; the GPU-side budget
shows it is reachable.

## Reproduce

```bash
make prepare    # build artifact from HF checkpoint (float, offline, once)
make test       # unit + golden + fast determinism checks
make test-all   # + 8k-context slow checks
make ppl        # full WikiText2 perplexity (int8 + fp16)
```
