# DetLLM — Fully Deterministic LLM Inference

An inference pipeline for [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
where the entire forward pass — every matmul, normalization, activation,
softmax, and rotary embedding — is exact integer arithmetic. No
floating-point operations anywhere between input ids and int32 logits.

**The point is determinism, not compression**: bit-identical token
sequences and logits across runs, batch sizes and compositions, prefill vs.
incremental decode, and hardware backends (A100 cuBLASLt int8 GEMM vs. CPU
reference produce identical bits).

## Why LLM inference is nondeterministic — and how integers fix it

Ask a served LLM the same question twice at temperature 0 and you can get
two different answers. The root cause is that **floating-point addition is
not associative** — `(a + b) + c ≠ a + (b + c)` because each addition
rounds — while high-performance kernels sum in whatever order maximizes
throughput. That order changes with batch size and composition (your
request's rows get tiled differently depending on who it shares a batch
with), prefill vs. incremental decode (one big GEMM vs. many small ones),
kernel selection (cuBLAS picks tile shapes by heuristic; split-K and
atomics can differ run to run), and hardware or library version. So the
"same" mathematical forward pass produces slightly different logits, and
one flipped argmax early in a generation cascades into a visibly different
completion. Framework "deterministic mode" flags only pin run-to-run order
on one machine at one shape — they do nothing for batch invariance or
cross-hardware reproducibility.

The common fix is to write **batch-invariant floating-point kernels**
that pin a fixed reduction order everywhere. That works, but it treats the
symptom: determinism holds only for those specific kernels on that
platform, and bit-identical results across different hardware are still
out of reach.

This project removes the cause instead: **integer addition is exactly
associative and commutative**, so every reduction — GEMM accumulation,
softmax sums, norm sums — yields the same bits in *any* order, on *any*
correct hardware. Once the entire forward pass is integer (this repo's
contribution: including softmax, RMSNorm, SwiGLU, and RoPE, via I-LLM's
dyadic-scale machinery), determinism is not an engineering discipline to
maintain — it is a property of the arithmetic. That is what makes the
strongest check here possible at all: an A100's tensor-core GEMMs and a
CPU's plain integer matmuls produce **identical logits, bit for bit**.

Integer-only inference itself is not new — I-LLM and its predecessors
(I-BERT, I-ViT) target *efficiency* on integer-only edge hardware, and
never examine determinism (batch invariance, prefill/decode invariance,
and cross-device bit-exactness are not claims that appear in that
literature). To our knowledge this is the first pipeline built and
verified end-to-end for **unconditional determinism** — bit-identical
logits across runs, batch compositions, prefill/decode splits, and CPU/GPU
backends simultaneously — with the test suite treating every one of those
as an exact, zero-tolerance acceptance criterion.

Based on a simplification of I-LLM (arXiv:2405.17849) — dyadic-number scale
arithmetic and shift-based integer non-linear operators — with analytic
(training-free) smoothing. WikiText2 perplexity: **20.72 (int8) vs 20.95
(fp16 baseline)**; CUDA-graphed integer decode runs at **3.6× the fp16
eager baseline at batch 1 and 3.4× at batch 8** (106 / 783 tok/s on an
A100) while staying bit-exact. The original specification is in [docs/SPEC.md](docs/SPEC.md);
all numbers in [docs/RESULTS.md](docs/RESULTS.md); design decisions and deviation
log in [docs/NOTES.md](docs/NOTES.md).

## Layout

```
src/detllm/
  intmath.py    canonical integer primitives (rounding rules frozen here)
  dyadic.py     (data, m, k) dyadic-scale quant/requant machinery
  backends.py   int_gemm dispatch — the ONLY backend-divergent code
  ops/          DI-Exp, DI-Softmax, DI-RMSNorm, DI-SwiGLU, int RoPE
  model.py      IntQwen3 runtime (loads only the int artifact)
  prepare.py    offline float → int artifact (calibration + quantization)
  guard.py      float-leak guard (TorchDispatchMode)
tests/          unit + golden + §9.3 determinism suite
scripts/        perplexity eval, ablations, benchmarks, diagnostics
```

## Demo: one prompt, seven execution paths, one hash

`scripts/demo.py` is the whole thesis in one table. It takes the first 64
tokens of Sonnet 18, greedy-generates 512 tokens, and chain-hashes the raw
int32 logits of every step (plus the token ids, separately). It then runs
that same computation through radically different execution paths:

- **batch 1** on CUDA (graphed decode);
- **batch 8**, the sonnet sharing a batch with 7 *random junk co-prompts*
  (row 0 extracted) — different GEMM shapes, different co-batched data;
- **split prefill/decode** — 32 tokens prefilled at once, 32 fed
  one-by-one, then generation;
- **pure-CPU reference** — no cuBLASLt, no Triton, no CUDA graphs; a
  completely independent implementation of the same integer semantics
  (slow: ~30 min for its 512 exact integer forwards).

All four print the **same hash**. The same variations applied to the fp16
model (including a batch of 8 *identical copies* of the prompt) each
diverge, and the demo prints the step at which they fork.

```bash
# full table (~35-40 min; the CPU row dominates)
uv run python scripts/demo.py

# skip the fp16 contrast rows
uv run python scripts/demo.py --skip-fp16

# one configuration — for running on OTHER machines
uv run python scripts/demo.py --config int8-cpu-b1
uv run python scripts/demo.py --config int8-cuda-b1
```

Cross-machine runs: **copy the artifact, never re-run `make prepare`** —
preparation is the float stage and is not required to be bit-reproducible
across machines. The demo prints `sha256(model.safetensors)` first; two
machines are only comparable when it matches. On Apple silicon, run the
`int8-cpu-b1` config (the reference backend is the Apple path per the
spec) — x86 CPU, ARM CPU, and NVIDIA tensor cores printing the same hash
is the three-architecture version of the claim.

## Usage

```bash
uv sync
make prepare    # one-time: build artifacts/qwen3-0.6b-int8 (needs GPU + HF)
make test       # unit + golden + fast determinism checks
make test-all   # + 8k-context checks
make ppl        # full WikiText2 perplexity
```

Generate (greedy, deterministic):

```python
from detllm.model import IntQwen3
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = IntQwen3("artifacts/qwen3-0.6b-int8", backend="cuda")  # or "reference"
ids = tok("The capital of France is", return_tensors="pt").input_ids
toks, _ = model.generate(ids.cuda(), max_new=50)   # use_graph=True for
print(tok.decode(toks[0]))                         # CUDA-graphed decode
```
