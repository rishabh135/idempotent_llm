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
two different answers. **Floating-point addition is not associative** —
`(a + b) + c ≠ a + (b + c)`, since each addition rounds — while
high-performance kernels sum in whatever order maximizes throughput. That
order shifts with batch size and composition (your rows get tiled
differently depending on who you share a batch with), prefill vs.
incremental decode (one big GEMM vs. many small ones), and kernel selection
(cuBLAS picks tile shapes by heuristic; split-K and atomics can differ run
to run). The "same" forward pass yields slightly different logits, and one
flipped argmax cascades into a visibly different completion.
"Deterministic mode" flags only pin run-to-run order on one machine at one
shape; they do nothing for batch invariance or cross-hardware
reproducibility.

Across hardware the problem is structural, not incidental: chips differ in
tensor-core design, internal accumulation precision, and which reduction
order is fastest, so bit-identical float across architectures means
abandoning either each chip's optimized paths or float itself. PyTorch's
own reproducibility notes say as much — no guarantee across releases or
platforms, and none between CPU and GPU even with identical seeds.

The usual fix, **batch-invariant floating-point kernels** with a pinned
reduction order, treats the symptom: determinism holds only for those
kernels on that platform. Reaching for int8 doesn't work either, because
mainstream int8 stacks aren't actually integer — I-LLM (arXiv:2405.17849)
notes that prior methods such as SmoothQuant and OmniQuant use *simulated*
quantization, integers at the edges with the compute-intensive operations
run on dequantized floats, leaving pipelines that "still involve partially
FP operations on non-linear operators such as Softmax, Normalization, and
SiLU". Every surviving float op, and every dequantize→compute→requantize
cycle around it, hands the result back to platform-dependent rounding.
Integer *storage* is not integer *arithmetic*.

This project removes the cause: **integer addition is exactly associative
and commutative**, so every reduction — GEMM accumulation, softmax sums,
norm sums — yields the same bits in *any* order on *any* correct hardware.
Cashing that in takes the *entire* forward pass: no float anywhere, softmax,
RMSNorm, SwiGLU and RoPE included (this repo's contribution, via I-LLM's
dyadic-scale machinery). Determinism stops being a discipline to maintain
and becomes a property of the arithmetic — which is what makes the
strongest check here possible at all: an A100's tensor cores, an H100's,
and a CPU's plain integer matmuls produce **identical logits, bit for
bit**.

Integer-only inference itself isn't new — I-LLM and its predecessors
(I-BERT, I-ViT) target *efficiency* on integer-only edge hardware and never
examine determinism; batch invariance, prefill/decode invariance and
cross-device bit-exactness are not claims in that literature. To our
knowledge this is the first pipeline built and verified end-to-end for
**unconditional determinism** — bit-identical logits across runs, batch
compositions, prefill/decode splits and CPU/GPU backends at once — with the
test suite treating each as an exact, zero-tolerance acceptance criterion.

It simplifies I-LLM — dyadic-scale arithmetic, shift-based integer
non-linear operators — with analytic (training-free) smoothing. WikiText2
perplexity **20.72 (int8) vs 20.95 (fp16)**; CUDA-graphed integer decode
runs at **3.6× the fp16 eager baseline at batch 1 and 3.4× at batch 8**
(106 / 783 tok/s on an A100), bit-exact throughout. Spec in
[docs/SPEC.md](docs/SPEC.md), all numbers in
[docs/RESULTS.md](docs/RESULTS.md), design decisions and deviation log in
[docs/NOTES.md](docs/NOTES.md).

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

## Demo: one prompt, three machines, one hash

`scripts/demo.py` is the whole thesis in one table. As an example prompt it
takes the first 64 tokens of Shakespeare's Sonnet 18 ("Shall I compare thee
to a summer's day?"), greedy-generates 512 tokens, and chain-hashes the raw
int32 logits of every step (plus the token ids, separately). It then runs
that same computation through radically different execution paths:

- **batch 1** on CUDA;
- **batch 8**, the sonnet sharing a batch with 7 *random junk co-prompts*
  (row 0 extracted) — different GEMM shapes, different co-batched data;
- **split prefill/decode** — 32 tokens prefilled at once, 32 fed
  one-by-one, then generation;
- **pure-CPU reference** — the same integer semantics executed through an
  entirely different kernel stack: no cuBLASLt, no Triton, no CUDA graphs
  (512 exact integer forwards — by far the slowest row to run).

Those variations are then run on three machines that share no silicon: an
**A100 box** (Ampere tensor cores + an AMD EPYC 7J13), an **H100 box**
(Hopper + an Intel Xeon Platinum 8480+), and an **Apple M5 Max** (ARM,
macOS). The two NVIDIA boxes run all four; the Mac has no CUDA, so it runs
the reference path only. All three CUDA configurations use CUDA-graphed
decode. Same artifact `6658cea4dd89c613…` for every row, 512 generated
tokens for every row, hashes abbreviated:

| machine | execution path | int8 hash @512 | fp16 hash @512 |
|---------|----------------|----------------|----------------|
| A100 box | batch 1 | `64430dd985f8` | `af3ffdc1592d` |
| A100 box | batch 8 (7 random co-prompts) | `64430dd985f8` | `12cf67ceecbb` |
| A100 box | split prefill + token-by-token | `64430dd985f8` | `229b14ee3995` |
| A100 box | CPU (EPYC 7J13) | `64430dd985f8` | `02880fd41404` |
| H100 box | batch 1 | `64430dd985f8` | `88b7a5544ac5` |
| H100 box | batch 8 (7 random co-prompts) | `64430dd985f8` | `b1a5a93c743d` |
| H100 box | split prefill + token-by-token | `64430dd985f8` | `246e001ebbfd` |
| H100 box | CPU (Xeon 8480+) | `64430dd985f8` | `22af55b4f422` |
| M5 Max | CPU (macOS, ARM) | `64430dd985f8` | `54ed3db507f9` |
| | | **1 unique hash** ✅ | **9 unique hashes** ❌ |

**Nine int8 runs, one logits hash. Nine fp16 runs, nine.**

The int8 column is the thesis at full strength: **NVIDIA Ampere, NVIDIA
Hopper, an AMD EPYC, an Intel Xeon, and an Apple M5 Max — four vendors and
three instruction sets — emit the same 512-step logits hash, bit for
bit.** Three of those rows share no kernel with the CUDA ones: the reference
backend runs the same integer semantics entirely through plain CPU integer
ops — no cuBLASLt, no Triton, no CUDA graphs. Nothing was tuned per-platform
to make it happen; the arithmetic simply has no freedom left to disagree.

Every fp16 run forks instead, and forks **at step 0** — different reduction
orders in different kernels, chosen by batch shape, by prefill-vs-decode
split, by GPU generation, and by vendor. The CPU rows additionally use a
different internal accumulation path, since CPUs don't natively execute
fp16 math the way tensor cores do. Nine runs of identical mathematics, nine
different answers.

Worth savoring: the demo also chain-hashes the token ids, and despite every
float row disagreeing on the logits from step 0, eight of the nine still
generate the **same 512 tokens** — greedy decoding hid the divergence below
the argmax the whole way, across a GPU generation and a CPU vendor change.
The measured mechanism: batch composition perturbs this prompt's step-0
logits by up to **0.094**, while the top-1/top-2 margin happens to be
**2.66** — a 28× cushion. The M5 Max is the ninth row, and it is where the
cushion runs out: same weights, same prompt, same greedy decode, same
dtype, genuinely different text. Every generation is one near-tie away from
two "identical" deployments quietly disagreeing — which is what makes float
nondeterminism so insidious, and what the int8 column proves is optional.

The demo prints the exact hardware (GPU model, CPU model, torch version)
and the artifact sha256 above its table, so results from different machines
are self-documenting and provably comparable. Running the int8 configs on
*any* other correct hardware must reproduce these hashes exactly, given the
same artifact; the fp16 hashes carry no such promise (that is the point).

Throughput is measured separately with `scripts/bench.py` — the demo is a
correctness harness, and its wall-clock is dominated by one-time setup
rather than decode. Decode/prefill grids for the A100 and H100 boxes, plus
WikiText2 perplexity, are in [docs/RESULTS.md](docs/RESULTS.md).

```bash
# full table (~35-40 min; the CPU row dominates)
uv run python scripts/demo.py

# skip the fp16 contrast rows
uv run python scripts/demo.py --skip-fp16

# one configuration — for running on OTHER machines
uv run python scripts/demo.py --config int8-cpu-b1
uv run python scripts/demo.py --config int8-cuda-b1
```

Cross-machine runs: **use the published artifact, never re-run
`make prepare`** — preparation is the float stage and is not required to
be bit-reproducible across machines. The exact artifact used for the
numbers in this repo is on the Hub:

```bash
hf download nathanbarry/detllm-qwen3-0.6b-int8 --local-dir artifacts/qwen3-0.6b-int8
```

The demo prints `sha256(model.safetensors)` on startup (this artifact:
`6658cea4dd89c613…`); two machines are only comparable when it matches.
On Apple silicon, run the `int8-cpu-b1` config — the reference backend is
the Apple path per the spec, and it is the M5 Max row in the table above.

## Usage

```bash
uv sync
make prepare    # one-time: build artifacts/qwen3-0.6b-int8 (needs GPU + HF)
make test       # unit + golden + fast determinism checks
make test-all   # + 4k-long-context checks
make ppl        # full WikiText2 perplexity
```

Generate (greedy, deterministic):

```python
from detllm.model import IntQwen3
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = IntQwen3("artifacts/qwen3-0.6b-int8", backend="cuda")  # or "reference"/"cpu"
ids = tok("The capital of France is", return_tensors="pt").input_ids
toks, _ = model.generate(ids.cuda(), max_new=50)   # use_graph=True for
print(tok.decode(toks[0]))                         # CUDA-graphed decode
```
