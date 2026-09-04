# llamacpp-quant-lab

A GGUF/llama.cpp companion to this repo's own dyadic int8 pipeline: quantize
the same Qwen3-0.6B checkpoint to several real GGUF precision levels
entirely locally, visualize exactly which tensors land at which
quantization type, host it with `llama-server`, and test what determinism
guarantee that setup actually gives you — versus what this repo's own
integer runtime gives you.

**Read [PRINCIPLES.md](PRINCIPLES.md) first** if you haven't done this
before — it builds up GGUF's block quantization scheme from a plain-language
bit-budget explanation through a fully worked numeric example, contrasts it
directly against this repo's own `(data, m, k)` dyadic scheme (same tensor,
both schemes, side by side), shows the real per-layer quantization pattern
on the actual Qwen3-0.6B model, and explains precisely why GGUF's kernels
don't give the cross-hardware bit-exact guarantee this repo's own
[`model.py`](../src/detllm/model.py)/[`dyadic.py`](../src/detllm/dyadic.py)
pipeline does. [PLAN.md](PLAN.md) is the engineering plan — what was built,
in what order, and the real numbers each step produced.

## Layout

```
llamacpp-quant-lab/
  README.md        this file — orientation and quick start
  PRINCIPLES.md     the math + the GGUF-vs-DetLLM comparison, worked examples, real data
  PLAN.md           engineering plan: what was built, file by file, with real output
  setup_llamacpp.py       clone + build llama.cpp with Metal support
  convert_quantize.py     HF checkpoint -> F16 GGUF -> 6 quantized variants
  gguf_report.py          per-layer quantization report + two-file diff (the screenshot, reproduced)
  visualize.py            matplotlib heatmaps/bar charts + plotly interactive HTML
  cookbook.py             determinism cookbook: repeated queries, SHA256-chained
  dashboard.py            lightweight tokens/sec + latency dashboard
```

## Quick start

```bash
uv sync --extra llamacpp-lab

# 1. clone + build llama.cpp with Metal (a few minutes)
make setup-llamacpp

# 2. convert Qwen3-0.6B to F16 GGUF, then quantize to F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q4_K_S
make llamacpp-convert

# 3. per-layer quantization report + the Q4_K_S-vs-Q4_K_M diff (the screenshot, on the real model)
make llamacpp-report
make llamacpp-diff

# 4. matplotlib/plotly visualizations
make llamacpp-visualize

# 5. host it — llama-server IS the front end (chat UI at http://localhost:8080,
#    OpenAI-API-compatible /v1/chat/completions)
make llamacpp-serve

# 6. in another terminal, with the server running:
make llamacpp-cookbook     # determinism cookbook (20 repeated queries, SHA256-chained)
make llamacpp-dashboard    # live tokens/sec + latency
```

Everything under `artifacts/`, `vendor/`, and `runs/` is gitignored (build
products, not source) — `report/` (the generated docs + PNG charts) is
committed, mirroring how [`quant-report/README.md`](../quant-report/README.md)
commits `scripts/quant_report.py`'s captured real output.
