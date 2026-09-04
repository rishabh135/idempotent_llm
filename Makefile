# Regression harness (§9.5)
#
# make test     — everything except full perplexity (unit + golden +
#                 determinism fast checks); any numerics change must
#                 re-freeze golden bounds deliberately.
# make test-all — includes the slow 4k-long-context determinism checks.
# make ppl      — full WikiText2 perplexity, int8 CUDA + fp16 baseline.
# make prepare  — rebuild the artifact from the HF checkpoint.
# make quant-report — per-layer quantization + storage breakdown.
# make setup-llamacpp / llamacpp-* — GGUF/llama.cpp quantization lab, see
#                 llamacpp-quant-lab/README.md.

.PHONY: test test-all ppl prepare eval quant-report \
	setup-llamacpp llamacpp-convert llamacpp-report llamacpp-diff \
	llamacpp-visualize llamacpp-serve llamacpp-cookbook llamacpp-dashboard

test:
	uv run pytest tests/ -q -m "not slow"

test-all:
	uv run pytest tests/ -q

ppl:
	uv run python scripts/eval_ppl.py --model int --backend cuda
	uv run python scripts/eval_ppl.py --model fp16

eval: test

quant-report:
	uv run python scripts/quant_report.py

prepare:
	uv run python -m detllm.prepare --out artifacts/qwen3-0.6b-int8 \
		--qk-alpha 0.3 --calib-samples 48 --calib-len 1024

setup-llamacpp:
	uv run python llamacpp-quant-lab/setup_llamacpp.py

llamacpp-convert:
	uv sync --extra llamacpp-lab
	uv run python llamacpp-quant-lab/convert_quantize.py

llamacpp-report:
	uv run python llamacpp-quant-lab/gguf_report.py \
		--gguf llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf \
		--out llamacpp-quant-lab/report/qwen3-0.6b-gguf-report.md

llamacpp-diff:
	uv run python llamacpp-quant-lab/gguf_report.py \
		--compare llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_S.gguf \
		          llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf \
		--assert-known-asymmetry \
		--out llamacpp-quant-lab/report/q4_k_s-vs-q4_k_m-diff.md

llamacpp-visualize:
	uv run python llamacpp-quant-lab/visualize.py \
		--gguf-dir llamacpp-quant-lab/artifacts --out-dir llamacpp-quant-lab/report/charts
	uv run python llamacpp-quant-lab/visualize.py \
		--compare llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_S.gguf \
		          llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf \
		--out-dir llamacpp-quant-lab/report/charts

# native binary, not a uv-run python step — llama-server IS the front end
# (built-in web UI at localhost:8080 + OpenAI-API-compatible /v1/chat/completions)
llamacpp-serve:
	llamacpp-quant-lab/vendor/llama.cpp/build/bin/llama-server \
		-m llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf --port 8080 --metrics

llamacpp-cookbook:
	uv run python llamacpp-quant-lab/cookbook.py --runs 20 --n-predict 128 \
		--with-int8-reference --int8-steps 64 \
		--out llamacpp-quant-lab/report/determinism-cookbook.md

llamacpp-dashboard:
	uv run python llamacpp-quant-lab/dashboard.py --mode live-metrics --interval 2
