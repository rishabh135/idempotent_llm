# Regression harness (§9.5)
#
# make test     — everything except full perplexity (unit + golden +
#                 determinism fast checks); any numerics change must
#                 re-freeze golden bounds deliberately.
# make test-all — includes the slow 4k-long-context determinism checks.
# make ppl      — full WikiText2 perplexity, int8 CUDA + fp16 baseline.
# make prepare  — rebuild the artifact from the HF checkpoint.
# make quant-report — per-layer quantization + storage breakdown.

.PHONY: test test-all ppl prepare eval quant-report

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
