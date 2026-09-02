# Abliteration for DetLLM

Directional-ablation ("abliteration") tooling for the Qwen3-0.6B checkpoint
this repo quantizes — estimate a behavioral direction in the residual
stream from contrastive prompts, then permanently orthogonalize the
residual-writer weights against it, **before** handing the result to the
existing [`prepare.py`](../src/detllm/prepare.py) integer-quantization
pipeline unchanged.

**Scope note:** this is a research/interpretability technique for studying
and modifying refusal-related behavior in an instruction-tuned model. It
weakens safety guardrails and does not make any generated content correct,
reliable, or safe. It ships here as tooling (a documented, reversible,
inspectable pipeline with an evaluation harness) so that anyone using it —
including to *study* refusal directions, red-team a model, or evaluate
robustness — can do so with the same rigor (immutable manifests, hashed
artifacts, before/after evaluation) this repo already applies to
quantization. No harmful example prompts ship in this repo; you supply your
own harmful/harmless prompt sets. See [PRINCIPLES.md](PRINCIPLES.md) §22-23
for the evaluation requirements and common failure modes before using this
on anything you intend to ship.

## Layout

```
abliteration/
  README.md        this file — orientation and quick start
  PRINCIPLES.md     the math: direction-finding, projection, idempotency proof,
                    which layers to touch, evaluation methodology
  PLAN.md           concrete engineering plan: how this integrates with
                    DetLLM's prepare.py -> model.py pipeline, file-by-file
  abliterate.py     implementation: find-direction, apply, verify subcommands
  evaluate.py       implementation: before/after refusal-rate + sanity checks
```

Read [PRINCIPLES.md](PRINCIPLES.md) first if you haven't done this before —
it derives the projection math from a 2x2 integer example up to the full
weight-orthogonalization update, and explains *why* only `embed_tokens` and
each layer's `o_proj`/`down_proj` are touched (§13), not `q_proj`/`k_proj`/
`v_proj`/`gate_proj`/`up_proj`. [PLAN.md](PLAN.md) is the DetLLM-specific
translation of that into concrete stages, files, and a CLI.

## Why this fits DetLLM specifically

DetLLM's entire integer runtime ([`model.py`](../src/detllm/model.py)) is
downstream of one float-permitted stage
([`prepare.py`](../src/detllm/prepare.py)): "quantize whatever BF16
checkpoint you hand it, deterministically." Abliteration is *also* a
float-stage weight edit (PRINCIPLES.md §19 calls this out directly: project
once in float, then quantize once). That means:

- **Zero changes to the integer runtime.** `model.py`, `dyadic.py`,
  `ops/*.py`, `backends.py` — none of it needs to know an ablated checkpoint
  exists. An ablated model is just a different BF16 source for
  `prepare.py`.
- **The determinism suite still applies unmodified.** Once quantized, an
  ablated artifact must pass the exact same §9.3 batch/prefill/decode/
  cross-backend invariance tests as any other artifact — abliteration
  changes *which* weights get quantized, not how quantization or the
  runtime works.
- **The one integration point is `prepare.py`'s hardcoded `MODEL_ID`.**
  [PLAN.md](PLAN.md) proposes the smallest possible change (a `--model-id`
  override, default unchanged) so it can load a local ablated checkpoint
  directory instead of always fetching `Qwen/Qwen3-0.6B`.

## Quick start (once implemented per PLAN.md)

```bash
# 1. Find a candidate direction per layer from your own prompt files
#    (one instruction per line; NOT shipped in this repo — see PLAN.md §1)
uv run python abliteration/abliterate.py find-direction \
    --harmful-file harmful_prompts.txt \
    --harmless-file harmless_prompts.txt \
    --out abliteration/artifacts/direction.pt

# 2. Apply the chosen layer's direction: orthogonalize embed_tokens +
#    every o_proj + every down_proj, save a new local float checkpoint
uv run python abliteration/abliterate.py apply \
    --direction abliteration/artifacts/direction.pt --layer 9 \
    --out artifacts/qwen3-0.6b-ablated-fp32

# 3. Sanity-check refusal rate before/after on held-out prompts
uv run python abliteration/evaluate.py \
    --baseline Qwen/Qwen3-0.6B --ablated artifacts/qwen3-0.6b-ablated-fp32 \
    --prompts held_out_prompts.txt

# 4. Feed the ablated float checkpoint into the existing quantization
#    pipeline, completely unmodified from here on
uv run python -m detllm.prepare --model-id artifacts/qwen3-0.6b-ablated-fp32 \
    --out artifacts/qwen3-0.6b-ablated-int8
```

`abliteration/artifacts/` (like the repo's own top-level `artifacts/`) is
gitignored — checkpoints and directions are build products, not source.
