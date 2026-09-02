# Implementation plan: abliteration for DetLLM

This translates [PRINCIPLES.md](PRINCIPLES.md)'s general math into concrete
stages, files, and interfaces for *this* repo's Qwen3-0.6B checkpoint and
its `prepare.py` -> `model.py` pipeline. See that document first for why
each step is shaped the way it is; this document is the "what to build,"
not the "why."

## 0. Goals and non-goals

**Goals:**
- A repeatable, manifest-addressed pipeline that turns a float Qwen3-0.6B
  checkpoint plus a chosen direction into a new float checkpoint with
  `embed_tokens` and every layer's `o_proj`/`down_proj` orthogonalized
  against that direction (PRINCIPLES.md §13).
- An evaluation harness that reports refusal rate, harmless-prompt
  over-refusal, and a basic capability/coherence check before and after,
  on held-out data the user supplies.
- Zero changes to the integer runtime (`model.py`, `dyadic.py`, `ops/*`,
  `backends.py`) — an ablated model is just a different float source for
  the existing `prepare.py`.

**Non-goals (out of scope for this plan):**
- Shipping any harmful prompts, a specific direction, or a pre-built
  ablated artifact in this repository.
- A production-grade safety/capability eval suite (MMLU harness, red-team
  rotation, etc.) — §4 below is a minimal, honest starting point, not a
  replacement for one.
- Multi-direction subspace removal (PRINCIPLES.md §16) or non-difference-
  of-means direction estimators (§17) — single difference-of-means
  direction, one layer, is the v1 target. Both are natural v2 extensions
  and the manifest schema (§3) already has room for them.
- Activation-time (runtime) ablation inside the integer path. Projecting
  an *activation* mid-forward-pass would need a new dyadic-scale-aware
  integer op family (like `ops/di_softmax.py`) and re-verification of every
  §9.3 determinism invariant for it. Permanent float-stage weight
  orthogonalization (PRINCIPLES.md §10-11) sidesteps all of that — it's a
  different checkpoint, not a different runtime — which is why it's the
  only mode this plan implements.

## 1. Stage 1: direction-finding (`abliterate.py find-direction`)

**Input:** two plain-text files, one prompt per line — `--harmful-file`
and `--harmless-file`. Not shipped in this repo. Conventional public
sources: a harmful-instruction set (e.g. AdvBench-style behavior prompts)
contrasted with a benign-instruction set (e.g. Alpaca-style prompts) of
comparable length/format, so the discovered direction tracks *refusal*,
not topic, register, or prompt length (PRINCIPLES.md §23's first failure
mode). Loading is left to the user (`datasets.load_dataset(...)` or a local
file) — same externalization pattern `prepare.py` already uses for
WikiText2 and `eval_extras.py` uses for C4.

**Procedure:**
1. Load the float checkpoint via `AutoModelForCausalLM.from_pretrained`
   (default `Qwen/Qwen3-0.6B`, override with `--model-id` for iterating on
   an already-modified checkpoint).
2. Apply the model's chat template to every prompt (a raw-string prompt and
   its chat-templated form land at very different residual-stream
   positions; mixing the two is PRINCIPLES.md §23's second failure mode).
3. Forward each prompt once, capturing `resid_post` (the block output,
   pre-next-block-norm) at every layer, at the **last token position**
   only — this is the conventional site (it's the token whose forward pass
   actually decides "do I refuse next").
4. Per layer: `direction[l] = mean(harmful_acts[l]) - mean(harmless_acts[l])`,
   then L2-normalize.
5. **Layer scoring** (so the tool doesn't just dump 28 candidates on the
   user): for each candidate layer, project it out via the activation-level
   `remove_direction` hook from PRINCIPLES.md §8 (non-destructive), generate
   a few tokens per harmful prompt, and score with the same refusal-phrase
   heuristic `evaluate.py` uses (§4 below). Report the layer with the
   biggest harmful-refusal drop that does **not** also spike on a small
   fixed harmless canary set (a proxy for §23's "harmless over-refusal"
   check) — cheap enough to run at find-direction time since it's forward
   passes only, no weight edits.
6. Save every per-layer direction plus the per-layer scores, not just the
   winner — picking a different layer later shouldn't require re-running
   step 3 (the expensive part).

**Output** (`--out abliteration/artifacts/direction.pt`, gitignored): a
`torch.save`d dict:
```python
{
  "directions": {layer_idx: tensor[hidden]},   # unit-norm, all candidate layers
  "scores": {layer_idx: {"harmful_refusal_before": float, "harmful_refusal_after": float,
                         "harmless_refusal_after": float}},
  "manifest": { ... },  # see §3
}
```

## 2. Stage 2: apply (`abliterate.py apply`)

1. Load the float checkpoint fresh from the **immutable source** named in
   the direction file's manifest (never from an already-ablated checkpoint
   — PRINCIPLES.md §10/§18 are explicit that repeated projection of an
   already-projected source is how bit-idempotency silently breaks, even
   though the ideal projector is idempotent in exact arithmetic).
2. Load `direction.pt`, select `--layer` (default: the winner from step
   1.5 above).
3. Orthogonalize, in this exact order, using
   `orthogonalize_residual_writer` (PRINCIPLES.md §12, adapted for the
   embedding's transposed orientation per §13.3):
   - `model.model.embed_tokens.weight` (`E_new = E - (E @ r) r^T`, row-space)
   - for every layer `i` in `range(n_layers)`:
     `model.model.layers[i].self_attn.o_proj.weight` (`W_new = P @ W`)
     `model.model.layers[i].mlp.down_proj.weight` (`W_new = P @ W`)
   - (Qwen3 ties `lm_head` to `embed_tokens` — orthogonalizing the
     embedding already covers the head; assert `tie_word_embeddings` and
     fail loudly if a future model doesn't tie them, since the head would
     then need its own pass too.)
4. Compute in `float64`, matching `prepare.py`'s own `W.double()` convention
   for the same reason (rounding error in the projection step should be
   negligible next to the *quantization* error `prepare.py` introduces
   afterward — no point being sloppier than the stage that follows it).
   Cast back to the source dtype (bf16) only at save time.
5. **Verify before saving** (§3's `verify_projection`): for a sample of
   held-out prompts, forward the new checkpoint, extract the same
   `resid_post[layer]` as step 1.3, and assert
   `(activations @ direction).abs().max() < 1e-4` (relative to activation
   norm) — a direct numerical check that the write path is actually gone,
   not just "the eval got better."
6. Save via `model.save_pretrained(out_dir)` /
   `tokenizer.save_pretrained(out_dir)` — a standard local HF checkpoint
   directory, so `prepare.py --model-id <out_dir>` (§5) loads it with no
   special-casing.

## 3. Manifest and addressing

Every artifact this stage produces is named by a manifest hash, exactly
matching the pattern `prepare.py`'s own artifact already uses (a
`config.json` alongside the weights, `sha256` used for identity —
PRINCIPLES.md §18 gives the schema this plan follows):

```json
{
  "source_model_id": "Qwen/Qwen3-0.6B",
  "source_revision": "<HF commit sha>",
  "harmful_file_sha256": "<hash>",
  "harmless_file_sha256": "<hash>",
  "chat_template_sha256": "<hash>",
  "selected_layer": 9,
  "residual_stream": "resid_post",
  "writer_policy": ["embed_tokens", "self_attn.o_proj", "mlp.down_proj"],
  "projection_dtype": "float64",
  "output_dtype": "bfloat16",
  "algorithm_version": 1
}
```

Write this JSON as `<out_dir>/abliteration_manifest.json` next to the saved
checkpoint. `abliterate.py apply` computes `sha256(canonical_json(manifest))`
up front and refuses to overwrite an existing `<out_dir>` whose manifest
hash already matches (idempotent build, matching `prepare.py`'s own
"never run preparation on an already prepared artifact" rule) — re-running
with the same inputs is a no-op, not a second projection.

## 4. Stage 3: evaluation (`evaluate.py`)

Minimal, explicitly-not-sufficient-alone harness (PRINCIPLES.md §22 lists
what a *real* one needs — this covers the first two rows only):

- **Refusal rate**: generate greedily (reuse the HF `fp16` generation path
  `scripts/demo.py::run_fp16` already exercises, not the int8 runtime — this
  harness runs *before* quantization) on a held-out harmful-prompt file,
  classify each completion with a small refusal-phrase substring list
  ("I cannot", "I'm not able to", "As an AI", "I won't", ...; the detector
  phrases are generic English refusal boilerplate, not harmful content, so
  they're safe to ship inline in this file).
- **Harmless over-refusal**: same classifier, run on a held-out
  harmless-prompt file. This number going *up* after ablation is the
  clearest sign something broke (PRINCIPLES.md §23's over-refusal risk).
- **Coherence sanity check**: WikiText2 perplexity via
  `scripts/eval_ppl.py::eval_float` (extended with optional `model_id`/
  `device` args so it can point at a local checkpoint dir instead of always
  `Qwen/Qwen3-0.6B`), before vs. after — a cheap proxy for "did the
  projection wreck general capability," not a substitute for
  the fuller capability suite PRINCIPLES.md §22 recommends before treating
  any ablated model as usable for anything beyond this kind of study.

Report a plain before/after table (baseline model vs. `--ablated`
checkpoint dir); exit nonzero if harmless over-refusal increases or
perplexity regresses past a `--max-ppl-regression` threshold, so this can
gate a build the same way `make test` gates a quantization change.

## 5. `prepare.py` integration

Add one argparse option, default unchanged:

```python
ap.add_argument("--model-id", default=MODEL_ID,
                help="HF hub id or local checkpoint dir to quantize "
                     "(e.g. an abliteration/ output directory)")
```

and thread `args.model_id` through to `prepare()` in place of the
module-level `MODEL_ID` constant (three call sites: the tokenizer load, the
model load, and `scalars["model_id"]`). `AutoModelForCausalLM.from_pretrained`
and `AutoTokenizer.from_pretrained` both already accept a local directory
path, so no other change is needed — `prepare.py`'s calibration,
quantization, and artifact format are completely untouched.

The only other outside change is the same pattern applied to
`scripts/eval_ppl.py::eval_float`, which hardcoded both the model id and
`.cuda()` — it now takes optional `model_id`/`device` arguments (default
unchanged) so §4's evaluation harness can point it at a local ablated
checkpoint on whatever device is available. These two are the only changes
this plan makes outside the `abliteration/` folder.

## 6. Testing strategy

- **Unit: projection idempotency.** `P @ (P @ W) == P @ W` and
  `direction @ (P @ W) ≈ 0` for random `W`, asserted in exact float64 —
  the direct translation of PRINCIPLES.md §7's `P² = P` proof into a test,
  belongs in `tests/test_abliteration.py` alongside this repo's other
  golden/property tests.
- **Smoke test: shape/orientation correctness, no real model.** Construct
  a tiny randomly-initialized `Qwen3ForCausalLM` from a small
  `Qwen3Config` (a handful of layers, small hidden size — no download, no
  real weights, no semantic content) and run `find-direction` -> `apply`
  ->`verify` end to end against a handful of synthetic (non-harmful,
  clearly placeholder) contrastive strings, purely to catch shape/dtype/
  orientation bugs before they meet the real 0.6B checkpoint. This is the
  same philosophy `scripts/quant_report.py` was smoke-tested with, against
  a synthetic artifact, before running it on the real download.
- **Real-checkpoint validation is manual, by design.** This plan
  deliberately does not include a CI job that downloads Qwen3-0.6B, runs a
  real harmful/harmless prompt set, and asserts a specific refusal-rate
  delta — that would mean committing a harmful-prompt fixture to the repo
  and treating "successfully reduced refusal" as a green checkmark, which
  is the wrong thing to automate. Running the real pipeline stays a manual,
  deliberate action with a human reading `evaluate.py`'s output.

## 7. Sequencing

1. `prepare.py --model-id` (§5) — trivial, unlocks everything else, useful
   independent of abliteration (anyone quantizing a fine-tuned Qwen3-0.6B
   variant benefits).
2. `abliterate.py apply` + the projection unit test (§2, §6) — the
   permanent-weight-update core, testable without any prompt data at all.
3. `abliterate.py find-direction` (§1) — needs real user-supplied prompt
   files to exercise for real, but its output shape is fixed by §2's input
   contract, so it can be built and smoke-tested independently.
4. `evaluate.py` (§4) — depends on §5 only for the "after" side; can be
   validated against the *baseline* model alone first (refusal rate on
   harmful prompts should be high, on harmless prompts near zero, before
   any ablation exists to compare against).

## 8. Open questions for whoever runs this for real

- Which harmful/harmless prompt sets, and under what access/handling
  policy for those files locally (they are not committed here).
- What refusal-rate delta, and what harmless-over-refusal ceiling, would
  actually justify shipping a resulting int8 artifact anywhere — this plan
  builds the measurement tools, not the decision threshold.
- Whether a single layer's direction generalizes across the prompt
  distribution you actually care about, or whether §16's multi-direction
  subspace removal is needed — decide from `find-direction`'s per-layer
  score table, not a priori.
