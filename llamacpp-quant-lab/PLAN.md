# Implementation plan: `llamacpp-quant-lab`

This is the engineering plan — what was built, why it's shaped this way, and
how it fits together. See [PRINCIPLES.md](PRINCIPLES.md) for the math/theory
(why GGUF quantizes the way it does, and how that compares to this repo's own
dyadic int8 scheme) and [README.md](README.md) for the quick start.

## 0. Goals and non-goals

**Goals:**
- Quantize the SAME Qwen3-0.6B checkpoint this repo already uses to several
  real GGUF precision levels, entirely locally ("complete control" — convert
  once, quantize N ways from that one immutable source, never download a
  pre-made community GGUF).
- Report and visualize, from the real produced files (never estimated), the
  exact per-layer/per-tensor quantization scheme GGUF uses — mirroring
  [`scripts/quant_report.py`](../scripts/quant_report.py)'s header-only
  introspection design for this repo's own artifact.
- Host the quantized model locally via `llama-server` and prove two things
  about it: single-machine determinism (the cookbook), and real throughput
  (the dashboard) — both against actually-running infrastructure, not
  simulated.
- Zero changes to this repo's own integer runtime
  ([`src/detllm/model.py`](../src/detllm/model.py),
  [`dyadic.py`](../src/detllm/dyadic.py), `ops/*.py`, `backends.py`) — this
  lab is a separate float/GGUF playground, entirely decoupled from the
  determinism-critical int8 path, the same zero-footprint property
  [`abliteration/PLAN.md`](../abliteration/PLAN.md) holds itself to.

**Non-goals:**
- A production monitoring stack (Prometheus/Grafana) — a lightweight custom
  poller is enough for a single-user local lab (see §6).
- Shipping a pre-built ablated *and* GGUF-quantized model — `prepare.py`'s
  `--model-id` and this folder's `--model-id` both already support pointing
  at an [`abliteration/`](../abliteration/) output directory, but building
  that combination is left as a manual follow-up, not automated here.
- Perfect symmetry claims about GGUF's Q4_K_S/Q4_K_M promotion heuristic —
  §4 below reports exactly what the real build does, which turned out to be
  less clean than the commonly-cited "S=uniform, M=mixed" simplification
  (see PRINCIPLES.md §4 and §8 for the real numbers).

## 1. Folder layout (as built)

```
llamacpp-quant-lab/
  README.md              orientation + quick start
  PRINCIPLES.md           the math/comparison doc
  PLAN.md                 this file
  setup_llamacpp.py       clone+build llama.cpp w/ Metal; idempotent
  convert_quantize.py     HF checkpoint -> F16 GGUF -> 6 quantized GGUFs + manifest.json
  gguf_report.py          single-file report + --compare diff mode
  visualize.py            matplotlib PNGs + plotly interactive HTML
  cookbook.py             determinism cookbook: N repeated /completion calls, SHA256-chained
  dashboard.py            lightweight tokens/sec + latency (+ optional RSS) poller/plotter
  artifacts/              [gitignored] qwen3-0.6b-hf/, 6 real .gguf files, manifest.json,
                          llamacpp_build_info.json
  vendor/llama.cpp/       [gitignored] git clone + Metal build (commit d230ddd763ff)
  runs/                   [gitignored] raw cookbook/dashboard logs
  report/                 COMMITTED docs + charts
    qwen3-0.6b-gguf-report.md   real single-file report (Q4_K_M)
    q4_k_s-vs-q4_k_m-diff.md    real compare-mode diff, --assert-known-asymmetry PASSED
    determinism-cookbook.md    real 20/20-identical cookbook run + int8 cross-check
    charts/                     committed PNGs (real data, see §5)
    interactive/                [gitignored] plotly HTML (real data, not GitHub-renderable)

tests/test_gguf_report.py   pure parsing/classification tests, synthetic GGUF fixture,
                            no network, no llama.cpp build, sub-second
```

`.gitignore` additions: `vendor/`, `llamacpp-quant-lab/runs/`,
`llamacpp-quant-lab/report/interactive/` (`llamacpp-quant-lab/artifacts/` is
covered for free — the existing `artifacts/` rule has no leading slash, so it
matches at any depth).

`pyproject.toml` addition — a `[project.optional-dependencies]` group (the
repo had none before):
```toml
[project.optional-dependencies]
llamacpp-lab = [
    "gguf>=0.10.0",
    "matplotlib>=3.9",
    "plotly>=5.24",
    "requests>=2.32",
    "psutil>=6.0",
    "sentencepiece>=0.2.0",
]
```
`sentencepiece` earned its place the hard way: `convert_hf_to_gguf.py`'s
`Qwen2Model.set_vocab()` tries `_set_vocab_sentencepiece()` first and only
falls back to the correct BPE path (`_set_vocab_gpt2()`) on a
`FileNotFoundError` — but without the package installed, the `import
sentencepiece` line itself raises `ModuleNotFoundError`, which isn't caught,
crashing the conversion before it ever gets to check whether a
`tokenizer.model` file exists. Qwen3 ships only a BPE `tokenizer.json`, so
installing the package just lets the probe fail the *right* way and fall
through. Install with `uv sync --extra llamacpp-lab`.

## 2. `setup_llamacpp.py`

Clones `ggml-org/llama.cpp` into `vendor/llama.cpp/` (gitignored — a large
C++ project has no business being committed; a `--llamacpp-dir` override
exists for anyone with an existing build, same override-with-default
pattern as `--model-id`/`--artifact` elsewhere in this repo). Preflights
`xcode-select -p` and `cmake >= 3.14` with an actionable remedy printed on
failure (this DID fire on the build machine — `cmake` wasn't installed;
`brew install cmake` fixed it in under a minute). Idempotency guard: skips
rebuilding if the three binaries + `convert_hf_to_gguf.py` already exist,
`--force` to redo — deliberately lighter than a content-addressed manifest,
since this is a build tool, not a data artifact.

Build: `cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DGGML_METAL=ON
-DCMAKE_BUILD_TYPE=Release`, then `cmake --build ... --config Release -j
$(sysctl -n hw.ncpu)`. On the M4 Pro build machine this took a few minutes
end to end (clone + full Release build with Metal). Records
`artifacts/llamacpp_build_info.json`:
```json
{
 "llamacpp_commit": "d230ddd763ffe27781c7ffd237ea78b639b36b6d",
 "cmake_flags": ["-DGGML_METAL=ON", "-DCMAKE_BUILD_TYPE=Release"],
 "cmake_version": "4.4.3",
 "host": {"platform": "darwin", "machine": "arm64"}
}
```
`convert_quantize.py` reads this back into its own manifest — the one piece
of cross-script state, kept as a single small JSON file, no database.

`brew install llama.cpp` is documented (docstring only, not executed) as a
faster hosting-only alternative — it ships the compiled binaries but not
`convert_hf_to_gguf.py`/`gguf-py`, so it can't do §3's conversion step.

## 3. `convert_quantize.py`

Real run, real output:
```
variant   file                                sizesha256
F16       qwen3-0.6b-F16.gguf              1.12 GB  31f0240dce446782…
Q8_0      qwen3-0.6b-Q8_0.gguf           609.82 MB  7792f1ba05270bc5…
Q6_K      qwen3-0.6b-Q6_K.gguf           472.17 MB  b4412b3df97d8174…
Q5_K_M    qwen3-0.6b-Q5_K_M.gguf         423.83 MB  18929718c7d70e0d…
Q4_K_M    qwen3-0.6b-Q4_K_M.gguf         378.33 MB  c974d814f88ecb3e…
Q4_K_S    qwen3-0.6b-Q4_K_S.gguf         365.51 MB  95bd2ee787f10039…
```
Steps: (1) `save_pretrained` the HF checkpoint to `artifacts/qwen3-0.6b-hf/`
if not already there (`--model-id` flag, same name/semantics as
`prepare.py`'s own override — an abliterated checkpoint could be fed through
this pipeline with zero code changes); (2) convert to F16 GGUF via
`sys.executable convert_hf_to_gguf.py <hf_dir> --outtype f16 --outfile ...`
(running via `sys.executable` guarantees the repo's own venv, with `gguf`
and `sentencepiece` installed, executes the converter — not whatever bare
`python3` happens to be on `$PATH`); (3) quantize the remaining 5 types via
`llama-quantize <f16> <out> <TYPE>`, always from the F16 baseline, never
variant-from-variant — the same "always rebuild from the immutable source"
rule `prepare.py` applies to its own artifact; (4) write `manifest.json`
(source model id + HF revision, llama.cpp commit/flags, per-variant
filename/sha256/bytes), reusing `sha256_file`/`human_bytes` imported
directly from [`scripts/quant_report.py`](../scripts/quant_report.py) — the
same cross-file reuse pattern `tests/test_abliteration.py` already uses for
`abliteration/abliterate.py`. Skip-if-exists at every step, `--force` to
redo; no heavier hash-guard needed for a single-user local lab.

## 4. `gguf_report.py`

Uses the official `gguf` PyPI package's `GGUFReader` — never a hand-rolled
binary parser. It memory-maps the file and reads only header/tensor-info
metadata, matching `quant_report.py`'s own "no weight data loaded" design.
Confirmed empirically (not assumed) against the real files: llama.cpp's
GGUF tensor names for Qwen3 are `token_embd.weight`, `output_norm.weight`
(no separate `output.weight` — see the tied-embedding note in
PRINCIPLES.md §8) plus per-block `blk.{i}.attn_norm/attn_q/attn_k/attn_v/
attn_output/attn_q_norm/attn_k_norm/ffn_norm/ffn_gate/ffn_up/ffn_down
.weight`. The classifier (`blk\.(\d+)\.(.+)` regex + a non-layer allowlist)
raises `ValueError` on anything else, matching `quant_report.py::classify()`'s
fail-loud discipline.

**Single-file mode** real output highlights (Q4_K_M): 310 tensors, 28
layers, 596,049,920 total elements, 372.65 MB tensor bytes, 5.24 real
bits/weight (computed, not the architecture-ballpark figure), 3.051x
smaller than an F16 file of the same shapes. The per-component table
surfaces mixed types directly — `V-proj weight` and `Down-proj weight` each
get two rows (Q4_K ×14, Q6_K ×14) — and the per-layer table names the exact
14 block indices (0,1,2,5,8,11,14,17,20,23,24,25,26,27).

**Compare mode** (`--compare A B [--assert-known-asymmetry] [--out path]`)
reproduces the user's original vimdiff screenshot: a KV-pair diff (real
run: `general.file_type: A=14 | B=15`, exactly the screenshot's `[14]` vs
`[18]`-style pattern) and a per-tensor diff table with an emoji-swatch
legend built from only the types actually observed (chosen over inline
HTML/CSS — GitHub's markdown sanitizer strips a lot of that; emoji-in-a-table
renders identically everywhere). `--assert-known-asymmetry` exits 1 unless
every changed tensor is `attn_v.weight`/`ffn_down.weight` — a live
regression check, not just a doc claim. Real Q4_K_S-vs-Q4_K_M run: **29
tensors changed** (15 `attn_v.weight`, 14 `ffn_down.weight`), assertion
PASSED. That 15-vs-14 split (not a clean 14-and-14) is itself a real finding
— see PRINCIPLES.md §4/§8 for why the "S is uniform" simplification doesn't
quite hold for this build.

## 5. `visualize.py`

Imports `gguf_report.py`'s row-loading/classification functions directly
(one parser, multiple consumers). Real charts produced, all from real data
(`os.path.getsize`, real `GGUFReader` output — nothing estimated):
- `heatmap_type_<variant>.png` — 28×11 grid (7 weight families + 4 norm
  families), discrete colormap per GGML type, for every variant with
  non-uniform layer typing. First attempt used `#e0e0e0` for F32, which was
  visually indistinguishable from the "missing cell" white — fixed to
  `#9e9e9e` after inspecting the rendered PNG.
- `heatmap_bpw_<variant>.png` — same grid, continuous colormap by real
  bits/weight per cell.
- `size_by_variant.png` — one bar per real produced file (F16 1143 MB down
  to Q4_K_S 366 MB).
- `family_stacked_compare_Q4_K_S_vs_Q4_K_M.png` — two stacked bars isolating
  exactly where the ~13 MB size delta between the two files comes from.
- Plotly interactive HTML versions of the heatmaps (hover tooltips per cell:
  exact tensor name, shape, bytes) — genuinely more useful than static
  labels on a 28×11 grid; gitignored, local-only (GitHub can't render live
  plotly).

## 6. `cookbook.py`

Reuses this repo's own hashing primitives directly: `sys.path.insert(0,
"scripts"); from demo import SONNET, HashChain, first_divergence`. Fixed
request to `llama-server`'s `/completion`: `temperature=0, top_k=1,
top_p=1.0, seed=42, cache_prompt=false, stream=true` — `cache_prompt=false`
deliberately rules out llama.cpp's prompt-cache reuse as a hidden source of
run-to-run variance. Streams SSE `content` deltas into a per-run
`HashChain`, exactly the way `demo.py` chains raw int32 logits. Real run: 20
requests, `n_predict=128`, against a live `llama-server` serving
`qwen3-0.6b-Q4_K_M.gguf` on this machine — **20/20 byte-identical**, final
hash `a380b042a7c6d771`, decode throughput 258–299 tok/s across runs.
`--with-int8-reference` additionally shells out to
`scripts/demo.py --config int8-cpu-b1 --steps 64` and prints a 2-row
contrast table naming the different guarantee classes explicitly. The
critical scope-limit caveat (single-machine/single-build "request-level
idempotency" only, per `deep-dive/README.md` §1.1 — not cross-hardware) is
stated in the module docstring, an `--explain` flag, and the committed
`report/determinism-cookbook.md`.

## 7. `dashboard.py`

Stdlib + `requests` + optional `psutil`/`matplotlib` only — no
Prometheus/Grafana/web framework, per the explicit choice to keep this
lightweight. `--mode benchmark`: fires its own N `/completion` requests
(non-streaming) and reads each response's own `timings` object — no server
flag needed. Real run: 10 requests, 128 tokens each, 288–302 tok/s decode,
~0.43–0.45s wall per request. `--mode live-metrics`: polls the server's
Prometheus-text `/metrics` endpoint (server started with `--metrics`) with a
plain regex, no client library; optionally samples `psutil` RSS if the
`llama-server` process is found. Real smoke run: RSS ~4.9 GB for
`llama-server` serving Q4_K_M with the default 4-slot × 40,960-token
context allocation. Both modes write a raw CSV to gitignored `runs/` and a
PNG to `report/charts/` (`--chart-name` for the stable, doc-referenced one:
`live_tokens_per_sec_demo.png`).

## 8. Makefile targets

Added to the root `Makefile` (flat, `llamacpp-`-prefixed, matching the
existing style): `setup-llamacpp`, `llamacpp-convert`, `llamacpp-report`,
`llamacpp-diff`, `llamacpp-visualize`, `llamacpp-serve`, `llamacpp-cookbook`,
`llamacpp-dashboard`. `llamacpp-serve` runs the native `llama-server` binary
directly, with `--metrics` on — **this is the front end**: its built-in web
UI at `http://localhost:8080` (confirmed working — the initial `curl` 415
was just curl not sending `Accept-Encoding: gzip` the way a real browser
does; `curl --compressed` returns 200 with the real static UI HTML) plus its
OpenAI-API-compatible `/v1/chat/completions` satisfy "point the front end"
with zero additional code.

## 9. Testing (`tests/test_gguf_report.py`)

Builds tiny real GGUF files with `gguf.GGUFWriter` in `tmp_path` fixtures —
raw zero-filled byte arrays of the exact size `GGML_QUANT_SIZES[type]`
implies, tagged with `raw_dtype`, no `raw_shape` override needed (the writer
derives the logical element shape from the byte array's own shape via
`quant_shape_from_byte_shape`). No network, no llama.cpp build, no real
weights — same "smoke-test against synthetic data first" philosophy
`quant_report.py` itself used before its real download. 9 tests: tensor
classification (including the fail-loud unrecognized-name case), mixed-type
loading, diff correctness, and both the pass and fail path of
`--assert-known-asymmetry`. All pass in well under a second and were folded
into the default (non-`slow`) `make test` tier automatically.

## 10. Verification performed

Every item below was run for real against the actual build/files, not
simulated:
- `setup_llamacpp.py`: binaries + `convert_hf_to_gguf.py` present;
  `llamacpp_build_info.json` written and parses.
- `convert_quantize.py`: 6 real `.gguf` files + `manifest.json` with 6
  entries, real sha256/bytes for each.
- `gguf_report.py --gguf ...Q4_K_M.gguf`: 28-row per-layer table,
  `general.architecture` reads `qwen3`.
- `gguf_report.py --compare ...Q4_K_S.gguf ...Q4_K_M.gguf
  --assert-known-asymmetry`: exit 0, PASSED, 29/310 changed.
- `visualize.py` (both modes): 14 PNGs (6 `heatmap_type_*` + 6
  `heatmap_bpw_*` + `family_stacked_compare_*` + `size_by_variant`) + 6
  interactive HTML files, all non-empty; visually inspected 3 of the PNGs
  directly (found and fixed the F32 color legibility issue in the process).
- `llama-server --metrics`: `curl --compressed /` → 200, real chat UI HTML;
  `/v1/models` → real metadata including `"n_params": 596049920`, which
  exactly matches this repo's own `quant-report/README.md`'s "original
  checkpoint parameters (tied, bf16)" figure — an independent cross-check
  that both projects agree on the true (tied) parameter count.
- `cookbook.py --with-int8-reference`: exit 0, PASS 20/20, joint contrast
  table printed, DetLLM int8 reference hash captured.
- `dashboard.py`: both modes each produced a real CSV under `runs/`; the
  benchmark-mode run's chart is the one kept as the canonical, doc-referenced
  PNG (`report/charts/live_tokens_per_sec_demo.png`) — the live-metrics
  mode's own short smoke-test chart was a throwaway verification artifact,
  cleaned up rather than committed.
- `uv run pytest tests/test_gguf_report.py -q`: 9 passed, well under a
  second.
- `uv run pytest tests/ -q -m "not slow"` (full existing suite): 85 passed,
  11 skipped (skips are the pre-existing artifact-required
  `test_determinism.py` cases, unrelated to this folder) — confirms zero
  regression and zero footprint on `src/detllm/*`.
