"""Determinism cookbook: fire the SAME prompt at an already-running
`llama-server` N times, SHA256-chain the generated text exactly the way
scripts/demo.py's HashChain chains raw int32 logits, and report whether all
N runs are byte-identical.

IMPORTANT SCOPE LIMIT (read this before citing the result of this script):
this only tests single-machine, single-build "request-level idempotency" —
the same narrower notion deep-dive/README.md's section 1.1 uses that name
for. It does NOT test cross-hardware, cross-batch-composition, or
cross-llama.cpp-version bit-exactness — the guarantee this repo's own
integer runtime (src/detllm/model.py + dyadic.py) provides and scripts/demo.py
proves across CPU/A100/H100. GGUF's K-quant kernels dequantize blocks to
float and accumulate in a hardware/SIMD-dependent order; float addition is
not associative, so that stability is expected only on one fixed
machine+build+thread-count, at temperature 0 (greedy). See
llamacpp-quant-lab/PRINCIPLES.md sections 11-13 for the full argument.

Run (llama-server must already be running, e.g. via `make llamacpp-serve`):
  uv run python llamacpp-quant-lab/cookbook.py --runs 20 --n-predict 128
  uv run python llamacpp-quant-lab/cookbook.py --runs 20 --n-predict 128 \\
      --with-int8-reference --int8-steps 64 \\
      --out llamacpp-quant-lab/report/determinism-cookbook.md
  uv run python llamacpp-quant-lab/cookbook.py --explain
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import requests

sys.path.insert(0, "scripts")
from demo import SONNET, HashChain, first_divergence  # noqa: E402

EXPLAIN_TEXT = __doc__.split("Run (llama-server")[0]

HOST_DEFAULT = "http://127.0.0.1:8080"


def run_once(host: str, n_predict: int) -> tuple[HashChain, dict]:
    body = {
        "prompt": SONNET, "n_predict": n_predict, "temperature": 0,
        "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": False,
        "stream": True,
    }
    chain = HashChain()
    timings = {}
    with requests.post(f"{host}/completion", json=body, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk = json.loads(line[len("data: "):])
            content = chunk.get("content", "")
            if content:
                chain.update(content.encode("utf-8"))
            if chunk.get("stop"):
                timings = chunk.get("timings", {})
    return chain, timings


def run_int8_reference(steps: int) -> str:
    out = subprocess.run(
        ["uv", "run", "python", "scripts/demo.py", "--config", "int8-cpu-b1",
        "--steps", str(steps)],
        capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.strip().startswith("@"):
            return line.strip().split("logits")[-1].strip()
    return "<no hash line found>"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--with-int8-reference", action="store_true")
    ap.add_argument("--int8-steps", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--explain", action="store_true",
                    help="print the scope-limit caveat and exit, no requests")
    args = ap.parse_args()

    if args.explain:
        print(EXPLAIN_TEXT)
        return

    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p(f"determinism cookbook: {args.runs} runs, n_predict={args.n_predict}, "
      f"host={args.host}")
    p(f"request: temperature=0, top_k=1, top_p=1.0, seed=42, cache_prompt=false")
    p()

    chains = []
    all_timings = []
    for i in range(args.runs):
        chain, timings = run_once(args.host, args.n_predict)
        chains.append(chain)
        all_timings.append(timings)
        final_hash = chain.h.hex()[:16]
        tps = timings.get("predicted_per_second", 0)
        p(f"  run {i:>3}: final_hash={final_hash}  "
          f"predicted_tok/s={tps:.1f}" if tps else
          f"  run {i:>3}: final_hash={final_hash}")

    base = chains[0]
    ok = True
    for i, c in enumerate(chains[1:], start=1):
        div = first_divergence(base, c)
        if div is not None:
            ok = False
            p(f"\nFAIL: run {i} diverged from run 0 at token-delta step {div}")
            break
    p()
    if ok:
        p(f"PASS: {args.runs}/{args.runs} runs byte-identical "
          f"(final hash {base.h.hex()[:16]})")
    else:
        p(f"FAIL: not all {args.runs} runs were byte-identical (see above)")

    if args.with_int8_reference:
        p()
        p("cross-check against this repo's own int8 pipeline "
          f"(scripts/demo.py --config int8-cpu-b1 --steps {args.int8_steps}, "
          "a quick sanity contrast, NOT the full 512-step suite):")
        int8_hash = run_int8_reference(args.int8_steps)
        p()
        p(f"{'pipeline':<42}{'runs':>6}{'identical?':>12}  guarantee class")
        p(f"{'llama.cpp GGUF (this lab)':<42}{args.runs:>6}"
          f"{'yes' if ok else 'no':>12}  request-level, THIS machine + build only")
        p(f"{'DetLLM int8 (scripts/demo.py)':<42}{1:>6}{'n/a (ref)':>12}  "
          f"request-level AND cross-hardware/cross-batch/cross-backend")
        p(f"    DetLLM int8-cpu-b1 hash: {int8_hash}")

    p()
    p("SCOPE LIMIT: this demonstrates single-machine, single-build "
      "'request-level idempotency' only (deep-dive/README.md §1.1) — not "
      "cross-hardware/cross-batch bit-exactness. See --explain for the full "
      "argument.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"# Determinism cookbook — real run "
                    f"({datetime.datetime.now(datetime.timezone.utc).isoformat()})\n\n")
            f.write("```text\n" + "\n".join(lines) + "\n```\n")
        print(f"\nwrote {args.out}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
