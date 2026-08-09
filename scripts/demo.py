"""Determinism demo: one prompt, many execution configurations, one hash.

Sonnet 18 (first 64 tokens) → greedy-generate N tokens → chain-hash the
int32 logits of every generated step (h = sha256(h_prev ‖ step_logits_bytes))
and separately the token ids. Checkpoints at 64/128/256/512 steps let slow
configs (CPU reference) stop early and still be compared exactly.

int8 configs must produce IDENTICAL hashes at every checkpoint:
  int8-cuda-b1       batch 1, CUDA-graphed decode
  int8-cuda-b8       batched with 7 RANDOM co-prompts (row 0 extracted)
  int8-cuda-split    prompt fed as prefill(32) + 32 token-by-token steps
  int8-cpu-b1        pure-CPU reference backend (eager, no triton, no cuda)

fp16 baselines (HF model, greedy, KV cache) demonstrate the contrast — the
same variations change the answer, and the demo reports the first step at
which each diverges from fp16-cuda-b1:
  fp16-cuda-b1 / fp16-cuda-b8dup (8 COPIES of the same prompt!) / fp16-cuda-split

Cross-machine use: copy the SAME artifact (never re-run prepare.py — the
float calibration stage is not required to be reproducible across machines;
the artifact sha256 is printed so runs are provably comparable), then:
    uv run python scripts/demo.py --config int8-cpu-b1 --steps 128
on e.g. an Apple-silicon Mac (the reference backend is the Apple path per
spec §2 — x86 CPU, ARM CPU and NVIDIA tensor cores all print the same hash).

Default full table runs in ~6-8 minutes on an A100 box.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch

ART_DEFAULT = "artifacts/qwen3-0.6b-int8"
CHECKPOINTS = (64, 128, 256, 512)
PROMPT_TOKENS = 64

SONNET = (
    "Shall I compare thee to a summer's day?\n"
    "Thou art more lovely and more temperate:\n"
    "Rough winds do shake the darling buds of May,\n"
    "And summer's lease hath all too short a date;\n"
    "Sometime too hot the eye of heaven shines,\n"
    "And often is his gold complexion dimm'd;\n"
    "And every fair from fair sometime declines,\n"
    "By chance or nature's changing course untrimm'd;\n"
    "But thy eternal summer shall not fade,\n"
    "Nor lose possession of that fair thou ow'st;\n"
)


class HashChain:
    """h_i = sha256(h_{i-1} ‖ bytes_i); checkpoint snapshots + full history."""

    def __init__(self):
        self.h = hashlib.sha256(b"detllm-demo-v1").digest()
        self.steps: list[str] = []          # per-step hex (for divergence search)
        self.at: dict[int, str] = {}        # checkpoint -> hex

    def update(self, data: bytes):
        self.h = hashlib.sha256(self.h + data).digest()
        self.steps.append(self.h.hex()[:16])
        n = len(self.steps)
        if n in CHECKPOINTS:
            self.at[n] = self.h.hex()[:16]


def _bytes_i32(t: torch.Tensor) -> bytes:
    return t.detach().to("cpu", torch.int32).contiguous().numpy().tobytes()


def _bytes_f16(t: torch.Tensor) -> bytes:
    return t.detach().to("cpu", torch.float16).contiguous().numpy().tobytes()


def get_prompt_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = tok(SONNET, return_tensors="pt").input_ids[0][:PROMPT_TOKENS]
    assert ids.shape[0] == PROMPT_TOKENS
    return tok, ids


def co_prompts(n: int):
    """Deterministic random junk co-prompts (same on every machine)."""
    g = torch.Generator().manual_seed(1234)
    return [torch.randint(100, 50000, (int(torch.randint(5, 40, (1,), generator=g)),),
                          generator=g) for _ in range(n)]


# ---------------------------------------------------------------------------
# int8 configs
# ---------------------------------------------------------------------------

def _hash_int8_steps(step_logits_row0, tok_chain, logit_chain):
    for lg in step_logits_row0:
        logit_chain.update(_bytes_i32(lg))
        # greedy token from these logits is recomputed by the caller's model;
        # token chain fed separately


def run_int8(config: str, steps: int, artifact: str):
    from detllm.model import IntQwen3
    backend = "cpu" if "cpu" in config else "cuda"
    if backend == "cuda":
        from detllm.compile import compile_ops
        compile_ops()
    model = IntQwen3(artifact, backend="reference" if backend == "cpu" else "cuda")
    _, ids = get_prompt_ids()
    dev = model.device
    tok_chain, logit_chain = HashChain(), HashChain()

    use_graph = backend == "cuda"
    if config.endswith("b8"):
        seqs = [ids] + co_prompts(7)
        T = max(len(s) for s in seqs)
        bids = torch.zeros(len(seqs), T, dtype=torch.int64)
        valid = torch.zeros(len(seqs), T, dtype=torch.bool)
        for i, s in enumerate(seqs):
            bids[i, : len(s)] = s
            valid[i, : len(s)] = True
        toks, step_logits = model.generate(bids.to(dev), steps,
                                           chunk_valid=valid.to(dev),
                                           use_graph=use_graph)
        row_toks, row_logits = toks[0], [l[0] for l in step_logits]
    elif config.endswith("split"):
        # prefill 32, then 32 forced token-by-token steps, then generate
        half = PROMPT_TOKENS // 2
        pos = torch.arange(half, device=dev).unsqueeze(0)
        logits, _, _, cache = model.forward(ids[:half].unsqueeze(0).to(dev), pos)
        last = logits[:, -1]
        for t in range(half, PROMPT_TOKENS):
            lg, _, _, cache = model.forward(
                ids[t:t + 1].unsqueeze(0).to(dev),
                torch.tensor([[t]], device=dev), cache)
            last = lg[:, 0]
        row_logits = [last[0]]
        cur = model.greedy_pick(last)
        row_toks_l = [cur[0]]
        if use_graph:
            from detllm.graphed import GraphedDecode
            lens = torch.full((1,), PROMPT_TOKENS, dtype=torch.int64, device=dev)
            gd = GraphedDecode(model, cache, lens)
            for _ in range(steps - 1):
                lg = gd.step(cur).clone()
                row_logits.append(lg[0])
                cur = model.greedy_pick(lg)
                row_toks_l.append(cur[0])
        else:
            for step in range(1, steps):
                p = torch.tensor([[PROMPT_TOKENS + step - 1]], device=dev)
                lg, _, _, cache = model.forward(cur.unsqueeze(1), p, cache)
                row_logits.append(lg[:, 0][0])
                cur = model.greedy_pick(lg[:, 0])
                row_toks_l.append(cur[0])
        row_toks = torch.stack(row_toks_l)
    else:  # b1
        toks, step_logits = model.generate(ids.unsqueeze(0).to(dev), steps,
                                           use_graph=use_graph)
        row_toks, row_logits = toks[0], [l[0] for l in step_logits]

    for t, lg in zip(row_toks, row_logits):
        tok_chain.update(int(t).to_bytes(8, "little"))
        logit_chain.update(_bytes_i32(lg))
    return tok_chain, logit_chain


# ---------------------------------------------------------------------------
# fp16 configs (HF model, hand-rolled greedy with KV cache)
# ---------------------------------------------------------------------------

def run_fp16(config: str, steps: int):
    from transformers import AutoModelForCausalLM
    dev = "cuda" if "cuda" in config else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=dtype).eval().to(dev)
    _, ids = get_prompt_ids()
    tok_chain, logit_chain = HashChain(), HashChain()

    def prefill(bids):
        out = model(bids.to(dev), use_cache=True)
        return out.logits[:, -1], out.past_key_values

    with torch.no_grad():
        if config.endswith("b8dup"):
            # 8 COPIES of the SAME prompt — even this changes fp16 results
            last, past = prefill(ids.unsqueeze(0).repeat(8, 1))
        elif config.endswith("split"):
            half = PROMPT_TOKENS // 2
            last, past = prefill(ids[:half].unsqueeze(0))
            for t in range(half, PROMPT_TOKENS):
                out = model(ids[t:t + 1].unsqueeze(0).to(dev),
                            past_key_values=past, use_cache=True)
                last, past = out.logits[:, -1], out.past_key_values
        else:
            last, past = prefill(ids.unsqueeze(0))

        for _ in range(steps):
            row = last[0]
            cur = torch.argmax(row).view(1)
            tok_chain.update(int(cur).to_bytes(8, "little"))
            logit_chain.update(_bytes_f16(row))
            nxt = cur.view(1, 1).expand(last.shape[0], 1).to(dev)
            out = model(nxt, past_key_values=past, use_cache=True)
            last, past = out.logits[:, -1], out.past_key_values
    return tok_chain, logit_chain


# ---------------------------------------------------------------------------

INT8_CONFIGS = ["int8-cuda-b1", "int8-cuda-b8", "int8-cuda-split", "int8-cpu-b1"]
FP16_CONFIGS = ["fp16-cuda-b1", "fp16-cuda-b8dup", "fp16-cuda-split"]


def run_config(name: str, steps: int, artifact: str):
    t0 = time.perf_counter()
    if name.startswith("int8"):
        tc, lc = run_int8(name, steps, artifact)
    else:
        tc, lc = run_fp16(name, steps)
    for c in (tc, lc):  # final step is always an implicit checkpoint
        c.at.setdefault(len(c.steps), c.h.hex()[:16])
    return tc, lc, time.perf_counter() - t0


def first_divergence(a: HashChain, b: HashChain):
    for i, (x, y) in enumerate(zip(a.steps, b.steps)):
        if x != y:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="run one configuration (for cross-machine "
                                     f"runs): {INT8_CONFIGS + FP16_CONFIGS}")
    ap.add_argument("--steps", type=int, default=0,
                    help="generated tokens (default 512; int8-cpu default 128)")
    ap.add_argument("--artifact", default=ART_DEFAULT)
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    import hashlib as _h
    with open(f"{args.artifact}/model.safetensors", "rb") as f:
        art_sha = _h.sha256(f.read()).hexdigest()
    print(f"artifact sha256: {art_sha[:16]}…  (runs are only comparable "
          f"across machines when this matches)\n")

    if args.config:
        steps = args.steps or (128 if args.config == "int8-cpu-b1" else 512)
        tc, lc, dt = run_config(args.config, steps, args.artifact)
        print(f"{args.config}  ({steps} tokens, {dt:.0f}s)")
        for n in CHECKPOINTS:
            if n in lc.at:
                print(f"  @{n:<4} tokens {tc.at[n]}   logits {lc.at[n]}")
        return

    results = {}
    configs = list(INT8_CONFIGS) + ([] if args.skip_fp16 else FP16_CONFIGS)
    for name in configs:
        steps = args.steps or (128 if name == "int8-cpu-b1" else 512)
        print(f"running {name} ({steps} tokens)...", flush=True)
        results[name] = run_config(name, steps, args.artifact)

    cols = sorted({n for _, lc, _ in results.values() for n in lc.at})
    hdr = "".join(f"{'@%d tok/logits' % n:<34}" for n in cols)
    print(f"\n{'config':<22} {hdr}")
    base_int = results["int8-cuda-b1"]
    for name in configs:
        tc, lc, dt = results[name]
        cells = "".join(
            f"{(tc.at[n][:12] + ' / ' + lc.at[n][:12]) if n in lc.at else '—':<34}"
            for n in cols)
        print(f"{name:<22} {cells} ({dt:.0f}s)")

    print()
    ok = True
    for name in INT8_CONFIGS[1:]:
        if name not in results:
            continue
        div = first_divergence(base_int[1], results[name][1])
        n = min(len(base_int[1].steps), len(results[name][1].steps))
        if div is None:
            print(f"int8: {name} == int8-cuda-b1 for all {n} compared steps ✓")
        else:
            print(f"int8: {name} DIVERGED from int8-cuda-b1 at step {div} ✗")
            ok = False
    if not args.skip_fp16:
        base_fp = results["fp16-cuda-b1"]
        for name in FP16_CONFIGS[1:]:
            div = first_divergence(base_fp[1], results[name][1])
            where = f"logits diverged at step {div}" if div is not None \
                else "identical (fp16 got lucky this time)"
            print(f"fp16: {name} vs fp16-cuda-b1 -> {where}")
    print("\nVERDICT:", "int8 pipeline bit-identical across every configuration"
          if ok else "INT8 DIVERGENCE — this is a bug, report it")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
