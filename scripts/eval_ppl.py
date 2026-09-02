"""WikiText2 perplexity (§9.2): int8 pipeline vs fp32/fp16 HF baseline.

Standard protocol: concatenate the raw test set, tokenize once, split into
non-overlapping segments of --seqlen, average NLL over all predicted tokens.
Float math here is ANALYSIS ONLY (log-softmax over the returned int32
logits); the model's numerical path stays integer.

Usage:
  uv run python scripts/eval_ppl.py --model int --backend cuda [--limit 20]
  uv run python scripts/eval_ppl.py --model fp16
"""

import argparse
import math

import torch

from detllm.backends import canonical_backend


def get_segments(seqlen: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = ids.shape[0] // seqlen
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]


def eval_int(segments, backend: str, artifact: str, model=None):
    from detllm.model import IntQwen3
    if model is None:
        model = IntQwen3(artifact, backend=backend)
    total_nll, total_tok = 0.0, 0
    head_m = model.head_m.double().to(model.device)
    for si, seg in enumerate(segments):
        ids = seg.unsqueeze(0).to(model.device)
        T = ids.shape[1]
        pos = torch.arange(T, device=model.device).unsqueeze(0)
        logits, rm, rk, _ = model.forward(ids, pos)
        # analysis-only float conversion (row scale is a per-row temperature —
        # must be applied before softmax)
        lf = logits.double() * head_m[None, None, :]
        lf = lf * rm.double() * torch.pow(2.0, -(rk.double() + model.head_k))
        lp = torch.log_softmax(lf[0, :-1], dim=-1)
        nll = -lp.gather(1, ids[0, 1:, None]).sum().item()
        total_nll += nll
        total_tok += T - 1
        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(segments)}] running ppl "
                  f"{math.exp(total_nll / total_tok):.4f}", flush=True)
    return math.exp(total_nll / total_tok)


def eval_float(segments, dtype, model_id="Qwen/Qwen3-0.6B",
              device="cuda" if torch.cuda.is_available() else "cpu"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype).eval().to(device)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for si, seg in enumerate(segments):
            ids = seg.unsqueeze(0).to(device)
            out = model(ids)
            lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            nll = -lp.gather(1, ids[0, 1:, None]).sum().item()
            total_nll += nll
            total_tok += ids.shape[1] - 1
            if (si + 1) % 20 == 0:
                print(f"  [{si+1}/{len(segments)}] running ppl "
                      f"{math.exp(total_nll / total_tok):.4f}", flush=True)
    return math.exp(total_nll / total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["int", "fp16", "fp32"], default="int")
    ap.add_argument("--backend", default="cuda", type=canonical_backend)
    ap.add_argument("--artifact", default="artifacts/qwen3-0.6b-int8")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="segments (0=all)")
    args = ap.parse_args()

    segs = get_segments(args.seqlen)
    if args.limit:
        segs = segs[: args.limit]
    print(f"{len(segs)} segments of {args.seqlen}")
    if args.model == "int":
        ppl = eval_int(segs, args.backend, args.artifact)
    else:
        ppl = eval_float(segs, torch.float16 if args.model == "fp16" else torch.float32)
    print(f"FINAL ppl[{args.model}] = {ppl:.4f}")


if __name__ == "__main__":
    main()
