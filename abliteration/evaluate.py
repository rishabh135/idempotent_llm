"""Minimal before/after evaluation for an abliterated checkpoint (PLAN.md
§4). NOT a substitute for the fuller capability/safety suite PRINCIPLES.md
§22 recommends before treating any ablated model as usable for anything
beyond studying the technique itself — this covers exactly three signals:

  - harmful-prompt refusal rate (should drop)
  - harmless-prompt refusal rate, i.e. over-refusal (should NOT rise)
  - WikiText2 perplexity (should not regress past --max-ppl-regression)

No harmful prompts ship in this repo; --harmful-file/--harmless-file point
at files you supply.

Usage:
  uv run python abliteration/evaluate.py \\
      --baseline Qwen/Qwen3-0.6B --ablated artifacts/qwen3-0.6b-ablated-fp32 \\
      --harmful-file harmful_prompts.txt --harmless-file harmless_prompts.txt
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "abliteration")
sys.path.insert(0, "scripts")
from abliterate import chat_ids, is_refusal, load_prompts  # noqa: E402


def refusal_rate(model_id: str, prompts: list[str], device: str, max_new: int) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval().to(device)
    hits = 0
    with torch.no_grad():
        for p in prompts:
            ids = chat_ids(tok, p, device)
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            hits += is_refusal(text)
    return hits / max(1, len(prompts))


def perplexity(model_id: str, device: str, seqlen: int, limit: int) -> float:
    import eval_ppl
    segs = eval_ppl.get_segments(seqlen)
    if limit:
        segs = segs[:limit]
    return eval_ppl.eval_float(segs, torch.float32, model_id=model_id, device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--ablated", required=True)
    ap.add_argument("--harmful-file", required=True)
    ap.add_argument("--harmless-file", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--ppl-limit", type=int, default=20,
                    help="WikiText2 segments (0 = full set)")
    ap.add_argument("--max-ppl-regression", type=float, default=0.10,
                    help="fail if ablated ppl exceeds baseline by this fraction")
    ap.add_argument("--skip-ppl", action="store_true")
    args = ap.parse_args()

    harmful = load_prompts(args.harmful_file)
    harmless = load_prompts(args.harmless_file)

    rows = []
    for label, model_id in (("baseline", args.baseline), ("ablated", args.ablated)):
        print(f"evaluating {label} ({model_id})...")
        hr = refusal_rate(model_id, harmful, args.device, args.max_new)
        hl = refusal_rate(model_id, harmless, args.device, args.max_new)
        ppl = None if args.skip_ppl else perplexity(
            model_id, args.device, args.ppl_seqlen, args.ppl_limit)
        rows.append((label, hr, hl, ppl))
        ppl_s = f"{ppl:.4f}" if ppl is not None else "skipped"
        print(f"  harmful refusal rate:  {hr:.2%}")
        print(f"  harmless refusal rate: {hl:.2%}  (over-refusal if this rises)")
        print(f"  WikiText2 ppl:         {ppl_s}")

    (_, hr0, hl0, ppl0), (_, hr1, hl1, ppl1) = rows
    print(f"\n{'metric':<28}{'baseline':>12}{'ablated':>12}{'delta':>12}")
    print(f"{'harmful refusal rate':<28}{hr0:>12.2%}{hr1:>12.2%}{hr1 - hr0:>12.2%}")
    print(f"{'harmless refusal rate':<28}{hl0:>12.2%}{hl1:>12.2%}{hl1 - hl0:>12.2%}")
    if ppl0 is not None:
        print(f"{'WikiText2 perplexity':<28}{ppl0:>12.4f}{ppl1:>12.4f}"
              f"{ppl1 - ppl0:>12.4f}")

    ok = True
    if hl1 > hl0 + 1e-9:
        print(f"\nFAIL: harmless refusal rate rose ({hl0:.2%} -> {hl1:.2%}) — "
              f"over-refusal, see PRINCIPLES.md §23")
        ok = False
    if ppl0 is not None and ppl1 > ppl0 * (1 + args.max_ppl_regression):
        print(f"\nFAIL: perplexity regressed more than "
              f"{args.max_ppl_regression:.0%} ({ppl0:.4f} -> {ppl1:.4f})")
        ok = False
    if ok:
        print("\nPASS: no over-refusal, no perplexity regression past threshold "
              "(this is NOT a safety clearance — see PRINCIPLES.md §22)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
