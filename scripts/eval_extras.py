"""§9.2 extras: C4 secondary PPL, softmax-clip ablation, qualitative prompts.

Usage:
  uv run python scripts/eval_extras.py --what c4 [--limit 20]
  uv run python scripts/eval_extras.py --what clip --limit 5
  uv run python scripts/eval_extras.py --what qual
"""

import argparse
import sys

import torch

sys.path.insert(0, "scripts")
import eval_ppl  # noqa: E402

ART = "artifacts/qwen3-0.6b-int8"


def c4_segments(seqlen, n_docs=2000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        texts.append(row["text"])
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = ids.shape[0] // seqlen
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]


def run_c4(limit):
    segs = c4_segments(2048)
    if limit:
        segs = segs[:limit]
    print(f"{len(segs)} C4 segments")
    ppl_i = eval_ppl.eval_int(segs, "cuda", ART)
    print(f"C4 ppl[int] = {ppl_i:.4f}")
    ppl_f = eval_ppl.eval_float(segs, torch.float16)
    print(f"C4 ppl[fp16] = {ppl_f:.4f}  ratio {ppl_i/ppl_f:.4f}")


def run_clip(limit):
    import detllm.model as M
    segs = eval_ppl.get_segments(2048)[:limit]
    orig = M.di_softmax
    for c in [12, 15, 20, None]:
        M.di_softmax = (lambda cc: lambda s, m, k, v, clip_c=None: orig(
            s, m, k, v, clip_c=cc))(c)
        ppl = eval_ppl.eval_int(segs, "cuda", ART)
        print(f"clip c={c if c is not None else '∞'}: ppl = {ppl:.4f}")
    M.di_softmax = orig


QUAL_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a village by the sea,",
    "The three laws of thermodynamics state that",
    "def quicksort(arr):",
    "In 1969, humans first landed on the Moon. The mission was called",
    "The recipe for a perfect omelette starts with",
    "Photosynthesis is the process by which",
    "Breaking news: scientists at CERN announced today",
    "The difference between machine learning and deep learning is",
    "Dear hiring manager, I am writing to apply for",
    "The stock market crashed in 1929 because",
    "To be or not to be, that is",
    "The fastest land animal is the cheetah, which can run",
    "import numpy as np\n\ndef matrix_multiply(",
    "The Great Wall of China was built to",
    "Water boils at 100 degrees Celsius at sea level, but",
    "The human brain contains approximately",
    "Once the rocket reached orbit, the astronauts",
    "The most spoken language in the world is",
    "Artificial intelligence will change society by",
]


def run_qual():
    from detllm.model import IntQwen3
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = IntQwen3(ART, backend="cuda")
    for p in QUAL_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        toks, _ = model.generate(ids.cuda(), 200)
        text = tok.decode(toks[0], skip_special_tokens=True)
        print(f"\n=== {p!r}\n{p}{text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["c4", "clip", "qual"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.what == "c4":
        run_c4(args.limit)
    elif args.what == "clip":
        run_clip(args.limit or 5)
    else:
        run_qual()


if __name__ == "__main__":
    main()
