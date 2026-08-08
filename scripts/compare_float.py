"""Milestone 3 diagnostic: run the integer model (reference backend) and the
HF float model side by side; report per-layer residual-stream divergence and
top-token agreement. Float appears here as an ANALYSIS oracle only.

Usage: uv run python scripts/compare_float.py [--layers-only]
"""

import argparse

import torch

from detllm.model import IntQwen3

ART = "artifacts/qwen3-0.6b-int8"
PROMPT = "The capital of France is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--backend", default="reference")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    fmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                                  dtype=torch.float32).eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        fout = fmodel(ids, output_hidden_states=True)
    fh = fout.hidden_states  # tuple [B,T,H] per layer boundary

    imodel = IntQwen3(ART, backend=args.backend)

    # instrument: capture residual h after each layer by monkeypatching
    captured = []
    orig_add = None
    import detllm.dyadic as dy
    orig = dy.residual_add
    calls = {"n": 0}

    def spy(h, k_res, d, dm, dk):
        h2, k2 = orig(h, k_res, d, dm, dk)
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # after MLP add = end of layer
            captured.append((h2.clone(), k2.clone()))
        return h2, k2

    dy.residual_add = spy
    try:
        T = ids.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        logits, xm, xk, cache = imodel.forward(ids, pos)
    finally:
        dy.residual_add = orig

    print(f"{'layer':>5} {'rel_rms_err':>12} {'cos_sim':>9}")
    for li, (h, k) in enumerate(captured):
        hv = h.double() * torch.pow(2.0, -k.double())
        fv = fh[li + 1].double()
        rel = ((hv - fv).pow(2).mean() / fv.pow(2).mean()).sqrt().item()
        cos = torch.nn.functional.cosine_similarity(
            hv.flatten(), fv.flatten(), dim=0).item()
        print(f"{li:>5} {rel:>12.4f} {cos:>9.5f}")

    fl = fout.logits[0, -1]
    il = logits[0, -1].double() * imodel.head_m.double()
    il = il * float(xm[0, -1]) * 2.0 ** -(float(xk[0, -1]) + imodel.head_k)
    ftop = torch.topk(fl, 5)
    itop = torch.topk(il, 5)
    print("float top5:", [(tok.decode([i]), round(v.item(), 2)) for v, i in zip(*ftop)])
    print("int   top5:", [(tok.decode([i]), round(v.item(), 2)) for v, i in zip(*itop)])
    print("argmax match:", ftop.indices[0].item() == itop.indices[0].item())


if __name__ == "__main__":
    main()
