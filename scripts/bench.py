"""§10 benchmarks: decode + prefill throughput, int8 pipeline vs fp16 eager.

Usage:
  uv run python scripts/bench.py --model int [--batch 1] [--prompt-len 128]
  uv run python scripts/bench.py --model fp16 --batch 8
"""

import argparse
import time

import torch

ART = "artifacts/qwen3-0.6b-int8"


def bench_int(batch, prompt_len, decode_steps, prefill_len, compiled=False,
              graphed=False):
    if compiled or graphed:
        from detllm.compile import compile_ops
        compile_ops()
    from detllm.model import IntQwen3
    model = IntQwen3(ART, backend="cuda")
    if compiled or graphed:  # trigger compilation outside the timed region
        model.generate(torch.randint(100, 50000, (batch, prompt_len)).cuda(), 3)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(100, 50000, (batch, prompt_len), generator=g).cuda()

    if graphed:
        # steady-state: one GraphedDecode session, timed after capture+warm
        from detllm.graphed import GraphedDecode
        pos = torch.arange(prompt_len, device="cuda").expand(batch, prompt_len)
        logits, _, _, cache = model.forward(ids, pos)
        lens = torch.full((batch,), prompt_len, dtype=torch.int64, device="cuda")
        gd = GraphedDecode(model, cache, lens)
        cur = model.greedy_pick(logits[:, -1])
        for _ in range(10):
            cur = model.greedy_pick(gd.step(cur))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            cur = model.greedy_pick(gd.step(cur))
        torch.cuda.synchronize()
        decode_tps = batch * decode_steps / (time.perf_counter() - t0)
    else:
        torch.cuda.synchronize()
        model.generate(ids, 3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids, decode_steps)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        decode_tps = batch * decode_steps / dt

    # prefill timing
    pids = torch.randint(100, 50000, (batch, prefill_len), generator=g).cuda()
    pos = torch.arange(prefill_len).cuda().expand(batch, prefill_len)
    model.forward(pids, pos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.forward(pids, pos)
    torch.cuda.synchronize()
    prefill_tps = batch * prefill_len / (time.perf_counter() - t0)
    return decode_tps, prefill_tps


def bench_fp16(batch, prompt_len, decode_steps, prefill_len):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16).eval().cuda()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(100, 50000, (batch, prompt_len), generator=g).cuda()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=3, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=decode_steps, do_sample=False,
                       min_new_tokens=decode_steps)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        decode_tps = batch * decode_steps / dt

        pids = torch.randint(100, 50000, (batch, prefill_len), generator=g).cuda()
        model(pids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(pids)
        torch.cuda.synchronize()
        prefill_tps = batch * prefill_len / (time.perf_counter() - t0)
    return decode_tps, prefill_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["int", "int-compiled", "int-graphed", "fp16"],
                    required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--prefill-len", type=int, default=2048)
    args = ap.parse_args()
    if args.model == "fp16":
        d, p = bench_fp16(args.batch, args.prompt_len, args.decode_steps,
                          args.prefill_len)
    else:
        d, p = bench_int(args.batch, args.prompt_len, args.decode_steps,
                         args.prefill_len,
                         compiled=args.model == "int-compiled",
                         graphed=args.model == "int-graphed")
    print(f"{args.model} batch={args.batch}: decode {d:.1f} tok/s, "
          f"prefill {p:.0f} tok/s")


if __name__ == "__main__":
    main()
