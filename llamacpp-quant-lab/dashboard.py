"""Lightweight local metrics dashboard for a running llama-server — tokens/sec
and latency, no Prometheus/Grafana/web framework, just stdlib + requests
(+ optional psutil/matplotlib).

Two modes:
  --mode benchmark      fires its own N /completion requests, reads each
                        response's own `timings` object (no server flag needed)
  --mode live-metrics    polls the server's Prometheus-text /metrics endpoint
                        on an interval (server must be started with --metrics)

Run:
  uv run python llamacpp-quant-lab/dashboard.py --mode benchmark \\
      --n-requests 10 --n-predict 128 --chart-name live_tokens_per_sec_demo.png
  uv run python llamacpp-quant-lab/dashboard.py --mode live-metrics --interval 2 --duration 60
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import sys
import time

import requests

sys.path.insert(0, "scripts")
from demo import SONNET  # noqa: E402

HOST_DEFAULT = "http://127.0.0.1:8080"
RUNS_DIR_DEFAULT = "llamacpp-quant-lab/runs"
CHARTS_DIR_DEFAULT = "llamacpp-quant-lab/report/charts"

METRIC_RE = re.compile(r'^(llamacpp:\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)')


def mode_benchmark(host: str, n_requests: int, n_predict: int):
    samples = []
    for i in range(n_requests):
        body = {"prompt": SONNET, "n_predict": n_predict, "temperature": 0,
               "stream": False}
        t0 = time.perf_counter()
        resp = requests.post(f"{host}/completion", json=body)
        resp.raise_for_status()
        wall_s = time.perf_counter() - t0
        timings = resp.json().get("timings", {})
        row = {"t": time.time(), "wall_s": wall_s,
              "prompt_per_second": timings.get("prompt_per_second", 0.0),
              "predicted_per_second": timings.get("predicted_per_second", 0.0),
              "predicted_n": timings.get("predicted_n", 0)}
        samples.append(row)
        print(f"  request {i:>3}: {row['predicted_per_second']:.1f} tok/s decode, "
             f"{row['predicted_n']} tokens, {wall_s:.2f}s wall")
    return samples


def parse_metrics_text(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if m:
            out.setdefault(m.group(1), []).append(float(m.group(3)))
    return {k: sum(v) / len(v) for k, v in out.items()}


def mode_live_metrics(host: str, interval: float, duration: float):
    samples = []
    psutil_proc = None
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name"]):
            if "llama-server" in (p.info.get("name") or ""):
                psutil_proc = psutil.Process(p.info["pid"])
                break
    except ImportError:
        pass

    t_start = time.time()
    print(f"{'elapsed':>8}{'prompt tok/s':>14}{'decode tok/s':>14}{'RSS MB':>10}")
    try:
        while True:
            elapsed = time.time() - t_start
            if duration and elapsed > duration:
                break
            resp = requests.get(f"{host}/metrics", timeout=5)
            resp.raise_for_status()
            metrics = parse_metrics_text(resp.text)
            rss_mb = None
            if psutil_proc is not None:
                try:
                    rss_mb = psutil_proc.memory_info().rss / (1024 * 1024)
                except Exception:
                    rss_mb = None
            row = {
                "t": time.time(), "elapsed_s": elapsed,
                "prompt_tokens_per_sec": metrics.get("llamacpp:prompt_tokens_seconds", 0.0),
                "predicted_tokens_per_sec": metrics.get("llamacpp:predicted_tokens_seconds", 0.0),
                "rss_mb": rss_mb,
            }
            samples.append(row)
            print(f"{elapsed:>7.0f}s{row['prompt_tokens_per_sec']:>14.1f}"
                 f"{row['predicted_tokens_per_sec']:>14.1f}"
                 f"{(rss_mb if rss_mb is not None else float('nan')):>10.0f}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C)")
    return samples


def save_csv(samples, path):
    if not samples:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        w.writeheader()
        w.writerows(samples)
    print(f"wrote {path}")


def save_chart(samples, mode, path):
    if not samples:
        print("no samples collected — skipping chart")
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if mode == "benchmark":
        xs = list(range(len(samples)))
        ax.plot(xs, [s["predicted_per_second"] for s in samples], marker="o",
               label="decode tok/s")
        ax.set_xlabel("request #")
    else:
        xs = [s["elapsed_s"] for s in samples]
        ax.plot(xs, [s["prompt_tokens_per_sec"] for s in samples], label="prompt tok/s")
        ax.plot(xs, [s["predicted_tokens_per_sec"] for s in samples], label="decode tok/s")
        ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("tokens/sec")
    ax.set_title("llama-server throughput")
    ax.legend()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["benchmark", "live-metrics"], required=True)
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--n-requests", type=int, default=10)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0,
                    help="0 = run until Ctrl-C (live-metrics mode only)")
    ap.add_argument("--chart-name", default=None,
                    help="stable filename under report/charts/ (default: timestamped)")
    args = ap.parse_args()

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.mode == "benchmark":
        samples = mode_benchmark(args.host, args.n_requests, args.n_predict)
    else:
        samples = mode_live_metrics(args.host, args.interval, args.duration)

    save_csv(samples, os.path.join(RUNS_DIR_DEFAULT, f"dashboard_{ts}.csv"))
    chart_name = args.chart_name or f"live_tokens_per_sec_{ts}.png"
    save_chart(samples, args.mode, os.path.join(CHARTS_DIR_DEFAULT, chart_name))


if __name__ == "__main__":
    main()
