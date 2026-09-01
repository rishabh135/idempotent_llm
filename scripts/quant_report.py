"""Quantization & storage report for a prepared int8 artifact (§7/§8).

Introspects ``<artifact>/model.safetensors`` (header only — tensor names,
dtypes, shapes; no weight data is loaded into memory) and
``<artifact>/config.json`` (architecture facts + the per-linear dyadic `k`
and worst-case-error `min_m` diagnostics recorded by `prepare.py`) to answer,
per transformer layer and in aggregate:

  - how many logical weight parameters live in each layer, and how many
    bytes they and their quantization bookkeeping (per-channel scales,
    activation-smoothing factors, KV-cache scales, RoPE tables) actually
    cost on disk;
  - what quantization scheme is in force for each tensor family (int8
    symmetric per-output-channel weights, int8 per-token/per-row dynamic
    activations at runtime, int64 dyadic (m, k) scale bookkeeping — see
    `dyadic.py` and `prepare.py`);
  - the achieved precision per layer (worst-case relative weight-scale
    error, from the `min_m` calibration diagnostic prepare.py records);
  - the net compression ratio against an equivalent fp16/fp32 checkpoint of
    the same architecture.

Run:
  uv run python scripts/quant_report.py
  uv run python scripts/quant_report.py --artifact artifacts/qwen3-0.6b-int8
  uv run python scripts/quant_report.py --full   # + one row per raw tensor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct

ART_DEFAULT = "artifacts/qwen3-0.6b-int8"

DTYPE_BITS = {"I8": 8, "I16": 16, "I32": 32, "I64": 64,
              "U8": 8, "F16": 16, "BF16": 16, "F32": 32, "F64": 64}

# tensor-name suffix -> (role, human label, quantization scheme)
WEIGHT_SCHEME = ("weight", "int8, symmetric, per-output-channel scale "
                 "(row of .m mantissa / 2^k, one shared k per tensor)")
SCALE_SCHEME = ("scale", "int64 per-output-channel mantissa (pairs with the "
                "tensor-wide shift k recorded in config.json)")
NORM_SCHEME = ("norm-gamma", "int64 per-channel dyadic RMSNorm affine "
               "(gamma folded out of the main path at prep time)")
SMOOTH_SCHEME = ("smoothing", "int64 per-channel dyadic multiplier "
                 "(analytic SmoothQuant-style activation smoothing, "
                 "folded into the consumer weight at prep time)")
KV_SCHEME = ("kv-scale", "int64 static per-(KV-head[,channel]) dyadic K/V "
             "cache scale (calibrated once offline)")

COMPONENT_INFO = {
    "q.w8t": ("attn", "Q-proj weight", *WEIGHT_SCHEME),
    "q.m": ("attn", "Q-proj scale", *SCALE_SCHEME),
    "k.w8t": ("attn", "K-proj weight", *WEIGHT_SCHEME),
    "k.m": ("attn", "K-proj scale", *SCALE_SCHEME),
    "v.w8t": ("attn", "V-proj weight", *WEIGHT_SCHEME),
    "v.m": ("attn", "V-proj scale", *SCALE_SCHEME),
    "o.w8t": ("attn", "O-proj weight", *WEIGHT_SCHEME),
    "o.m": ("attn", "O-proj scale", *SCALE_SCHEME),
    "gate.w8t": ("mlp", "Gate-proj weight", *WEIGHT_SCHEME),
    "gate.m": ("mlp", "Gate-proj scale", *SCALE_SCHEME),
    "up.w8t": ("mlp", "Up-proj weight", *WEIGHT_SCHEME),
    "up.m": ("mlp", "Up-proj scale", *SCALE_SCHEME),
    "down.w8t": ("mlp", "Down-proj weight", *WEIGHT_SCHEME),
    "down.m": ("mlp", "Down-proj scale", *SCALE_SCHEME),
    "q_norm.m": ("attn", "Q-Norm gamma", *NORM_SCHEME),
    "k_norm.m": ("attn", "K-Norm gamma", *NORM_SCHEME),
    "qs_m": ("attn", "QK-smoothing factor", *SMOOTH_SCHEME),
    "as_m": ("attn", "attn-input smoothing", *SMOOTH_SCHEME),
    "ms_m": ("mlp", "mlp-input smoothing", *SMOOTH_SCHEME),
    "ds_m": ("mlp", "down-input smoothing", *SMOOTH_SCHEME),
    "os_m": ("attn", "o-input smoothing", *SMOOTH_SCHEME),
    "kq_m": ("attn", "K-cache scale mantissa", *KV_SCHEME),
    "kq_k": ("attn", "K-cache scale shift", *KV_SCHEME),
    "ks_m": ("attn", "K-score scale mantissa", *KV_SCHEME),
    "ks_k": ("attn", "K-score scale shift", *KV_SCHEME),
    "vq_m": ("attn", "V-cache scale mantissa", *KV_SCHEME),
    "vq_k": ("attn", "V-cache scale shift", *KV_SCHEME),
}
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def read_safetensors_header(path: str) -> dict:
    """Parse just the JSON header (tensor name -> dtype/shape/offsets); the
    multi-hundred-MB data payload that follows is never read into memory."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def numel(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def classify(name: str):
    """-> (layer_idx or None, suffix, section, label, role, scheme)."""
    m = LAYER_RE.match(name)
    if m:
        li, suffix = int(m.group(1)), m.group(2)
        section, label, role, scheme = COMPONENT_INFO[suffix]
        return li, suffix, section, label, role, scheme
    if name in ("embed.e8", "embed.m", "embed.k"):
        scheme = ("int8, symmetric, per-ROW (per-token-id) scale — each row "
                  "has its OWN shift k too (embed.k), unlike weight matrices"
                  if name == "embed.e8" else
                  "int64 per-row mantissa/shift (paired with embed.e8)")
        return None, name, "embed", "Token embedding", "weight" if name == "embed.e8" else "scale", scheme
    if name in ("head.w8t", "head.m"):
        role, scheme = WEIGHT_SCHEME if name == "head.w8t" else SCALE_SCHEME
        return None, name, "head", "LM head (gamma-folded, untied on disk)", role, scheme
    if name in ("rope.cos", "rope.sin"):
        return None, name, "rope", "RoPE lookup table", "constant", (
            "int16, global fixed-point cos/sin table, single shared shift "
            "k=rope_frac_bits (see config.json)")
    raise ValueError(f"unrecognized tensor name: {name!r}")


def original_fp_params(c: dict) -> int:
    """Parameter count of the ORIGINAL tied-embedding bf16 checkpoint this
    artifact was quantized from (no biases: attention_bias=false, no MLP
    bias) — for a size comparison, not read from any tensor here."""
    hidden, inter = c["hidden"], c["intermediate"]
    n_q, n_kv, hd = c["n_q_heads"], c["n_kv_heads"], c["head_dim"]
    per_layer = (
        hidden * (n_q * hd)      # q_proj
        + hidden * (n_kv * hd)   # k_proj
        + hidden * (n_kv * hd)   # v_proj
        + (n_q * hd) * hidden    # o_proj
        + hidden * inter         # gate_proj
        + hidden * inter         # up_proj
        + inter * hidden         # down_proj
        + hd + hd                # q_norm, k_norm gamma
        + hidden + hidden         # input/post-attn layernorm gamma
    )
    return c["vocab"] * hidden + c["n_layers"] * per_layer + hidden  # + final norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=ART_DEFAULT)
    ap.add_argument("--full", action="store_true",
                    help="also dump one row per raw tensor (~760 rows)")
    args = ap.parse_args()

    st_path = f"{args.artifact}/model.safetensors"
    with open(f"{args.artifact}/config.json") as f:
        c = json.load(f)
    header = read_safetensors_header(st_path)

    rows = []  # (name, li, suffix, section, label, role, scheme, dtype, shape, numel, bytes)
    for name, meta in header.items():
        li, suffix, section, label, role, scheme = classify(name)
        n = numel(meta["shape"])
        bits = DTYPE_BITS[meta["dtype"]]
        nbytes = n * bits // 8
        off0, off1 = meta["data_offsets"]
        assert off1 - off0 == nbytes, (name, off1 - off0, nbytes)
        rows.append(dict(name=name, li=li, suffix=suffix, section=section,
                         label=label, role=role, scheme=scheme,
                         dtype=meta["dtype"], shape=meta["shape"],
                         numel=n, bytes=nbytes))

    total_bytes = sum(r["bytes"] for r in rows)
    total_weight_params = sum(r["numel"] for r in rows if r["role"] == "weight")
    orig_params = original_fp_params(c)

    art_sha = sha256_file(st_path)
    print(f"artifact: {args.artifact}")
    print(f"model: {c.get('model_id', '?')}  |  sha256(model.safetensors): "
          f"{art_sha[:16]}…")
    print(f"architecture: {c['n_layers']}L  hidden={c['hidden']}  "
          f"heads={c['n_q_heads']}/{c['n_kv_heads']} (Q/KV, GQA)  "
          f"head_dim={c['head_dim']}  intermediate={c['intermediate']}  "
          f"vocab={c['vocab']}  weight_bits={c.get('weight_bits', 8)}")
    print()
    print("Dyadic-scale convention (see dyadic.py): every quantized tensor is")
    print("carried as (data, m, k) with real value ≈ data · m / 2^k. `data` is")
    print("the narrow integer stored below; `m` (mantissa) and `k` (binary")
    print("shift) are either a tensor alongside it or a scalar in config.json.")
    print()

    # ---- Table 1: global summary -----------------------------------------
    fp16_bytes = orig_params * 2
    fp32_bytes = orig_params * 4
    eff_bits_per_weight = total_bytes * 8 / total_weight_params
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'logical weight parameters (this artifact)':<48}{total_weight_params:>15,}")
    print(f"{'  of which embed.e8 + head.w8t (untied on disk)':<48}"
          f"{sum(r['numel'] for r in rows if r['section'] in ('embed', 'head') and r['role'] == 'weight'):>15,}")
    print(f"{'original checkpoint parameters (tied, bf16)':<48}{orig_params:>15,}")
    print("-" * 63)
    print(f"{'artifact size on disk (all tensors, incl. scales)':<48}"
          f"{human_bytes(total_bytes):>15}")
    print(f"{'  equivalent fp16 checkpoint size':<48}{human_bytes(fp16_bytes):>15}")
    print(f"{'  equivalent fp32 checkpoint size':<48}{human_bytes(fp32_bytes):>15}")
    print(f"{'this artifact is smaller than fp16 by':<48}"
          f"{fp16_bytes / total_bytes:>14.3f}x")
    print(f"{'this artifact is smaller than fp32 by':<48}"
          f"{fp32_bytes / total_bytes:>14.3f}x")
    print(f"{'effective bits / logical weight parameter':<48}"
          f"{eff_bits_per_weight:>14.2f}b  (8b weight + "
          f"{eff_bits_per_weight - 8:.2f}b scale/bookkeeping overhead)")
    print()

    # ---- Table 2: by raw dtype --------------------------------------------
    print("-" * 78)
    print("BY STORAGE DTYPE")
    print("-" * 78)
    print(f"{'dtype':<8}{'tensors':>10}{'elements':>16}{'bytes':>14}{'% of size':>12}")
    for dt in sorted({r["dtype"] for r in rows}):
        sub = [r for r in rows if r["dtype"] == dt]
        b = sum(r["bytes"] for r in sub)
        print(f"{dt:<8}{len(sub):>10}{sum(r['numel'] for r in sub):>16,}"
              f"{human_bytes(b):>14}{100 * b / total_bytes:>11.1f}%")
    print()

    # ---- Table 3: by component, aggregated across all layers -------------
    print("-" * 78)
    print("BY COMPONENT (summed across all layers)")
    print("-" * 78)
    n_layers = c["n_layers"]
    print(f"{'component':<26}{'dtype':<7}{'shape/layer':<16}{'x' + str(n_layers):>5}"
          f"{'total elems':>14}{'bytes':>12}")
    for suffix in COMPONENT_INFO:
        sub = [r for r in rows if r["suffix"] == suffix]
        if not sub:
            continue
        label, shape, dtype = sub[0]["label"], sub[0]["shape"], sub[0]["dtype"]
        tot_n = sum(r["numel"] for r in sub)
        tot_b = sum(r["bytes"] for r in sub)
        print(f"{label:<26}{dtype:<7}{str(shape):<16}{len(sub):>5}"
              f"{tot_n:>14,}{human_bytes(tot_b):>12}")
    for name in ("embed.e8", "embed.m", "embed.k", "head.w8t", "head.m",
                "rope.cos", "rope.sin"):
        if name not in header:
            continue
        r = next(r for r in rows if r["name"] == name)
        print(f"{name:<26}{r['dtype']:<7}{str(r['shape']):<16}{1:>5}"
              f"{r['numel']:>14,}{human_bytes(r['bytes']):>12}")
    print()
    print("Legend (scheme per component):")
    printed = set()
    for suffix in list(COMPONENT_INFO) + ["embed.e8", "head.w8t", "rope.cos"]:
        info = COMPONENT_INFO.get(suffix)
        scheme = info[3] if info else classify(suffix)[5]
        label = info[1] if info else classify(suffix)[3]
        if scheme in printed:
            continue
        printed.add(scheme)
        print(f"  - {label}-style: {scheme}")
    print()

    # ---- Table 4: per-layer ------------------------------------------------
    print("=" * 78)
    print("PER-LAYER BREAKDOWN (the answer to 'how much does each layer cost')")
    print("=" * 78)
    hdr = (f"{'L':>3}{'attn params':>14}{'mlp params':>13}{'total params':>14}"
           f"{'bytes':>12}{'b/param':>9}{'worst attn err':>16}{'worst mlp err':>15}")
    print(hdr)
    n_layers = c["n_layers"]
    for li in range(n_layers):
        layer_rows = [r for r in rows if r["li"] == li]
        attn_params = sum(r["numel"] for r in layer_rows
                          if r["section"] == "attn" and r["role"] == "weight")
        mlp_params = sum(r["numel"] for r in layer_rows
                         if r["section"] == "mlp" and r["role"] == "weight")
        layer_bytes = sum(r["bytes"] for r in layer_rows)
        layer_params = attn_params + mlp_params
        bpp = layer_bytes * 8 / layer_params
        Lc = c["layers"][li]
        attn_min_m = min(Lc[k]["min_m"] for k in ("q", "k", "v", "o"))
        mlp_min_m = min(Lc[k]["min_m"] for k in ("gate", "up", "down"))
        print(f"{li:>3}{attn_params:>14,}{mlp_params:>13,}{layer_params:>14,}"
              f"{human_bytes(layer_bytes):>12}{bpp:>8.2f}b"
              f"{100 * 0.5 / attn_min_m:>15.3f}%{100 * 0.5 / mlp_min_m:>14.3f}%")
    print("-" * len(hdr))
    print("'worst attn/mlp err' = 0.5/min_m, the worst-case relative "
          "per-channel weight-scale\nerror observed at prep time for that "
          "layer's linears (prepare.py asserts min_m >= 2^8,\ni.e. <= 0.2%, "
          "for every channel of every weight in the model).")
    print()
    extra_rows = [r for r in rows if r["li"] is None]
    extra_bytes = sum(r["bytes"] for r in extra_rows)
    extra_params = sum(r["numel"] for r in extra_rows if r["role"] == "weight")
    print(f"{'non-layer tensors (embed+head+rope)':<48}"
          f"{extra_params:>14,} params  {human_bytes(extra_bytes):>10}")

    if args.full:
        print()
        print("=" * 78)
        print("FULL TENSOR DUMP")
        print("=" * 78)
        print(f"{'name':<28}{'dtype':<7}{'shape':<18}{'elements':>12}{'bytes':>10}")

        def sort_key(r):
            order = {"embed": 0, "head": 1, "rope": 2}
            return (order.get(r["section"], 3), r["li"] if r["li"] is not None else -1, r["name"])
        for r in sorted(rows, key=sort_key):
            print(f"{r['name']:<28}{r['dtype']:<7}{str(r['shape']):<18}"
                  f"{r['numel']:>12,}{human_bytes(r['bytes']):>10}")


if __name__ == "__main__":
    main()
