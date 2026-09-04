"""GGUF quantization/storage report — the llamacpp-quant-lab analog of
scripts/quant_report.py, for llama.cpp's GGUF format instead of this repo's
own safetensors artifact.

Introspects a `.gguf` file's header/metadata ONLY via the official `gguf`
package's `GGUFReader` (numpy-memmapped, no weight data loaded) and reports,
per transformer block and in aggregate: parameter counts, on-disk bytes, the
exact GGML quantization type per tensor family, a dtype breakdown, and (in
--compare mode) a diff against a second `.gguf` file — mirroring a
vimdiff-style two-column comparison of key-value pairs and per-tensor
`shape | size | quantization type`.

Run:
  uv run python llamacpp-quant-lab/gguf_report.py --gguf <path.gguf>
  uv run python llamacpp-quant-lab/gguf_report.py --gguf <path.gguf> --full
  uv run python llamacpp-quant-lab/gguf_report.py \\
      --compare A.gguf B.gguf [--assert-known-asymmetry] [--out report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import re

import gguf

LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
NON_LAYER_NAMES = {"token_embd.weight", "output_norm.weight", "output.weight"}

# suffix -> (family label, section) — section groups attn vs ffn for the
# per-layer param-count split; norms are counted with their parent section.
FAMILY_INFO = {
    "attn_norm.weight": ("attn-norm gamma", "attn"),
    "attn_q.weight": ("Q-proj weight", "attn"),
    "attn_k.weight": ("K-proj weight", "attn"),
    "attn_v.weight": ("V-proj weight", "attn"),
    "attn_output.weight": ("O-proj weight", "attn"),
    "attn_q_norm.weight": ("Q-Norm gamma", "attn"),
    "attn_k_norm.weight": ("K-Norm gamma", "attn"),
    "ffn_norm.weight": ("ffn-norm gamma", "mlp"),
    "ffn_gate.weight": ("Gate-proj weight", "mlp"),
    "ffn_up.weight": ("Up-proj weight", "mlp"),
    "ffn_down.weight": ("Down-proj weight", "mlp"),
}

# quant-type name -> emoji swatch. Not exhaustive by design — only types
# actually observed in a given report get shown in that report's legend.
TYPE_SWATCH = {
    "F32": "⬜", "F16": "🟦", "BF16": "🟦",
    "Q4_0": "🟥", "Q4_1": "🟥", "Q5_0": "🟧", "Q5_1": "🟧", "Q8_0": "🟪",
    "Q2_K": "🟫", "Q3_K": "🟫", "Q4_K": "🟥", "Q5_K": "🟧", "Q6_K": "🟩",
}

ASYMMETRY_ALLOWED_SUFFIXES = {"attn_v.weight", "ffn_down.weight"}


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


def classify(name: str):
    """-> (layer_idx or None, suffix). Raises on an unrecognized tensor name
    (fail loud rather than silently mis-report — same discipline as
    scripts/quant_report.py's own classify())."""
    m = LAYER_RE.match(name)
    if m:
        return int(m.group(1)), m.group(2)
    if name in NON_LAYER_NAMES:
        return None, name
    raise ValueError(f"unrecognized GGUF tensor name: {name!r}")


def load_tensor_rows(path: str):
    """-> (GGUFReader, list of per-tensor dicts). Header/metadata only."""
    reader = gguf.GGUFReader(path)
    rows = []
    for t in reader.tensors:
        li, suffix = classify(t.name)
        rows.append(dict(name=t.name, li=li, suffix=suffix,
                         dtype=t.tensor_type.name, shape=[int(s) for s in t.shape],
                         numel=int(t.n_elements), bytes=int(t.n_bytes)))
    return reader, rows


def kv_pairs(reader) -> dict:
    """-> {key: value}, long array-valued fields summarized as '[N items]'."""
    out = {}
    for key, field in reader.fields.items():
        try:
            val = field.contents()
        except Exception:
            val = "<unparsed>"
        if isinstance(val, list) and len(val) > 8:
            val = f"[{len(val)} items]"
        elif isinstance(val, str) and len(val) > 120:
            val = f"{val[:80]!r}… [{len(val)} chars]"
        out[key] = val
    return out


def n_layers_of(rows) -> int:
    idxs = [r["li"] for r in rows if r["li"] is not None]
    return (max(idxs) + 1) if idxs else 0


def diff_tensors(rows_a, rows_b):
    """-> list of dicts, one per tensor name (union of both files), each with
    a/b dtype+bytes and whether the type changed. Raises if the two files
    don't share the same tensor-name set (they should, for two quant variants
    of the same architecture)."""
    by_name_a = {r["name"]: r for r in rows_a}
    by_name_b = {r["name"]: r for r in rows_b}
    names_a, names_b = set(by_name_a), set(by_name_b)
    if names_a != names_b:
        only_a = sorted(names_a - names_b)[:5]
        only_b = sorted(names_b - names_a)[:5]
        raise ValueError(f"tensor name sets differ between files: "
                         f"only in A (sample): {only_a}, only in B (sample): {only_b}")
    out = []
    for name in sorted(names_a):
        a, b = by_name_a[name], by_name_b[name]
        out.append(dict(name=name, li=a["li"], suffix=a["suffix"], shape=a["shape"],
                        dtype_a=a["dtype"], bytes_a=a["bytes"],
                        dtype_b=b["dtype"], bytes_b=b["bytes"],
                        changed=a["dtype"] != b["dtype"]))
    return out


def _print(*args, file=None):
    print(*args, file=file)


def report_single(path: str, full: bool, out):
    reader, rows = load_tensor_rows(path)
    kv = kv_pairs(reader)
    n_layers = n_layers_of(rows)
    total_bytes = sum(r["bytes"] for r in rows)
    total_numel = sum(r["numel"] for r in rows)
    on_disk = None
    try:
        import os
        on_disk = os.path.getsize(path)
    except OSError:
        pass

    _print(f"gguf file: {path}", file=out)
    _print(f"sha256: {sha256_file(path)[:16]}…", file=out)
    _print(f"general.architecture: {kv.get('general.architecture')}  "
          f"general.file_type: {kv.get('general.file_type')}  "
          f"general.quantization_version: {kv.get('general.quantization_version')}", file=out)
    _print(f"tensors: {len(rows)}  layers: {n_layers}  "
          f"total elements: {total_numel:,}  total bytes (tensors): {human_bytes(total_bytes)}"
          + (f"  file size: {human_bytes(on_disk)}" if on_disk is not None else ""), file=out)
    _print(file=out)

    _print("=" * 78, file=out)
    _print("KEY-VALUE PAIRS", file=out)
    _print("=" * 78, file=out)
    for k, v in kv.items():
        _print(f"  {k:<48}: {v}", file=out)
    _print(file=out)

    _print("-" * 78, file=out)
    _print("BY COMPONENT (summed across all layers; a family with mixed types", file=out)
    _print("across layers gets one row per (family, type) pair)", file=out)
    _print("-" * 78, file=out)
    _print(f"{'component':<20}{'dtype':<8}{'x N':>5}{'total elems':>16}{'bytes':>12}", file=out)
    seen = set()
    for suffix in FAMILY_INFO:
        for dtype in sorted({r["dtype"] for r in rows if r["suffix"] == suffix}):
            sub = [r for r in rows if r["suffix"] == suffix and r["dtype"] == dtype]
            if not sub:
                continue
            label = FAMILY_INFO[suffix][0]
            tot_n = sum(r["numel"] for r in sub)
            tot_b = sum(r["bytes"] for r in sub)
            _print(f"{label:<20}{dtype:<8}{len(sub):>5}{tot_n:>16,}{human_bytes(tot_b):>12}", file=out)
            seen.add(suffix)
    for name in sorted(NON_LAYER_NAMES):
        sub = [r for r in rows if r["name"] == name]
        for r in sub:
            _print(f"{name:<20}{r['dtype']:<8}{1:>5}{r['numel']:>16,}{human_bytes(r['bytes']):>12}", file=out)
    _print(file=out)

    _print("-" * 78, file=out)
    _print(f"PER-LAYER QUANTIZATION TYPE (0-{n_layers - 1})", file=out)
    _print("-" * 78, file=out)
    families = [s for s in FAMILY_INFO if any(r["suffix"] == s for r in rows)]
    hdr = "".join(f"{FAMILY_INFO[s][0][:10]:<12}" for s in families)
    _print(f"{'L':>3} {hdr}", file=out)
    for li in range(n_layers):
        cells = []
        for s in families:
            match = [r for r in rows if r["li"] == li and r["suffix"] == s]
            cells.append(match[0]["dtype"] if match else "-")
        _print(f"{li:>3} " + "".join(f"{c:<12}" for c in cells), file=out)
    _print(file=out)

    eff_bpw = total_bytes * 8 / total_numel if total_numel else 0.0
    f16_bytes = total_numel * 2
    _print("=" * 78, file=out)
    _print("SUMMARY", file=out)
    _print("=" * 78, file=out)
    _print(f"{'total weight elements':<44}{total_numel:>20,}", file=out)
    _print(f"{'total bytes (tensors)':<44}{human_bytes(total_bytes):>20}", file=out)
    _print(f"{'effective bits / weight (real, computed)':<44}{eff_bpw:>19.2f}b", file=out)
    _print(f"{'equivalent F16 size (same shapes)':<44}{human_bytes(f16_bytes):>20}", file=out)
    _print(f"{'this file is smaller than F16 by':<44}{f16_bytes / total_bytes:>19.3f}x", file=out)
    _print(file=out)

    _print("-" * 78, file=out)
    _print("BY GGML QUANT TYPE (whole file)", file=out)
    _print("-" * 78, file=out)
    _print(f"{'type':<10}{'tensors':>10}{'elements':>16}{'bytes':>14}{'% of size':>12}", file=out)
    for dt in sorted({r["dtype"] for r in rows}):
        sub = [r for r in rows if r["dtype"] == dt]
        b = sum(r["bytes"] for r in sub)
        _print(f"{dt:<10}{len(sub):>10}{sum(r['numel'] for r in sub):>16,}"
              f"{human_bytes(b):>14}{100 * b / total_bytes:>11.1f}%", file=out)

    if full:
        _print(file=out)
        _print("=" * 78, file=out)
        _print("FULL TENSOR DUMP", file=out)
        _print("=" * 78, file=out)
        _print(f"{'name':<28}{'dtype':<8}{'shape':<18}{'elements':>12}{'bytes':>10}", file=out)

        def sort_key(r):
            return (0 if r["li"] is None else 1, r["li"] or -1, r["name"])
        for r in sorted(rows, key=sort_key):
            _print(f"{r['name']:<28}{r['dtype']:<8}{str(r['shape']):<18}"
                  f"{r['numel']:>12,}{human_bytes(r['bytes']):>10}", file=out)


def report_compare(path_a: str, path_b: str, assert_known_asymmetry: bool, out):
    reader_a, rows_a = load_tensor_rows(path_a)
    reader_b, rows_b = load_tensor_rows(path_b)
    kv_a, kv_b = kv_pairs(reader_a), kv_pairs(reader_b)

    _print(f"A: {path_a}  (sha256 {sha256_file(path_a)[:16]}…)", file=out)
    _print(f"B: {path_b}  (sha256 {sha256_file(path_b)[:16]}…)", file=out)
    _print(file=out)

    _print("=" * 78, file=out)
    _print("KEY-VALUE PAIR DIFF", file=out)
    _print("=" * 78, file=out)
    keys = sorted(set(kv_a) | set(kv_b))
    for k in keys:
        va, vb = kv_a.get(k, "<missing>"), kv_b.get(k, "<missing>")
        if va == vb:
            _print(f"  {k:<48}: {va}", file=out)
        else:
            _print(f"  {k:<48}: A={va} | B={vb}", file=out)
    _print(file=out)

    diffs = diff_tensors(rows_a, rows_b)
    changed = [d for d in diffs if d["changed"]]
    types_seen = sorted({d["dtype_a"] for d in diffs} | {d["dtype_b"] for d in diffs})
    legend = "  ".join(f"{TYPE_SWATCH.get(t, '❓')} {t}" for t in types_seen)

    _print("=" * 78, file=out)
    _print(f"PER-TENSOR DIFF  (legend: {legend})", file=out)
    _print("=" * 78, file=out)
    _print(f"{'name':<26}{'shape':<16}{'A size':>10}{'A':>4}{'B size':>10}{'B':>4}"
          f"{'Δbytes':>12}  changed?", file=out)
    for d in diffs:
        swatch_a = TYPE_SWATCH.get(d["dtype_a"], "❓")
        swatch_b = TYPE_SWATCH.get(d["dtype_b"], "❓")
        delta = d["bytes_b"] - d["bytes_a"]
        mark = "  <-- CHANGED" if d["changed"] else ""
        _print(f"{d['name']:<26}{str(d['shape']):<16}{human_bytes(d['bytes_a']):>10}"
              f"{swatch_a:>4}{human_bytes(d['bytes_b']):>10}{swatch_b:>4}"
              f"{delta:>+12,}{mark}", file=out)
    _print(file=out)

    total_a = sum(d["bytes_a"] for d in diffs)
    total_b = sum(d["bytes_b"] for d in diffs)
    numel_a = sum(r["numel"] for r in rows_a)
    numel_b = sum(r["numel"] for r in rows_b)
    _print("-" * 78, file=out)
    _print("SUMMARY", file=out)
    _print("-" * 78, file=out)
    _print(f"{'changed tensors':<40}{len(changed):>10} / {len(diffs)}", file=out)
    _print(f"{'total size A':<40}{human_bytes(total_a):>20}", file=out)
    _print(f"{'total size B':<40}{human_bytes(total_b):>20}", file=out)
    _print(f"{'Δ size (B - A)':<40}{total_b - total_a:>+20,} bytes", file=out)
    _print(f"{'bits/weight A':<40}{total_a * 8 / numel_a:>19.2f}b", file=out)
    _print(f"{'bits/weight B':<40}{total_b * 8 / numel_b:>19.2f}b", file=out)

    if assert_known_asymmetry:
        offenders = [d["name"] for d in changed if d["suffix"] not in ASYMMETRY_ALLOWED_SUFFIXES]
        if offenders:
            _print(file=out)
            _print(f"ASSERTION FAILED: changed tensors outside the expected "
                  f"attn_v/ffn_down asymmetry: {offenders}", file=out)
            return 1
        _print(file=out)
        _print(f"ASSERTION PASSED: all {len(changed)} changed tensors are "
              f"attn_v.weight/ffn_down.weight, as expected for a Q4_K_S-vs-"
              f"Q4_K_M-style comparison.", file=out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="single-file report mode: path to a .gguf file")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare mode: two .gguf paths")
    ap.add_argument("--full", action="store_true",
                    help="single-file mode: also dump one row per raw tensor")
    ap.add_argument("--assert-known-asymmetry", action="store_true",
                    help="compare mode: exit 1 unless every changed tensor is "
                         "attn_v.weight or ffn_down.weight (the documented "
                         "Q4_K_M-vs-Q4_K_S promotion pattern)")
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args()
    if not args.gguf and not args.compare:
        ap.error("pass --gguf PATH or --compare A B")

    import sys as _sys
    outputs = [_sys.stdout]
    fh = None
    if args.out:
        fh = open(args.out, "w")
        outputs.append(fh)

    class _Tee:
        def write(self, s):
            for o in outputs:
                o.write(s)

        def flush(self):
            for o in outputs:
                o.flush()

    tee = _Tee()
    exit_code = 0
    if args.compare:
        exit_code = report_compare(args.compare[0], args.compare[1],
                                   args.assert_known_asymmetry, tee) or 0
    else:
        report_single(args.gguf, args.full, tee)
    if fh:
        fh.close()
    _sys.exit(exit_code)


if __name__ == "__main__":
    main()
