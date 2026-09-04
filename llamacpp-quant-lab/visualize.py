"""matplotlib/plotly visualizations of GGUF per-layer quantization, built
directly from the real produced .gguf files (never estimated/synthetic
data) via gguf_report.py's tensor-loading/classification functions.

Outputs (committed PNGs under --out-dir, e.g. llamacpp-quant-lab/report/charts/):
  heatmap_type_<variant>.png    28 blocks x 7 families, colored by quant type
  heatmap_bpw_<variant>.png     same grid, colored by real bits/weight per cell
  size_by_variant.png           one bar per --gguf-dir variant, real file size
  family_stacked_compare_<A>_vs_<B>.png   (--compare mode only)

Interactive HTML (gitignored, local-only — GitHub can't render live plotly):
  llamacpp-quant-lab/report/interactive/heatmap_<variant>.html

Run:
  uv run python llamacpp-quant-lab/visualize.py --gguf-dir llamacpp-quant-lab/artifacts \\
      --out-dir llamacpp-quant-lab/report/charts
  uv run python llamacpp-quant-lab/visualize.py \\
      --compare llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_S.gguf \\
                llamacpp-quant-lab/artifacts/qwen3-0.6b-Q4_K_M.gguf \\
      --out-dir llamacpp-quant-lab/report/charts
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from gguf_report import FAMILY_INFO, load_tensor_rows, n_layers_of

QUANT_TYPE_COLORS = {
    "F32": "#9e9e9e", "F16": "#7fb3ff", "Q8_0": "#c39bd3",
    "Q4_K": "#ff6b6b", "Q5_K": "#ffa94d", "Q6_K": "#51cf66",
    "Q2_K": "#a0785a", "Q3_K": "#a0785a",
}


def _grid_for(rows, n_layers):
    families = [s for s in FAMILY_INFO if any(r["suffix"] == s for r in rows)]
    grid_type, grid_bpw, hover = [], [], []
    for li in range(n_layers):
        row_type, row_bpw, row_hover = [], [], []
        for s in families:
            match = [r for r in rows if r["li"] == li and r["suffix"] == s]
            if match:
                r = match[0]
                row_type.append(r["dtype"])
                row_bpw.append(r["bytes"] * 8 / r["numel"] if r["numel"] else 0)
                row_hover.append(f"{r['name']}<br>shape={r['shape']}<br>"
                                 f"{r['dtype']}, {r['bytes']} bytes")
            else:
                row_type.append(None)
                row_bpw.append(None)
                row_hover.append("")
        grid_type.append(row_type)
        grid_bpw.append(row_bpw)
        hover.append(row_hover)
    return families, grid_type, grid_bpw, hover


def heatmap_type_png(rows, n_layers, variant: str, out_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    families, grid_type, _, _ = _grid_for(rows, n_layers)
    types_present = sorted({t for row in grid_type for t in row if t})
    type_idx = {t: i for i, t in enumerate(types_present)}
    arr = np.full((n_layers, len(families)), -1, dtype=int)
    for li, row in enumerate(grid_type):
        for j, t in enumerate(row):
            if t:
                arr[li, j] = type_idx[t]
    colors = [QUANT_TYPE_COLORS.get(t, "#888888") for t in types_present]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff"] + colors)

    fig, ax = plt.subplots(figsize=(1.2 * len(families) + 2, 0.28 * n_layers + 1.5))
    ax.imshow(arr + 1, cmap=cmap, aspect="auto", vmin=0, vmax=len(colors))
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([FAMILY_INFO[s][0] for s in families], rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_ylabel("transformer block index")
    ax.set_title(f"quantization type per block/family — {variant}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, types_present, loc="upper left", bbox_to_anchor=(1.02, 1),
             title="GGML type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap_bpw_png(rows, n_layers, variant: str, out_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    families, _, grid_bpw, _ = _grid_for(rows, n_layers)
    arr = np.array([[v if v is not None else np.nan for v in row] for row in grid_bpw])
    fig, ax = plt.subplots(figsize=(1.2 * len(families) + 2, 0.28 * n_layers + 1.5))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([FAMILY_INFO[s][0] for s in families], rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_ylabel("transformer block index")
    ax.set_title(f"real bits/weight per block/family — {variant}")
    fig.colorbar(im, ax=ax, label="bits/weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap_html(rows, n_layers, variant: str, out_path: str):
    import plotly.graph_objects as go

    families, grid_type, grid_bpw, hover = _grid_for(rows, n_layers)
    types_present = sorted({t for row in grid_type for t in row if t})
    type_idx = {t: i for i, t in enumerate(types_present)}
    z = [[type_idx.get(t, -1) for t in row] for row in grid_type]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[FAMILY_INFO[s][0] for s in families], y=list(range(n_layers)),
        text=hover, hoverinfo="text",
        colorscale=[[i / max(1, len(types_present) - 1),
                    QUANT_TYPE_COLORS.get(t, "#888888")] for i, t in enumerate(types_present)],
    ))
    fig.update_layout(title=f"quantization type per block/family — {variant}",
                      yaxis_title="transformer block index")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path)


def size_by_variant_png(manifest_path: str, out_path: str):
    import matplotlib.pyplot as plt

    with open(manifest_path) as f:
        manifest = json.load(f)
    variants = manifest["quant_variants"]
    names = list(variants)
    sizes_mb = [variants[n]["bytes"] / (1024 * 1024) for n in names]
    colors = ["#333333" for _ in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, sizes_mb, color=colors)
    ax.set_ylabel("size (MB)")
    ax.set_title("qwen3-0.6b GGUF file size by quantization variant")
    for i, v in enumerate(sizes_mb):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def family_stacked_compare_png(rows_a, rows_b, name_a, name_b, out_path: str):
    import matplotlib.pyplot as plt

    families = list(FAMILY_INFO)
    families = [s for s in families if any(r["suffix"] == s for r in rows_a)]
    totals_a = [sum(r["bytes"] for r in rows_a if r["suffix"] == s) / (1024 * 1024)
               for s in families]
    totals_b = [sum(r["bytes"] for r in rows_b if r["suffix"] == s) / (1024 * 1024)
               for s in families]
    fig, ax = plt.subplots(figsize=(6, 5))
    bottom_a = bottom_b = 0
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(families):
        ax.bar([name_a], [totals_a[i]], bottom=bottom_a, color=cmap(i % 10),
              label=FAMILY_INFO[s][0])
        ax.bar([name_b], [totals_b[i]], bottom=bottom_b, color=cmap(i % 10))
        bottom_a += totals_a[i]
        bottom_b += totals_b[i]
    ax.set_ylabel("size (MB)")
    ax.set_title(f"per-tensor-family size: {name_a} vs {name_b}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf-dir", help="dir containing manifest.json + .gguf files")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    interactive_dir = "llamacpp-quant-lab/report/interactive"

    if args.gguf_dir:
        manifest_path = os.path.join(args.gguf_dir, "manifest.json")
        for path in sorted(glob.glob(os.path.join(args.gguf_dir, "*.gguf"))):
            variant = os.path.basename(path).replace("qwen3-0.6b-", "").replace(".gguf", "")
            reader, rows = load_tensor_rows(path)
            n_layers = n_layers_of(rows)
            if n_layers == 0:
                continue
            types_present = {r["dtype"] for r in rows if r["li"] is not None}
            if len(types_present) > 1:
                heatmap_type_png(rows, n_layers, variant,
                                os.path.join(args.out_dir, f"heatmap_type_{variant}.png"))
                heatmap_html(rows, n_layers, variant,
                            os.path.join(interactive_dir, f"heatmap_{variant}.html"))
            heatmap_bpw_png(rows, n_layers, variant,
                           os.path.join(args.out_dir, f"heatmap_bpw_{variant}.png"))
            print(f"visualized {variant} ({n_layers} layers, types={sorted(types_present)})")
        if os.path.exists(manifest_path):
            size_by_variant_png(manifest_path, os.path.join(args.out_dir, "size_by_variant.png"))
            print("wrote size_by_variant.png")

    if args.compare:
        path_a, path_b = args.compare
        _, rows_a = load_tensor_rows(path_a)
        _, rows_b = load_tensor_rows(path_b)
        name_a = os.path.basename(path_a).replace("qwen3-0.6b-", "").replace(".gguf", "")
        name_b = os.path.basename(path_b).replace("qwen3-0.6b-", "").replace(".gguf", "")
        out = os.path.join(args.out_dir, f"family_stacked_compare_{name_a}_vs_{name_b}.png")
        family_stacked_compare_png(rows_a, rows_b, name_a, name_b, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
