"""HF checkpoint -> F16 GGUF -> N quantized GGUF variants, all built from the
SAME immutable F16 baseline (never variant-from-variant) — the same
"always rebuild from the immutable source" discipline src/detllm/prepare.py
already applies to this repo's own int8 artifact.

Produces, under --out-dir (default llamacpp-quant-lab/artifacts/):
  qwen3-0.6b-hf/                 local HF checkpoint (save_pretrained)
  qwen3-0.6b-F16.gguf             conversion baseline
  qwen3-0.6b-{Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_K_S}.gguf   quantized from the F16 file
  manifest.json                   source/build provenance + per-file sha256

Run:
  uv sync --extra llamacpp-lab
  uv run python llamacpp-quant-lab/convert_quantize.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, "scripts")
from quant_report import human_bytes, sha256_file  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
LLAMACPP_DIR_DEFAULT = "llamacpp-quant-lab/vendor/llama.cpp"
OUT_DIR_DEFAULT = "llamacpp-quant-lab/artifacts"
BUILD_INFO_DEFAULT = "llamacpp-quant-lab/artifacts/llamacpp_build_info.json"
QUANT_TYPES_DEFAULT = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S"]


def ensure_hf_checkpoint(model_id: str, hf_dir: str, force: bool) -> str:
    if os.path.exists(os.path.join(hf_dir, "config.json")) and not force:
        print(f"HF checkpoint already at {hf_dir} — skip")
        return hf_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_id!r} from Hugging Face...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir)
    tok.save_pretrained(hf_dir)
    print(f"wrote {hf_dir}")
    return hf_dir


def source_revision(model_id: str) -> str:
    if os.path.isdir(model_id):
        return "local-checkpoint"
    try:
        from huggingface_hub import model_info
        return model_info(model_id).sha or "unknown"
    except Exception:
        return "unknown"


def convert_to_f16(llamacpp_dir: str, hf_dir: str, out_path: str, force: bool):
    if os.path.exists(out_path) and not force:
        print(f"{out_path} already exists — skip conversion")
        return
    converter = os.path.join(llamacpp_dir, "convert_hf_to_gguf.py")
    cmd = [sys.executable, converter, hf_dir, "--outtype", "f16", "--outfile", out_path]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def quantize(llamacpp_dir: str, f16_path: str, out_path: str, quant_type: str, force: bool):
    if os.path.exists(out_path) and not force:
        print(f"{out_path} already exists — skip quantize")
        return
    quantize_bin = os.path.join(llamacpp_dir, "build", "bin", "llama-quantize")
    cmd = [quantize_bin, f16_path, out_path, quant_type]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_info(build_info_path: str) -> dict:
    if os.path.exists(build_info_path):
        with open(build_info_path) as f:
            return json.load(f)
    return {"llamacpp_commit": "unknown", "cmake_flags": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID,
                    help="HF hub id or local checkpoint dir to convert "
                         "(e.g. an abliteration/ output directory)")
    ap.add_argument("--llamacpp-dir", default=LLAMACPP_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--quant-types", nargs="+", default=QUANT_TYPES_DEFAULT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hf_dir = os.path.join(args.out_dir, "qwen3-0.6b-hf")
    ensure_hf_checkpoint(args.model_id, hf_dir, args.force_download)
    revision = source_revision(args.model_id)

    f16_path = os.path.join(args.out_dir, "qwen3-0.6b-F16.gguf")
    convert_to_f16(args.llamacpp_dir, hf_dir, f16_path, args.force)

    variants = {}
    for qtype in args.quant_types:
        out_path = os.path.join(args.out_dir, f"qwen3-0.6b-{qtype}.gguf")
        if qtype == "F16":
            pass  # already produced by conversion
        else:
            quantize(args.llamacpp_dir, f16_path, out_path, qtype, args.force)
        variants[qtype] = {
            "file": os.path.basename(out_path),
            "sha256": sha256_file(out_path),
            "bytes": os.path.getsize(out_path),
        }

    info = build_info(os.path.join(args.out_dir, "llamacpp_build_info.json"))
    manifest = {
        "source_model_id": args.model_id,
        "source_revision": revision,
        "llamacpp_commit": info.get("llamacpp_commit", "unknown"),
        "llamacpp_build_flags": info.get("cmake_flags", []),
        "conversion": {"outtype": "f16", "converter": "convert_hf_to_gguf.py"},
        "quant_variants": variants,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print()
    print(f"{'variant':<10}{'file':<28}{'size':>12}{'sha256':<20}")
    for qtype, v in variants.items():
        print(f"{qtype:<10}{v['file']:<28}{human_bytes(v['bytes']):>12}  {v['sha256'][:16]}…")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
