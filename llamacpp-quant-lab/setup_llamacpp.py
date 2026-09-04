"""Clone + build llama.cpp with Metal support (Apple Silicon GPU backend),
for the conversion/quantization/serving steps the rest of this folder's
scripts depend on.

Deliberately idempotent (skips a rebuild if the binaries already exist) and
NOT a content-addressed manifest system — this is a build tool, not a data
artifact, so "skip if already built, --force to redo" is enough (see
llamacpp-quant-lab/PLAN.md for why this is intentionally lighter-weight than
abliteration/'s hash-guarded apply step).

A prebuilt `brew install llama.cpp` also exists and is faster, but ships only
the compiled binaries (llama-cli/llama-server/llama-quantize) — no
convert_hf_to_gguf.py, no gguf-py — so it cannot do this lab's conversion
step (llamacpp-quant-lab/convert_quantize.py). This script is the "complete
control" path: clone, build locally, know exactly what commit/flags were
used (recorded below), quantize from it directly.

Run:
  uv run python llamacpp-quant-lab/setup_llamacpp.py
  uv run python llamacpp-quant-lab/setup_llamacpp.py --ref b4523 --force
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys

LLAMACPP_DIR_DEFAULT = "llamacpp-quant-lab/vendor/llama.cpp"
BUILD_INFO_DEFAULT = "llamacpp-quant-lab/artifacts/llamacpp_build_info.json"
REPO_URL = "https://github.com/ggml-org/llama.cpp.git"


def _run(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def preflight():
    if subprocess.run(["xcode-select", "-p"], capture_output=True).returncode != 0:
        print("Xcode Command Line Tools not found. Run: xcode-select --install")
        sys.exit(1)
    cmake = shutil.which("cmake")
    if not cmake:
        print("cmake not found. Run: brew install cmake")
        sys.exit(1)
    out = subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout
    version = out.splitlines()[0].split()[-1]
    major, minor = (int(x) for x in version.split(".")[:2])
    if (major, minor) < (3, 14):
        print(f"cmake {version} is too old (need >= 3.14). Run: brew install cmake")
        sys.exit(1)
    return version


def already_built(llamacpp_dir: str) -> bool:
    build_bin = os.path.join(llamacpp_dir, "build", "bin")
    needed = ["llama-cli", "llama-quantize", "llama-server"]
    return all(os.path.exists(os.path.join(build_bin, b)) for b in needed) and \
        os.path.exists(os.path.join(llamacpp_dir, "convert_hf_to_gguf.py"))


def setup(llamacpp_dir: str, ref: str | None, force: bool, build_info_out: str):
    cmake_version = preflight()

    if already_built(llamacpp_dir) and not force:
        print(f"already built at {llamacpp_dir} — skip (--force to rebuild)")
        return

    if not os.path.isdir(llamacpp_dir):
        os.makedirs(os.path.dirname(llamacpp_dir) or ".", exist_ok=True)
        _run(["git", "clone", REPO_URL, llamacpp_dir])
    if ref:
        _run(["git", "-C", llamacpp_dir, "checkout", ref])

    ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True,
                          text=True).stdout.strip() or "4"
    build_dir = os.path.join(llamacpp_dir, "build")
    cmake_flags = ["-DGGML_METAL=ON", "-DCMAKE_BUILD_TYPE=Release"]
    _run(["cmake", "-B", build_dir, "-S", llamacpp_dir, *cmake_flags])
    _run(["cmake", "--build", build_dir, "--config", "Release", "-j", ncpu])

    if not already_built(llamacpp_dir):
        print("build finished but expected binaries/scripts are missing — "
             "check the build log above")
        sys.exit(1)

    commit = subprocess.run(["git", "-C", llamacpp_dir, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    info = {
        "llamacpp_commit": commit,
        "llamacpp_ref_requested": ref or "default-branch-HEAD",
        "cmake_flags": cmake_flags,
        "cmake_version": cmake_version,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {"platform": sys.platform, "machine": platform.machine(),
                "processor": platform.processor() or platform.machine()},
    }
    os.makedirs(os.path.dirname(build_info_out) or ".", exist_ok=True)
    with open(build_info_out, "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {build_info_out} (llama.cpp commit {commit[:12]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llamacpp-dir", default=LLAMACPP_DIR_DEFAULT)
    ap.add_argument("--ref", default=None, help="git ref to check out (default: leave "
                    "on the clone's default branch HEAD)")
    ap.add_argument("--force", action="store_true", help="rebuild even if already built")
    ap.add_argument("--build-info-out", default=BUILD_INFO_DEFAULT)
    args = ap.parse_args()
    setup(args.llamacpp_dir, args.ref, args.force, args.build_info_out)


if __name__ == "__main__":
    main()
