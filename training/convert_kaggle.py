#!/usr/bin/env python3
"""Kaggle conversion notebook (ONLINE): merged HF weights -> quantised GGUF, fast.

With internet available, this avoids the slow from-source compile: it clones llama.cpp for
the pure-Python converter and downloads llama.cpp's prebuilt Linux release binary for the
quantiser. Falls back to a source build only if the prebuilt binary cannot run.

Kernel metadata:
    kernel_sources : ["oversightnode/kilobyte-train"]   (the merged weights)
    enable_internet: true
    enable_gpu     : false
Output:
    /kaggle/working/kilobyte.gguf     Q4_K_M brain, ready to download
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import urllib.request

WORK = "/kaggle/working"
LLAMA = f"{WORK}/llama.cpp"
F16 = f"{WORK}/kilobyte-f16.gguf"
OUT = f"{WORK}/kilobyte.gguf"


def run(cmd: str, **kw) -> int:
    print("+", cmd, flush=True)
    return subprocess.call(cmd, shell=True, **kw)


def find_merged() -> str:
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if any(os.path.getsize(w) > 200 * 1024 * 1024
               for w in glob.glob(os.path.join(d, "*.safetensors")) + glob.glob(os.path.join(d, "*.bin"))):
            return d
    for cfg in glob.glob("/kaggle/input/**/merged/config.json", recursive=True):
        return os.path.dirname(cfg)
    raise SystemExit("merged weights not found under /kaggle/input")


def prebuilt_quantize(f16: str, out: str) -> bool:
    """Download the latest llama.cpp Ubuntu release binaries and quantise with them."""
    try:
        rel = json.load(urllib.request.urlopen(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", timeout=30))
        asset = next(a for a in rel["assets"]
                     if "ubuntu-x64" in a["name"] and a["name"].endswith(".zip"))
        print("prebuilt:", asset["name"], flush=True)
        run(f"curl -sSL '{asset['browser_download_url']}' -o {WORK}/llbin.zip")
        run(f"cd {WORK} && unzip -oq llbin.zip -d llbin")
    except Exception as exc:
        print("prebuilt fetch failed:", repr(exc)[:200], flush=True)
        return False
    bins = glob.glob(f"{WORK}/llbin/**/llama-quantize", recursive=True)
    if not bins:
        return False
    q = bins[0]
    run(f"chmod +x {q}")
    # the release ships shared libs alongside; point the loader at them
    libdir = os.path.dirname(q)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
    return run(f"{q} {f16} {out} Q4_K_M", env=env) == 0 and os.path.exists(out)


def source_quantize(f16: str, out: str) -> bool:
    if run(f"cmake -S {LLAMA} -B {LLAMA}/build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF "
           f"-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF") != 0:
        return False
    if run(f"cmake --build {LLAMA}/build --target llama-quantize -j4") != 0:
        return False
    for cand in (f"{LLAMA}/build/bin/llama-quantize", f"{LLAMA}/build/llama-quantize"):
        if os.path.exists(cand):
            return run(f"{cand} {f16} {out} Q4_K_M") == 0 and os.path.exists(out)
    return False


def main() -> int:
    merged = find_merged()
    print("merged weights at:", merged, flush=True)
    for w in glob.glob(os.path.join(merged, "*.safetensors")):
        print(f"  weight {os.path.getsize(w) / 1e9:.2f} GB  {os.path.basename(w)}", flush=True)

    run(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
    run("pip install -q gguf sentencepiece protobuf")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{LLAMA}/gguf-py:" + env.get("PYTHONPATH", "")
    if run(f"python {LLAMA}/convert_hf_to_gguf.py {merged} --outfile {F16} --outtype f16", env=env) != 0 \
            or not os.path.exists(F16):
        raise SystemExit("convert_hf_to_gguf.py failed")
    print(f"f16 GGUF: {os.path.getsize(F16) / 1e9:.2f} GB", flush=True)

    if prebuilt_quantize(F16, OUT) or source_quantize(F16, OUT):
        os.remove(F16)
        print(f"GGUF READY (Q4_K_M): {os.path.getsize(OUT) / 1e9:.2f} GB", flush=True)
    else:
        print("quantiser failed — leaving f16 GGUF for downstream quantisation", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
