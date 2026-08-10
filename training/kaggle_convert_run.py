#!/usr/bin/env python3
"""Submit and retrieve the pinned merged-HF to Q4_K_M conversion on Kaggle."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from kaggle_run import authenticate, download, wait


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--username", default="oversightnode"); parser.add_argument("--training-kernel", default="oversightnode/kilobyte-train"); parser.add_argument("--slug", default="kilobyte-convert"); parser.add_argument("--out", type=Path, default=Path("output/conversion")); parser.add_argument("--no-wait", action="store_true"); args = parser.parse_args()
    api = authenticate(); ref = f"{args.username}/{args.slug}"
    with tempfile.TemporaryDirectory() as raw:
        staging = Path(raw); shutil.copy2(Path(__file__).with_name("convert_kaggle.py"), staging / "kilobyte-convert.py")
        (staging / "kernel-metadata.json").write_text(json.dumps({"id": ref, "title": "Kilobyte GGUF Convert", "code_file": "kilobyte-convert.py", "language": "python", "kernel_type": "script", "enable_gpu": False, "enable_internet": True, "dataset_sources": [], "model_sources": [], "competition_sources": [], "kernel_sources": [args.training_kernel]}, indent=2))
        api.kernels_push(str(staging))
    print(f"pushed conversion notebook {ref} from {args.training_kernel}")
    if args.no_wait: return 0
    state = wait(api, ref)
    if "complete" not in state.lower(): print(f"conversion did not complete cleanly: {state}"); return 1
    download(api, ref, args.out); return 0


if __name__ == "__main__": raise SystemExit(main())
