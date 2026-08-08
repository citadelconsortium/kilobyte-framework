#!/usr/bin/env python3
"""Drive a Kilobyte training run on Kaggle via the official Kaggle API.

Uploads the dataset, pushes the training notebook with GPU enabled, starts it, polls for
completion and downloads the output. Uses the current Kaggle Python API; authentication is
read the supported way (KAGGLE_USERNAME/KAGGLE_KEY env or ~/.kaggle/kaggle.json) and the
key is never printed, logged, committed, or written into any artifact.

    export KAGGLE_USERNAME=... KAGGLE_KEY=...
    python kaggle_run.py --config config.json

The notebook itself does the GPU work; this script only orchestrates. It verifies
authentication before submitting anything, so a bad credential fails fast and clearly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("kaggle", {}).get("username", "").startswith("PASTE_"):
        raise SystemExit("set kaggle.username in the config before running")
    return config


def authenticate():
    """Return an authenticated Kaggle API client, or exit with a clear message.

    Never prints credential contents; on failure it reports that authentication failed and
    which mechanisms were checked, not the values.
    """
    try:
        from kaggle import KaggleApi
    except ImportError as exc:
        raise SystemExit("the kaggle package is required: pip install kaggle") from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise SystemExit(
            "Kaggle authentication failed. Provide credentials via KAGGLE_API_TOKEN (for a "
            "KGAT_ token), or KAGGLE_USERNAME and KAGGLE_KEY, or ~/.kaggle/kaggle.json (mode "
            "600). The token itself is never read from or written to this repository."
        ) from exc
    return api


def push_dataset(api, config: dict, data_dir: Path) -> str:
    username = config["kaggle"]["username"]
    slug = config["kaggle"]["dataset_slug"]
    ref = f"{username}/{slug}"
    with tempfile.TemporaryDirectory() as raw:
        staging = Path(raw)
        for name in ("kilobyte-sft.jsonl", "kilobyte-sft.val.jsonl"):
            src = data_dir / name
            if src.is_file():
                shutil.copy2(src, staging / name)
        # Bundle the config so the notebook finds it at /kaggle/input/<slug>/config.json.
        shutil.copy2(config["_config_path"], staging / "config.json")
        metadata = {
            "title": "Kilobyte SFT",
            "id": ref,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
        try:
            api.dataset_create_version(str(staging), version_notes=f"kilobyte {int(time.time())}", dir_mode="zip")
            print(f"updated dataset {ref}")
        except Exception:
            api.dataset_create_new(str(staging), dir_mode="zip")
            print(f"created dataset {ref}")
    return ref


def push_notebook(api, config: dict, dataset_ref: str) -> str:
    username = config["kaggle"]["username"]
    slug = config["kaggle"]["notebook_slug"]
    ref = f"{username}/{slug}"
    here = Path(__file__).parent
    with tempfile.TemporaryDirectory() as raw:
        staging = Path(raw)
        # Push the training script itself as the kernel source; no launcher wrapper.
        shutil.copy2(here / "kaggle_notebook.py", staging / "kilobyte-train.py")
        metadata = {
            "id": ref,
            "title": "Kilobyte Train",
            "code_file": "kilobyte-train.py",
            "language": "python",
            "kernel_type": "script",
            "enable_gpu": bool(config["kaggle"].get("enable_gpu", True)),
            "enable_internet": bool(config["kaggle"].get("enable_internet", True)),
            "dataset_sources": [dataset_ref],
            "model_sources": config["kaggle"].get("model_sources", []),
            "competition_sources": [],
            "kernel_sources": [],
        }
        (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        api.kernels_push(str(staging))
        print(f"pushed notebook {ref} (GPU={metadata['enable_gpu']})")
    return ref


def wait(api, notebook_ref: str, poll: int = 60, timeout: int = 6 * 3600) -> str:
    """Poll until the run completes; return its final status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = api.kernels_status(notebook_ref)
        state = str(getattr(status, "status", status))
        print(f"status: {state}")
        if any(word in state.lower() for word in ("complete", "error", "cancel")):
            return state
        time.sleep(poll)
    return "timeout"


def download(api, notebook_ref: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output(notebook_ref, path=str(out_dir))
    print(f"downloaded output to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kilobyte training on Kaggle")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--no-wait", action="store_true", help="submit and exit without polling")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    if not (args.data / "kilobyte-sft.jsonl").is_file():
        raise SystemExit(f"no dataset at {args.data}/kilobyte-sft.jsonl — run build_dataset.py first")

    api = authenticate()
    print("Kaggle authentication verified")
    dataset_ref = push_dataset(api, config, args.data)
    notebook_ref = push_notebook(api, config, dataset_ref)
    if args.no_wait:
        print("submitted; not waiting. Check status on kaggle.com or rerun with polling.")
        return 0
    state = wait(api, notebook_ref)
    if "complete" not in state.lower():
        print(f"run did not complete cleanly: {state}", file=sys.stderr)
        return 1
    download(api, notebook_ref, args.out)
    print("done. Evaluate the candidate with evaluate.py before promoting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
