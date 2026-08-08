#!/usr/bin/env python3
"""Validate, deduplicate and split the Kilobyte SFT dataset — all CPU-side.

Run this before any Kaggle GPU session so the training environment receives data that is
already correct: Kaggle's free GPU time is for training, not for catching a malformed
example. Every conversation is checked against dataset_spec.md; anything invalid is
reported with its id and reason and excluded rather than silently shipped.

    python build_dataset.py --seed seed --out data/kilobyte-sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DOMAINS = {"coding", "tools", "linux", "security", "planning", "recovery", "general", "persona"}
KNOWN_TOOLS = {
    "read_file", "write_file", "list_files", "search_files", "run_command", "system_info",
    "web_search", "web_fetch", "remember", "recall", "save_skill", "list_skills",
}


def _tool_ok(name: str) -> bool:
    return name in KNOWN_TOOLS or name.startswith("mcp__")


def validate(conv: dict[str, Any]) -> list[str]:
    """Return a list of problems with one conversation; empty means valid."""
    problems: list[str] = []
    if conv.get("domain") not in DOMAINS:
        problems.append(f"domain must be one of {sorted(DOMAINS)}")
    messages = conv.get("messages")
    if not isinstance(messages, list) or not messages:
        return problems + ["messages must be a non-empty list"]

    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system or non_system[0].get("role") != "user":
        problems.append("conversation must start with a user turn (after an optional system turn)")

    last_tool_call: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            problems.append(f"message {index}: unknown role {role!r}")
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                name = call.get("name")
                if not _tool_ok(str(name)):
                    problems.append(f"message {index}: unknown tool {name!r}")
                if not isinstance(call.get("arguments"), dict):
                    problems.append(f"message {index}: tool arguments must be an object")
                last_tool_call = str(name)
        elif role == "tool":
            if last_tool_call is None:
                problems.append(f"message {index}: tool result without a preceding tool call")
            if message.get("name") and message["name"] != last_tool_call:
                problems.append(f"message {index}: tool result name does not match the last call")
            last_tool_call = None
    if not any(m.get("role") == "assistant" for m in messages):
        problems.append("no assistant turn")
    return problems


def _fingerprint(conv: dict[str, Any]) -> str:
    text = "\n".join(f"{m.get('role')}:{m.get('content','')}" for m in conv.get("messages", []))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(paths: list[Path]) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
    return conversations


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the Kilobyte SFT dataset")
    parser.add_argument("--seed", type=Path, default=Path(__file__).parent / "seed", help="directory of seed .jsonl files")
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="additional .jsonl files")
    parser.add_argument("--out", type=Path, default=Path("data/kilobyte-sft.jsonl"))
    parser.add_argument("--val-out", type=Path, default=None, help="defaults to <out>.val")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    sources = sorted(args.seed.glob("*.jsonl")) if args.seed.is_dir() else []
    sources += list(args.extra)
    if not sources:
        print("no input files found", file=sys.stderr)
        return 1

    conversations = load(sources)
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected = duplicates = 0
    for conv in conversations:
        problems = validate(conv)
        if problems:
            rejected += 1
            print(f"reject {conv.get('id', '?')}: {problems[0]}", file=sys.stderr)
            continue
        fingerprint = _fingerprint(conv)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        valid.append(conv)

    if not valid:
        print("no valid conversations after validation", file=sys.stderr)
        return 1

    # Stratify the validation split by domain so every domain is represented in eval.
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for conv in valid:
        by_domain.setdefault(conv["domain"], []).append(conv)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for group in by_domain.values():
        cut = max(1, int(len(group) * args.val_fraction)) if len(group) > 1 else 0
        val.extend(group[:cut])
        train.extend(group[cut:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in train), encoding="utf-8")
    val_out = args.val_out or args.out.with_suffix(".val.jsonl")
    val_out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in val), encoding="utf-8")

    mix = Counter(c["domain"] for c in valid)
    total = len(valid)
    print(f"valid {total}  train {len(train)}  val {len(val)}  rejected {rejected}  duplicates {duplicates}")
    print("domain mix:")
    for domain in sorted(mix):
        print(f"  {domain:<10} {mix[domain]:>5}  {mix[domain] / total * 100:4.1f}%")
    print(f"wrote {args.out} and {val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
