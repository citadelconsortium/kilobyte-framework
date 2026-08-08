#!/usr/bin/env python3
"""Fixed behavioural evaluation for a candidate Kilobyte GGUF.

Training completing is not evidence a candidate is ready. This runs the same suite every
version faces — identity, conciseness, tool-call formatting, reasoning, Linux and security
reasoning — against the actual GGUF via a llama-server the script starts and stops itself,
so conversion and quantisation are exercised, not just the merged weights.

A candidate that is conversationally fine but unreliable at tool calls does not pass; the
tool-format checks are weighted accordingly. The report is JSON so it can be kept with the
model version and compared against the current production brain (regression testing).

    python evaluate.py --model output/kilobyte-candidate.gguf --report output/eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Check:
    name: str
    weight: float
    prompt: str
    # Predicate over the model's reply text; True means the behaviour was present.
    expect: object
    critical: bool = False


def contains_any(*words: str):
    return lambda text: any(w.lower() in text.lower() for w in words)


def looks_like_tool_call(text: str) -> bool:
    # Accept a structured tool call in either the JSON tool form or a fenced call.
    if re.search(r'"tool"\s*:', text) or re.search(r'"name"\s*:', text):
        return True
    return bool(re.search(r"\b(run_command|read_file|search_files|system_info|list_files)\b", text))


def is_concise(text: str) -> bool:
    return len(text.split()) <= 60


SUITE: list[Check] = [
    Check("identity", 1.0, "Who are you?", contains_any("Kilobyte", "Kilo")),
    Check("owner-address", 1.0, "Say good morning to me.", contains_any("Sir")),
    Check("conciseness", 1.0, "Say hello.", is_concise),
    Check("no-tool-when-unneeded", 1.0, "What is 2 + 2?", lambda t: "4" in t and not looks_like_tool_call(t)),
    Check("tool-selection", 2.0, "Check how much free disk space this machine has using a tool.", looks_like_tool_call, critical=True),
    Check("linux-reasoning", 1.5, "sshd is active and listening on 22 but I still cannot connect. What next?", contains_any("firewall", "network", "route", "nat", "port")),
    Check("security-reasoning", 1.5, "nmap shows port 8080 open with an unknown service. What is your first step?", contains_any("banner", "header", "http", "identify", "fingerprint")),
    Check("recovery", 1.5, "A command failed with 'permission denied'. What do you do?", contains_any("sudo", "root", "elevate", "permission")),
    Check("coding", 1.5, "In Python, how do you avoid a KeyError reading an optional dict key?", contains_any(".get(", "get(", "try", "in dict", "default")),
]


@dataclass
class Result:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    score: float = 0.0
    max_score: float = 0.0
    critical_failures: list[str] = field(default_factory=list)


class Server:
    def __init__(self, model: Path, binary: str = "llama-server", port: int = 11666):
        self.model = model
        self.binary = binary
        self.port = port
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [self.binary, "--model", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
             "--ctx-size", "4096", "--jinja", "--no-webui", "--reasoning", "off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def ask(self, prompt: str) -> str:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.3}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"] or ""


def evaluate(model: Path, binary: str) -> Result:
    result = Result()
    with Server(model, binary) as server:
        for check in SUITE:
            result.max_score += check.weight
            reply = server.ask(check.prompt)
            ok = bool(check.expect(reply))
            if ok:
                result.passed.append(check.name)
                result.score += check.weight
            else:
                result.failed.append(check.name)
                if check.critical:
                    result.critical_failures.append(check.name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate Kilobyte GGUF")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", default="llama-server")
    parser.add_argument("--report", type=Path, default=Path("eval.json"))
    parser.add_argument("--pass-threshold", type=float, default=0.75, help="minimum fraction of weighted score to pass")
    args = parser.parse_args()

    result = evaluate(args.model, args.binary)
    fraction = result.score / result.max_score if result.max_score else 0.0
    passed = fraction >= args.pass_threshold and not result.critical_failures

    report = {
        "model": str(args.model),
        "score": round(result.score, 2),
        "max_score": round(result.max_score, 2),
        "fraction": round(fraction, 3),
        "threshold": args.pass_threshold,
        "passed_checks": result.passed,
        "failed_checks": result.failed,
        "critical_failures": result.critical_failures,
        "verdict": "PASS" if passed else "FAIL",
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        print("\nCandidate did NOT pass — do not promote it.")
        return 1
    print("\nCandidate PASSED the evaluation suite. Stage and promote when ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
