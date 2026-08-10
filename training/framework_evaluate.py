#!/usr/bin/env python3
"""End-to-end acceptance suite for a candidate brain through Kilo's real RPC agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Exchange:
    prompt: str
    text: str = ""
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    session_id: str | None = None


async def exchange(socket: Path, prompt: str, cwd: Path, session_id: str | None = None) -> Exchange:
    reader, writer = await asyncio.open_unix_connection(socket)
    writer.write((json.dumps({"command": "chat", "text": prompt, "session_id": session_id, "cwd": str(cwd), "fresh": session_id is None}) + "\n").encode())
    await writer.drain()
    result = Exchange(prompt=prompt, session_id=session_id)
    try:
        while raw := await asyncio.wait_for(reader.readline(), timeout=1800):
            event = json.loads(raw); kind = event.get("type")
            if kind == "session": result.session_id = event.get("session_id")
            elif kind == "token": result.text += event.get("text", "")
            elif kind == "response_reset": result.text = ""
            elif kind == "tool_start": result.tools.append(str(event.get("name")))
            elif kind == "tool_end" and not event.get("ok"): result.errors.append(str(event.get("summary") or "tool failed"))
            elif kind == "permission":
                writer.write((json.dumps({"type": "permission_response", "id": event["id"], "allow": True}) + "\n").encode()); await writer.drain()
            elif kind == "error": result.errors.append(str(event.get("error")))
            elif kind == "done": break
    finally:
        writer.close(); await writer.wait_closed()
    return result


def require(report: dict, name: str, condition: bool, detail: str) -> None:
    report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
    if not condition: report["failures"].append(name)


async def run(socket: Path) -> dict:
    report: dict = {"socket": str(socket), "checks": [], "failures": [], "exchanges": []}
    with tempfile.TemporaryDirectory(prefix="kilo-framework-eval-") as raw:
        cwd = Path(raw)
        async def ask(prompt: str, sid: str | None = None) -> Exchange:
            item = await exchange(socket, prompt, cwd, sid)
            report["exchanges"].append({"prompt": prompt, "tools": item.tools, "errors": item.errors, "text": item.text[:1000]})
            return item
        plain = await ask("Reply with exactly KILO_READY and do not use a tool."); require(report, "plain-directive", "KILO_READY" in plain.text and not plain.tools, repr(plain.text))
        system = await ask("Use the system_info tool and report this machine's total memory."); require(report, "system-info", "system_info" in system.tools and bool(system.text.strip()), str(system.tools))
        write = await ask("Create framework-check.txt containing exactly framework-pass using write_file, then confirm it."); created = cwd / "framework-check.txt"; require(report, "write-file", "write_file" in write.tools and created.is_file() and created.read_text().strip() == "framework-pass", str(write.tools))
        read = await ask("Read framework-check.txt with read_file and tell me its exact contents.", write.session_id); require(report, "read-follow-through", "read_file" in read.tools and "framework-pass" in read.text, str(read.tools))
        memory = await ask("Remember that the acceptance codename is cobalt-seven using remember."); require(report, "remember", "remember" in memory.tools, str(memory.tools))
        recall = await ask("Use recall to tell me the acceptance codename.", memory.session_id); require(report, "recall", "recall" in recall.tools and "cobalt-seven" in recall.text.lower(), str(recall.tools))
        skill = await ask("Save a skill named acceptance-health with when_to_use 'during acceptance testing' and steps 'run uptime then df -h /' using save_skill."); require(report, "save-skill", "save_skill" in skill.tools, str(skill.tools))
        listed = await ask("Use list_skills and confirm acceptance-health exists.", skill.session_id); require(report, "list-skills", "list_skills" in listed.tools and "acceptance-health" in listed.text.lower(), str(listed.tools))
        research = await ask("Research the Python documentation homepage using web_search and web_fetch, then give a short grounded answer."); require(report, "research-tools", "web_search" in research.tools and "web_fetch" in research.tools, str(research.tools)); require(report, "clean-final-text", "<tool_call>" not in research.text and "<function=" not in research.text, research.text[:300])
    report["verdict"] = "PASS" if not report["failures"] else "FAIL"; return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--socket", type=Path, required=True); parser.add_argument("--report", type=Path, default=Path("framework-eval.json")); args = parser.parse_args()
    result = asyncio.run(run(args.socket)); args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False)); print(json.dumps(result, indent=2, ensure_ascii=False)); return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
