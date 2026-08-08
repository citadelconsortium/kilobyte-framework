from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .agent import Agent
from .errors import KilobyteError
from .memory import MemoryStore
from .resources import ResourceManager
from .runtime import LlamaRuntime
from .security import Risk


class RPCServer:
    def __init__(self, socket_path: Path, agent: Agent, runtime: LlamaRuntime, resources: ResourceManager, memory: MemoryStore):
        self.socket_path = socket_path
        self.agent = agent
        self.runtime = runtime
        self.resources = resources
        self.memory = memory
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.server = await asyncio.start_unix_server(self._handle, path=self.socket_path)
        os.chmod(self.socket_path, 0o660)

    async def close(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.socket_path.unlink(missing_ok=True)

    async def _send(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload, ensure_ascii=False).encode() + b"\n")
        await writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=30)
            request = json.loads(raw)
            command = request.get("command")
            if command == "status":
                status = self.runtime.status()
                status["healthy"] = await self.runtime.healthy()
                status["memory"] = self.memory.stats()
                await self._send(writer, {"type": "result", "data": status})
            elif command == "resources":
                await self._send(writer, {"type": "result", "data": self.resources.profile().to_dict()})
            elif command == "model_info":
                await self._send(writer, {"type": "result", "data": await self.runtime.metadata()})
            elif command == "sessions":
                await self._send(writer, {"type": "result", "data": {"sessions": self.memory.list_sessions()}})
            elif command == "session_history":
                sid = str(request.get("session_id", ""))
                await self._send(writer, {"type": "result", "data": {"messages": self.memory.history(sid, 200)}})
            elif command == "rotate_circuit":
                from . import net
                if not net.tor_available():
                    await self._send(writer, {"type": "result", "data": {"ok": False, "error": "Tor is not available"}})
                else:
                    ok = await asyncio.to_thread(net.rotate_circuit)
                    ip = await asyncio.to_thread(net.exit_ip) if ok else None
                    await self._send(writer, {"type": "result", "data": {"ok": ok, "exit_ip": ip}})
            elif command == "tor_status":
                from . import net
                await self._send(writer, {"type": "result", "data": {"available": net.tor_available()}})
            elif command == "provider_models":
                try:
                    models = await asyncio.to_thread(self.agent.providers.list_models, None, bool(request.get("only_free", True)))
                    await self._send(writer, {"type": "result", "data": {"ok": True, "models": models}})
                except Exception as exc:
                    await self._send(writer, {"type": "result", "data": {"ok": False, "error": str(exc)}})
            elif command == "provider_info":
                await self._send(writer, {"type": "result", "data": self.agent.providers.info()})
            elif command == "set_model":
                try:
                    m = self.agent.providers.set_model(str(request.get("name", "")), str(request.get("model", "")))
                    await self._send(writer, {"type": "result", "data": {"ok": True, "model": m}})
                except Exception as exc:
                    await self._send(writer, {"type": "result", "data": {"ok": False, "error": str(exc)}})
            elif command == "providers_catalog":
                from .providers import KNOWN_PROVIDERS
                configured = self.agent.providers.providers()
                await self._send(writer, {"type": "result", "data": {
                    "known": {n: {"label": v["label"], "model": v["model"]} for n, v in KNOWN_PROVIDERS.items()},
                    "configured": sorted(configured),
                    "default": self.agent.providers.default_name(),
                }})
            elif command == "configure_provider":
                try:
                    prov = self.agent.providers.configure(
                        str(request.get("name", "")), str(request.get("api_key", "")),
                        request.get("model") or None,
                    )
                    await self._send(writer, {"type": "result", "data": {"ok": True, "label": prov.label, "name": prov.name}})
                except Exception as exc:
                    await self._send(writer, {"type": "result", "data": {"ok": False, "error": str(exc)}})
            elif command == "chat":
                async def permission(capability: str, detail: str, risk: Risk) -> bool:
                    permission_id = uuid.uuid4().hex
                    await self._send(writer, {"type": "permission", "id": permission_id, "capability": capability, "detail": detail, "risk": risk.value})
                    raw_answer = await asyncio.wait_for(reader.readline(), timeout=300)
                    answer = json.loads(raw_answer)
                    if answer.get("type") != "permission_response" or answer.get("id") != permission_id:
                        return False
                    allow = bool(answer.get("allow"))
                    # "yes for this session": grant this capability for the daemon's lifetime
                    # (not persisted to disk), so the same kind of action stops re-prompting.
                    if allow and answer.get("remember"):
                        self.agent.tools.permissions.rules[capability] = "allow"
                    return allow
                run = self.agent.run(
                    str(request.get("text", "")),
                    request.get("session_id"),
                    Path(request.get("cwd") or self.agent.settings.home),
                    bool(request.get("remote", False)),
                    permission,
                    request.get("provider"),
                    request.get("effort"),
                    request.get("agent_profile"),
                    bool(request.get("private", False)),
                )
                try:
                    async for event in run:
                        await self._send(writer, event)
                finally:
                    # A disconnected client (Ctrl-C, killed process) must not leave the
                    # generator running and holding llama-server's single inference slot.
                    await run.aclose()
            else:
                await self._send(writer, {"type": "error", "error": f"unknown command: {command}"})
        except Exception as exc:
            try:
                await self._send(writer, {"type": "error", "error": str(exc)})
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()


class RPCClient:
    def __init__(self, socket_path: Path):
        self.socket_path = socket_path

    async def request(self, command: str, **kwargs: Any) -> dict[str, Any]:
        async for event in self.stream(command, **kwargs):
            if event.get("type") == "result":
                return event.get("data", {})
            if event.get("type") == "error":
                raise KilobyteError(event.get("error", "unknown daemon error"))
        return {}

    async def stream(self, command: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        writer.write(json.dumps({"command": command, **kwargs}).encode() + b"\n")
        await writer.drain()
        try:
            while raw := await reader.readline():
                yield json.loads(raw)
        finally:
            writer.close()
            await writer.wait_closed()
