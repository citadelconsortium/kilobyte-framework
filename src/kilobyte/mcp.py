"""Model Context Protocol client (stdio transport).

Implements the client side of MCP 2025-06-18 so tools published by external servers can
be offered to the brain alongside the built-in ones. Only the stdio transport is
implemented: the specification says clients should support it whenever possible, it is
what local servers use, and it needs no listening socket on a machine whose whole point
is that it does not expose services.

Framing, per the transport specification: UTF-8 JSON-RPC messages, one per line,
newline-delimited, and a message must not contain embedded newlines. A server may write
free-form logging to stderr, so stderr is captured for diagnostics and never parsed.

Servers are untrusted. Their tools are namespaced, their schemas are checked before being
shown to the model, and their results pass through the same compaction and the same
permission layer as everything else. A server that hangs must not hang Kilobyte, so every
request has a timeout and a failed server is disabled rather than retried forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

from .errors import ToolError


log = logging.getLogger("kilobyte.mcp")

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "Kilobyte", "version": "0.1.0"}

# A tool name the model sees, e.g. "mcp__files__read". Keeps server tools distinguishable
# from built-ins and from each other when two servers publish the same name.
NAME_SEPARATOR = "__"


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "MCPServerConfig":
        command = raw.get("command")
        if isinstance(command, str):
            command = [command, *[str(a) for a in raw.get("args", [])]]
        elif isinstance(command, list):
            command = [str(part) for part in command]
        else:
            raise ValueError(f"server {name}: 'command' must be a string or a list")
        if not command:
            raise ValueError(f"server {name}: 'command' is empty")
        env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        return cls(name=name, command=command, env=env, enabled=bool(raw.get("enabled", True)))


class MCPServer:
    """One server subprocess and the JSON-RPC session running over its pipes."""

    def __init__(self, config: MCPServerConfig, request_timeout: float = 30.0):
        self.config = config
        self.request_timeout = request_timeout
        self.process: asyncio.subprocess.Process | None = None
        self.tools: list[dict[str, Any]] = []
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def _drain_stderr(self) -> None:
        """Servers may log freely to stderr; read it so the pipe cannot fill and block
        the server, and surface it at debug level for diagnosis."""
        assert self.process is not None and self.process.stderr is not None
        try:
            while line := await self.process.stderr.readline():
                log.debug("[%s] %s", self.config.name, line.decode("utf-8", "replace").rstrip())
        except (asyncio.CancelledError, ValueError):
            pass

    async def start(self) -> None:
        binary = shutil.which(self.config.command[0])
        if not binary:
            raise ToolError(f"MCP server {self.config.name}: command not found: {self.config.command[0]}")
        # A server inherits only what it is given plus PATH, so a misbehaving one cannot
        # read secrets that happen to be in the daemon's environment.
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", **self.config.env}
        self.process = await asyncio.create_subprocess_exec(
            binary, *self.config.command[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        result = await self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        negotiated = str(result.get("protocolVersion", ""))
        if negotiated != PROTOCOL_VERSION:
            # The server may answer with a version it prefers. Log it and continue: the
            # messages this client sends are stable across these revisions.
            log.info("[%s] server negotiated protocol %s", self.config.name, negotiated or "unknown")
        await self.notify("notifications/initialized")
        self.tools = await self._list_tools()
        log.info("[%s] ready with %d tool(s)", self.config.name, len(self.tools))

    async def _list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        # Bounded so a server that returns a cursor forever cannot spin here.
        for _ in range(20):
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            for tool in result.get("tools", []):
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    tools.append(tool)
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def _send(self, message: dict[str, Any]) -> None:
        if not self.running or self.process is None or self.process.stdin is None:
            raise ToolError(f"MCP server {self.config.name} is not running")
        # Messages must not contain embedded newlines: compact separators and no literal
        # newlines in the encoding keep each message on exactly one line.
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")).replace("\n", " ")
        self.process.stdin.write(line.encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        await self._send(message)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and wait for the response with a matching id.

        Requests are serialised: this client sends one at a time, so anything arriving
        with a different id is a server-initiated message and is skipped rather than
        mistaken for the answer.
        """
        async with self._lock:
            if self.process is None or self.process.stdout is None:
                raise ToolError(f"MCP server {self.config.name} is not running")
            self._next_id += 1
            request_id = self._next_id
            message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params:
                message["params"] = params
            await self._send(message)

            async def await_response() -> dict[str, Any]:
                assert self.process is not None and self.process.stdout is not None
                while raw := await self.process.stdout.readline():
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("[%s] non-JSON line on stdout, ignored", self.config.name)
                        continue
                    if payload.get("id") != request_id:
                        continue
                    if "error" in payload:
                        error = payload["error"] or {}
                        raise ToolError(f"{self.config.name}: {error.get('message', 'unknown error')}")
                    return payload.get("result") or {}
                raise ToolError(f"MCP server {self.config.name} closed its output stream")

            try:
                return await asyncio.wait_for(await_response(), timeout=self.request_timeout)
            except asyncio.TimeoutError as exc:
                raise ToolError(f"MCP server {self.config.name} timed out on {method}") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(
            str(block.get("text", ""))
            for block in result.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        # A tool-execution failure is reported in the result, not as a JSON-RPC error,
        # and has to reach the model as a failure rather than as a normal answer.
        if result.get("isError"):
            raise ToolError(text or f"{self.config.name}: tool reported an error")
        payload: dict[str, Any] = {"content": text}
        if isinstance(result.get("structuredContent"), dict):
            payload["structured"] = result["structuredContent"]
        return payload

    async def stop(self) -> None:
        """Shut down as the specification directs: close stdin, then escalate."""
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ConnectionError, OSError):
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                with_kill = getattr(process, "kill", None)
                if with_kill:
                    with_kill()
                await process.wait()


class MCPRegistry:
    """Owns the configured servers and exposes their tools under namespaced names."""

    def __init__(self, config_path, request_timeout: float = 30.0):
        self.config_path = config_path
        self.request_timeout = request_timeout
        self.servers: dict[str, MCPServer] = {}

    def configured(self) -> list[MCPServerConfig]:
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("mcp config unreadable (%s); no servers started", exc)
            return []
        servers = raw.get("servers") or raw.get("mcpServers") or {}
        configs: list[MCPServerConfig] = []
        for name, entry in servers.items():
            if not isinstance(entry, dict):
                continue
            try:
                config = MCPServerConfig.from_dict(str(name), entry)
            except ValueError as exc:
                log.warning("ignoring mcp server: %s", exc)
                continue
            if config.enabled:
                configs.append(config)
        return configs

    async def start(self) -> None:
        for config in self.configured():
            server = MCPServer(config, self.request_timeout)
            try:
                await server.start()
                self.servers[config.name] = server
            except Exception:
                # One bad server must not stop the others or the daemon.
                log.exception("mcp server %s failed to start; skipping", config.name)
                await server.stop()

    async def stop(self) -> None:
        for server in list(self.servers.values()):
            await server.stop()
        self.servers.clear()

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the model, namespaced and validated.

        A server controls these strings, so anything without a usable object schema is
        dropped rather than passed to the model as-is.
        """
        schemas: list[dict[str, Any]] = []
        for server_name, server in self.servers.items():
            for tool in server.tools:
                parameters = tool.get("inputSchema")
                if not isinstance(parameters, dict) or parameters.get("type") != "object":
                    log.warning("[%s] tool %s has no object input schema; skipped", server_name, tool.get("name"))
                    continue
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp{NAME_SEPARATOR}{server_name}{NAME_SEPARATOR}{tool['name']}",
                        "description": str(tool.get("description") or f"{tool['name']} from {server_name}")[:400],
                        "parameters": parameters,
                    },
                })
        return schemas

    def resolve(self, namespaced: str) -> tuple[MCPServer, str] | None:
        if not namespaced.startswith(f"mcp{NAME_SEPARATOR}"):
            return None
        remainder = namespaced[len(f"mcp{NAME_SEPARATOR}"):]
        server_name, separator, tool_name = remainder.partition(NAME_SEPARATOR)
        if not separator:
            return None
        server = self.servers.get(server_name)
        if server is None:
            return None
        return server, tool_name

    async def call(self, namespaced: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resolved = self.resolve(namespaced)
        if resolved is None:
            raise ToolError(f"unknown MCP tool: {namespaced}")
        server, tool_name = resolved
        return await server.call_tool(tool_name, arguments)
