from __future__ import annotations

import asyncio
import html
import ipaddress
import os
import platform
import re
import shutil
import signal
import socket
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import net, reference
from .config import Settings
from .errors import SecurityError, ToolError
from .mcp import MCPRegistry
from .memory import MemoryStore
from .security import (
    CommandPolicy,
    PathPolicy,
    PermissionCallback,
    PermissionManager,
    Risk,
)


@dataclass(slots=True)
class ToolContext:
    session_id: str
    cwd: Path
    remote: bool = False
    permission_callback: PermissionCallback | None = None
    private: bool = False  # route web tools through Tor (fail-closed)


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _assert_not_local_literal(url: str) -> None:
    """Private-mode guard that does NOT resolve DNS (a local lookup would leak the site).
    Tor resolves the name; here we only reject obviously-local hosts and private IP literals."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SecurityError("only public HTTP(S) URLs are allowed")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local") or host.endswith(".onion"):
        raise SecurityError("refusing a local or onion host in private web mode")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise SecurityError("refusing a non-public IP literal in private web mode")
    except ValueError:
        pass  # a hostname; Tor will resolve it


def _assert_public(url: str) -> None:
    """Refuse a URL that resolves to anything other than a globally routable address."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SecurityError("only public HTTP(S) URLs are allowed")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        }
    except socket.gaierror as exc:
        raise ToolError(f"DNS lookup failed: {parsed.hostname}") from exc
    for raw in addresses:
        ip = ipaddress.ip_address(raw)
        if not ip.is_global:
            raise SecurityError(f"private or local address blocked: {ip}")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-check every redirect target.

    Validating only the requested URL is not enough: a public host may redirect to the
    local network, and urllib follows redirects by default, which would turn web_fetch
    into a way to reach services the path and command policies exist to protect.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _assert_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _object_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


class ToolRegistry:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        permissions: PermissionManager,
        mcp: "MCPRegistry | None" = None,
    ):
        self.settings = settings
        self.memory = memory
        self.permissions = permissions
        self.mcp = mcp
        self.paths = PathPolicy(settings.allowed_roots)
        self.commands = CommandPolicy()
        self._tools: dict[str, ToolDefinition] = {}
        self._register_defaults()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def schemas(
        self, remote: bool = False, request: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a stable tool set.

        The list deliberately does not vary with the request text. Tools are rendered
        into the prompt prefix, so selecting them per request changes that prefix and
        misses llama-server's cache, forcing a full reprocess of the system prompt on
        every message -- minutes of work on CPU-only hardware. A fixed set keeps the
        prefix cacheable, and letting the model choose from all tools is also what the
        agent design calls for.
        """
        del request
        schemas = [tool.openai_schema() for tool in self._tools.values()]
        # Tools published by MCP servers are external code; remote callers never get them.
        if self.mcp is not None and not remote:
            schemas.extend(self.mcp.schemas())
        return schemas

    async def execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> Any:
        if self.mcp is not None and self.mcp.resolve(name) is not None:
            if context.remote:
                raise SecurityError(f"{name} is unavailable over Telegram")
            # An MCP server is third-party code reached from this machine, so calling one
            # is gated like any other outward action rather than treated as safe.
            await self.permissions.authorize(
                "mcp.call",
                name,
                Risk.WRITE,
                context.remote,
                context.permission_callback,
            )
            started = time.monotonic()
            try:
                result = await self.mcp.call(name, arguments)
                self.memory.audit(
                    context.session_id,
                    name,
                    arguments,
                    f"ok in {time.monotonic() - started:.3f}s",
                    context.remote,
                )
                return result
            except Exception as exc:
                self.memory.audit(
                    context.session_id, name, arguments, f"error: {exc}", context.remote
                )
                raise
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        required = [
            str(field) for field in tool.parameters.get("required", [])
            if field not in arguments
        ]
        if required:
            supplied = ", ".join(sorted(arguments)) or "none"
            raise ToolError(
                f"{name} missing required argument(s): {', '.join(required)}; "
                f"supplied: {supplied}. Call {name} again with every required field."
            )
        started = time.monotonic()
        try:
            result = await tool.handler(arguments, context)
            outcome = f"ok in {time.monotonic() - started:.3f}s"
            self.memory.audit(
                context.session_id, name, arguments, outcome, context.remote
            )
            return result
        except Exception as exc:
            self.memory.audit(
                context.session_id, name, arguments, f"error: {exc}", context.remote
            )
            raise

    def _register_defaults(self) -> None:
        string = {"type": "string"}
        self.register(
            ToolDefinition(
                "read_file",
                "Read a UTF-8 text file inside allowed roots.",
                _object_schema({"path": string}, ["path"]),
                self._read_file,
            )
        )
        self.register(
            ToolDefinition(
                "list_files",
                "List a directory inside allowed roots.",
                _object_schema({"path": string}, ["path"]),
                self._list_files,
            )
        )
        self.register(
            ToolDefinition(
                "search_files",
                "Search file contents with ripgrep.",
                _object_schema({"query": string, "path": string}, ["query", "path"]),
                self._search_files,
            )
        )
        self.register(
            ToolDefinition(
                "write_file",
                "Atomically write a UTF-8 file after permission.",
                _object_schema(
                    {"path": string, "content": string}, ["path", "content"]
                ),
                self._write_file,
            )
        )
        self.register(
            ToolDefinition(
                "run_command",
                "Run one program without a shell, with time/output limits.",
                _object_schema(
                    {
                        "command": string,
                        "cwd": string,
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                    },
                    ["command"],
                ),
                self._run_command,
            )
        )
        self.register(
            ToolDefinition(
                "system_info",
                "Inspect operating system, CPU, memory, disk and uptime.",
                _object_schema({}),
                self._system_info,
            )
        )
        self.register(
            ToolDefinition(
                "web_search",
                "Search the public web and return titles, URLs and snippets.",
                _object_schema(
                    {
                        "query": string,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    ["query"],
                ),
                self._web_search,
            )
        )
        self.register(
            ToolDefinition(
                "web_fetch",
                "Fetch a public HTTP(S) page as bounded text.",
                _object_schema({"url": string}, ["url"]),
                self._web_fetch,
            )
        )
        self.register(
            ToolDefinition(
                "reference",
                "Look up the offline reference bank — tool usage, coding, systems and security cheat-sheets. Works with no internet; use it to ground how-to before acting.",
                _object_schema({"query": string}, ["query"]),
                self._reference,
            )
        )
        self.register(
            ToolDefinition(
                "remember",
                "Persist a useful user preference or stable fact.",
                _object_schema(
                    {
                        "content": string,
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    ["content"],
                ),
                self._remember,
            )
        )
        self.register(
            ToolDefinition(
                "recall",
                "Search persistent long-term memory.",
                _object_schema({"query": string}, ["query"]),
                self._recall,
            )
        )
        self.register(
            ToolDefinition(
                "save_skill",
                "Record a reusable procedure once you have completed a multi-step task, so the same work can be repeated without replanning it.",
                _object_schema(
                    {"name": string, "when_to_use": string, "steps": string},
                    ["name", "when_to_use", "steps"],
                ),
                self._save_skill,
            )
        )
        self.register(
            ToolDefinition(
                "search_history",
                "Search your past conversations across earlier sessions to recall something the user said or you did before.",
                _object_schema({"query": string}, ["query"]),
                self._search_history,
            )
        )
        self.register(
            ToolDefinition(
                "list_skills",
                "List the procedures already learned, with how reliable each has been.",
                _object_schema({}),
                self._list_skills,
            )
        )

    async def _read_file(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = self.paths.resolve(str(args["path"]), ctx.cwd, must_exist=True)
        if not path.is_file():
            raise ToolError(f"not a regular file: {path}")
        # Bounded local reads are short and avoid keeping an executor alive in the daemon.
        with path.open("rb") as handle:
            data = handle.read(self.settings.max_read_bytes + 1)
        truncated = len(data) > self.settings.max_read_bytes
        return {
            "path": str(path),
            "content": data[: self.settings.max_read_bytes].decode("utf-8", "replace"),
            "truncated": truncated,
        }

    async def _list_files(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = self.paths.resolve(str(args["path"]), ctx.cwd, must_exist=True)
        if not path.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(
            path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )[:500]
        return {
            "path": str(path),
            "entries": [
                {"name": p.name, "type": "dir" if p.is_dir() else "file"}
                for p in entries
            ],
        }

    async def _search_files(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = self.paths.resolve(str(args["path"]), ctx.cwd, must_exist=True)
        query = str(args["query"])
        binary = shutil.which("rg")
        if not binary:
            raise ToolError("ripgrep is not installed")
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--",
            query,
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await self._bounded_output(proc, self.settings.command_timeout)
        return {
            "matches": stdout[: self.settings.max_tool_output].decode(
                "utf-8", "replace"
            ),
            "stderr": stderr[:4096].decode("utf-8", "replace"),
            "exit_code": proc.returncode,
        }

    async def _bounded_output(
        self, proc: asyncio.subprocess.Process, timeout: int
    ) -> tuple[bytes, bytes]:
        limit = self.settings.max_tool_output + 1

        async def drain(reader: asyncio.StreamReader | None) -> bytes:
            kept = bytearray()
            if reader is None:
                return bytes(kept)
            while chunk := await reader.read(16 * 1024):
                if len(kept) < limit:
                    kept.extend(chunk[: limit - len(kept)])
            return bytes(kept)

        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(drain(proc.stdout), drain(proc.stderr)), timeout=timeout
            )
            await proc.wait()
            return stdout, stderr
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                await proc.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ToolError(f"command timed out after {timeout}s") from exc

    async def _write_file(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = self.paths.resolve(str(args["path"]), ctx.cwd)
        await self.permissions.authorize(
            "filesystem.write",
            str(path),
            Risk.WRITE,
            ctx.remote,
            ctx.permission_callback,
        )
        content = str(args["content"])
        if len(content.encode()) > 8 * 1024 * 1024:
            raise ToolError("write exceeds 8 MiB limit")

        def atomic_write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(
                f".{path.name}.kilo-tmp-{os.getpid()}-{time.time_ns()}"
            )
            try:
                temp.write_text(content, encoding="utf-8")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)

        # The payload is bounded to 8 MiB and the rename is local/atomic. Performing this
        # directly avoids a thread-pool stall after subprocess tools on constrained VMs.
        atomic_write()
        return {"path": str(path), "bytes": len(content.encode())}

    async def _run_command(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        assessment = self.commands.assess(str(args["command"]), ctx.remote)
        capability = f"terminal.execute.{assessment.risk.value}"
        await self.permissions.authorize(
            capability,
            " ".join(assessment.argv),
            assessment.risk,
            ctx.remote,
            ctx.permission_callback,
        )
        cwd = self.paths.resolve(
            str(args.get("cwd") or ctx.cwd), ctx.cwd, must_exist=True
        )
        timeout = min(int(args.get("timeout", self.settings.command_timeout)), 600)
        try:
            proc = await asyncio.create_subprocess_exec(
                *assessment.argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await self._bounded_output(proc, timeout)
        except OSError as exc:
            raise ToolError(f"could not execute {assessment.argv[0]}: {exc}") from exc
        limit = self.settings.max_tool_output
        return {
            "exit_code": proc.returncode,
            "stdout": stdout[:limit].decode("utf-8", "replace"),
            "stderr": stderr[:limit].decode("utf-8", "replace"),
            "truncated": len(stdout) > limit or len(stderr) > limit,
        }

    async def _system_info(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        del args, ctx
        mem = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
                key, value = line.split(":", 1)
                mem[key] = value.strip()
        except OSError:
            pass
        usage = shutil.disk_usage("/")
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory": mem,
            "disk": {"total": usage.total, "used": usage.used, "free": usage.free},
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        }

    @staticmethod
    def _public_url(url: str) -> str:
        _assert_public(url)
        return url

    @staticmethod
    def _request_text(
        url: str, max_bytes: int = 1_000_000, private: bool = False
    ) -> tuple[str, str]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Kilobyte/0.1 (+local-first terminal assistant)",
                "Accept": "text/html,text/plain,application/json",
            },
        )
        # Redirects are followed by default, so validating only the URL we were given
        # leaves the private-address block bypassable: a public host can answer 302 with
        # a location on the local network. Every hop is re-checked instead.
        if private:
            # Fail-closed: if privacy was asked for but Tor is not reachable, do NOT send —
            # a silent direct connection would expose the real IP, the opposite of intent.
            if not net.tor_available():
                raise ToolError(
                    "private mode is on but Tor is not reachable — request NOT sent so your real IP is not exposed. Start tor, or turn it off with /private off."
                )
            opener = urllib.request.build_opener(
                net.SocksHTTPHandler, net.SocksHTTPSHandler, _ValidatingRedirectHandler
            )
            timeout = 45
        else:
            opener = urllib.request.build_opener(_ValidatingRedirectHandler)
            timeout = 15
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            data = response.read(max_bytes + 1)
        return data[:max_bytes].decode("utf-8", "replace"), content_type

    async def _web_search(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        if ctx.private and not net.tor_available():
            raise ToolError(
                "private mode is on but Tor is not reachable — request NOT sent"
            )
        query = str(args["query"]).strip()
        limit = min(int(args.get("limit", 5)), 10)
        url = "https://www.bing.com/search?" + urllib.parse.urlencode(
            {"q": query, "format": "rss"}
        )
        body, _ = await asyncio.to_thread(self._request_text, url, 800_000, ctx.private)
        return {
            "query": query,
            "results": self._parse_search_rss(body, limit),
            "private": ctx.private,
        }

    @staticmethod
    def _parse_search_rss(body: str, limit: int) -> list[dict[str, str]]:
        results = []
        # ElementTree does not resolve external entities, but it does expand internal
        # ones, so a hostile or compromised provider could return a small document that
        # expands to gigabytes. Bounding the response does not bound the expansion, and
        # this runs on machines with about 2 GB to spare, so documents carrying a
        # document type declaration are refused outright.
        if re.search(r"<!DOCTYPE", body[:4096], re.IGNORECASE):
            raise ToolError(
                "search provider returned a document type declaration; refused"
            )
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise ToolError("search provider returned invalid RSS") from exc
        strip = lambda text: re.sub(
            r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))
        ).strip()
        for item in root.findall("./channel/item")[:limit]:
            link = (item.findtext("link") or "").strip()
            if link.startswith(("http://", "https://")):
                results.append(
                    {
                        "title": strip(item.findtext("title") or ""),
                        "url": link,
                        "snippet": strip(item.findtext("description") or ""),
                    }
                )
        return results

    async def _web_fetch(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        raw = str(args["url"])
        # Private mode skips local DNS resolution (it would leak the lookup) — Tor resolves
        # it — and applies a DNS-free local-host guard instead of the resolving SSRF check.
        if ctx.private:
            _assert_not_local_literal(raw)
            if not net.tor_available():
                raise ToolError(
                    "private mode is on but Tor is not reachable — request NOT sent"
                )
            url = raw
        else:
            url = self._public_url(raw)
        body, content_type = await asyncio.to_thread(
            self._request_text, url, 1_000_000, ctx.private
        )
        if content_type == "text/html":
            body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", body)
            body = re.sub(r"(?s)<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", html.unescape(body)).strip()
        return {
            "url": url,
            "content_type": content_type,
            "content": body[:200_000],
            "truncated": len(body) > 200_000,
        }

    async def _reference(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        del ctx
        entries = reference.search(str(args["query"]))
        return {
            "query": args["query"],
            "entries": entries,
            "note": "offline reference bank"
            if entries
            else "no matching entry; use a tool or say you are not certain",
        }

    async def _remember(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        fact_id = self.memory.remember(
            str(args["content"]),
            scope="global",
            importance=float(args.get("importance", 0.5)),
        )
        return {"remembered": True, "id": fact_id}

    async def _recall(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        return {"facts": self.memory.recall(str(args["query"]))}

    async def _save_skill(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        del ctx
        name = str(args["name"]).strip()
        if not name:
            raise ToolError("a skill needs a name")
        self.memory.save_skill(name, str(args["when_to_use"]), str(args["steps"]))
        return {"saved": True, "name": name}

    async def _list_skills(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        del args, ctx
        return {"skills": self.memory.list_skills()}

    async def _search_history(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        del ctx
        hits = self.memory.search_messages(str(args["query"]))
        return {
            "matches": [
                {"role": h["role"], "content": h["content"][:300]} for h in hits
            ]
        }
