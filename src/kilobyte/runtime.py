from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from typing import Any

from .config import Settings
from .errors import ModelUnavailable, RuntimeUnavailable
from .resources import ResourceManager, ResourceProfile


class LlamaRuntime:
    """Owns exactly one persistent llama-server process and model instance."""

    def __init__(self, settings: Settings, resources: ResourceManager):
        self.settings = settings
        self.resources = resources
        self.profile: ResourceProfile | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None
        self.log_path = settings.log_dir / "llama-server.log"
        self._log_handle: Any = None
        self._lock = asyncio.Lock()
        self.cache_restored = False
        # Warmup holds the single inference slot for as long as the prefix takes to
        # process. Requests arriving meanwhile queue behind it, so callers need to be
        # able to say so rather than appearing to hang and then timing out.
        self.warming = False

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.llama_host}:{self.settings.llama_port}"

    def command(self, profile: ResourceProfile) -> list[str]:
        # Persist slot KV cache to disk so a warmed prefix survives restarts instead of
        # being reprocessed from scratch, and let partially matching prefixes be reused.
        slot_cache = self.settings.data_dir / "kv-cache"
        slot_cache.mkdir(parents=True, exist_ok=True)
        return [
            self.settings.llama_binary,
            "--model", str(self.settings.model_path),
            "--slot-save-path", str(slot_cache),
            "--cache-reuse", "256",
            "--host", self.settings.llama_host,
            "--port", str(self.settings.llama_port),
            "--ctx-size", str(profile.context_size),
            "--threads", str(profile.threads),
            "--threads-batch", str(profile.threads),
            "--batch-size", str(profile.batch_size),
            "--ubatch-size", str(min(profile.batch_size, 128)),
            "--n-gpu-layers", str(profile.gpu_layers),
            "--parallel", "1",
            "--jinja",
            "--metrics",
            "--no-webui",
            "--reasoning", "off",
        ]

    async def start(self, timeout: float = 240.0) -> None:
        async with self._lock:
            if self.process and self.process.returncode is None:
                return
            if not self.settings.model_path.is_file():
                raise ModelUnavailable(f"model not installed: {self.settings.model_path}")
            binary = shutil.which(self.settings.llama_binary)
            if not binary:
                raise RuntimeUnavailable(f"llama-server not found: {self.settings.llama_binary}")
            self.profile = self.resources.profile()
            enough, reason = self.resources.enough_to_start(self.profile)
            if not enough:
                raise RuntimeUnavailable(reason)
            self.settings.log_dir.mkdir(parents=True, exist_ok=True)
            if self._log_handle:
                self._log_handle.close()
            self._log_handle = self.log_path.open("ab", buffering=0)
            env = os.environ.copy()
            env.setdefault("LLAMA_CACHE", str(self.settings.data_dir / "cache"))
            self.process = await asyncio.create_subprocess_exec(
                *self.command(self.profile),
                stdout=self._log_handle,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            self.started_at = time.monotonic()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process and self.process.returncode is not None:
                raise RuntimeUnavailable(f"llama-server exited {self.process.returncode}; see {self.log_path}")
            if await self.healthy():
                return
            await asyncio.sleep(0.5)
        await self.stop()
        raise RuntimeUnavailable(f"llama-server did not become healthy within {timeout:.0f}s")

    async def ensure_ready(self) -> None:
        """Recover a crashed model process while preserving the one-instance invariant."""
        if self.process is None or self.process.returncode is not None:
            await self.start()
            return
        headroom_ok, available_mb = self.resources.live_headroom()
        if not headroom_ok:
            raise RuntimeUnavailable(
                f"inference paused to protect the system: only {available_mb} MiB memory available"
            )

    async def stop(self) -> None:
        async with self._lock:
            process = self.process
            self.process = None
            if process and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    await asyncio.wait_for(process.wait(), timeout=15)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None

    async def healthy(self) -> bool:
        def check() -> bool:
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=1.5) as response:
                    return response.status == 200
            except (OSError, urllib.error.URLError):
                return False
        return await asyncio.to_thread(check)

    # Kept well under systemd's stop timeout: these run in worker threads that shutdown
    # cannot interrupt, so a long timeout here delays daemon exit.
    def _slot_action(self, action: str, filename: str, timeout: float = 25.0) -> bool:
        request = urllib.request.Request(
            f"{self.base_url}/slots/0?action={action}",
            data=json.dumps({"filename": filename}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    def _cache_filename(self, system_prompt: str, tools: list[dict[str, Any]] | None) -> str:
        """Key the saved cache to everything that makes it valid.

        The prompt and tool set define the token prefix. The context size has to be in
        here too: the install is portable, so the same data directory can be carried to
        a machine with different memory, where the resource manager picks a different
        context and a slot saved under the old one is not restorable.
        """
        digest = hashlib.sha256()
        digest.update(system_prompt.encode())
        digest.update(json.dumps(tools or [], sort_keys=True).encode())
        digest.update(str(self.settings.model_path).encode())
        digest.update(str(self.profile.context_size if self.profile else 0).encode())
        return f"kilobyte-prefix-{digest.hexdigest()[:16]}.bin"

    def _prune_cache(self, keep: str) -> None:
        """Each saved slot is large and machine-specific; carrying the install between
        machines would otherwise leave a stale one behind for every context size used."""
        cache_dir = self.settings.data_dir / "kv-cache"
        try:
            for stale in cache_dir.glob("kilobyte-prefix-*.bin"):
                if stale.name != keep:
                    stale.unlink(missing_ok=True)
        except OSError:
            pass

    async def warmup(self, system_prompt: str, tools: list[dict[str, Any]] | None = None) -> None:
        """Make the prefix real requests use resident in the KV cache.

        The tool schemas must match what the agent sends, otherwise the prefix differs
        and every real message still pays full prompt processing -- minutes on a CPU-only
        machine. The warmed slot is saved to disk and restored on the next start, so this
        cost is paid once for a given prompt/tool set rather than on every boot.
        """
        filename = self._cache_filename(system_prompt, tools)
        if await asyncio.to_thread(self._slot_action, "restore", filename):
            self.cache_restored = True
            return
        self.warming = True
        payload: dict[str, Any] = {
            "model": "kilobyte",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Reply with just: ready"}],
            "max_tokens": 4,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            async for _ in self.chat_stream(payload):
                pass
        finally:
            self.warming = False
        self.warming = False
        if await asyncio.to_thread(self._slot_action, "save", filename):
            await asyncio.to_thread(self._prune_cache, filename)

    async def metadata(self) -> dict[str, Any]:
        def fetch() -> dict[str, Any]:
            try:
                with urllib.request.urlopen(self.base_url + "/props", timeout=3) as response:
                    return json.load(response)
            except Exception:
                return {}
        return await asyncio.to_thread(fetch)

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream decoded OpenAI SSE deltas without exposing reasoning_content."""
        payload = dict(payload)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        def open_request():
            request = urllib.request.Request(
                self.base_url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            # Generous: on a CPU-only host a request can legitimately sit behind warmup
            # for the single slot before its own generation even starts.
            return urllib.request.urlopen(request, timeout=2400)

        try:
            response = await asyncio.to_thread(open_request)
        except Exception as exc:
            raise RuntimeUnavailable(f"inference request failed: {exc}") from exc
        try:
            while True:
                raw = await asyncio.to_thread(response.readline)
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if choices:
                    delta = dict(choices[0].get("delta") or {})
                    delta.pop("reasoning_content", None)
                    yield {"delta": delta, "finish_reason": choices[0].get("finish_reason")}
                if event.get("usage"):
                    yield {"usage": event["usage"]}
        finally:
            response.close()

    def status(self) -> dict[str, Any]:
        running = bool(self.process and self.process.returncode is None)
        return {
            "running": running,
            "pid": self.process.pid if running and self.process else None,
            "healthy": None,
            "uptime_seconds": int(time.monotonic() - self.started_at) if running and self.started_at else 0,
            "model": str(self.settings.model_path),
            "warming": self.warming,
            "profile": self.profile.to_dict() if self.profile else None,
        }
