from __future__ import annotations

import asyncio
import logging
import signal

from .agent import Agent
from .config import Settings
from .mcp import MCPRegistry
from .memory import MemoryStore
from .prompt import SYSTEM_PROMPT
from .providers import ProviderRegistry
from .resources import ResourceManager
from .rpc import RPCServer
from .runtime import LlamaRuntime
from .security import PermissionManager
from .telegram import TelegramBridge
from .tools import ToolRegistry


async def serve() -> None:
    settings = Settings()
    settings.ensure_user_dirs()
    logging.basicConfig(
        filename=settings.log_dir / "kilobyte.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("kilobyte")
    memory = MemoryStore(settings.database_path, settings.memory_message_limit, settings.memory_fact_limit, settings.memory_skill_limit)
    resources = ResourceManager(settings)
    permissions = PermissionManager(settings.policy_path)
    mcp = MCPRegistry(settings.mcp_path)
    tools = ToolRegistry(settings, memory, permissions, mcp)
    runtime = LlamaRuntime(settings, resources)
    providers = ProviderRegistry(settings.providers_path)
    agent = Agent(settings, runtime, memory, tools, providers)
    rpc = RPCServer(settings.socket_path, agent, runtime, resources, memory)
    telegram = TelegramBridge(settings.telegram_path, agent)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, stop_event.set)

    telegram_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None
    warmup_task: asyncio.Task[None] | None = None

    async def warmup() -> None:
        try:
            log.info("preparing model cache (restore from disk, else process the prefix)")
            await runtime.warmup(SYSTEM_PROMPT, tools.schemas())
            log.info("model cache warm (%s)", "restored from disk" if runtime.cache_restored else "processed and saved")
        except Exception:
            log.exception("warmup failed; first real request will pay the cold cost")

    async def monitor_runtime() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(5)
            if runtime.process is not None and runtime.process.returncode is not None:
                log.error("llama-server exited %s; restarting the same model", runtime.process.returncode)
                try:
                    await runtime.start()
                except Exception:
                    log.exception("model runtime restart failed; retrying")
    async def start_when_memory_allows() -> None:
        """Wait out transient memory pressure instead of crash-looping.

        Available memory can dip below the start threshold for reasons unrelated to Kilo
        (a large file copy, another process spiking). Exiting immediately made systemd
        restart the daemon in a tight loop and, because the bridge lives in the daemon,
        took Telegram down with it. Retrying with backoff lets a temporary dip clear;
        only a persistent shortage eventually gives up to systemd."""
        from .errors import RuntimeUnavailable

        delay = 5
        for _attempt in range(12):
            try:
                await runtime.start()
                return
            except RuntimeUnavailable as exc:
                log.warning("model runtime not startable yet (%s); retry in %ss", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        # Persistent shortage: start once more and let the exception propagate to systemd.
        await runtime.start()

    try:
        log.info("starting persistent model runtime")
        await start_when_memory_allows()
        # Started before warmup so any server tools are part of the prefix that gets
        # primed and cached, rather than changing it on the first real request.
        await mcp.start()
        await rpc.start()
        telegram_task = asyncio.create_task(telegram.run(), name="telegram-bridge")
        monitor_task = asyncio.create_task(monitor_runtime(), name="runtime-monitor")
        warmup_task = asyncio.create_task(warmup(), name="model-warmup")
        log.info("ready on %s", settings.socket_path)
        await stop_event.wait()
    finally:
        telegram.stop()
        tasks = [task for task in (telegram_task, monitor_task, warmup_task) if task]
        for task in tasks:
            task.cancel()
        await rpc.close()
        # Stop llama-server before awaiting the tasks. Warmup, telegram and inference
        # do blocking HTTP in worker threads; cancelling a task does not interrupt its
        # thread, and asyncio.run() waits for the default executor at exit. Killing the
        # server first makes those requests fail immediately, so shutdown stays within
        # systemd's stop timeout instead of being escalated to SIGKILL.
        await runtime.stop()
        await mcp.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        memory.close()
        log.info("stopped cleanly")


def main() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
