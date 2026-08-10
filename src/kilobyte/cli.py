from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import MODEL_QUANTIZATION, MODEL_REPOSITORY, MODEL_SHA256, Settings
from .doctor import run_checks
from .errors import KilobyteError
from .resources import ResourceManager
from .rpc import RPCClient
from .theme import BOLD, RED
from .tui import DIM, GREEN, RESET, YELLOW, TerminalUI


def json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _duration(seconds: int | float) -> str:
    total = max(0, int(seconds))
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def print_status(status: dict[str, Any] | None) -> None:
    """Render status for people; JSON is an implementation detail, not a UI."""
    print(f"{BOLD}KILOBYTE STATUS{RESET}")
    if not status:
        print(f"{'STATE':<12} {RED}STOPPED{RESET}")
        print(f"{'daemon':<12} {RED}INACTIVE{RESET}")
        print(f"{'action':<12} sudo systemctl start kilobyte")
        return

    running = bool(status.get("running"))
    healthy = bool(status.get("healthy"))
    ready = running and healthy
    state = "READY" if ready else "DEGRADED" if running else "FAILED"
    state_color = GREEN if ready else YELLOW if running else RED
    profile = status.get("profile") or {}
    memory = status.get("memory") or {}
    model = Path(str(status.get("model") or "unknown")).name
    cache = "WARMING" if status.get("warming") else "READY"
    cache_color = YELLOW if status.get("warming") else GREEN

    print(f"{'STATE':<12} {state_color}{state}{RESET}")
    print(f"{'daemon':<12} {GREEN}ACTIVE{RESET}  pid {status.get('pid', '?')}")
    print(f"{'brain':<12} {(GREEN if healthy else RED)}{'HEALTHY' if healthy else 'UNHEALTHY'}{RESET}  {model}")
    print(f"{'cache':<12} {cache_color}{cache}{RESET}")
    print(f"{'uptime':<12} {_duration(status.get('uptime_seconds', 0))}")
    print(
        f"{'runtime':<12} {profile.get('threads', '?')} threads · "
        f"{profile.get('context_size', '?')} context · {profile.get('gpu_layers', 0)} GPU layers"
    )
    print(
        f"{'memory':<12} {profile.get('total_mb', '?')} MiB total · "
        f"{profile.get('available_mb', '?')} MiB available · "
        f"{memory.get('sessions', 0)} sessions · {memory.get('facts', 0)} facts · "
        f"{memory.get('skills', 0)} skills"
    )


def runtime_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep model-info useful without printing llama.cpp's multi-page template."""
    caps = metadata.get("chat_template_caps") or {}
    defaults = metadata.get("default_generation_settings") or {}
    return {
        "build": metadata.get("build_info"),
        "model": metadata.get("model_alias") or metadata.get("model_path"),
        "context_size": defaults.get("n_ctx"),
        "tool_calling": bool(caps.get("supports_tool_calls")),
        "sleeping": bool(metadata.get("is_sleeping", False)),
    }


def service_action(action: str) -> int:
    command = ["systemctl", action, "kilobyte.service"]
    if os.geteuid() != 0:
        command.insert(0, "sudo")
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kilo", description="Kilobyte local-first terminal AI")
    parser.add_argument("--version", action="version", version=f"Kilobyte {__version__}")
    sub = parser.add_subparsers(dest="command")
    chat = sub.add_parser("chat", help="send one prompt and stream the answer")
    chat.add_argument("text", nargs="+")
    sub.add_parser("status", help="show daemon, model and resource status")
    doctor = sub.add_parser("doctor", help="run installation and health checks")
    doctor.add_argument("--verify-model", action="store_true", help="read the full GGUF and verify SHA-256")
    sub.add_parser("resources", help="show the live resource profile")
    sub.add_parser("model-info", help="show the one installed brain")
    sub.add_parser("version", help="show Kilobyte and runtime versions")
    logs = sub.add_parser("logs", help="show service logs")
    logs.add_argument("-n", "--lines", type=int, default=100)
    for action in ("restart", "stop", "start"):
        sub.add_parser(action, help=f"{action} the Kilobyte service")
    benchmark = sub.add_parser("benchmark", help="measure a short real inference")
    benchmark.add_argument("--prompt", default="Reply with exactly: Kilobyte is ready.")
    brain = sub.add_parser("brain", help="manage the trained brain (candidate/current/previous)")
    brain_sub = brain.add_subparsers(dest="brain_command")
    brain_sub.add_parser("status", help="show the installed, candidate and previous brains")
    stage = brain_sub.add_parser("stage", help="stage a trained GGUF as the candidate brain")
    stage.add_argument("source", help="path to the candidate .gguf")
    stage.add_argument("--sha256", help="expected checksum to verify before staging")
    promote = brain_sub.add_parser("promote", help="promote the candidate to the current brain")
    promote.add_argument("--sha256", help="expected checksum to verify before promoting")
    promote.add_argument("--brain-version", default="unknown", help="version label to record for this brain")
    brain_sub.add_parser("rollback", help="restore the previous brain after a bad promotion")
    brain_sub.add_parser("versions", help="show the brain version history")
    deploy = brain_sub.add_parser("deploy", help="stage, promote, restart and smoke-test a brain, rolling back on failure")
    deploy.add_argument("source", help="path to the candidate .gguf")
    deploy.add_argument("--brain-version", default="unknown")
    deploy.add_argument("--sha256", help="expected checksum")

    tg = sub.add_parser("telegram", help="manage the Telegram bot (token and allowed chats)")
    tg_sub = tg.add_subparsers(dest="telegram_command")
    tg_sub.add_parser("status", help="show whether Telegram is enabled and which chats are allowed")
    tg_token = tg_sub.add_parser("set-token", help="set the bot token")
    tg_token.add_argument("token", help="the @BotFather bot token")
    tg_allow = tg_sub.add_parser("allow", help="authorise a chat id")
    tg_allow.add_argument("chat_id", type=int, help="numeric chat id (send /id to the bot to find it)")
    tg_deny = tg_sub.add_parser("disallow", help="remove a chat id")
    tg_deny.add_argument("chat_id", type=int)
    tg_sub.add_parser("disable", help="turn Telegram off by clearing the token")
    return parser


def brain_command(args: argparse.Namespace, settings: Settings) -> int:
    """Manage the trained brain. Model moves are deliberate and never automatic: a
    candidate is staged and only becomes current on explicit promotion, and the previous
    brain is always kept so a bad promotion can be rolled back."""
    from .brains import BrainError, BrainManager

    manager = BrainManager(settings.data_dir / "models")
    action = getattr(args, "brain_command", None)
    try:
        if action in (None, "status"):
            status = manager.status()
            for slot in ("current", "candidate", "previous"):
                info = status[slot]
                where = str(info["path"])
                if info["exists"]:
                    print(f"{slot:<10} {info['size'] // (1024 * 1024)} MiB   {where}")
                else:
                    print(f"{slot:<10} {DIM}absent{RESET}   {where}")
            return 0
        if action == "stage":
            info = manager.stage_candidate(Path(args.source), args.sha256)
            print(f"{GREEN}staged candidate{RESET} ({info.size // (1024 * 1024)} MiB, sha256 {info.sha256})")
            print("evaluate it, then: kilo brain promote")
            return 0
        if action == "promote":
            info = manager.promote(args.sha256, brain_version=args.brain_version, framework_version=__version__)
            print(f"{GREEN}promoted{RESET} brain {args.brain_version} to current ({info.sha256}).")
            print("restart Kilo to load it: sudo systemctl restart kilobyte")
            return 0
        if action == "rollback":
            info = manager.rollback()
            print(f"{YELLOW}rolled back{RESET} to the previous brain ({info.sha256}). Restart Kilo: sudo systemctl restart kilobyte")
            return 0
        if action == "deploy":
            return brain_deploy(manager, args, settings)
        if action == "versions":
            history = manager.versions()
            if not history:
                print(f"{DIM}no brain versions recorded yet{RESET}")
                return 0
            import datetime
            for entry in history:
                when = datetime.datetime.fromtimestamp(entry.get("at", 0)).strftime("%Y-%m-%d %H:%M")
                mark = GREEN if entry.get("event") == "promote" else YELLOW
                print(f"{when}  {mark}{entry.get('event'):<9}{RESET}brain {entry.get('brain_version')}  fw {entry.get('framework_version')}  {DIM}{str(entry.get('sha256'))[:12]}{RESET}")
            print(f"\ncurrent brain version: {manager.current_version() or 'unknown'}")
            return 0
    except BrainError as exc:
        print(f"{YELLOW}brain error:{RESET} {exc}", file=sys.stderr)
        return 1
    return 0


def telegram_command(args: argparse.Namespace, settings: Settings) -> int:
    """Manage the Telegram bot without hand-editing JSON. The bridge polls its config
    after each long poll, so these changes take effect without a restart. The file is written 0600
    because it holds the bot token."""
    import json
    import os

    path = settings.telegram_path
    action = getattr(args, "telegram_command", None) or "status"
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        config = {}
    config.setdefault("token", "")
    allowed = [int(x) for x in config.get("allowed_chat_ids", []) if str(x).lstrip("-").isdigit()]

    def save() -> None:
        config["allowed_chat_ids"] = sorted(set(allowed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    if action == "status":
        token = str(config.get("token", "")).strip()
        enabled = bool(token) and token != "PASTE_BOT_TOKEN_HERE" and bool(allowed)
        print(f"telegram   {GREEN + 'enabled' + RESET if enabled else YELLOW + 'disabled' + RESET}")
        print(f"token      {'set (' + token[:6] + '…)' if token else DIM + 'unset' + RESET}")
        print(f"allowed    {', '.join(map(str, allowed)) if allowed else DIM + 'none' + RESET}")
        if not enabled:
            print(f"\n{DIM}enable with: kilo telegram set-token <token> && kilo telegram allow <chat_id>{RESET}")
        return 0
    if action == "set-token":
        config["token"] = args.token.strip()
        save()
        print(f"{GREEN}token set{RESET}. The bridge picks it up after the current poll (about 10s).")
        return 0
    if action == "allow":
        allowed.append(int(args.chat_id))
        save()
        print(f"{GREEN}authorised{RESET} chat {args.chat_id}.")
        return 0
    if action == "disallow":
        allowed[:] = [c for c in allowed if c != int(args.chat_id)]
        save()
        print(f"{YELLOW}removed{RESET} chat {args.chat_id}.")
        return 0
    if action == "disable":
        config["token"] = ""
        save()
        print(f"{YELLOW}telegram disabled{RESET} (token cleared).")
        return 0
    return 0


def brain_deploy(manager, args: argparse.Namespace, settings: Settings) -> int:
    """The promotion gate: stage, promote, restart, smoke-test, and roll back on failure.

    A trained brain becomes the one Kilo runs only if it actually loads and answers after
    promotion. If the smoke test fails, the previous brain is restored automatically, so a
    bad candidate never leaves Kilo without a working brain.
    """
    from .brains import BrainError

    try:
        manager.stage_candidate(Path(args.source), args.sha256)
        manager.promote(args.sha256, brain_version=args.brain_version, framework_version=__version__)
    except BrainError as exc:
        print(f"{YELLOW}deploy aborted:{RESET} {exc}", file=sys.stderr)
        return 1
    print(f"{GREEN}promoted{RESET} brain {args.brain_version}; restarting to load it")
    if service_action("restart") != 0:
        print(f"{YELLOW}could not restart the service; rolling back{RESET}", file=sys.stderr)
        manager.rollback()
        service_action("restart")
        return 1

    ok, detail = smoke_test(settings)
    if ok:
        print(f"{GREEN}KILOBYTE BRAIN PROMOTION SUCCESSFUL{RESET} — {detail}")
        return 0
    print(f"{YELLOW}smoke test failed ({detail}); rolling back to the previous brain{RESET}", file=sys.stderr)
    try:
        manager.rollback()
    except BrainError as exc:
        print(f"{YELLOW}rollback failed:{RESET} {exc}", file=sys.stderr)
        return 2
    service_action("restart")
    print(f"{YELLOW}rolled back{RESET}; Kilo is running the previous brain")
    return 1


def smoke_test(settings: Settings, timeout: float = 900.0) -> tuple[bool, str]:
    """Confirm the freshly promoted brain loads and answers. Waits for the daemon to come
    healthy, then checks a reply mentions the Kilobyte identity."""
    client = RPCClient(settings.socket_path)

    async def probe() -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = await client.request("status")
                if status.get("healthy"):
                    break
            except (FileNotFoundError, ConnectionError, OSError):
                pass
            await asyncio.sleep(3)
        else:
            return False, "daemon did not become healthy"
        reply = ""
        try:
            async for event in client.stream("chat", text="State your name in one short sentence.", cwd=str(Path.cwd())):
                if event.get("type") == "token":
                    reply += event.get("text", "")
                if event.get("type") == "done":
                    break
        except (FileNotFoundError, ConnectionError, OSError) as exc:
            return False, f"inference failed: {exc}"
        if "kilo" in reply.lower():
            return True, "identity confirmed"
        return False, f"unexpected reply: {reply[:80]!r}"

    return asyncio.run(probe())


async def async_main(args: argparse.Namespace, settings: Settings) -> int:
    client = RPCClient(settings.socket_path)
    if args.command is None:
        # Prefer the full-screen interface; fall back to the streaming line UI when
        # prompt_toolkit is missing or there is no real terminal (piped, dumb term).
        if sys.stdout.isatty() and not os.environ.get("KILO_SIMPLE_TUI"):
            try:
                from .tui_full import run_full_tui
                if await run_full_tui(client):
                    return 0
            except ImportError:
                pass
        await TerminalUI(client).run()
        return 0
    if args.command == "chat":
        await TerminalUI(client).ask(" ".join(args.text))
    elif args.command == "status":
        print_status(await client.request("status"))
    elif args.command == "resources":
        try:
            json_print(await client.request("resources"))
        except (FileNotFoundError, ConnectionError):
            json_print(ResourceManager(settings).profile().to_dict())
    elif args.command == "model-info":
        data = {"repository": MODEL_REPOSITORY, "quantization": MODEL_QUANTIZATION, "path": str(settings.model_path), "sha256": MODEL_SHA256, "installed": settings.model_path.is_file()}
        try:
            data["runtime"] = runtime_summary(await client.request("model_info"))
        except (FileNotFoundError, ConnectionError):
            data["runtime"] = None
        json_print(data)
    elif args.command == "version":
        print(f"Kilobyte {__version__}")
        try:
            metadata = await client.request("model_info")
            if metadata:
                print(f"runtime model: {metadata.get('model_alias', metadata.get('model_path', 'loaded'))}")
        except (FileNotFoundError, ConnectionError):
            pass
    elif args.command == "doctor":
        checks = await asyncio.to_thread(run_checks, settings, args.verify_model)
        for check in checks:
            icon = f"{GREEN}PASS" if check.ok else f"{YELLOW}{'WARN' if check.warning else 'FAIL'}"
            print(f"{icon}{RESET}  {check.name:<20} {check.detail}")
        return 0 if all(item.ok or item.warning for item in checks) else 1
    elif args.command == "benchmark":
        started = time.monotonic()
        count = 0
        async for event in client.stream("chat", text=args.prompt, cwd=str(Path.cwd())):
            if event.get("type") == "token":
                piece = event.get("text", "")
                count += len(piece.split())
                print(piece, end="", flush=True)
        elapsed = time.monotonic() - started
        print(f"\n\n{count} approximate word-tokens in {elapsed:.2f}s ({count / max(elapsed, 0.001):.2f}/s end-to-end)")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    if args.command in {"start", "stop", "restart"}:
        raise SystemExit(service_action(args.command))
    if args.command == "logs":
        raise SystemExit(subprocess.run(["journalctl", "-u", "kilobyte.service", "-n", str(args.lines), "--no-pager"], check=False).returncode)
    if args.command == "brain":
        # Local filesystem operation; it does not need the daemon.
        raise SystemExit(brain_command(args, settings))
    if args.command == "telegram":
        raise SystemExit(telegram_command(args, settings))
    try:
        raise SystemExit(asyncio.run(async_main(args, settings)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Quitting the TUI (Ctrl-Q/Ctrl-C) must exit cleanly, not dump a traceback.
        raise SystemExit(0) from None
    except (FileNotFoundError, ConnectionRefusedError):
        if args.command == "status":
            print_status(None)
            raise SystemExit(2) from None
        print(f"{YELLOW}Kilobyte daemon is not running.{RESET} Try: sudo systemctl start kilobyte", file=sys.stderr)
        raise SystemExit(2) from None
    except KilobyteError as exc:
        print(f"{YELLOW}Kilobyte error:{RESET} {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
