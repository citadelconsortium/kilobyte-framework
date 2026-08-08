"""Kilo's terminal interface.

A dependency-free streaming TUI. Presentation is kept out of the control flow: colours,
glyphs and box characters live in ``theme``, Markdown formatting lives in ``render``, and
the panel border is one reusable component here, so the interface reads as one system
rather than formatting scattered through f-strings.

The activity indicator shows a rotating human-readable phase (thinking, reasoning,
working) with a spinner and an elapsed clock, redrawn on a timer so a slow step never
looks frozen. It never shows a raw step counter and never exposes model reasoning.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .render import MarkdownStream
from .rpc import RPCClient
from .theme import (
    ACTIVITY_WORDS,
    BOLD,
    Box,
    CLOUD,
    CYAN,
    DIM,
    DOT_OFF,
    DOT_ON,
    FAIL,
    GREEN,
    OK,
    PURPLE,
    RESET,
    SPINNER,
    TOOL,
    WARN,
    YELLOW,
    visible_len,
)

# Kept for callers that import colours from here (cli.py).
__all__ = ["TerminalUI", "CYAN", "DIM", "GREEN", "RESET", "YELLOW"]

# 3D shadow-block "KILO" wordmark, shown beside the live status in the header.
KILO_ART = (
    "██╗  ██╗██╗██╗      ██████╗ ",
    "██║ ██╔╝██║██║     ██╔═══██╗",
    "█████╔╝ ██║██║     ██║   ██║",
    "██╔═██╗ ██║██║     ██║   ██║",
    "██║  ██╗██║███████╗╚██████╔╝",
    "╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ",
)


class TerminalUI:
    def __init__(self, client: RPCClient):
        self.client = client
        self.session_id: str | None = None
        self.model_name: str | None = None
        # Set only for the next message, by /cloud. Escalation is never sticky.
        self.provider: str | None = None

    # ---- layout primitives -------------------------------------------------

    @staticmethod
    def _width() -> int:
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            # No controlling terminal (piped, service context): assume a sane width.
            cols = 80
        return max(48, min(cols, 100))

    def _rule_top(self, label: str, color: str) -> None:
        width = self._width()
        head = f"{Box.h} {label} "
        print(f"{color}{Box.tl}{head}{Box.h * max(0, width - visible_len(head) - 2)}{Box.tr}{RESET}")

    def _rule_bottom(self, color: str, note: str = "") -> None:
        width = self._width()
        tail = f" {note} {Box.h}" if note else ""
        print(f"{color}{Box.bl}{Box.h * max(0, width - visible_len(tail) - 2)}{tail}{Box.br}{RESET}")

    def _line(self, color: str, body: str) -> None:
        """A single framed line, padded to the panel width."""
        width = self._width()
        pad = max(0, width - 4 - visible_len(body))
        print(f"{color}{Box.v}{RESET} {body}{' ' * pad} {color}{Box.v}{RESET}")

    def _panel(self, label: str, color: str, lines: list[str]) -> None:
        self._rule_top(label, color)
        for body in lines:
            self._line(color, body)
        self._rule_bottom(color)
        print()

    # ---- header ------------------------------------------------------------

    async def banner(self) -> None:
        online = True
        model_name = "unknown"
        context_size = threads = gpu_layers = "?"
        try:
            status = await self.client.request("status")
            profile = status.get("profile") or {}
            model_name = Path(str(status.get("model", ""))).stem or "unknown"
            self.model_name = model_name
            context_size = profile.get("context_size", "?")
            threads = profile.get("threads", "?")
            gpu_layers = profile.get("gpu_layers", "?")
        except (ConnectionError, FileNotFoundError, OSError):
            online = False

        dot = DOT_ON if online else DOT_OFF
        state = f"{GREEN}online{RESET}" if online else f"{YELLOW}offline{RESET}"
        info = [
            f"{BOLD}{GREEN}KILOBYTE{RESET}  {dot} {state}",
            f"{DIM}local-first · one model · no cloud by default{RESET}",
            f"{DIM}brain   {RESET}{model_name}" if online else f"{YELLOW}sudo systemctl start kilobyte{RESET}",
            f"{DIM}context {RESET}{context_size}  {DIM}threads {RESET}{threads}  {DIM}gpu {RESET}{gpu_layers}" if online else "",
            f"{DIM}tools   {RESET}files · shell · web · memory · skills" if online else "",
            f"{DIM}made by 0v3r51ght{RESET}",
        ]
        width = self._width()
        art_width = visible_len(KILO_ART[0])
        # On a terminal too narrow for the wordmark and status side by side, stack the
        # status below the wordmark instead of letting it overflow the border.
        stacked = width - 4 < art_width + 3 + 24
        print()
        print(f"  {GREEN}{Box.tl}{Box.h * (width - 4)}{Box.tr}{RESET}")
        if stacked:
            rows = [f"{GREEN}{BOLD}{art}{RESET}" for art in KILO_ART] + [m for m in info if m]
        else:
            padded = info + [""] * (len(KILO_ART) - len(info))
            rows = [f"{GREEN}{BOLD}{art}{RESET}   {self._fit(meta, width - 6 - art_width - 3)}" for art, meta in zip(KILO_ART, padded, strict=True)]
        rows.append(f"{DIM}/help  /status  /new  /cloud  /clear  /exit{RESET}")
        for body in rows:
            pad = max(0, width - 6 - visible_len(body))
            print(f"  {GREEN}{Box.v}{RESET} {body}{' ' * pad} {GREEN}{Box.v}{RESET}")
        print(f"  {GREEN}{Box.bl}{Box.h * (width - 4)}{Box.br}{RESET}\n")

    @staticmethod
    def _fit(text: str, limit: int) -> str:
        """Truncate styled text to a visible width, so it never overruns a border."""
        if visible_len(text) <= limit or limit <= 1:
            return text
        # Strip styling and cut; the trailing reset keeps later output clean.
        from .theme import _ANSI
        plain = _ANSI.sub("", text)
        return plain[: max(0, limit - 1)] + "…"

    # ---- activity indicator ------------------------------------------------

    async def _animate(self, state: dict[str, Any]) -> None:
        """Redraw the activity line on a timer. The phase word rotates while the model
        works, which reads as a live operator rather than a stuck counter."""
        frame = 0
        while True:
            if state["streaming"]:
                await asyncio.sleep(0.1)
                continue
            now = time.monotonic()
            elapsed = now - state["started"]
            glyph = SPINNER[frame % len(SPINNER)]
            phase = state["phase"] or ACTIVITY_WORDS[(frame // 12) % len(ACTIVITY_WORDS)]
            # Trailing dots drift under the word so a long, quiet step still looks alive.
            dots = ("." * ((frame // 3) % 4)).ljust(3)
            model = state.get("model") or "local brain"
            sys.stdout.write(
                f"\r\033[2K{GREEN}{Box.v}{RESET} {GREEN}{glyph}{RESET} {BOLD}{phase}{RESET}{DIM}{dots}{RESET}"
                f" {DIM}{elapsed:0.0f}s · {model}{RESET}  {DIM}(ctrl-c to cancel){RESET}"
            )
            sys.stdout.flush()
            frame += 1
            await asyncio.sleep(0.1)

    async def _permission(self, event: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        prompt = f"\n{YELLOW}Permission required [{event['risk']}]:{RESET} {event['detail']}\nAllow once? [y/N] "
        answer = await asyncio.to_thread(input, prompt)
        writer.write((json.dumps({"type": "permission_response", "id": event["id"], "allow": answer.lower() in {"y", "yes"}}) + "\n").encode())
        await writer.drain()

    # ---- the exchange ------------------------------------------------------

    def _gutter(self, text: str) -> str:
        """Prefix every line of already-formatted text with the panel border."""
        return text.replace("\n", f"\n{GREEN}{Box.v}{RESET} ")

    async def ask(self, text: str) -> None:
        reader, writer = await asyncio.open_unix_connection(self.client.socket_path)
        request: dict[str, Any] = {"command": "chat", "text": text, "session_id": self.session_id, "cwd": str(Path.cwd())}
        if self.provider is not None:
            request["provider"] = self.provider
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()

        started = time.monotonic()
        state: dict[str, Any] = {"phase": "", "started": started, "streaming": False, "model": self.model_name}
        animator = asyncio.create_task(self._animate(state))
        markdown = MarkdownStream()
        printed = False       # true once the answer body has started printing
        at_line_start = True  # cursor sits after a gutter, ready for content
        cancelled = False
        tool_started = started
        tools_used: list[str] = []
        first_token_at: float | None = None

        def emit(formatted: str) -> None:
            """Write formatted text into the panel, keeping the border gutter."""
            nonlocal at_line_start
            if not formatted:
                return
            sys.stdout.write(self._gutter(formatted))
            at_line_start = formatted.endswith("\n")

        try:
            while raw := await reader.readline():
                event = json.loads(raw)
                kind = event.get("type")

                if kind == "session":
                    self.session_id = event["session_id"]
                    state["phase"] = "thinking"
                elif kind == "brain":
                    state["model"] = event.get("label")
                    if event.get("location") == "cloud":
                        sys.stdout.write(f"\r\033[2K{GREEN}{Box.v}{RESET} {CLOUD} {YELLOW}escalated to {event.get('label')}{RESET}\n")
                elif kind == "warming":
                    state["phase"] = "warming the model cache"
                    sys.stdout.write(f"\r\033[2K{GREEN}{Box.v}{RESET} {YELLOW}first run after a change: warming the prompt cache, once-off{RESET}\n")
                elif kind == "thinking":
                    # No step number: the animator's rotating word carries the sense of
                    # progress, and a counter that only ever reached 1 read as stuck.
                    state["phase"] = "thinking"
                    state["streaming"] = False
                elif kind == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    if not printed:
                        state["streaming"] = True
                        sys.stdout.write(f"\r\033[2K{GREEN}{Box.v}{RESET} ")
                        printed = True
                        at_line_start = True
                    emit(markdown.feed(event.get("text", "")))
                    sys.stdout.flush()
                elif kind == "tool_start":
                    emit(markdown.flush())
                    if not at_line_start:
                        sys.stdout.write("\n")
                    tool_started = time.monotonic()
                    tools_used.append(event["name"])
                    state["phase"] = f"running {event['name']}"
                    state["streaming"] = False
                    args = event.get("arguments") or {}
                    detail = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
                    sys.stdout.write(f"\r\033[2K{GREEN}{Box.v}{RESET} {TOOL} {BOLD}{event['name']}{RESET}{(' ' + DIM + detail + RESET) if detail else ''}\n")
                    sys.stdout.flush()
                    printed = False
                    at_line_start = True
                elif kind == "tool_end":
                    took = time.monotonic() - tool_started
                    icon = OK if event.get("ok") else WARN
                    summary = str(event.get("summary", ""))[:140]
                    print(f"\r\033[2K{GREEN}{Box.v}{RESET}   {icon} {DIM}{event['name']} · {took:0.1f}s · {summary}{RESET}")
                    state["phase"] = "interpreting result"
                    state["streaming"] = False
                    printed = False
                    at_line_start = True
                elif kind == "permission":
                    state["streaming"] = True
                    await self._permission(event, writer)
                    state["streaming"] = False
                elif kind == "error":
                    emit(markdown.flush())
                    if not at_line_start:
                        sys.stdout.write("\n")
                    print(f"\r\033[2K{GREEN}{Box.v}{RESET} {FAIL} {YELLOW}{event.get('error')}{RESET}")
                    at_line_start = True
                elif kind == "done":
                    break

            emit(markdown.flush())
            sys.stdout.flush()
        except (asyncio.CancelledError, KeyboardInterrupt):
            cancelled = True
            raise
        finally:
            animator.cancel()
            # Dropping the connection tells the daemon to close the run and free the slot.
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            # Only clear the line when it still holds the spinner; clearing after streamed
            # content would erase the answer's last line (which has no trailing newline).
            if printed and not at_line_start:
                sys.stdout.write(f"{RESET}\n")
            elif printed:
                sys.stdout.write(RESET)
            else:
                sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
            elapsed = time.monotonic() - started
            if cancelled:
                self._rule_bottom(YELLOW, f"cancelled after {elapsed:0.0f}s")
            else:
                parts = [f"{elapsed:0.0f}s"]
                if first_token_at is not None:
                    parts.append(f"first token {first_token_at - started:0.1f}s")
                if tools_used:
                    parts.append(f"tools: {', '.join(dict.fromkeys(tools_used))}")
                self._rule_bottom(GREEN, " · ".join(parts))
            print()

    # ---- command loop ------------------------------------------------------

    async def _run_ask(self, text: str) -> None:
        """Run one exchange as a task so SIGINT cancels the generation, not the app."""
        self._rule_top("kilo", GREEN)
        task = asyncio.create_task(self.ask(text))
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        except (NotImplementedError, RuntimeError):
            loop = None
        try:
            await task
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"{DIM}generation cancelled — type /exit to leave{RESET}\n")
        except (ConnectionError, FileNotFoundError) as exc:
            print(f"{YELLOW}Daemon unavailable:{RESET} {exc}\nTry: sudo systemctl restart kilobyte")
            self._rule_bottom(YELLOW, "not delivered")
            print()
        finally:
            if loop is not None:
                loop.remove_signal_handler(signal.SIGINT)

    async def run(self) -> None:
        await self.banner()
        while True:
            try:
                self._rule_top("you", CYAN)
                text = (await asyncio.to_thread(input, f"{CYAN}{Box.v}{RESET} ")).strip()
                self._rule_bottom(CYAN)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}bye{RESET}")
                return
            if not text:
                continue
            if text in {"/exit", "/quit", "/q", "exit", "quit"}:
                print(f"{DIM}bye{RESET}")
                return
            if text == "/new":
                self.session_id = None
                self._panel("session", PURPLE, [f"{DIM}new session started; previous context is not carried over{RESET}"])
                continue
            if text == "/help":
                self._panel("commands", PURPLE, [
                    f"{GREEN}{'/new':<8}{RESET} {DIM}start a separate session{RESET}",
                    f"{GREEN}{'/status':<8}{RESET} {DIM}show daemon, model and resource status{RESET}",
                    f"{GREEN}{'/cloud':<8}{RESET} {DIM}send one message to a configured cloud model{RESET}",
                    f"{GREEN}{'/clear':<8}{RESET} {DIM}clear the screen and redraw the header{RESET}",
                    f"{GREEN}{'/exit':<8}{RESET} {DIM}leave Kilobyte{RESET}",
                    f"{DIM}anything else is sent to the local brain · ctrl-c cancels{RESET}",
                ])
                continue
            if text == "/clear":
                sys.stdout.write("\033[2J\033[H")
                await self.banner()
                continue
            if text.startswith("/cloud"):
                parts = text.split(maxsplit=2)
                named = parts[1] if len(parts) > 1 and not parts[1].startswith("/") else ""
                question = parts[2] if len(parts) > 2 else ""
                if not question and len(parts) == 2:
                    named, question = "", parts[1]
                if not question:
                    self._panel("cloud", YELLOW, [
                        "usage: /cloud [provider] <question>",
                        f"{DIM}sends this one message to a configured cloud model{RESET}",
                        f"{DIM}local stays the default; nothing escalates automatically{RESET}",
                    ])
                    continue
                self.provider = named or ""
                try:
                    await self._run_ask(question)
                finally:
                    self.provider = None
                continue
            if text == "/status":
                try:
                    status = await self.client.request("status")
                except (ConnectionError, FileNotFoundError, OSError) as exc:
                    self._panel("status", YELLOW, [f"daemon unavailable: {exc}"])
                    continue
                profile = status.get("profile") or {}
                memory = status.get("memory") or {}
                self._panel("status", PURPLE, [
                    f"{DIM}{'healthy':<9}{RESET}{status.get('healthy')}",
                    f"{DIM}{'uptime':<9}{RESET}{status.get('uptime_seconds', 0)}s",
                    f"{DIM}{'model':<9}{RESET}{Path(str(status.get('model', ''))).name}",
                    f"{DIM}{'context':<9}{RESET}{profile.get('context_size')}   threads {profile.get('threads')}   gpu {profile.get('gpu_layers')}",
                    f"{DIM}{'memory':<9}{RESET}{profile.get('available_mb')} MiB of {profile.get('total_mb')} MiB free",
                    f"{DIM}{'sessions':<9}{RESET}{memory.get('sessions')} · {memory.get('messages')} messages · {memory.get('facts')} facts",
                ])
                continue
            await self._run_ask(text)
