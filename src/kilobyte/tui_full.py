"""Full-screen Kilo terminal application.

A persistent layout that fills the window: a banner on top, a live stats bar, a scrollable
conversation that streams character by character, a runtime panel toggled with F2, and an
input box fixed at the bottom. The stats bar shows what Kilo is doing plus live numeric
counters — elapsed runtime, tools used, and tokens produced.

Everything visible animates so the interface always reads as alive: a light sweeps across
the wordmark, the status dot breathes, the activity glyph and word rotate with trailing
dots while Kilo works, and an idle wave drifts when it is not. There is no step counter —
a raw number that usually only reached "1" read as frozen.

Built on prompt_toolkit's widgets (TextArea) so input focus and scrolling are handled
robustly. When prompt_toolkit or a real terminal is unavailable, cli.py falls back to the
streaming line-based UI, so nothing here is a hard requirement.

Inference happens in the daemon over the Unix socket; this process only renders and
forwards keystrokes, so the interface stays responsive while a reply streams.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .rpc import RPCClient

KILO_ART = (
    "██╗  ██╗██╗██╗      ██████╗ ",
    "██║ ██╔╝██║██║     ██╔═══██╗",
    "█████╔╝ ██║██║     ██║   ██║",
    "██╔═██╗ ██║██║     ██║   ██║",
    "██║  ██╗██║███████╗╚██████╔╝",
    "╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ",
)

# Kilo's README mascot, redrawn in terminal-safe pixels. It is a little green
# machine rather than a generic eye: the shell breathes, pupils scan, and the
# eyelids blink from the shared animation tick.
MASCOT_OPEN = (
    "    ▄▄▄▄▄    ", "   ▟█████▙   ", " ▄██●▓●██▄ ",
    "▐██▓ ▾ ▓██▌", " ▜██▓═▓██▛ ", "  ▀██▓██▀  ",
)
MASCOT_BLINK = (
    "    ▄▄▄▄▄    ", "   ▟█████▙   ", " ▄██━▓━██▄ ",
    "▐██▓ ▾ ▓██▌", " ▜██▓═▓██▛ ", "  ▀██▓██▀  ",
)

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"       # default "thinking" spinner
MOON = "◐◓◑◒"                    # a tool is running: a turning quarter
ELLIPSIS = ("·  ", "·· ", "···", " ··", "  ·")  # "interpreting" drift
PULSE = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
ACTIVITY = ("thinking", "reasoning", "planning", "working", "composing", "considering")

# Which glyph set animates for a given activity phase. Falls back to the braille spinner.
def _phase_frames(phase: str) -> str:
    if phase.startswith(("running", "warming")):
        return MOON
    return SPINNER


STYLE = Style.from_dict({
    "banner": "#3fa869 bold",
    "banner.hi": "#d7ffd7 bold",
    "evil": "#ff5f5f bold",   # the butler's eyes   # the bright band that sweeps across the wordmark
    "tagline": "#8a8a8a",
    "on": "#5fd787 bold",
    "off": "#ffd75f bold",
    "sep": "#3a3a3a",
    "stat": "#5fd787",
    "stat.k": "#8a8a8a",
    "you": "#5fafff bold",
    "kilo": "#5fd787",
    "dim": "#8a8a8a",
    "warn": "#ffd75f",
    "err": "#ff5f5f",
    "panel.title": "#af87ff bold",
    "panel.key": "#8a8a8a",
    "output": "bg:#0a0a0a",
    "box": "#3fa869",
    "box.you": "#5fafff bold",
    "box.kilo": "#5fd787 bold",
    "diff.add": "#5fd787",
    "diff.del": "#ff5f5f",
    "code": "#87d7ff",
    "toolline": "#8a8a8a",
    "prompt": "#5fafff bold",
})


_COMMANDS = [
    ("/commands", "show every TUI command"),
    ("/help", "show command help"),
    ("/effort ", "high | medium | low — reply depth vs speed"),
    ("/agent ", "force a specialist: orchestrator research coding security systems private"),
    ("/chats", "list past sessions to resume"),
    ("/kilochats", "browse past chats; type a number to continue one"),
    ("/cloud key", "add or change a provider API key"),
    ("/botkey", "set or change the Telegram bot token"),
    ("/cloud ", "set up or use a cloud model (provider picker)"),
    ("/cloudswitch", "select or reconfigure the active cloud provider"),
    ("/switch", "flip between cloud and local Kilo (Kilo default)"),
    ("/model ", "change the cloud model"),
    ("/chat ", "open a past chat by number"),
    ("/kchats", "browse past chats (alias for /kilochats)"),
    ("/gguf", "browse downloaded .gguf files and load one as the brain"),
    ("/private ", "on | off | rotate — mask web through Tor"),
    ("/cancel", "stop the running request and clear the queue"),
    ("/new", "start a fresh session"),
    ("/clear", "clear the screen"),
    ("/help", "list commands"),
    ("/quit", "exit Kilo"),
    ("/exit", "exit Kilo (alias)"),
    ("/q", "exit Kilo (alias)"),
]


class _ChatLexer(Lexer):
    """Colours the conversation: turn borders, +/- diff lines, destructive warnings,
    tool lines, and code inside ``` fences."""

    def lex_document(self, document):
        lines = document.lines

        def get_line(lineno):
            line = lines[lineno]
            stripped = line.strip()
            if stripped and set(stripped) <= set("\u2500") or line.startswith("\u2500\u2500\u2500"):
                cls = "class:box.you" if " Sir " in line or line.startswith("\u2500\u2500\u2500Sir") else (
                    "class:box.kilo" if ("Kilo" in line or "\u2601" in line) else "class:box")
                return [(cls, line)]
            body = line[2:] if line.startswith("\u2502 ") else line
            b = body.lstrip()
            fences = 0
            for i in range(lineno):
                if lines[i].lstrip("\u2502 ").startswith("```"):
                    fences += 1
            if "\u26a0" in line or "destructive" in body.lower():
                return [("class:diff.del", line)]
            if b.startswith("+") and not b.startswith("+++"):
                return [("class:diff.add", line)]
            if b.startswith("-") and not b.startswith("---"):
                return [("class:diff.del", line)]
            if b.startswith(("\u25c8", "\u2713", "!")):
                return [("class:toolline", line)]
            if fences % 2 == 1 or b.startswith("```"):
                return [("class:code", line)]
            return [("", line)]

        return get_line


class _SlashCompleter(Completer):
    """Pops up a menu of / commands as soon as the line starts with a slash."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _COMMANDS:
            if cmd.startswith(text) or cmd.strip() == text.strip():
                yield Completion(cmd, start_position=-len(text), display=cmd.strip(), display_meta=desc)


class KiloApp:
    def __init__(self, client: RPCClient):
        self.client = client
        self.session_id: str | None = None
        self.model_name = "local brain"
        self.status: dict[str, Any] = {}

        self.busy = False          # a request is in flight
        self.streaming = False     # tokens are currently arriving
        self.phase = ""
        self.started = 0.0
        self.spin = 0
        # Live numeric counters for the stats bar.
        self.tokens = 0
        self.tools_used = 0
        self.show_panel = False
        self.effort = "medium"
        self.agent_name = ""          # profile active this turn (auto-selected or forced)
        self.forced_profile = ""      # set by /agent; overrides auto-selection
        self._sessions: list[dict[str, Any]] = []
        self._active: asyncio.Task | None = None
        # Cloud escalation state. Local Kilo is always the default; /switch flips the
        # active brain to the last-configured cloud provider and back.
        self.cloud_active = False
        self.cloud_provider = ""
        self.private_mode = False   # route web tools through Tor
        self.current_task = ""      # text of the request being worked on
        self._model_options: list[str] = []
        self._gguf_options: list[str] = []
        self._perm_future: asyncio.Future | None = None   # resolved by 1/2/3 approval input
        self._line_buf = ""   # accumulates a streamed line until it can be boxed
        self.usage: dict[str, Any] = {}   # token usage from the last reply
        self._answered = False            # whether the current reply has started printing
        self._pending: dict[str, Any] | None = None   # awaited inline input (selector / key)
        self._catalog: dict[str, Any] = {}
        self._cloud_options: list[tuple[str, dict[str, Any]]] = []
        # Strong refs to spawned background tasks. Without this, asyncio can garbage-
        # collect a task that is awaiting (e.g. an RPC round-trip) and silently cancel
        # it mid-run — which is why the /cloud setup appeared to do nothing.
        self._bg_tasks: set[asyncio.Task] = set()
        self._queue: asyncio.Queue | None = None   # pending (text, provider) requests
        self._worker: asyncio.Task | None = None

        self.output = TextArea(
            text="", read_only=True, scrollbar=True, wrap_lines=True,
            focusable=False, style="class:output", lexer=_ChatLexer(),
        )
        self.input = TextArea(
            height=1, multiline=False, wrap_lines=False, prompt=self._input_prompt,
            style="class:prompt", accept_handler=self._accept,
            completer=_SlashCompleter(), complete_while_typing=True,
        )
        self._build_layout()

    def _cw(self) -> int:
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except Exception:
            cols = 80
        if self.show_panel:
            cols -= 31
        return max(20, cols - 2)

    def _rule(self, label: str = "") -> str:
        w = self._cw()
        if label:
            head = "\u256d\u2500 " + label + " "
            return head + "\u2500" * max(0, w - len(head) - 1) + "\u256e"
        return "\u2570" + "\u2500" * max(0, w - 2) + "\u256f"

    def _bline(self, text: str = "") -> None:
        """Emit one fully-closed box line: | text ... | padded to the box width."""
        inner = max(1, self._cw() - 3)
        self._append("\u2502 " + text[:inner].ljust(inner) + "\u2502\n")

    def _stream_boxed(self, text: str) -> None:
        """Buffer streamed tokens and emit complete closed-box lines as they fill."""
        inner = max(1, self._cw() - 3)
        self._line_buf += text
        while True:
            nl = self._line_buf.find("\n")
            if nl >= 0:
                line, self._line_buf = self._line_buf[:nl], self._line_buf[nl + 1:]
                self._bline(self._clean_md(line))
            elif len(self._line_buf) >= inner:
                self._bline(self._line_buf[:inner])
                self._line_buf = self._line_buf[inner:]
            else:
                break

    def _clean_md(self, line: str) -> str:
        """Strip markdown noise (**, *, #, >) so replies read clean; the lexer colours
        the rest. Runs per already-buffered line, so no newlines to worry about."""
        s = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
        s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', s)
        s = re.sub(r'^(\s{0,3})#{1,6}\s+', r'\1', s)
        s = re.sub(r'^(\s{0,3})>\s?', r'\1', s)
        return s

    def _flush_boxed(self) -> None:
        if self._line_buf:
            self._bline(self._clean_md(self._line_buf))
            self._line_buf = ""

    def _input_prompt(self):
        if self.busy:
            glyph = PULSE[self.spin % len(PULSE)]
        else:
            glyph = "\u203a" if (self.spin // 6) % 2 else "\u276f"
        return [("class:prompt", f"  {glyph} ")]

    # ---- layout -------------------------------------------------------------

    def _shimmer(self, art: str, row: int):
        """Split one wordmark line into segments with a bright band that sweeps across,
        giving the logo a diagonal light-sweep. Cheap: a handful of short segments."""
        span = len(art) + 14  # sweep a little past both edges so there is a brief pause
        head = self.spin % span
        out: list[tuple[str, str]] = [("class:banner", "  ")]
        for col, ch in enumerate(art):
            # Diagonal: the band leads on lower rows, so the highlight tilts as it moves.
            lit = ch != " " and abs(col - (head - row)) <= 1
            out.append(("class:banner.hi" if lit else "class:banner", ch))
        out.append(("class:banner", "   "))
        return out

    @staticmethod
    def _seg_len(segs) -> int:
        return sum(len(txt) for _, txt in segs)

    def _mascot(self, row: int):
        """One row of the pixel eyeball: open most of the time, a quick blink now and
        then, with the pupil glinting red on a beat."""
        blink = (self.spin % 44) < 4
        frames = MASCOT_BLINK if blink else MASCOT_OPEN
        art = (frames[row] if row < len(frames) else "").ljust(13)
        glint = (self.spin // 4) % 6 == 0
        segs = []
        for ch in art:
            if ch == "●":
                segs.append(("class:banner.hi" if glint else "class:evil", ch))
            elif ch == "\u2593":                 # ▓ iris
                segs.append(("class:banner.hi", ch))
            elif ch in {"━", "▾"}:
                segs.append(("class:evil", ch))
            else:
                segs.append(("class:banner", ch))
        return segs

    def _banner_text(self):
        online = bool(self.status.get("healthy"))
        prof = self.status.get("profile") or {}
        # A breathing dot animates even when idle, so the header never looks frozen.
        pulse = PULSE[self.spin % len(PULSE)]
        dot = f"{pulse} online" if online else "○ offline"
        info = [
            [("class:banner.hi", "KILOBYTE  "), ("class:on" if online else "class:off", dot)],
            [("class:tagline", "local-first · one model · no cloud by default")],
            [("class:on", f"brain   {self.model_name}")],
            [("class:dim", f"context {prof.get('context_size','?')}   threads {prof.get('threads','?')}   gpu {prof.get('gpu_layers','?')}")],
            [("class:dim", "tools   files · shell · web · memory · skills")],
            [("class:tagline", "made by 0v3r51ght  ·  /help · F2 runtime · Ctrl-Q quit")],
        ]
        rows: list[tuple[str, str]] = []
        show_mascot = False  # banner intentionally stays mascot-free
        for i, art in enumerate(KILO_ART):
            rows += self._shimmer(art, i)
            line = info[i] if i < len(info) else []
            rows += line
            if show_mascot:
                # Keep the animated mascot on the actual right edge, not at a
                # hard-coded column that drifts with terminal size/font width.
                pad = max(2, self._cw() - len(art) - self._seg_len(line) - 13 - 4)
                rows.append(("class:banner", " " * pad))
                rows += self._mascot(i)
            rows.append(("", "\n"))
        return rows

    def _stats_bar(self):
        elapsed = (time.monotonic() - self.started) if (self.busy and self.started) else 0
        if self.busy:
            phase = self.phase or ACTIVITY[(self.spin // 10) % len(ACTIVITY)]
            if self.streaming:
                # A blinking caret shows tokens are actively arriving.
                caret = "▌" if (self.spin // 3) % 2 else " "
                head = [("class:stat", " ▌ "), ("class:kilo", "responding"), ("class:stat", caret)]
            else:
                frames = _phase_frames(phase)
                glyph = frames[self.spin % len(frames)]
                dots = ELLIPSIS[(self.spin // 3) % len(ELLIPSIS)]
                head = [("class:stat", f" {glyph} "), ("class:kilo", phase), ("class:dim", f" {dots}")]
        else:
            # A gentle wave drifts while idle so the bar is never static.
            wave = "".join(PULSE[(self.spin + i) % len(PULSE)] for i in range(3))
            head = [("class:stat", f" {wave} "), ("class:dim", "ready")]
        bar = head + [
            ("class:stat.k", "   ⏱ "), ("class:stat", f"{elapsed:0.0f}s"),
            ("class:stat.k", "   ↗ requests "), ("class:stat", str((self.status.get("memory") or {}).get("requests", 0))),
            ("class:stat.k", "   🔧 tools "), ("class:stat", f"{self.tools_used}"),
            ("class:stat.k", "   ⇥ tokens "), ("class:stat", f"{self.tokens}"),
            ("class:stat.k", "   effort "), ("class:stat", f"{self.effort}"),
            ("class:stat.k", "   ⬡ "),
            ("class:kilo", self._short_model()),
        ]
        if self.agent_name:
            bar += [("class:stat.k", "   ◆ "), ("class:kilo", self.agent_name)]
        qn = self._queue.qsize() if getattr(self, "_queue", None) else 0
        if qn:
            bar += [("class:stat.k", "   ⧉ queued "), ("class:stat", f"{qn}")]
        if self.private_mode:
            bar += [("class:stat.k", "   🛡 "), ("class:kilo", "private")]
        ctx = (self.status.get("profile") or {}).get("context_size")
        if ctx:
            used = (self.usage or {}).get("total_tokens")
            cloud_ctx = self.status.get("cloud_context_limit")
            if self.cloud_active and cloud_ctx:
                used = int((self.usage or {}).get("total_tokens") or 0)
                ratio = min(1.0, used / max(1, int(cloud_ctx)))
                filled = round(ratio * 8)
                meter = "".join("█" if i < filled else "░" for i in range(8))
                bar += [("class:stat.k", "   ▤ ctx "), ("class:stat", f"{meter} {used}/{cloud_ctx}")]
            elif self.cloud_active:
                bar += [("class:stat.k", "   ▤ ctx "), ("class:stat", "cloud ?")]
            else:
                used = int(used or 0)
                ratio = min(1.0, used / max(1, int(ctx)))
                filled = round(ratio * 8)
                meter = "".join("█" if i < filled else "░" for i in range(8))
                bar += [("class:stat.k", "   ▤ ctx "), ("class:stat", f"{meter} {used}/{ctx}")]
        return bar

    def _panel_text(self):
        prof = self.status.get("profile") or {}
        mem = self.status.get("memory") or {}
        return [
            ("class:panel.title", " RUNTIME\n\n"),
            ("class:panel.key", " model    "), ("", f"{self.model_name}\n"),
            ("class:panel.key", " healthy  "), ("", f"{self.status.get('healthy')}\n"),
            ("class:panel.key", " uptime   "), ("", f"{self.status.get('uptime_seconds',0)}s\n"),
            ("class:panel.key", " context  "), ("", f"{prof.get('context_size','?')}\n"),
            ("class:panel.key", " threads  "), ("", f"{prof.get('threads','?')}\n"),
            ("class:panel.key", " gpu      "), ("", f"{prof.get('gpu_layers','?')} layers\n"),
            ("class:panel.key", " memory   "), ("", f"{prof.get('available_mb','?')} MiB\n\n"),
            ("class:panel.title", " THIS TURN\n\n"),
            ("class:panel.key", " reply tok"), ("", f" {self.tokens}\n"),
            ("class:panel.key", " prompt   "), ("", f"{(self.usage or {}).get('prompt_tokens','-')}\n"),
            ("class:panel.key", " total    "), ("", f"{(self.usage or {}).get('total_tokens','-')}\n"),
            ("class:panel.key", " ctx limit"), ("", f" {prof.get('context_size','?')}\n"),
            ("class:panel.key", " tools    "), ("", f"{self.tools_used}\n\n"),
            ("class:panel.title", " MEMORY\n\n"),
            ("class:panel.key", " sessions "), ("", f"{mem.get('sessions','?')}\n"),
            ("class:panel.key", " facts    "), ("", f"{mem.get('facts','?')}\n"),
        ]

    def _build_layout(self) -> None:
        panel = ConditionalContainer(
            VSplit([
                Window(width=1, char="│", style="class:sep"),
                Window(FormattedTextControl(self._panel_text), width=30),
            ]),
            filter=Condition(lambda: self.show_panel),
        )
        root = HSplit([
            Window(FormattedTextControl(self._banner_text), height=len(KILO_ART)),
            Window(height=1, char="─", style="class:sep"),
            VSplit([self.output, panel]),
            Window(height=1, char="─", style="class:sep"),
            Window(FormattedTextControl(self._stats_bar), height=1),
            Window(height=1, char="─", style="class:sep"),
            self.input,
        ])
        root = FloatContainer(root, floats=[
            Float(xcursor=True, ycursor=True,
                  content=CompletionsMenu(max_height=8, scroll_offset=1)),
        ])
        self.layout = Layout(root, focused_element=self.input)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _worker_loop(self) -> None:
        """Process queued requests one at a time. There is a single inference slot, so
        running requests concurrently clobbers shared state and loses output — which
        looked like 'some messages never responded'. Serialising fixes that."""
        while True:
            text, provider = await self._queue.get()
            self.current_task = text
            try:
                await self._ask(text, provider)
            except Exception as exc:  # noqa: BLE001
                self._append(f"\n⚠ {exc}\n")
            finally:
                self.current_task = ""
                self._queue.task_done()
                self.app.invalidate()

    def _enqueue(self, text: str, provider: str | None = None) -> None:
        inner = max(1, self._cw() - 3)
        self._append("\n" + self._rule("Sir") + "\n")
        for para in text.split("\n"):
            if not para:
                self._bline("")
            while para:
                self._bline(para[:inner])
                para = para[inner:]
        self._append(self._rule() + "\n")
        ahead = (self._queue.qsize() if self._queue else 0) + (1 if self.busy else 0)
        if ahead > 0:
            self._append(f"⏳ queued — {ahead} task(s) ahead\n")
        if self._queue is not None:
            self._queue.put_nowait((text, provider))

    def _append(self, text: str) -> None:
        buff = self.output.buffer
        new = buff.text + text
        buff.set_document(Document(new, len(new)), bypass_readonly=True)

    # ---- interaction --------------------------------------------------------

    def _accept(self, buff) -> bool:
        text = buff.text.strip()
        # Returning False clears the input for the next message.
        if not text:
            return False
        # An awaited answer (cloud provider pick or API key) is consumed here rather than
        # being sent to the model. Returning False also wipes the key from the input line.
        if self._pending is not None:
            if self._pending.get("kind") == "permission" and self._perm_future and not self._perm_future.done():
                self._perm_future.set_result(text)
                self._pending = None
                return False
            self._spawn(self._resume_pending(text))
            return False
        if self._handle_command(text):
            return False
        self._enqueue(text)
        return False

    def _handle_command(self, text: str) -> bool:
        if text in {"/quit", "/exit", "/q", "quit", "exit"}:
            self.app.exit()
            return True
        if text == "/clear":
            self.output.buffer.set_document(Document("", 0), bypass_readonly=True)
            return True
        if text == "/new":
            self.session_id = None
            self.output.buffer.set_document(Document("", 0), bypass_readonly=True)
            self._append("— new session · the previous chat is saved (use /kilochats to reopen it) —\n")
            return True
        if text == "/help":
            self._append(
                "\ncommands:\n"
                "  /effort high|medium|low   depth vs speed of replies\n"
                "  /agent <name>|off         force research|coding|security|systems, or auto\n"
                "  /chats · /kilochats       list past chats; type a number to continue one\n"
                "  /chat <n>                 open a past session by number\n"
                "  /cloud [question]         set up / use a cloud model (key selector)\n"
                "  /botkey [token]          set or change the Telegram bot token\n"
                "  /switch                   flip between cloud and local Kilo (Kilo default)\n"
                "  /private [on|off|rotate]  mask web via Tor — hide IP, rotate exit\n"
                "  /model [name]             show or change the cloud model\n"
                "  /cancel                   stop the running request and clear the queue\n"
                "  /new · /clear · /quit\n"
                "keys: F2 runtime panel · Ctrl-C cancel · Ctrl-Q quit\n"
            )
            return True
        if text == "/chats":
            self._spawn(self._list_chats())
            return True
        if text == "/commands":
            self._append("\ncommands:\n" + "\n".join(f"  {name:<18} {description}" for name, description in _COMMANDS) + "\n")
            return True
        if text.startswith("/botkey"):
            token = text[len("/botkey"):].strip()
            if token:
                self._spawn(self._set_botkey(token))
            else:
                self._append("\n🔐 paste the Telegram bot token and press Enter (it will not be echoed back):\n")
                self._pending = {"kind": "telegram_key"}
            return True
        if text in ("/kilochats", "/kchats"):
            self._spawn(self._kilochats())
            return True
        if text.startswith("/chat "):
            self._spawn(self._open_chat(text.split(maxsplit=1)[1].strip()))
            return True
        if text.startswith("/agent"):
            parts = text.split()
            name = parts[1].lower() if len(parts) > 1 else ""
            name = {"hacking": "security", "hack": "security", "pentest": "security",
                    "chat": "conversation", "convo": "conversation", "anon": "private",
                    "tor": "private", "orchestrate": "orchestrator", "router": "orchestrator"}.get(name, name)
            valid = {"research", "coding", "security", "systems", "general", "conversation",
                     "private", "orchestrator"}
            if name in {"", "off", "auto"}:
                self.forced_profile = ""
                self._append("\n— agent auto-selection restored —\n")
            elif name in valid:
                self.forced_profile = name
                self.agent_name = name
                self._append(f"\n— forced {name} agent (use /agent off to auto-select) —\n")
            else:
                self._append(f"\n— unknown agent; choose {', '.join(sorted(valid))} —\n")
            return True
        if text.startswith("/effort"):
            parts = text.split()
            level = parts[1].lower() if len(parts) > 1 else ""
            if level in {"high", "medium", "low"}:
                self.effort = level
                self._append(f"\n— effort set to {level} —\n")
            else:
                self._append("\n— use /effort high|medium|low —\n")
            return True
        if text.startswith("/cloudswitch"):
            self._spawn(self._cloud_setup(force_key=False))
            return True
        if text.startswith("/cloud"):
            rest = text[len("/cloud"):].strip()
            # Re-run the provider picker to add or change an API key at any time.
            if rest.lower() in ("add", "key", "keys", "change", "new", "setup"):
                self._spawn(self._cloud_setup(force_key=True))
                return True
            # No provider yet: run the pick-and-key setup, carrying any question along.
            if not self.cloud_provider:
                self._spawn(self._cloud_setup(pending_question=rest or None))
                return True
            if not rest:
                self._append(
                    f"\n— cloud provider: {self.cloud_provider}. /switch to route here, "
                    f"or /cloud <question> for one message —\n"
                )
                return True
            self._enqueue(rest, provider=self.cloud_provider)
            return True
        if text == "/switch":
            if not self.cloud_provider:
                self._append("\n— no cloud provider yet; run /cloud to set one up —\n")
                return True
            self.cloud_active = not self.cloud_active
            where = f"cloud · {self.cloud_provider}" if self.cloud_active else "local · Kilo"
            self._append(f"\n— switched to {where} —\n")
            self._spawn(self._refresh_brain_label())
            return True
        if text.startswith("/private"):
            arg = text[len("/private"):].strip().lower()
            if arg == "off":
                self.private_mode = False
                self._append("\n— private mode OFF · web requests go direct again —\n")
            elif arg == "rotate":
                self._spawn(self._rotate_circuit())
            elif arg == "status":
                self._spawn(self._private_status())
            else:  # "" or "on"
                self.private_mode = True
                self._append("\n🛡 private mode ON · web searches and fetches route through Tor.\n"
                             "   Your IP is hidden; if Tor is down the request is refused, never sent\n"
                             "   unmasked. Exit with /private off · new IP with /private rotate\n")
                self._spawn(self._private_status())
            return True
        if text == "/gguf":
            self._spawn(self._gguf_menu())
            return True
        if text.startswith("/model"):
            arg = text[len("/model"):].strip()
            if arg:
                self._spawn(self._model_cmd(arg))
            else:
                self._spawn(self._model_picker())
            return True
        if text == "/cancel":
            cancelled = False
            if self._queue is not None:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                        cancelled = True
                    except Exception:
                        break
            if self._active and not self._active.done():
                self._active.cancel()
                cancelled = True
            self._append("\n— cancelled —\n" if cancelled else "\n— nothing to cancel —\n")
            return True
        return False

    async def _list_chats(self) -> None:
        try:
            data = await self.client.request("sessions")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ could not list sessions: {exc}\n")
            return
        self._sessions = data.get("sessions", [])
        if not self._sessions:
            self._append("\n— no past sessions yet —\n")
            return
        lines = ["\npast sessions — /chat <n> to resume:"]
        for i, s in enumerate(self._sessions, 1):
            title = (s.get("title") or "").strip() or "(untitled)"
            lines.append(f"  {i:>2}. {title[:56]}  · {s.get('messages',0)} msgs")
        self._append("\n".join(lines) + "\n")

    async def _kilochats(self) -> None:
        """List past chats and arm a selector: the next number typed opens and continues it."""
        await self._list_chats()
        if self._sessions:
            self._append("  \u2192 type a number to open and continue that chat, or keep typing to stay here\n")
            self._pending = {"kind": "chat_pick"}

    async def _open_chat(self, arg: str) -> None:
        try:
            session = self._sessions[int(arg) - 1]
        except (ValueError, IndexError):
            self._append("\n— unknown chat number; run /chats first —\n")
            return
        self.session_id = session["id"]
        try:
            data = await self.client.request("session_history", session_id=self.session_id)
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ could not load session: {exc}\n")
            return
        self.output.buffer.set_document(Document("", 0), bypass_readonly=True)
        self._append(f"— resumed session · {session.get('messages',0)} messages —\n")
        for m in data.get("messages", []):
            self._append(f"\n{_you(m['content']) if m['role']=='user' else m['content']}\n")
        self._append("\n— continue below —\n")

    async def _cloud_setup(self, pending_question: str | None = None, force_key: bool = False) -> None:
        """Show the provider catalog and await a pick. Users only ever supply an API key:
        the base URL and default model come from the catalog."""
        try:
            data = await self.client.request("providers_catalog")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ could not load providers: {exc}\n")
            return
        self._catalog = data
        self._cloud_options = list((data.get("known") or {}).items())
        configured = set(data.get("configured", []))
        lines = ["\n☁ choose a cloud provider — type its number, then paste your API key:"]
        for i, (name, meta) in enumerate(self._cloud_options, 1):
            mark = "  ✓ configured" if name in configured else ""
            lines.append(f"  {i:>2}. {meta['label']:<12} {meta.get('model','')}{mark}")
        lines.append("  (type the number or name · blank line cancels)")
        self._append("\n".join(lines) + "\n")
        self._pending = {"kind": "cloud_pick", "question": pending_question, "force_key": force_key}

    def _run_cloud(self, name: str, question: str | None) -> None:
        """Activate a configured provider and, if a question was queued, send it now."""
        self.cloud_provider = name
        self.cloud_active = True
        self._append(f"\n— routing to cloud · {name} (use /switch for local Kilo) —\n")
        self._spawn(self._refresh_brain_label())
        if question:
            self._enqueue(question, provider=name)

    async def _resume_pending(self, text: str) -> None:
        pending = self._pending or {}
        self._pending = None
        kind = pending.get("kind")
        if kind == "gguf_pick":
            if text.strip().isdigit() and 1 <= int(text) <= len(self._gguf_options):
                path = self._gguf_options[int(text) - 1]
                self._append(f"\n\U0001f4e6 deploying {os.path.basename(path)} as the brain \u2014 this "
                             "restarts and warms the model; reconnect after it is ready.\n")
                import subprocess
                subprocess.Popen(["kilo", "brain", "deploy", path, "--brain-version", "custom"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self._append("\n\u2014 cancelled \u2014\n")
            return
        if kind == "chat_pick":
            if text.strip().isdigit():
                await self._open_chat(text.strip())
            else:
                self._append("— stayed in the current chat —\n")
            return
        if kind == "model_pick":
            name = None
            if text.isdigit() and 1 <= int(text) <= len(self._model_options):
                name = self._model_options[int(text) - 1]
            elif text.strip() in self._model_options:
                name = text.strip()
            if not name:
                self._append("\n— cancelled —\n")
                return
            await self._model_cmd(name)
            return
        if kind == "cloud_pick":
            name = None
            names = [n for n, _ in self._cloud_options]
            if text.isdigit() and 1 <= int(text) <= len(names):
                name = names[int(text) - 1]
            elif text.strip().lower() in names:
                name = text.strip().lower()
            if not name:
                self._append("\n— cancelled cloud setup —\n")
                return
            if name in set(self._catalog.get("configured", [])) and not pending.get("force_key"):
                self._run_cloud(name, pending.get("question"))
                return
            if name == "cloudflare":
                self._append("\n☁ enter your Cloudflare account ID:\n")
                self._pending = {"kind": "cloud_account", "name": name, "question": pending.get("question")}
            else:
                self._append(f"\n☁ paste your {name} API key and press Enter:\n")
                self._pending = {"kind": "cloud_key", "name": name, "question": pending.get("question")}
            return
        if kind == "cloud_account":
            if not text.strip() or not text.strip().replace("-", "").isalnum():
                self._append("\n⚠ invalid Cloudflare account ID\n")
                return
            self._append("\n☁ paste your Cloudflare API token and press Enter:\n")
            self._pending = {"kind": "cloud_key", "name": "cloudflare", "account_id": text.strip(), "question": pending.get("question")}
            return
        if kind == "cloud_key":
            name = pending["name"]
            try:
                res = await self.client.request("configure_provider", name=name, api_key=text, account_id=pending.get("account_id"))
            except (ConnectionError, FileNotFoundError, OSError) as exc:
                self._append(f"\n⚠ could not save key: {exc}\n")
                return
            if not res.get("ok"):
                self._append(f"\n⚠ {res.get('error', 'could not configure provider')}\n")
                return
            self._append(f"\n✓ {res.get('label', name)} configured.\n")
            self._run_cloud(name, pending.get("question"))
            return
        if kind == "telegram_key":
            await self._set_botkey(text.strip())

    async def _set_botkey(self, token: str) -> None:
        try:
            res = await self.client.request("set_telegram_token", token=token)
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ could not save Telegram bot token: {exc}\n")
            return
        if not res.get("ok"):
            self._append(f"\n⚠ {res.get('error', 'invalid Telegram bot token')}\n")
            return
        self._append("\n✓ Telegram bot token saved securely. Restart or wait for the bridge to reload it.\n")

    async def _rotate_circuit(self) -> None:
        try:
            res = await self.client.request("rotate_circuit")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ could not rotate: {exc}\n")
            return
        if res.get("ok"):
            self._append(f"\n🛡 new Tor circuit · exit IP {res.get('exit_ip') or 'unknown'}\n")
        else:
            self._append(f"\n⚠ {res.get('error', 'could not rotate circuit')}\n")

    async def _private_status(self) -> None:
        try:
            res = await self.client.request("tor_status")
        except (ConnectionError, FileNotFoundError, OSError):
            return
        if not res.get("available"):
            self._append("\n⚠ Tor is not reachable — private requests will be refused (fail-closed), "
                         "not sent unmasked. Start it: sudo systemctl start tor\n")

    async def _refresh_brain_label(self) -> None:
        """Keep the displayed model name in sync with the active brain: the local gguf stem,
        or the cloud provider's current model."""
        try:
            if self.cloud_active and self.cloud_provider:
                info = await self.client.request("provider_info")
                model = info.get("model")
                self.status["cloud_context_limit"] = info.get("context_limit")
                self.model_name = f"{info.get('default')}:{model}" if model else f"cloud·{self.cloud_provider}"
            else:
                st = await self.client.request("status")
                self.model_name = Path(str(st.get("model", ""))).stem or self.model_name
        except (ConnectionError, FileNotFoundError, OSError):
            pass
        # Unit tests and headless callers can exercise cloud switching before the
        # prompt_toolkit Application is attached; refreshing the label must stay safe.
        app = getattr(self, "app", None)
        if app is not None:
            app.invalidate()

    async def _model_picker(self) -> None:
        try:
            info = await self.client.request("provider_info")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ {exc}\n")
            return
        if not info.get("default"):
            self._append("\n— no cloud provider configured; run /cloud first —\n")
            return
        self._append("\n☁ fetching available models…\n")
        try:
            res = await self.client.request("provider_models")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ {exc}\n")
            return
        if not res.get("ok"):
            self._append(f"\n⚠ {res.get('error', 'could not list models')} — use /model <name>\n")
            return
        await self._refresh_brain_label()
        models = res.get("models", [])[:40]
        if not models:
            self._append("\n— no models returned; use /model <name> —\n")
            return
        self._model_options = models
        lines = [f"\n☁ {info['default']} · current {info.get('model') or '(unset)'} — type a number to switch:"]
        for i, m in enumerate(models, 1):
            lines.append(f"  {i:>2}. {m}")
        lines.append("  (blank line cancels)")
        self._append("\n".join(lines) + "\n")
        self._pending = {"kind": "model_pick"}

    def _gguf_scan(self) -> list[str]:
        """Find .gguf files in the usual places a downloaded model would land."""
        import glob
        home = Path.home()
        roots = [home, home / "Downloads", home / "models", Path.cwd(),
                 Path("/var/lib/kilobyte/models")]
        found: list[str] = []
        for r in roots:
            for pat in ("*.gguf", "*/*.gguf"):
                for f in glob.glob(str(r / pat)):
                    if f not in found and os.path.isfile(f):
                        found.append(f)
        return found[:40]

    async def _gguf_menu(self) -> None:
        files = self._gguf_scan()
        if not files:
            self._append("\n\u2014 no .gguf files found in ~, ~/Downloads, ~/models, cwd or the models dir. "
                         "Download a GGUF first, then run /gguf again. \u2014\n")
            return
        self._gguf_options = files
        free_gb = 0.0
        try:
            for _ln in Path('/proc/meminfo').read_text().splitlines():
                if _ln.startswith('MemAvailable'):
                    free_gb = int(_ln.split()[1]) / 1024 / 1024
                    break
        except Exception:
            pass
        self._append(
            f'\n\u26a0 Only load a model your machine can run. This box has about '
            f'{free_gb:.1f} GB free RAM - a GGUF larger than that will fail to load or run '
            'unusably slow. A bad load auto-rolls-back to the previous brain.\n')
        lines = ["\n\U0001f4e6 pick a GGUF to load as the brain \u2014 type its number:"]
        for i, f in enumerate(files, 1):
            mb = os.path.getsize(f) // (1024 * 1024)
            lines.append(f"  {i:>2}. {mb:>5} MB  {f}")
        lines.append("  (blank line cancels)")
        self._append("\n".join(lines) + "\n")
        self._pending = {"kind": "gguf_pick"}

    async def _model_cmd(self, arg: str) -> None:
        try:
            info = await self.client.request("provider_info")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ {exc}\n")
            return
        if not info.get("default"):
            self._append("\n— no cloud provider configured; run /cloud first —\n")
            return
        if not arg:
            self._append(f"\n☁ {info['default']} · model: {info.get('model') or '(unset)'}\n"
                         f"   change it with /model <model-name>\n")
            return
        try:
            res = await self.client.request("set_model", name=info["default"], model=arg)
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ {exc}\n")
            return
        if res.get("ok"):
            self._append(f"\n✓ {info['default']} model set to {res.get('model')}\n")
            self._spawn(self._refresh_brain_label())
        else:
            self._append(f"\n⚠ {res.get('error', 'could not set model')}\n")

    async def _ask(self, text: str, provider: str | None = None) -> None:
        # A plain message follows the active brain: local Kilo by default, the last cloud
        # provider after /switch.
        if provider is None and self.cloud_active and self.cloud_provider:
            provider = self.cloud_provider
        self.busy = True
        self._active = asyncio.current_task()
        self.tokens = self.tools_used = 0
        self._answered = False
        self._work_split = False
        self._had_work = False
        self.usage = {}
        self.streaming = False
        self.agent_name = ""
        self.phase = "thinking"
        self.started = time.monotonic()
        reader = writer = None
        try:
            reader, writer = await asyncio.open_unix_connection(self.client.socket_path)
            req: dict[str, Any] = {"command": "chat", "text": text, "session_id": self.session_id, "cwd": str(Path.cwd()), "effort": self.effort}
            if self.forced_profile:
                req["agent_profile"] = self.forced_profile
            if provider is not None:
                req["provider"] = provider
            if self.private_mode:
                req["private"] = True
            writer.write((json.dumps(req) + "\n").encode())
            await writer.drain()
            while raw := await reader.readline():
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "session":
                    self.session_id = event["session_id"]
                elif kind == "brain":
                    self.model_name = event.get("label", self.model_name)
                    if event.get("location") == "cloud":
                        self._open_box()
                        self._bline(f"\u2601 escalated to {event.get('label')}")
                        self._had_work = True
                elif kind == "agent":
                    self.agent_name = event.get("profile", "")
                    self._open_box()
                    self._bline(f"\u25c7 orchestrator \u2192 {event.get('profile','')} agent")
                    self._had_work = True
                elif kind == "warming":
                    self.phase = "warming cache (one-off)"
                    self._open_box()
                    self._bline("\u23f3 warming the prompt cache (one-off after a change)")
                    self._had_work = True
                elif kind == "thinking":
                    self.phase = "thinking"
                    self.streaming = False
                elif kind == "token":
                    self._open_box()
                    if self._had_work and not self._work_split:
                        # a faint divider separates the work section from the reply
                        self._flush_boxed()
                        self._bline("\u2508" * max(4, self._cw() - 4))
                        self._work_split = True
                    self.streaming = True
                    self.tokens += 1
                    self._stream_boxed(event.get("text", ""))
                elif kind == "tool_start":
                    self.tools_used += 1
                    self.phase = f"running {event['name']}"
                    self.streaming = False
                    args = event.get("arguments") or {}
                    detail = ", ".join(f"{k}={str(v)[:32]}" for k, v in list(args.items())[:2])
                    self._open_box()
                    self._flush_boxed()
                    self._bline(f"\u25c8 {event['name']} {detail}")
                    self._had_work = True
                elif kind == "tool_end":
                    ok = "✓" if event.get("ok") else "!"
                    self._open_box()
                    self._bline(f"{ok} {event.get('name')} \u00b7 {str(event.get('summary',''))[:90]}")
                    self.phase = "interpreting"
                elif kind == "error":
                    self._append(f"\n⚠ {event.get('error')}\n")
                elif kind == "permission":
                    allow, remember = await self._ask_permission(event)
                    writer.write((json.dumps({"type": "permission_response", "id": event.get("id"), "allow": allow, "remember": remember}) + "\n").encode())
                    await writer.drain()
                elif kind == "done":
                    if self._answered:
                        self._flush_boxed()
                        self._append(self._rule() + "\n")
                    self.usage = event.get("usage") or {}
                    break
                self.app.invalidate()
        except asyncio.CancelledError:
            self._append("\n[cancelled]\n")
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            self._append(f"\n⚠ daemon unavailable: {exc}\n")
        finally:
            if writer is not None:
                writer.close()
            self.busy = self.streaming = False
            self.phase = ""
            self._append("\n")
            self.app.invalidate()

    def _open_box(self) -> None:
        """Open Kilo's response box exactly once, so every part of his turn — escalation
        notices, agent hand-offs, tool work, and the reply — renders INSIDE his border,
        never floating under the owner's input box."""
        if self._answered:
            return
        label = f"\u2601 {self.cloud_provider}" if self.cloud_active else "Kilo"
        self._append("\n" + self._rule(label) + "\n")
        self._answered = True
        self._work_split = False
        self._line_buf = ""

    def _short_model(self) -> str:
        """A compact brain label for the status bar so long cloud ids never crowd it."""
        name = self.model_name or (("cloud\u00b7" + self.cloud_provider) if self.cloud_active else "kilo")
        name = name.replace(":free", "")
        if "/" in name:
            name = name.rsplit("/", 1)[-1]
        elif ":" in name and not name.startswith("cloud"):
            name = name.split(":", 1)[-1]
        return (name[:20] + "\u2026") if len(name) > 21 else name

    async def _ask_permission(self, event: dict) -> tuple[bool, bool]:
        """Ask the owner to approve a risky action. Type 1 (yes), 2 (yes, all this
        session) or 3 (no). Times out to a denial so a walked-away prompt fails safe."""
        risk = str(event.get("risk", "?"))
        self._append(
            f"\n\u26a0 approval needed \u2014 {event.get('capability')} [{risk}]\n"
            f"   {event.get('detail', '')}\n"
            f"   1) yes   2) yes, all this session   3) no\n   \u203a "
        )
        self._perm_future = asyncio.get_event_loop().create_future()
        self._pending = {"kind": "permission"}
        self.app.invalidate()
        try:
            ans = (await asyncio.wait_for(self._perm_future, timeout=280)).strip().lower()
        except (asyncio.TimeoutError, asyncio.CancelledError):
            ans = "3"
        allow = ans in {"1", "2", "y", "yes"}
        remember = ans == "2"
        self._append(f"   \u2192 {'approved' if allow else 'denied'}{' (session)' if remember else ''}\n")
        return allow, remember

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-q")
        @kb.add("c-d")
        def _(event):
            event.app.exit()

        @kb.add("c-l")
        def _(event):
            self.output.buffer.set_document(Document("", 0), bypass_readonly=True)

        @kb.add("f2")
        def _(event):
            self.show_panel = not self.show_panel

        @kb.add("c-c")
        def _(event):
            if self._active and not self._active.done():
                self._active.cancel()
            else:
                event.app.exit()

        return kb

    async def _tick(self) -> None:
        """Animate the spinner and refresh status so the interface always feels alive."""
        n = 0
        while True:
            self.spin += 1
            n += 1
            if n % 25 == 0:  # ~ every 2.5s
                try:
                    self.status = await self.client.request("status")
                    # Don't overwrite the cloud model label with the local model name while a
                    # cloud provider is the active brain.
                    if not self.cloud_active:
                        self.model_name = Path(str(self.status.get("model", ""))).stem or self.model_name
                except Exception:
                    pass
            # Always invalidate so the header dot and idle wave keep moving; the rate is
            # modest, so this is cheap even while nothing is happening.
            self.app.invalidate()
            await asyncio.sleep(0.12)

    async def run(self) -> None:
        try:
            self.status = await self.client.request("status")
            self.model_name = Path(str(self.status.get("model", ""))).stem or self.model_name
        except Exception:
            pass
        self.app = Application(
            layout=self.layout,
            key_bindings=self._bindings(),
            style=STYLE,
            full_screen=True,
            mouse_support=True,
        )
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._worker_loop())
        ticker = asyncio.create_task(self._tick())
        try:
            await self.app.run_async()
        finally:
            ticker.cancel()
            self._worker.cancel()


def _you(text: str) -> str:
    return f"› {text}"


async def run_full_tui(client: RPCClient) -> bool:
    """Run the full-screen UI. Returns False if it could not start, so the caller can fall
    back to the line-based UI."""
    try:
        await KiloApp(client).run()
        return True
    except Exception:
        return False
