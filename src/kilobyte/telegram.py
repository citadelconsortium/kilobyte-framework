from __future__ import annotations

import asyncio
import html
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .activity import format_arguments, format_summary
from .agent import Agent
from .profiles import PROFILES
from .telegram_render import telegram_html, telegram_html_chunks

log = logging.getLogger("kilobyte.telegram")


class TelegramBridge:
    """Optional long-polling bridge; every message uses the daemon's same Agent/runtime."""

    CONFIG_POLL_SECONDS = 30
    # How often the progress message is rewritten. Frequent enough that the sender can
    # see it is alive, slow enough to stay clear of Telegram's edit rate limits.
    PROGRESS_SECONDS = 1.2

    def __init__(self, config_path: Path, agent: Agent):
        self.config_path = config_path
        self.agent = agent
        self.offset = 0
        self.running = False
        self._wake = asyncio.Event()
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._replies: set[asyncio.Task[None]] = set()
        self._sessions: dict[int, str] = {}
        self._fresh: set[int] = set()
        self._chat_providers: dict[int, str] = {}
        self._chat_profiles: dict[int, str] = {}

    def config(self) -> dict[str, Any] | None:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("telegram config unreadable (%s); staying disabled", exc)
            return None
        token = str(config.get("token", "")).strip()
        try:
            allowed = {int(item) for item in config.get("allowed_chat_ids", [])}
        except (TypeError, ValueError):
            log.warning("telegram allowed_chat_ids must be integers; staying disabled")
            return None
        if not token or token == "PASTE_BOT_TOKEN_HERE":
            log.warning("telegram config has no bot token; staying disabled")
            return None
        if not allowed:
            log.warning(
                "telegram config has an empty allowed_chat_ids; staying disabled"
            )
            return None
        return {"token": token, "allowed": allowed}

    # Shown under the message box by Telegram once registered with setMyCommands.
    COMMANDS = (
        ("start", "what Kilo is and how to use it"),
        ("status", "model, route and resource status"),
        ("new", "start a fresh conversation"),
        ("local", "use the private local GGUF"),
        ("cloud", "use the default or named cloud model"),
        ("switch", "switch between local and cloud"),
        ("models", "list cloud models"),
        ("model", "show or select a cloud model"),
        ("agent", "show or select a specialist agent"),
        ("id", "show this chat's id"),
        ("help", "list commands"),
    )

    MENU = {
        "inline_keyboard": [
            [
                {"text": "📊 Status", "callback_data": "status"},
                {"text": "✨ New chat", "callback_data": "new"},
            ],
            [
                {"text": "🏠 Local", "callback_data": "local"},
                {"text": "☁️ Cloud", "callback_data": "cloud"},
            ],
            [
                {"text": "🧠 Models", "callback_data": "models"},
                {"text": "🧩 Agents", "callback_data": "agent"},
            ],
            [
                {"text": "🆔 My ID", "callback_data": "id"},
                {"text": "❓ Help", "callback_data": "help"},
            ],
        ]
    }

    # Rotating glyphs for the live progress card, so a long-running reply visibly animates
    # rather than sitting on a static line that reads as a hang.
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    # A small icon per phase makes the progress card scannable at a glance.
    PHASE_ICONS = {"thinking": "💭", "warming": "🔥", "running": "⚙️", "reading": "📖"}
    BAR_FILLED = "█"
    BAR_EMPTY = "░"

    @staticmethod
    def _bar(fraction: float, width: int = 10) -> str:
        """A compact unicode meter, e.g. used for free-memory in the status card."""
        fraction = max(0.0, min(1.0, fraction))
        filled = round(fraction * width)
        return TelegramBridge.BAR_FILLED * filled + TelegramBridge.BAR_EMPTY * (
            width - filled
        )

    @staticmethod
    def _call(
        token: str,
        method: str,
        data: dict[str, Any] | None = None,
        request_timeout: int | None = None,
    ) -> dict[str, Any]:
        payload = dict(data or {})
        # Nested structures (keyboards, allowed_updates) must be JSON, not form values.
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                payload[key] = json.dumps(value)
        encoded = urllib.parse.urlencode(payload).encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/{method}", data=encoded
        )
        long_poll = int(payload.get("timeout", 0) or 0) if method == "getUpdates" else 0
        timeout = request_timeout or max(10, long_poll + 10)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not result.get("ok", False):
            raise RuntimeError(
                str(result.get("description") or "Telegram Bot API request failed")
            )
        return result

    async def send(
        self,
        token: str,
        chat_id: int,
        text: str,
        keyboard: dict[str, Any] | None = None,
    ) -> None:
        chunks = telegram_html_chunks(text)
        for index, chunk in enumerate(chunks):
            data: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk or "(empty response)",
                "parse_mode": "HTML",
            }
            # Attach the menu only to the final chunk so it appears once, at the end.
            if keyboard and index == len(chunks) - 1:
                data["reply_markup"] = keyboard
            await asyncio.to_thread(self._call, token, "sendMessage", data)

    async def _send_progress(self, token: str, chat_id: int, text: str) -> int | None:
        # Progress is bounded best-effort: show the thinking indicator when Telegram
        # responds, but never hold the agent behind a network/DNS outage.
        if ":" not in token:  # malformed/test token; avoid pointless network work
            return None
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call,
                    token,
                    "sendMessage",
                    {"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                ),
                timeout=3,
            )
            return int(response["result"]["message_id"])
        except Exception:
            return None

    async def _edit_progress(
        self, token: str, chat_id: int, message_id: int | None, text: str
    ) -> None:
        """Rewrite the live status line. Telegram rejects an edit that would not change
        the text, and that rejection is not worth surfacing."""
        if message_id is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._call,
                    token,
                    "editMessageText",
                    {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                ),
                timeout=1,
            )
        except Exception:
            pass

    async def _delete(self, token: str, chat_id: int, message_id: int | None) -> None:
        if message_id is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._call,
                    token,
                    "deleteMessage",
                    {"chat_id": chat_id, "message_id": message_id},
                ),
                timeout=1,
            )
        except Exception:
            pass

    async def _keep_typing(self, token: str, chat_id: int) -> None:
        """Telegram clears the typing indicator after ~5s, and a reply here can take
        minutes, so refresh it until the answer is ready."""
        try:
            while True:
                try:
                    await asyncio.to_thread(
                        self._call,
                        token,
                        "sendChatAction",
                        {"chat_id": chat_id, "action": "typing"},
                    )
                except Exception:
                    pass
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    def _start_reply(self, token: str, chat_id: int, text: str) -> None:
        """Run one reply per chat off the poll loop.

        Replies are serialised per chat so two questions cannot interleave in the same
        conversation, while a slow answer in one chat never stops the bridge from
        servicing commands, buttons, or another chat.
        """

        async def serialised() -> None:
            lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
            # There is one inference slot, so a message sent while another is generating
            # waits. Acknowledge immediately when that happens, otherwise the wait reads
            # as the bot ignoring the message.
            if lock.locked():
                try:
                    await self.send(
                        token,
                        chat_id,
                        "⏳ <i>queued — finishing the previous request first</i>",
                    )
                except Exception:
                    pass
            async with lock:
                await self._reply(token, chat_id, text)

        task = asyncio.create_task(serialised())
        self._replies.add(task)
        task.add_done_callback(self._reply_finished)

    def _reply_finished(self, task: asyncio.Task[None]) -> None:
        self._replies.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("telegram reply task failed: %s", error)

    async def _tick_progress(
        self,
        token: str,
        chat_id: int,
        message_id: int | None,
        work_message_id: int | None,
        state: dict[str, Any],
    ) -> None:
        """Rewrite the progress message on a timer.

        Driving it from agent events alone leaves it frozen on whatever happened last:
        a step can run for minutes without emitting anything, so the sender sees
        'step 1' indefinitely and assumes the bot is stuck. Telegram also rejects an
        edit that would not change the text, so the elapsed counter doubles as the
        thing that makes each edit distinct.
        """
        frame = 0
        while True:
            await asyncio.sleep(self.PROGRESS_SECONDS)
            frame += 1
            spin = self.SPINNER[frame % len(self.SPINNER)]
            icon = self.PHASE_ICONS.get(state.get("phase_kind", "thinking"), "💭")
            elapsed = int(time.monotonic() - state["started"])
            minutes, seconds = divmod(elapsed, 60)
            clock = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
            tools = list(dict.fromkeys(state["tools"]))
            body = "\n".join(
                [
                    f"{spin} <b>Kilo is working</b>",
                    "",
                    f"{icon} {html.escape(state['phase'])}",
                    f"⏱ <i>{clock}</i>",
                    *([f"🔧 <i>{html.escape(' → '.join(tools))}</i>"] if tools else []),
                ]
            )
            await self._edit_progress(token, chat_id, message_id, body)
            if frame % 2 == 0:
                await self._edit_progress(
                    token, chat_id, work_message_id, self._live_work_body(state)
                )

    @staticmethod
    def _live_work_body(state: dict[str, Any], finished: bool = False) -> str:
        """A separate persistent card for safe machine actions and streamed output."""
        title = "✅ <b>Work log</b>" if finished else "📟 <b>Live machine output</b>"
        entries = list(state.get("work", []))[-16:]
        body = title
        if entries:
            body += "\n<pre>" + html.escape("\n".join(entries)[-2200:]) + "</pre>"
        else:
            body += "\n<i>waiting for the first machine action…</i>"
        preview = "".join(state.get("output", []))[-1100:].strip()
        if preview and not finished:
            body += "\n\n<b>Live reply</b>\n<pre>" + html.escape(preview) + "</pre>"
        return body

    async def _reply(self, token: str, chat_id: int, text: str) -> None:
        typing = asyncio.create_task(self._keep_typing(token, chat_id))
        # A reply can take minutes here; a status message that is edited as work
        # progresses is the only way the sender can tell it is alive.
        progress = await self._send_progress(
            token, chat_id, "⠋ <b>Kilo is working</b>\n\n💭 thinking"
        )
        work_message = await self._send_progress(
            token,
            chat_id,
            "📟 <b>Live machine output</b>\n<i>waiting for the first machine action…</i>",
        )
        state: dict[str, Any] = {
            "phase": "thinking",
            "phase_kind": "thinking",
            "started": time.monotonic(),
            "tools": [],
            "output": [],
            "work": [],
        }
        ticker = asyncio.create_task(
            self._tick_progress(token, chat_id, progress, work_message, state)
        )
        output: list[str] = []
        agent_label: str | None = None
        brain_label: str | None = None
        provider = self._chat_providers.get(chat_id)
        profile = self._chat_profiles.get(chat_id)
        session_id = self._sessions.get(chat_id, f"telegram-{chat_id}")
        fresh = chat_id in self._fresh
        tool_started: dict[str, float] = {}
        try:
            async for event in self.agent.run(
                str(text),
                session_id,
                remote=True,
                provider=provider,
                agent_profile=profile,
                fresh=fresh,
            ):
                self._fresh.discard(chat_id)
                kind = event.get("type")
                if kind == "token":
                    token_text = event.get("text", "")
                    output.append(token_text)
                    state["output"].append(token_text)
                elif kind == "response_reset":
                    output.clear()
                    state["output"].clear()
                    state["work"].append("↻ intercepted model tool markup; dispatching safely")
                elif kind == "agent":
                    agent_label = str(event.get("profile") or "")
                    state["work"].append(f"◇ agent  {agent_label}")
                elif kind == "brain":
                    brain_label = str(event.get("label") or "")
                    state["phase"] = f"using {brain_label}"
                    state["work"].append(f"◇ brain  {brain_label}")
                elif kind == "warming":
                    state["phase"], state["phase_kind"] = (
                        "warming the model cache (one-off)",
                        "warming",
                    )
                    state["work"].append("◌ cache  warming model prefix")
                elif kind == "thinking":
                    # No step number: it read as stuck. The timer conveys progress.
                    state["phase"], state["phase_kind"] = "thinking", "thinking"
                elif kind == "tool_start":
                    name = str(event.get("name"))
                    state["tools"].append(name)
                    state["phase"], state["phase_kind"] = f"running {name}", "running"
                    tool_started[name] = time.monotonic()
                    detail = format_arguments(event.get("arguments") or {}, 420)
                    state["work"].append(f"▶ {name}{'  ' + detail if detail else ''}")
                elif kind == "tool_end":
                    name = str(event.get("name"))
                    elapsed = time.monotonic() - tool_started.pop(name, time.monotonic())
                    state["phase"], state["phase_kind"] = (
                        f"read {name} · interpreting",
                        "reading",
                    )
                    mark = "✓" if event.get("ok") else "✗"
                    summary = format_summary(event.get("summary", ""), 520)
                    state["work"].append(
                        f"{mark} {name}  {elapsed:0.1f}s{'  ' + summary if summary else ''}"
                    )
                elif kind == "error":
                    error = format_summary(event.get("error"), 520)
                    output.append(f"\n[error: {error}]")
                    state["work"].append(f"✗ error  {error}")
        except Exception as exc:
            # Silence looks identical to a hung bot, so always tell the user.
            log.exception("telegram request failed for chat %s", chat_id)
            ticker.cancel()
            await self._delete(token, chat_id, progress)
            await self._edit_progress(
                token, chat_id, work_message, self._live_work_body(state, finished=True)
            )
            await self.send(
                token,
                chat_id,
                f"⚠️ <b>Kilo hit an error</b>\n<code>{html.escape(str(exc))}</code>",
                self.MENU,
            )
            return
        finally:
            typing.cancel()
            ticker.cancel()
            await asyncio.gather(typing, ticker, return_exceptions=True)
        await self._delete(token, chat_id, progress)
        await self._edit_progress(
            token, chat_id, work_message, self._live_work_body(state, finished=True)
        )
        answer = telegram_html("".join(output).strip())
        took = int(time.monotonic() - state["started"])
        minutes, seconds = divmod(took, 60)
        clock = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        # A thin divider then a compact meta line: which agent answered, how long it took,
        # and which tools it actually used. Reads like a signed answer, not a raw dump.
        footer_bits = []
        if agent_label and agent_label not in ("", "general", "conversation"):
            footer_bits.append(f"◆ {html.escape(agent_label)}")
        if brain_label:
            footer_bits.append(f"🧠 {html.escape(brain_label)}")
        footer_bits.append(f"⏱ {clock}")
        if state["tools"]:
            footer_bits.append(
                f"🔧 {html.escape(', '.join(dict.fromkeys(state['tools'])))}"
            )
        body = (
            f"{answer}\n\n<i>{' · '.join(footer_bits)}</i>"
            if answer
            else "🤔 <i>(no response — try rephrasing)</i>"
        )
        await self.send(token, chat_id, body, self.MENU)

    async def _command(self, token: str, chat_id: int, command: str) -> bool:
        """Handle a slash command or menu button. Returns True when handled."""
        raw = command.strip().lstrip("/")
        if not raw:
            return False
        head, _, argument = raw.partition(" ")
        name = head.split("@", 1)[0].lower()
        argument = argument.strip()

        if name == "start":
            lines = [
                "🤖 <b>Kilo</b> — ready, Sir.",
                "Private local inference is the default. Cloud is used only after /cloud.",
                "🔒 <i>Telegram tools stay read-only.</i>  ·  /help for commands",
            ]
            await self.send(token, chat_id, "\n".join(lines), self.MENU)
            return True
        if name in {"help", "commands"}:
            lines = [
                "<b>Commands</b>",
                *[
                    f"• <code>/{name}</code> — {description}"
                    for name, description in self.COMMANDS
                ],
                "",
                "🔒 <b>Telegram remains read-only</b>: no terminal commands, file writes,",
                "service control, or destructive tools. /cloud explicitly routes prompts to",
                "the selected configured provider; /local keeps everything on Kilobase.",
            ]
            await self.send(token, chat_id, "\n".join(lines), self.MENU)
            return True
        if name == "new":
            self._sessions[chat_id] = self.agent.memory.new_session(
                "telegram", "fresh Telegram chat"
            )
            self._fresh.add(chat_id)
            await self.send(
                token,
                chat_id,
                "✨ <b>Fresh conversation started.</b>\n<i>Earlier context is set aside.</i>",
                self.MENU,
            )
            return True
        if name == "local":
            self._chat_providers.pop(chat_id, None)
            await self.send(
                token,
                chat_id,
                "🏠 <b>Local brain selected.</b>\n<i>No prompt leaves Kilobase.</i>",
                self.MENU,
            )
            return True
        if name in {"cloud", "switch"}:
            if name == "switch" and not argument:
                argument = "local" if chat_id in self._chat_providers else ""
            if argument.lower() == "local":
                self._chat_providers.pop(chat_id, None)
                await self.send(
                    token, chat_id, "🏠 <b>Switched to local.</b>", self.MENU
                )
                return True
            try:
                provider = self.agent.providers.resolve(argument or None)
            except Exception as exc:
                await self.send(
                    token, chat_id, f"⚠️ <code>{html.escape(str(exc))}</code>", self.MENU
                )
                return True
            self._chat_providers[chat_id] = provider.name
            await self.send(
                token,
                chat_id,
                f"☁️ <b>Cloud selected</b>\n<code>{html.escape(provider.label)}</code>\n"
                "<i>Future prompts use it until /local or /switch.</i>",
                self.MENU,
            )
            return True
        if name == "models":
            try:
                provider_name = (
                    argument
                    or self._chat_providers.get(chat_id)
                    or self.agent.providers.default_name()
                )
                models = await asyncio.to_thread(
                    self.agent.providers.list_models, provider_name, False
                )
                provider = self.agent.providers.resolve(provider_name)
                shown = models[:30]
                lines = [
                    f"🧠 <b>{html.escape(provider.name)} models</b>",
                    *[f"• <code>{html.escape(model)}</code>" for model in shown],
                ]
                if len(models) > len(shown):
                    lines.append(f"<i>…and {len(models) - len(shown)} more</i>")
                lines.append("\nSelect with <code>/model MODEL_ID</code>.")
                await self.send(token, chat_id, "\n".join(lines), self.MENU)
            except Exception as exc:
                await self.send(
                    token,
                    chat_id,
                    f"⚠️ Could not list models: <code>{html.escape(str(exc))}</code>",
                    self.MENU,
                )
            return True
        if name == "model":
            provider_name = (
                self._chat_providers.get(chat_id) or self.agent.providers.default_name()
            )
            if not argument:
                try:
                    provider = self.agent.providers.resolve(provider_name)
                    await self.send(
                        token,
                        chat_id,
                        f"🧠 <code>{html.escape(provider.label)}</code>",
                        self.MENU,
                    )
                except Exception as exc:
                    await self.send(
                        token,
                        chat_id,
                        f"⚠️ <code>{html.escape(str(exc))}</code>",
                        self.MENU,
                    )
                return True
            first, separator, remainder = argument.partition(" ")
            configured = self.agent.providers.providers()
            if separator and first in configured:
                provider_name, model = first, remainder.strip()
            else:
                model = argument
            try:
                if not provider_name:
                    raise RuntimeError("no cloud provider is configured")
                selected = self.agent.providers.set_model(provider_name, model)
                self._chat_providers[chat_id] = provider_name
                await self.send(
                    token,
                    chat_id,
                    f"✅ <b>Cloud model selected</b>\n<code>{html.escape(provider_name)}:{html.escape(selected)}</code>",
                    self.MENU,
                )
            except Exception as exc:
                await self.send(
                    token, chat_id, f"⚠️ <code>{html.escape(str(exc))}</code>", self.MENU
                )
            return True
        if name == "agent":
            if not argument:
                current = self._chat_profiles.get(chat_id, "auto")
                lines = [f"🧩 <b>Agent: {html.escape(current)}</b>"]
                lines.extend(
                    f"• <code>{profile.name}</code> — {html.escape(profile.hint)}"
                    for profile in PROFILES.values()
                )
                lines.append(
                    "\nSelect with <code>/agent NAME</code>; use <code>/agent auto</code> to route automatically."
                )
                await self.send(token, chat_id, "\n".join(lines), self.MENU)
                return True
            selected = argument.lower()
            if selected == "auto":
                self._chat_profiles.pop(chat_id, None)
            elif selected in PROFILES:
                self._chat_profiles[chat_id] = selected
            else:
                await self.send(
                    token,
                    chat_id,
                    f"⚠️ Unknown agent: <code>{html.escape(selected)}</code>",
                    self.MENU,
                )
                return True
            await self.send(
                token,
                chat_id,
                f"🧩 <b>Agent selected:</b> <code>{html.escape(selected)}</code>",
                self.MENU,
            )
            return True
        if name == "id":
            await self.send(
                token,
                chat_id,
                f"🆔 This chat's id is <code>{chat_id}</code>\n"
                f"<i>Add it to allowed_chat_ids to authorise it.</i>",
                self.MENU,
            )
            return True
        if name == "status":
            try:
                status = self.agent.runtime.status()
                profile = status.get("profile") or {}
                running = bool(status.get("running"))
                total = profile.get("total_mb") or 0
                avail = profile.get("available_mb") or 0
                mem_line = f"{avail} / {total} MiB free"
                if total:
                    mem_line = f"{self._bar(avail / total)}  {mem_line}"
                uptime = int(status.get("uptime_seconds", 0) or 0)
                um, us = divmod(uptime, 60)
                uh, um = divmod(um, 60)
                uptime_str = (
                    f"{uh}h {um}m" if uh else (f"{um}m {us}s" if um else f"{us}s")
                )
                selected_provider = self._chat_providers.get(chat_id)
                if selected_provider:
                    route = self.agent.providers.resolve(selected_provider).label
                    context_limit = self.agent.providers.context_limit(selected_provider)
                    if not context_limit:
                        try:
                            await asyncio.wait_for(
                                asyncio.to_thread(
                                    self.agent.providers.list_models,
                                    selected_provider,
                                    False,
                                ),
                                timeout=10,
                            )
                        except Exception:
                            pass
                        context_limit = self.agent.providers.context_limit(selected_provider)
                    context_text = (
                        f"{context_limit} tokens"
                        if context_limit
                        else "provider-managed (not advertised)"
                    )
                    compute_text = "hosted cloud"
                else:
                    route = f"local:{Path(str(status.get('model', ''))).stem}"
                    context_text = f"{profile.get('context_size')} tokens"
                    compute_text = (
                        f"threads {profile.get('threads')}   ·   gpu layers {profile.get('gpu_layers')}"
                    )
                lines = [
                    f"{'🟢' if running else '🔴'} <b>Kilo — {'running' if running else 'stopped'}</b>",
                    "",
                    f"🧠 <b>route</b>    <code>{html.escape(route)}</code>",
                    f"🧩 <b>agent</b>    <code>{html.escape(self._chat_profiles.get(chat_id, 'auto'))}</code>",
                    f"⏱ <b>uptime</b>   {uptime_str}",
                    f"📐 <b>context</b>  {context_text}",
                    f"⚙️ <b>compute</b>  {compute_text}",
                    f"💾 <b>memory</b>   {mem_line}",
                ]
                await self.send(token, chat_id, "\n".join(lines), self.MENU)
            except Exception as exc:
                await self.send(
                    token,
                    chat_id,
                    f"⚠️ Could not read status: <code>{html.escape(str(exc))}</code>",
                    self.MENU,
                )
            return True
        return False

    async def _publish_bot_ui(self, token: str) -> None:
        requests = (
            (
                "setMyCommands",
                {
                    "commands": [
                        {"command": name, "description": description}
                        for name, description in self.COMMANDS
                    ]
                },
            ),
            ("setChatMenuButton", {"menu_button": {"type": "commands"}}),
            (
                "setMyShortDescription",
                {
                    "short_description": "Private local AI with explicit cloud routing and specialist agents."
                },
            ),
            (
                "setMyDescription",
                {
                    "description": "Kilo is Sir's Kilobase assistant: local by default, cloud only when selected, with persistent memory and read-only Telegram tools."
                },
            ),
        )
        for method, data in requests:
            try:
                await asyncio.to_thread(self._call, token, method, data)
            except Exception:
                log.warning("could not publish Telegram %s", method, exc_info=True)

    async def run(self) -> None:
        self.running = True
        self._wake.clear()
        published_token: str | None = None
        try:
            while self.running:
                # Reload on every poll so token and allow-list edits take effect live.
                config = self.config()
                if config is None:
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(), timeout=self.CONFIG_POLL_SECONDS
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                token, allowed = config["token"], config["allowed"]
                if token != published_token:
                    self.offset = 0
                    await self._publish_bot_ui(token)
                    published_token = token
                    log.info(
                        "telegram bridge enabled for %d authorised chat(s)",
                        len(allowed),
                    )
                try:
                    response = await asyncio.to_thread(
                        self._call,
                        token,
                        "getUpdates",
                        {
                            "offset": self.offset,
                            "timeout": 10,
                            "allowed_updates": ["message", "callback_query"],
                        },
                    )
                    for update in response.get("result", []):
                        self.offset = max(self.offset, int(update["update_id"]) + 1)

                        query = update.get("callback_query")
                        if query:
                            chat_id = int(
                                ((query.get("message") or {}).get("chat") or {}).get(
                                    "id", 0
                                )
                            )
                            if chat_id not in allowed:
                                log.warning(
                                    "ignored telegram button from unauthorised chat %s",
                                    chat_id,
                                )
                                continue
                            # Acknowledge promptly or the client shows a spinner on the button.
                            try:
                                await asyncio.to_thread(
                                    self._call,
                                    token,
                                    "answerCallbackQuery",
                                    {"callback_query_id": query.get("id")},
                                )
                            except Exception:
                                pass
                            await self._command(
                                token, chat_id, str(query.get("data", ""))
                            )
                            continue

                        message = update.get("message") or {}
                        chat_id = int((message.get("chat") or {}).get("id", 0))
                        text = message.get("text")
                        if not text:
                            continue
                        if chat_id not in allowed:
                            log.warning(
                                "ignored telegram message from unauthorised chat %s",
                                chat_id,
                            )
                            continue
                        if text.startswith("/") and await self._command(
                            token, chat_id, text
                        ):
                            continue
                        # Keep polling while the one inference slot works in the background.
                        self._start_reply(token, chat_id, text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("telegram poll failed; retrying")
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
        finally:
            replies = list(self._replies)
            for task in replies:
                task.cancel()
            if replies:
                await asyncio.gather(*replies, return_exceptions=True)

    def stop(self) -> None:
        self.running = False
        self._wake.set()
        for task in list(self._replies):
            task.cancel()
