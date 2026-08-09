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

from .agent import Agent


log = logging.getLogger("kilobyte.telegram")


class TelegramBridge:
    """Optional long-polling bridge; every message uses the daemon's same Agent/runtime."""

    CONFIG_POLL_SECONDS = 30
    # How often the progress message is rewritten. Frequent enough that the sender can
    # see it is alive, slow enough to stay clear of Telegram's edit rate limits.
    PROGRESS_SECONDS = 3

    def __init__(self, config_path: Path, agent: Agent):
        self.config_path = config_path
        self.agent = agent
        self.offset = 0
        self.running = False
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._replies: set[asyncio.Task[None]] = set()

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
            log.warning("telegram config has an empty allowed_chat_ids; staying disabled")
            return None
        return {"token": token, "allowed": allowed}

    # Shown under the message box by Telegram once registered with setMyCommands.
    COMMANDS = (
        ("start", "what Kilo is and how to use it"),
        ("status", "model, backend and resource status"),
        ("new", "start a fresh conversation"),
        ("id", "show this chat's id"),
        ("help", "list commands"),
    )

    MENU = {
        "inline_keyboard": [
            [{"text": "📊 Status", "callback_data": "status"}, {"text": "✨ New chat", "callback_data": "new"}],
            [{"text": "🆔 My ID", "callback_data": "id"}, {"text": "❓ Help", "callback_data": "help"}],
        ]
    }

    # Rotating glyphs for the live progress card, so a long-running reply visibly animates
    # rather than sitting on a static line that reads as a hang.
    SPINNER = "◐◓◑◒"
    # A small icon per phase makes the progress card scannable at a glance.
    PHASE_ICONS = {"thinking": "💭", "warming": "🔥", "running": "⚙️", "reading": "📖"}
    BAR_FILLED = "█"
    BAR_EMPTY = "░"

    @staticmethod
    def _bar(fraction: float, width: int = 10) -> str:
        """A compact unicode meter, e.g. used for free-memory in the status card."""
        fraction = max(0.0, min(1.0, fraction))
        filled = round(fraction * width)
        return TelegramBridge.BAR_FILLED * filled + TelegramBridge.BAR_EMPTY * (width - filled)

    @staticmethod
    def _call(token: str, method: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(data or {})
        # Nested structures (keyboards, allowed_updates) must be JSON, not form values.
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                payload[key] = json.dumps(value)
        encoded = urllib.parse.urlencode(payload).encode()
        request = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=encoded)
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    async def send(self, token: str, chat_id: int, text: str, keyboard: dict[str, Any] | None = None) -> None:
        chunks = [text[start : start + 3900] for start in range(0, len(text) or 1, 3900)] or ["(empty response)"]
        for index, chunk in enumerate(chunks):
            data: dict[str, Any] = {"chat_id": chat_id, "text": chunk or "(empty response)", "parse_mode": "HTML"}
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
                asyncio.to_thread(self._call, token, "sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}),
                timeout=3,
            )
            return int(response["result"]["message_id"])
        except Exception:
            return None

    async def _edit_progress(self, token: str, chat_id: int, message_id: int | None, text: str) -> None:
        """Rewrite the live status line. Telegram rejects an edit that would not change
        the text, and that rejection is not worth surfacing."""
        if message_id is None:
            return
        try:
            await asyncio.wait_for(asyncio.to_thread(
                self._call, token, "editMessageText",
                {"chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": "HTML"},
            ), timeout=1)
        except Exception:
            pass

    async def _delete(self, token: str, chat_id: int, message_id: int | None) -> None:
        if message_id is None:
            return
        try:
            await asyncio.wait_for(asyncio.to_thread(self._call, token, "deleteMessage", {"chat_id": chat_id, "message_id": message_id}), timeout=1)
        except Exception:
            pass

    async def _keep_typing(self, token: str, chat_id: int) -> None:
        """Telegram clears the typing indicator after ~5s, and a reply here can take
        minutes, so refresh it until the answer is ready."""
        try:
            while True:
                try:
                    await asyncio.to_thread(self._call, token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
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
                    await self.send(token, chat_id, "⏳ <i>queued — finishing the previous request first</i>")
                except Exception:
                    pass
            async with lock:
                await self._reply(token, chat_id, text)

        task = asyncio.create_task(serialised())
        self._replies.add(task)
        task.add_done_callback(self._replies.discard)

    async def _tick_progress(self, token: str, chat_id: int, message_id: int | None, state: dict[str, Any]) -> None:
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
            body = "\n".join([
                f"{spin} <b>Kilo is working</b>",
                "",
                f"{icon} {html.escape(state['phase'])}",
                f"⏱ <i>{clock}</i>",
                *([f"🔧 <i>{html.escape(' → '.join(tools))}</i>"] if tools else []),
            ])
            preview = "".join(state.get("output", []))[-1200:]
            if preview:
                body += f"\n\n<b>live reply</b>\n<code>{html.escape(preview)}</code>"
            await self._edit_progress(token, chat_id, message_id, body)

    async def _reply(self, token: str, chat_id: int, text: str) -> None:
        typing = asyncio.create_task(self._keep_typing(token, chat_id))
        # A reply can take minutes here; a status message that is edited as work
        # progresses is the only way the sender can tell it is alive.
        progress = await self._send_progress(token, chat_id, "◐ <b>Kilo is working</b>\n\n💭 thinking")
        state: dict[str, Any] = {"phase": "thinking", "phase_kind": "thinking", "started": time.monotonic(), "tools": [], "output": []}
        ticker = asyncio.create_task(self._tick_progress(token, chat_id, progress, state))
        output: list[str] = []
        agent_label: str | None = None
        try:
            async for event in self.agent.run(str(text), f"telegram-{chat_id}", remote=True):
                kind = event.get("type")
                if kind == "token":
                    token_text = event.get("text", "")
                    output.append(token_text)
                    state["output"].append(token_text)
                elif kind == "agent":
                    agent_label = str(event.get("profile") or "")
                elif kind == "warming":
                    state["phase"], state["phase_kind"] = "warming the model cache (one-off)", "warming"
                elif kind == "thinking":
                    # No step number: it read as stuck. The timer conveys progress.
                    state["phase"], state["phase_kind"] = "thinking", "thinking"
                elif kind == "tool_start":
                    name = str(event.get("name"))
                    state["tools"].append(name)
                    state["phase"], state["phase_kind"] = f"running {name}", "running"
                elif kind == "tool_end":
                    state["phase"], state["phase_kind"] = f"read {event.get('name')} · interpreting", "reading"
                elif kind == "error":
                    output.append(f"\n[error: {event.get('error')}]")
        except Exception as exc:
            # Silence looks identical to a hung bot, so always tell the user.
            log.exception("telegram request failed for chat %s", chat_id)
            ticker.cancel()
            await self._delete(token, chat_id, progress)
            await self.send(
                token, chat_id,
                f"⚠️ <b>Kilo hit an error</b>\n<code>{html.escape(str(exc))}</code>",
                self.MENU,
            )
            return
        finally:
            typing.cancel()
            ticker.cancel()
        await self._delete(token, chat_id, progress)
        answer = html.escape("".join(output).strip())
        took = int(time.monotonic() - state["started"])
        minutes, seconds = divmod(took, 60)
        clock = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        # A thin divider then a compact meta line: which agent answered, how long it took,
        # and which tools it actually used. Reads like a signed answer, not a raw dump.
        footer_bits = []
        if agent_label and agent_label not in ("", "general", "conversation"):
            footer_bits.append(f"◆ {html.escape(agent_label)}")
        footer_bits.append(f"⏱ {clock}")
        if state["tools"]:
            footer_bits.append(f"🔧 {html.escape(', '.join(dict.fromkeys(state['tools'])))}")
        body = f"{answer}\n\n<i>{' · '.join(footer_bits)}</i>" if answer else "🤔 <i>(no response — try rephrasing)</i>"
        await self.send(token, chat_id, body, self.MENU)

    async def _command(self, token: str, chat_id: int, command: str) -> bool:
        """Handle a slash command or menu button. Returns True when handled."""
        command = command.lstrip("/").split("@")[0].split()[0].lower() if command.strip() else ""
        if command == "start":
            # Minimal welcome — details live in /help.
            lines = [
                "🤖 <b>Kilo</b> — ready, Sir.",
                "Send a message; the same local brain the terminal uses answers.",
                "🔒 <i>read-only here.</i>  ·  /help for commands",
            ]
            await self.send(token, chat_id, "\n".join(lines), self.MENU)
            return True
        if command == "help":
            lines = [
                "<b>Commands</b>",
                *[f"• <code>/{name}</code> — {description}" for name, description in self.COMMANDS],
                "",
                "🔒 <b>Kilo is read-only over Telegram</b> — it looks things up, searches the web",
                "and remembers facts, but never runs commands, writes files or changes the",
                "system, and has no cloud access here. Use the terminal for that.",
            ]
            await self.send(token, chat_id, "\n".join(lines), self.MENU)
            return True
        if command == "new":
            self.agent.memory.new_session("telegram", "reset")
            await self.send(token, chat_id, "✨ <b>Fresh conversation started.</b>\n<i>Earlier context is set aside.</i>", self.MENU)
            return True
        if command == "id":
            await self.send(
                token, chat_id,
                f"🆔 This chat's id is <code>{chat_id}</code>\n"
                f"<i>Add it to allowed_chat_ids to authorise it.</i>",
                self.MENU,
            )
            return True
        if command == "status":
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
                uptime_str = f"{uh}h {um}m" if uh else (f"{um}m {us}s" if um else f"{us}s")
                lines = [
                    f"{'🟢' if running else '🔴'} <b>Kilo — {'running' if running else 'stopped'}</b>",
                    "",
                    f"🧠 <b>model</b>    <code>{html.escape(Path(str(status.get('model', ''))).stem)}</code>",
                    f"⏱ <b>uptime</b>   {uptime_str}",
                    f"📐 <b>context</b>  {profile.get('context_size')} tokens",
                    f"⚙️ <b>threads</b>  {profile.get('threads')}   ·   gpu layers {profile.get('gpu_layers')}",
                    f"💾 <b>memory</b>   {mem_line}",
                ]
                await self.send(token, chat_id, "\n".join(lines), self.MENU)
            except Exception as exc:
                await self.send(token, chat_id, f"⚠️ Could not read status: <code>{html.escape(str(exc))}</code>", self.MENU)
            return True
        return False

    async def run(self) -> None:
        self.running = True
        config = self.config()
        while self.running and config is None:
            # Let the operator enable Telegram by writing the config file, without
            # having to restart the daemon to be noticed.
            await asyncio.sleep(self.CONFIG_POLL_SECONDS)
            config = self.config()
        if not self.running or config is None:
            return
        token, allowed = config["token"], config["allowed"]
        log.info("telegram bridge enabled for %d authorised chat(s)", len(allowed))
        try:
            # Publishes the command list into Telegram's UI menu.
            await asyncio.to_thread(
                self._call, token, "setMyCommands",
                {"commands": [{"command": name, "description": description} for name, description in self.COMMANDS]},
            )
        except Exception:
            log.warning("could not publish telegram command menu", exc_info=True)
        while self.running:
            try:
                response = await asyncio.to_thread(self._call, token, "getUpdates", {"offset": self.offset, "timeout": 30, "allowed_updates": ["message", "callback_query"]})
                for update in response.get("result", []):
                    self.offset = max(self.offset, int(update["update_id"]) + 1)

                    query = update.get("callback_query")
                    if query:
                        chat_id = int(((query.get("message") or {}).get("chat") or {}).get("id", 0))
                        if chat_id not in allowed:
                            log.warning("ignored telegram button from unauthorised chat %s", chat_id)
                            continue
                        # Acknowledge promptly or the client shows a spinner on the button.
                        try:
                            await asyncio.to_thread(self._call, token, "answerCallbackQuery", {"callback_query_id": query.get("id")})
                        except Exception:
                            pass
                        await self._command(token, chat_id, str(query.get("data", "")))
                        continue

                    message = update.get("message") or {}
                    chat_id = int((message.get("chat") or {}).get("id", 0))
                    text = message.get("text")
                    if not text:
                        continue
                    if chat_id not in allowed:
                        log.warning("ignored telegram message from unauthorised chat %s", chat_id)
                        continue
                    if text.startswith("/") and await self._command(token, chat_id, text):
                        continue
                    # Answering inline would block this loop for the whole generation,
                    # which on slow hardware is minutes. Commands and button presses
                    # arriving meanwhile would sit unread and look broken, so the reply
                    # runs on its own task and polling continues.
                    self._start_reply(token, chat_id, text)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("telegram poll failed; retrying")
                await asyncio.sleep(5)

    def stop(self) -> None:
        self.running = False
        for task in list(self._replies):
            task.cancel()
