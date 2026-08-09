import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from kilobyte.security import Risk
from kilobyte.telegram import TelegramBridge


def _config(raw: str, payload: dict) -> Path:
    path = Path(raw) / "telegram.json"
    path.write_text(json.dumps(payload))
    return path


class TelegramConfigTests(unittest.TestCase):
    def test_disabled_without_allowlist(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "secret", "allowed_chat_ids": []})
            self.assertIsNone(TelegramBridge(path, object()).config())  # type: ignore[arg-type]

    def test_disabled_without_token(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "", "allowed_chat_ids": [42]})
            self.assertIsNone(TelegramBridge(path, object()).config())  # type: ignore[arg-type]

    def test_disabled_with_placeholder_token(self):
        """The shipped example must never accidentally authorise a live bot."""
        with tempfile.TemporaryDirectory() as raw:
            path = _config(
                raw, {"token": "PASTE_BOT_TOKEN_HERE", "allowed_chat_ids": [42]}
            )
            self.assertIsNone(TelegramBridge(path, object()).config())  # type: ignore[arg-type]

    def test_disabled_when_missing_or_malformed(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertIsNone(
                TelegramBridge(Path(raw) / "absent.json", object()).config()
            )  # type: ignore[arg-type]
            bad = Path(raw) / "telegram.json"
            bad.write_text("{not json")
            self.assertIsNone(TelegramBridge(bad, object()).config())  # type: ignore[arg-type]
            ids = _config(raw, {"token": "secret", "allowed_chat_ids": ["not-an-id"]})
            self.assertIsNone(TelegramBridge(ids, object()).config())  # type: ignore[arg-type]

    def test_loads_explicit_allowlist(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "secret", "allowed_chat_ids": [42]})
            self.assertEqual(TelegramBridge(path, object()).config()["allowed"], {42})  # type: ignore[union-attr,arg-type]

    def test_long_poll_socket_timeout_exceeds_api_timeout(self):
        response = io.BytesIO(b'{"ok": true, "result": []}')
        with patch("urllib.request.urlopen", return_value=response) as opened:
            TelegramBridge._call("token", "getUpdates", {"timeout": 30})
        self.assertEqual(opened.call_args.kwargs["timeout"], 40)


class TelegramDeliveryTests(IsolatedAsyncioTestCase):
    async def test_reply_resets_tool_markup_and_renders_clean_research(self):
        class ResearchAgent:
            def run(self, *args, **kwargs):
                async def generate():
                    yield {"type": "brain", "label": "cloud:test"}
                    yield {"type": "token", "text": "Sir, let me check"}
                    yield {"type": "response_reset"}
                    yield {"type": "tool_start", "name": "web_search", "arguments": {"query": "citadel"}}
                    yield {"type": "tool_end", "name": "web_search", "ok": True, "summary": "2 sources"}
                    yield {"type": "token", "text": "## Findings\n- **Verified:** result, Sir."}

                return generate()

        with tempfile.TemporaryDirectory() as raw:
            bridge = TelegramBridge(
                _config(raw, {"token": "secret", "allowed_chat_ids": [42]}),
                ResearchAgent(),
            )  # type: ignore[arg-type]
            sent: list[str] = []
            progress: list[str] = []
            edits: list[str] = []

            async def capture_send(token, chat_id, text, keyboard=None):
                sent.append(text)

            async def capture_progress(token, chat_id, text):
                progress.append(text)
                return len(progress)

            async def capture_edit(token, chat_id, message_id, text):
                edits.append(text)

            async def noop(*args, **kwargs):
                return None

            bridge.send = capture_send  # type: ignore[method-assign]
            bridge._send_progress = capture_progress  # type: ignore[method-assign]
            bridge._edit_progress = capture_edit  # type: ignore[method-assign]
            bridge._delete = noop  # type: ignore[method-assign]
            await bridge._reply("secret", 42, "research")
            self.assertEqual(len(progress), 2)
            self.assertNotIn("let me check", sent[-1])
            self.assertIn("<b>Findings</b>", sent[-1])
            self.assertIn("• <b>Verified:</b> result", sent[-1])
            self.assertTrue(any("web_search" in edit for edit in edits))

    async def test_failure_is_reported_instead_of_silence(self):
        """A crash mid-generation must still produce a message; silence is
        indistinguishable from a hung bot."""

        class FailingAgent:
            def run(self, *args, **kwargs):
                async def generate():
                    raise RuntimeError("model unavailable")
                    yield  # pragma: no cover - makes this an async generator

                return generate()

        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "secret", "allowed_chat_ids": [42]})
            bridge = TelegramBridge(path, FailingAgent())  # type: ignore[arg-type]
            sent: list[str] = []

            async def capture(token, chat_id, text, keyboard=None):
                sent.append(text)

            bridge.send = capture  # type: ignore[method-assign]
            await bridge._reply("secret", 42, "hello")
            self.assertTrue(sent)
            self.assertIn("model unavailable", sent[0])


class TelegramCommandTests(IsolatedAsyncioTestCase):
    def _bridge(self, raw):
        path = _config(raw, {"token": "secret", "allowed_chat_ids": [42]})
        bridge = TelegramBridge(path, object())  # type: ignore[arg-type]
        self.sent: list[str] = []

        async def capture(token, chat_id, text, keyboard=None):
            self.sent.append(text)

        bridge.send = capture  # type: ignore[method-assign]
        return bridge

    async def test_start_and_help_explain_approval_gated_machine_tools(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(raw)
            for command in ("/start", "/help", "help"):
                self.sent.clear()
                self.assertTrue(await bridge._command("secret", 42, command))
                self.assertIn("approv", self.sent[0].lower())

    async def test_approval_button_is_bound_to_the_requesting_chat(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(raw)
            delivered = []

            async def capture(token, chat_id, text, keyboard=None):
                delivered.append((text, keyboard))

            bridge.send = capture  # type: ignore[method-assign]
            task = asyncio.create_task(
                bridge._request_approval(
                    "secret", 42, "terminal.execute.write", "git push", Risk.WRITE
                )
            )
            await asyncio.sleep(0)
            callback = delivered[0][1]["inline_keyboard"][0][0]["callback_data"]
            self.assertTrue(bridge._resolve_approval(99, callback))
            self.assertFalse(task.done())
            self.assertTrue(bridge._resolve_approval(42, callback))
            self.assertTrue(await task)

    async def test_group_style_command_suffix_is_accepted(self):
        """In groups Telegram delivers '/status@BotName'."""
        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(raw)
            self.assertTrue(await bridge._command("secret", 42, "/help@KiloBot"))

    async def test_unknown_command_is_not_swallowed(self):
        """An unhandled slash command must fall through to the model, not vanish."""
        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(raw)
            self.assertFalse(await bridge._command("secret", 42, "/summarise this"))
            self.assertEqual(self.sent, [])

    async def test_route_agent_and_new_session_reach_the_agent(self):
        captured = {}

        class Memory:
            def new_session(self, source, title):
                return "fresh-session"

        class Providers:
            def default_name(self):
                return "cloud"

            def providers(self):
                return {"cloud": self.resolve("cloud")}

            def resolve(self, name=None):
                return SimpleNamespace(
                    name=name or "cloud",
                    model="large",
                    label=f"{name or 'cloud'}:large",
                )

        class Agent:
            memory = Memory()
            providers = Providers()

            def run(self, text, session_id, **kwargs):
                captured.update(text=text, session_id=session_id, **kwargs)

                async def generate():
                    yield {"type": "brain", "label": "cloud:large"}
                    yield {"type": "token", "text": "Sir, done, Sir."}

                return generate()

        with tempfile.TemporaryDirectory() as raw:
            bridge = TelegramBridge(
                _config(raw, {"token": "secret", "allowed_chat_ids": [42]}), Agent()
            )  # type: ignore[arg-type]

            async def capture(*args, **kwargs):
                return None

            bridge.send = capture  # type: ignore[method-assign]
            bridge._send_progress = capture  # type: ignore[method-assign]
            bridge._delete = capture  # type: ignore[method-assign]
            bridge._edit_progress = capture  # type: ignore[method-assign]
            self.assertTrue(await bridge._command("secret", 42, "/cloud"))
            self.assertTrue(await bridge._command("secret", 42, "/agent security"))
            self.assertTrue(await bridge._command("secret", 42, "/new"))
            await bridge._reply("secret", 42, "inspect")
            self.assertEqual(captured["session_id"], "fresh-session")
            self.assertEqual(captured["provider"], "cloud")
            self.assertEqual(captured["agent_profile"], "security")
            self.assertTrue(captured["fresh"])
            self.assertTrue(callable(captured["permission_callback"]))


class TelegramConcurrencyTests(IsolatedAsyncioTestCase):
    async def test_cancel_stops_only_this_chats_active_work(self):
        import asyncio

        started = asyncio.Event()

        class SlowAgent:
            def run(self, *args, **kwargs):
                async def generate():
                    started.set()
                    await asyncio.sleep(30)
                    yield {"type": "token", "text": "late"}

                return generate()

        with tempfile.TemporaryDirectory() as raw:
            bridge = TelegramBridge(
                _config(raw, {"token": "secret", "allowed_chat_ids": [42]}),
                SlowAgent(),
            )  # type: ignore[arg-type]
            sent: list[str] = []

            async def capture(token, chat_id, text, keyboard=None):
                sent.append(text)

            async def noop(*args, **kwargs):
                return None

            bridge.send = capture  # type: ignore[method-assign]
            bridge._send_progress = noop  # type: ignore[method-assign]
            bridge._keep_typing = noop  # type: ignore[method-assign]
            bridge._start_reply("secret", 42, "slow research")
            await asyncio.wait_for(started.wait(), timeout=5)
            self.assertTrue(await bridge._command("secret", 42, "/cancel"))
            await asyncio.sleep(0)
            self.assertNotIn(42, bridge._chat_replies)
            self.assertTrue(any("Cancelled" in message for message in sent))

    async def test_a_slow_reply_does_not_block_commands(self):
        """A generation takes minutes on this hardware. If replies ran inline, a button
        press or command arriving meanwhile would sit unread and look broken."""
        import asyncio

        started = asyncio.Event()

        class SlowAgent:
            def run(self, *args, **kwargs):
                async def generate():
                    started.set()
                    await asyncio.sleep(30)
                    yield {"type": "token", "text": "late"}

                return generate()

        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "secret", "allowed_chat_ids": [42]})
            bridge = TelegramBridge(path, SlowAgent())  # type: ignore[arg-type]
            sent: list[str] = []

            async def capture(token, chat_id, text, keyboard=None):
                sent.append(text)

            async def noop(*args, **kwargs):
                return None

            bridge.send = capture  # type: ignore[method-assign]
            bridge._send_progress = noop  # type: ignore[method-assign]
            bridge._keep_typing = noop  # type: ignore[method-assign]

            bridge._start_reply("secret", 42, "something slow")
            await asyncio.wait_for(started.wait(), timeout=5)
            # The slow reply is in flight; a command must still be answered promptly.
            await asyncio.wait_for(bridge._command("secret", 42, "/help"), timeout=5)
            self.assertTrue(any("approv" in message.lower() for message in sent))
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
