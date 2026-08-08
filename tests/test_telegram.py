import json
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

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
            path = _config(raw, {"token": "PASTE_BOT_TOKEN_HERE", "allowed_chat_ids": [42]})
            self.assertIsNone(TelegramBridge(path, object()).config())  # type: ignore[arg-type]

    def test_disabled_when_missing_or_malformed(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertIsNone(TelegramBridge(Path(raw) / "absent.json", object()).config())  # type: ignore[arg-type]
            bad = Path(raw) / "telegram.json"
            bad.write_text("{not json")
            self.assertIsNone(TelegramBridge(bad, object()).config())  # type: ignore[arg-type]
            ids = _config(raw, {"token": "secret", "allowed_chat_ids": ["not-an-id"]})
            self.assertIsNone(TelegramBridge(ids, object()).config())  # type: ignore[arg-type]

    def test_loads_explicit_allowlist(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"token": "secret", "allowed_chat_ids": [42]})
            self.assertEqual(TelegramBridge(path, object()).config()["allowed"], {42})  # type: ignore[union-attr,arg-type]


class TelegramDeliveryTests(IsolatedAsyncioTestCase):
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

    async def test_start_and_help_explain_the_read_only_policy(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(raw)
            for command in ("/start", "/help", "help"):
                self.sent.clear()
                self.assertTrue(await bridge._command("secret", 42, command))
                self.assertIn("read-only", self.sent[0])

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


class TelegramConcurrencyTests(IsolatedAsyncioTestCase):
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
            self.assertTrue(any("read-only" in message for message in sent))
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
