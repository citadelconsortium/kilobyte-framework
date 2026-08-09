import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from kilobyte.tui_full import KiloApp
except ModuleNotFoundError as exc:
    if exc.name != "prompt_toolkit":
        raise
    KiloApp = None


@unittest.skipIf(KiloApp is None, "prompt_toolkit is not installed in the raw source-test environment")
class FullTUIRenderingTests(unittest.TestCase):
    def app(self):
        return KiloApp(SimpleNamespace(socket_path=Path("/tmp/not-used.sock")))

    def test_work_and_answer_share_one_box_and_whitespace_is_collapsed(self):
        app = self.app()
        app._work_line("◇ research agent active")
        app._work_line("▶ web_search  query=Kilo")
        app._open_answer_box()
        app._stream_boxed("\n\nFirst paragraph\n\n\nSecond paragraph\n\n")
        app._close_answer_box()
        app._close_box()

        rendered = app.output.buffer.text
        self.assertEqual(rendered.count("╭─ Kilo "), 1)
        self.assertNotIn("Live work", rendered)
        self.assertIn("web_search", rendered)
        self.assertIn("First paragraph", rendered)
        self.assertIn("Second paragraph", rendered)
        payloads = [line[2:-1].strip() for line in rendered.splitlines() if line.startswith("│ ")]
        for left, right in zip(payloads, payloads[1:]):
            self.assertFalse(left == right == "")

    def test_response_reset_retracts_preamble_inside_same_box(self):
        app = self.app()
        app._work_line("◇ tools active")
        app._open_answer_box()
        app._stream_boxed("I will check this now")
        app._flush_boxed()
        app._reset_answer()
        app._work_line("▶ web_search")
        app._open_answer_box()
        app._stream_boxed("Verified result")
        app._close_answer_box()
        app._close_box()

        rendered = app.output.buffer.text
        self.assertNotIn("I will check", rendered)
        self.assertIn("Verified result", rendered)
        self.assertEqual(rendered.count("╭─ Kilo "), 1)

@unittest.skipIf(KiloApp is None, "prompt_toolkit is not installed in the raw source-test environment")
class FullTUIDirectChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_rpc_event_stream_stays_in_one_clean_box(self):
        events = [
            {"type": "session", "session_id": "direct-chat"},
            {"type": "agent", "profile": "research"},
            {"type": "capabilities", "agent": "research", "tools": ["web_search", "web_fetch"]},
            {"type": "thinking"},
            {"type": "token", "text": "\n\nI will research this"},
            {"type": "response_reset"},
            {"type": "tool_start", "name": "web_search", "arguments": {"query": "Kilo"}},
            {"type": "tool_end", "name": "web_search", "ok": True, "summary": "2 results"},
            {"type": "tool_start", "name": "web_fetch", "arguments": {"url": "https://example.com"}},
            {"type": "tool_end", "name": "web_fetch", "ok": True, "summary": "source opened"},
            {"type": "thinking"},
            {"type": "token", "text": "\n\nVerified answer\n\n\nwith source."},
            {"type": "done", "usage": {"total_tokens": 42}},
        ]
        reader = asyncio.StreamReader()
        for event in events:
            reader.feed_data((json.dumps(event) + "\n").encode())
        reader.feed_eof()

        class Writer:
            def write(self, data):
                self.request = data

            async def drain(self):
                pass

            def close(self):
                pass

        writer = Writer()

        async def connect(path):
            del path
            return reader, writer

        app = KiloApp(SimpleNamespace(socket_path=Path("/tmp/in-memory.sock")))
        app.app = SimpleNamespace(invalidate=lambda: None)
        with patch("asyncio.open_unix_connection", side_effect=connect):
                await app._ask("research Kilo")

        rendered = app.output.buffer.text
        self.assertEqual(rendered.count("╭─ Kilo "), 1)
        self.assertNotIn("Live work", rendered)
        self.assertNotIn("I will research", rendered)
        self.assertIn("web_search", rendered)
        self.assertIn("web_fetch", rendered)
        self.assertIn("Verified answer", rendered)


if __name__ == "__main__":
    unittest.main()
