import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from prompt_toolkit.document import Document
    from kilobyte.tui_full import KiloApp, _ChatLexer
except ModuleNotFoundError as exc:
    if exc.name != "prompt_toolkit":
        raise
    KiloApp = None
    Document = None


@unittest.skipIf(KiloApp is None, "prompt_toolkit is not installed in the raw source-test environment")
class FullTUIDirectChatTests(unittest.IsolatedAsyncioTestCase):
    def test_python_fence_receives_language_aware_syntax_styles(self):
        document = Document(
            "\u2502 ```python                         \u2502\n"
            "\u2502 def greet(name): return f'Hi {name}' \u2502\n"
            "\u2502 ```                               \u2502"
        )
        fragments = _ChatLexer().lex_document(document)(1)
        styles = {style for style, value in fragments if value.strip()}
        self.assertIn("class:pygments.keyword", styles)
        self.assertIn("class:pygments.name.function", styles)
        self.assertTrue(any(style.startswith("class:pygments.literal.string") for style in styles))
        self.assertEqual("".join(value for _style, value in fragments), document.lines[1])

    async def test_old_live_work_and_response_share_the_same_kilo_box(self):
        events = [
            {"type": "session", "session_id": "direct-chat"},
            {"type": "agent", "profile": "research"},
            {"type": "thinking"},
            {"type": "tool_start", "name": "web_search", "arguments": {"query": "Kilo"}},
            {"type": "tool_end", "name": "web_search", "ok": True, "summary": "2 results"},
            {"type": "tool_start", "name": "web_fetch", "arguments": {"url": "https://example.com"}},
            {"type": "tool_end", "name": "web_fetch", "ok": True, "summary": "source opened"},
            {"type": "thinking"},
            {"type": "token", "text": "Verified answer with source."},
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
        self.assertIn("◇ orchestrator → research agent", rendered)
        self.assertIn("◈ web_search query=Kilo", rendered)
        self.assertIn("web_search", rendered)
        self.assertIn("web_fetch", rendered)
        self.assertIn("Verified answer", rendered)
        top = rendered.index("╭─ Kilo ")
        bottom = rendered.index("╰", top)
        for expected in ("research agent", "web_search", "web_fetch", "Verified answer"):
            self.assertLess(top, rendered.index(expected))
            self.assertLess(rendered.index(expected), bottom)


if __name__ == "__main__":
    unittest.main()
