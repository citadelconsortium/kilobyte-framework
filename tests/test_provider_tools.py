import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from kilobyte.providers import Provider, ProviderRegistry


class _StreamResponse:
    def __init__(self):
        self.lines = iter(
            [
                b'data: {"choices":[{"delta":{"content":"done"}}]}\n',
                b"data: [DONE]\n",
            ]
        )

    def readline(self):
        return next(self.lines, b"")

    def close(self):
        pass


class ProviderToolCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_rejecting_model_retries_with_text_tool_protocol(self):
        provider = Provider("example", "https://example.com/v1", "secret", "model")
        registry = ProviderRegistry(Path(tempfile.gettempdir()) / "unused-providers.json")
        rejected = urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            400,
            "bad request",
            {},
            io.BytesIO(b"this model does not support tools"),
        )
        requests = []

        def open_url(request, timeout):
            del timeout
            requests.append(json.loads(request.data))
            if len(requests) == 1:
                raise rejected
            return _StreamResponse()

        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]
        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch("urllib.request.urlopen", side_effect=open_url), patch(
            "asyncio.to_thread", side_effect=inline_to_thread
        ):
            events = [
                event
                async for event in registry.stream(
                    provider, [{"role": "user", "content": "research"}], 200, tools
                )
            ]

        self.assertEqual(events[0]["delta"]["content"], "done")
        self.assertIn("tools", requests[0])
        self.assertNotIn("tools", requests[1])
        fallback = requests[1]["messages"][-1]["content"]
        self.assertIn("<tool_call>", fallback)
        self.assertIn("web_search", fallback)
        rejected.close()


if __name__ == "__main__":
    unittest.main()
