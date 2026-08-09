import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from kilobyte.rpc import RPCServer


class RPCDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnected_chat_closes_inference_generator(self):
        started = asyncio.Event()
        closed = asyncio.Event()

        class Agent:
            settings = SimpleNamespace(home=Path("/tmp"))

            def run(self, *args, **kwargs):
                async def generate():
                    try:
                        yield {"type": "session", "session_id": "test"}
                        started.set()
                        await asyncio.Event().wait()
                    finally:
                        closed.set()

                return generate()

        class Reader:
            eof = False
            delivered = False

            async def readline(self):
                if not self.delivered:
                    self.delivered = True
                    return (
                        json.dumps({"command": "chat", "text": "hello"}).encode()
                        + b"\n"
                    )
                await asyncio.Event().wait()

            def at_eof(self):
                return self.eof

        class Writer:
            closing = False

            def __init__(self, reader):
                self.reader = reader
                self.output = bytearray()

            def write(self, data):
                self.output.extend(data)

            async def drain(self):
                # The client receives the first event and then disappears while the
                # generator is silent.
                self.reader.eof = True

            def is_closing(self):
                return self.closing

            def close(self):
                self.closing = True

            async def wait_closed(self):
                return None

        reader = Reader()
        writer = Writer(reader)
        server = RPCServer(
            Path("/tmp/not-used.sock"), Agent(), object(), object(), object()
        )  # type: ignore[arg-type]
        await asyncio.wait_for(server._handle(reader, writer), timeout=2)  # type: ignore[arg-type]
        await started.wait()
        await asyncio.wait_for(closed.wait(), timeout=2)
        self.assertIn(b'"type": "session"', writer.output)


if __name__ == "__main__":
    unittest.main()
