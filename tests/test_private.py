import tempfile
import unittest
from pathlib import Path

from kilobyte import net
from kilobyte.config import Settings
from kilobyte.errors import ToolError
from kilobyte.memory import MemoryStore
from kilobyte.security import PermissionManager
from kilobyte.tools import ToolContext, ToolRegistry


def _reg(root):
    s = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
    m = MemoryStore(root / "m.db")
    return ToolRegistry(s, m, PermissionManager(root / "p.json")), m


class PrivateModeTests(unittest.IsolatedAsyncioTestCase):
    def test_toolcontext_private_defaults_false(self):
        self.assertFalse(ToolContext(session_id="s", cwd=Path(".")).private)

    async def test_private_fetch_is_fail_closed_when_tor_down(self):
        """The whole point: if privacy was requested but Tor is unreachable, the request
        must be refused, never sent unmasked."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reg, m = _reg(root)
            original = net.tor_available
            net.tor_available = lambda *a, **k: False
            try:
                ctx = ToolContext(session_id="s", cwd=root, private=True)
                with self.assertRaises(ToolError) as caught:
                    await reg._web_fetch({"url": "https://example.com"}, ctx)
                self.assertIn("not sent", str(caught.exception).lower())
            finally:
                net.tor_available = original
                m.close()

    async def test_private_search_is_fail_closed_when_tor_down(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reg, m = _reg(root)
            original = net.tor_available
            net.tor_available = lambda *a, **k: False
            try:
                ctx = ToolContext(session_id="s", cwd=root, private=True)
                with self.assertRaises(ToolError):
                    await reg._web_search({"query": "anything"}, ctx)
            finally:
                net.tor_available = original
                m.close()

    async def test_private_rejects_local_host_without_dns(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reg, m = _reg(root)
            try:
                ctx = ToolContext(session_id="s", cwd=root, private=True)
                for bad in ("http://127.0.0.1/x", "http://localhost/x", "http://10.0.0.5/x"):
                    with self.assertRaises(Exception):
                        await reg._web_fetch({"url": bad}, ctx)
            finally:
                m.close()


if __name__ == "__main__":
    unittest.main()
