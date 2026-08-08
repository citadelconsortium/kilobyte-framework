"""The /cloud provider setup flow in the full TUI: pick -> key -> configure -> active."""
import asyncio
import unittest

try:
    from kilobyte.tui_full import KiloApp
    HAVE_PTK = True
except Exception:
    HAVE_PTK = False


class FakeClient:
    def __init__(self):
        self.socket_path = "/tmp/none.sock"
        self.calls = []

    async def request(self, command, **kw):
        self.calls.append((command, kw))
        if command == "providers_catalog":
            return {"known": {
                "openrouter": {"label": "OpenRouter", "model": "m1"},
                "groq": {"label": "Groq", "model": "m2"},
            }, "configured": [], "default": None}
        if command == "configure_provider":
            return {"ok": True, "label": kw["name"] + ":m", "name": kw["name"]}
        return {}


@unittest.skipUnless(HAVE_PTK, "prompt_toolkit not installed")
class CloudFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_pick_then_key_configures_and_activates(self):
        app = KiloApp(FakeClient())
        await app._cloud_setup(pending_question=None)
        self.assertEqual(app._pending["kind"], "cloud_pick")
        self.assertEqual(len(app._cloud_options), 2)
        await app._resume_pending("2")  # groq
        self.assertEqual(app._pending["kind"], "cloud_key")
        self.assertEqual(app._pending["name"], "groq")
        await app._resume_pending("gsk-testkey")
        self.assertTrue(app.cloud_active)
        self.assertEqual(app.cloud_provider, "groq")
        self.assertTrue(any(c[0] == "configure_provider" and c[1]["api_key"] == "gsk-testkey"
                            for c in app.client.calls))

    async def test_spawn_keeps_task_reference(self):
        app = KiloApp(FakeClient())
        t = app._spawn(asyncio.sleep(0))
        self.assertIn(t, app._bg_tasks)
        await t
        self.assertNotIn(t, app._bg_tasks)

    async def test_queue_processes_requests_sequentially(self):
        import types
        app = KiloApp(FakeClient())
        app.app = types.SimpleNamespace(invalidate=lambda: None)  # run() supplies the real one
        app._queue = asyncio.Queue()
        order = []
        active = []

        async def fake_ask(text, provider=None):
            active.append(text)
            self.assertEqual(len(active), 1, "two requests ran at once")
            await asyncio.sleep(0.01)
            order.append(text)
            active.remove(text)

        app._ask = fake_ask
        worker = asyncio.create_task(app._worker_loop())
        for msg in ("a", "b", "c"):
            app._enqueue(msg)
        await asyncio.wait_for(app._queue.join(), timeout=2)
        worker.cancel()
        self.assertEqual(order, ["a", "b", "c"])  # in order, never overlapping


if __name__ == "__main__":
    unittest.main()
