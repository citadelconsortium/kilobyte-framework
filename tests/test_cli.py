import unittest
from contextlib import redirect_stdout
from io import StringIO

from kilobyte.cli import print_status, runtime_summary


class CLITests(unittest.TestCase):
    def test_runtime_summary_omits_large_chat_template(self):
        result = runtime_summary({
            "build_info": "b123-test",
            "model_alias": "kilobyte",
            "chat_template": "very large template",
            "chat_template_caps": {"supports_tool_calls": True},
            "default_generation_settings": {"n_ctx": 8192},
            "is_sleeping": False,
        })
        self.assertEqual(result["build"], "b123-test")
        self.assertEqual(result["context_size"], 8192)
        self.assertTrue(result["tool_calling"])
        self.assertNotIn("chat_template", result)

    def test_status_is_a_readable_typed_health_summary(self):
        output = StringIO()
        with redirect_stdout(output):
            print_status({
                "running": True,
                "healthy": True,
                "pid": 42,
                "uptime_seconds": 125,
                "model": "/models/kilobyte.gguf",
                "warming": False,
                "profile": {"threads": 2, "context_size": 8192, "gpu_layers": 0, "available_mb": 2048},
                "memory": {"sessions": 3, "facts": 4, "skills": 5},
            })
        text = output.getvalue()
        self.assertIn("KILOBYTE STATUS", text)
        self.assertIn("STATE        READY", text)
        self.assertIn("daemon       ACTIVE  pid 42", text)
        self.assertIn("brain        HEALTHY  kilobyte.gguf", text)
        self.assertIn("uptime       2m 5s", text)

    def test_stopped_status_is_explicit(self):
        output = StringIO()
        with redirect_stdout(output):
            print_status(None)
        text = output.getvalue()
        self.assertIn("STATE        STOPPED", text)
        self.assertIn("daemon       INACTIVE", text)


if __name__ == "__main__":
    unittest.main()
