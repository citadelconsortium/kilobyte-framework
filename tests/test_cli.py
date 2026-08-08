import unittest

from kilobyte.cli import runtime_summary


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


if __name__ == "__main__":
    unittest.main()
