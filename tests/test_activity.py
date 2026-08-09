import unittest

from kilobyte.activity import format_arguments, format_summary


class ActivityRenderingTests(unittest.TestCase):
    def test_all_arguments_are_visible_but_secrets_are_redacted(self):
        text = format_arguments({"path": "/tmp/a", "query": "hello", "token": "secret-value"})
        self.assertIn('path="/tmp/a"', text)
        self.assertIn('query="hello"', text)
        self.assertIn("token=[redacted]", text)
        self.assertNotIn("secret-value", text)

    def test_summaries_redact_bearer_and_telegram_tokens(self):
        text = format_summary("Bearer abcdefghijklmnop 8877081234:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd")
        self.assertNotIn("abcdefghijklmnop", text)
        self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUVWXYZ", text)
        self.assertIn("[redacted", text)


if __name__ == "__main__":
    unittest.main()
