import json
import unittest

from kilobyte.context import CHARS_PER_TOKEN, as_tool_message, compact, shorten


class ShortenTests(unittest.TestCase):
    def test_short_text_is_untouched(self):
        text, cut = shorten("small", 100)
        self.assertEqual(text, "small")
        self.assertFalse(cut)

    def test_keeps_head_and_tail(self):
        text, cut = shorten("A" * 500 + "B" * 500 + "C" * 500, 600)
        self.assertTrue(cut)
        self.assertTrue(text.startswith("A"))
        self.assertTrue(text.rstrip().endswith("C"))
        self.assertIn("removed from the middle", text)


class CompactTests(unittest.TestCase):
    def test_command_output_is_bounded_but_stays_structured(self):
        """A directory listing tokenises to several times the context window; the exit
        code has to survive the shortening so the model can still tell success apart
        from failure."""
        result = {"exit_code": 0, "stdout": "x" * 200_000, "stderr": "", "truncated": False}
        reduced, cut = compact(result, 900)
        self.assertTrue(cut)
        self.assertIsInstance(reduced, dict)
        self.assertEqual(reduced["exit_code"], 0)
        self.assertLessEqual(len(json.dumps(reduced)), 900 * CHARS_PER_TOKEN + 400)

    def test_budget_is_shared_between_large_fields(self):
        result = {"stdout": "o" * 100_000, "stderr": "e" * 100_000}
        reduced, _ = compact(result, 900)
        self.assertTrue(reduced["stdout"])
        self.assertTrue(reduced["stderr"])

    def test_long_entry_lists_are_capped_to_what_the_budget_affords(self):
        listing = {"entries": [{"name": f"file-{i}", "type": "file"} for i in range(1000)]}
        reduced, cut = compact(listing, 900)
        self.assertTrue(cut)
        self.assertIsInstance(reduced, dict)
        self.assertLess(len(reduced["entries"]), 1000)
        self.assertEqual(reduced["entries_omitted"], 1000 - len(reduced["entries"]))
        self.assertLessEqual(len(json.dumps(reduced)), 900 * CHARS_PER_TOKEN)

    def test_small_result_is_passed_through_unchanged(self):
        result = {"platform": "Linux", "cpu_count": 2}
        reduced, cut = compact(result, 900)
        self.assertFalse(cut)
        self.assertEqual(reduced, result)


class ToolMessageTests(unittest.TestCase):
    def test_truncation_is_declared_to_the_model(self):
        message = as_tool_message({"stdout": "x" * 200_000}, 900)
        self.assertIn("shortened", message)
        self.assertLess(len(message), 900 * CHARS_PER_TOKEN + 500)

    def test_untruncated_result_has_no_notice(self):
        message = as_tool_message({"ok": True}, 900)
        self.assertNotIn("shortened", message)
        self.assertEqual(json.loads(message), {"ok": True})


if __name__ == "__main__":
    unittest.main()
