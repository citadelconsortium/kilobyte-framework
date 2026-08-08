import unittest

from kilobyte.render import MarkdownStream
from kilobyte.theme import visible_len


class MarkdownStreamTests(unittest.TestCase):
    def test_only_completed_lines_are_emitted(self):
        """Streaming: a line without its newline is held back, not printed half-formatted."""
        stream = MarkdownStream()
        self.assertEqual(stream.feed("partial"), "")
        out = stream.feed(" line\n")
        self.assertIn("partial line", out)
        self.assertTrue(out.endswith("\n"))

    def test_bold_split_across_chunks_renders_once_whole(self):
        stream = MarkdownStream()
        self.assertEqual(stream.feed("this is **bo"), "")
        out = stream.feed("ld** text\n")
        # The asterisks are consumed, the word survives.
        self.assertNotIn("**", out)
        self.assertIn("bold", out)

    def test_code_fence_content_is_verbatim(self):
        stream = MarkdownStream()
        rendered = stream.feed("```python\ndef f():\n    return **1**\n```\n")
        # Inside a fence, markdown is not applied: the asterisks stay.
        self.assertIn("return **1**", rendered)

    def test_flush_returns_trailing_partial_line(self):
        stream = MarkdownStream()
        stream.feed("no newline here")
        self.assertIn("no newline here", stream.flush())
        self.assertEqual(stream.flush(), "")

    def test_headings_and_bullets_are_transformed(self):
        stream = MarkdownStream()
        out = stream.feed("# Title\n- item\n1. first\n")
        self.assertNotIn("#", out)
        self.assertIn("Title", out)
        self.assertIn("item", out)
        self.assertIn("first", out)


class ThemeTests(unittest.TestCase):
    def test_visible_len_ignores_escape_sequences(self):
        self.assertEqual(visible_len("\033[38;5;84mKILO\033[0m"), 4)


if __name__ == "__main__":
    unittest.main()
