import unittest

from kilobyte.telegram_render import telegram_html, telegram_html_chunks


class TelegramRenderTests(unittest.TestCase):
    def test_renders_research_markdown_as_safe_readable_html(self):
        source = (
            "## Verified findings\n"
            "- **Owner:** Sir & Co\n"
            "- [Official source](https://example.com/a?x=1&y=2)\n"
            "Use `web_search`, not <guessing>."
        )
        rendered = telegram_html(source)
        self.assertIn("<b>Verified findings</b>", rendered)
        self.assertIn("• <b>Owner:</b> Sir &amp; Co", rendered)
        self.assertIn('<a href="https://example.com/a?x=1&amp;y=2">Official source</a>', rendered)
        self.assertIn("<code>web_search</code>", rendered)
        self.assertIn("&lt;guessing&gt;", rendered)

    def test_strips_tool_protocol_as_defence_in_depth(self):
        rendered = telegram_html(
            "Intro<tool_call><function=web_search></function></tool_call>Done"
        )
        self.assertEqual(rendered, "IntroDone")
        self.assertNotIn("tool_call", rendered)

    def test_long_html_is_split_with_balanced_tags_and_whole_entities(self):
        chunks = telegram_html_chunks("<b>" + ("word &amp; " * 900) + "</b>", 300)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))
        self.assertTrue(all(chunk.startswith("<b>") and chunk.endswith("</b>") for chunk in chunks))
        self.assertTrue(all("&am" not in chunk.replace("&amp;", "") for chunk in chunks))

    def test_excess_blank_lines_are_collapsed_outside_code(self):
        rendered = telegram_html("First\n\n\n\nSecond\n```\na\n\nb\n```")
        self.assertIn("First\n\nSecond", rendered)
        self.assertIn("<pre>a\n\nb</pre>", rendered)


if __name__ == "__main__":
    unittest.main()
