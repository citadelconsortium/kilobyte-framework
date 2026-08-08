import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kilobyte.rpc import RPCClient
from kilobyte.theme import Box, visible_len
from kilobyte.tui import TerminalUI


class PanelRenderTests(unittest.TestCase):
    """Rendering must not depend on a real terminal; the service and tests run without one."""

    def setUp(self):
        self.ui = TerminalUI(RPCClient(Path("/nonexistent")))

    def test_width_has_a_sane_fallback_without_a_tty(self):
        self.assertGreaterEqual(self.ui._width(), 48)

    def test_panel_lines_are_padded_to_a_consistent_width(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.ui._panel("demo", "", ["short", "a somewhat longer line of text"])
        lines = [line for line in out.getvalue().splitlines() if line]
        widths = {visible_len(line) for line in lines}
        # Every rendered row of the panel is the same visible width.
        self.assertEqual(len(widths), 1, f"ragged panel: {sorted(widths)}")

    def test_panel_has_all_four_corners(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.ui._panel("demo", "", ["body"])
        text = out.getvalue()
        for corner in (Box.tl, Box.tr, Box.bl, Box.br):
            self.assertIn(corner, text)


if __name__ == "__main__":
    unittest.main()
