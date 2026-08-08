"""One place for every colour, glyph and box character the interface uses.

Styling lived inline in many f-strings, which is how a terminal UI drifts into
inconsistency. Centralising it means a panel, a spinner and a status line all speak the
same visual language, and a change to the palette is one edit rather than twenty.

Everything degrades: with no TTY or NO_COLOR set, the colour codes become empty strings
and the box characters fall back to ASCII, so output stays legible when piped or on a
terminal without Unicode.
"""

from __future__ import annotations

import os
import re
import sys


def _supports_style() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("KILO_PLAIN"):
        return False
    return sys.stdout.isatty()


def _supports_unicode() -> bool:
    if not _supports_style():
        return False
    encoding = (sys.stdout.encoding or "").lower()
    return "utf" in encoding


STYLE = _supports_style()
UNICODE = _supports_unicode()


def _c(code: str) -> str:
    return f"\033[{code}m" if STYLE else ""


# Palette. Green is Kilo's signal colour; the rest are used sparingly and consistently.
GREEN = _c("38;5;84")
CYAN = _c("38;5;51")
PURPLE = _c("38;5;141")
YELLOW = _c("38;5;220")
RED = _c("38;5;203")
GREY = _c("38;5;245")
DIM = _c("2")
BOLD = _c("1")
RESET = _c("0")


class Box:
    """Border pieces, with an ASCII fallback for terminals without line-drawing."""

    if UNICODE:
        h, v = "─", "│"
        tl, tr, bl, br = "╭", "╮", "╰", "╯"
    else:
        h, v = "-", "|"
        tl, tr, bl, br = "+", "+", "+", "+"


# Activity spinner. The braille cycle reads as motion; the ASCII fallback still turns.
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if UNICODE else "|/-\\"

# Status markers, each a fixed meaning so the eye learns them.
TOOL = f"{PURPLE}◈{RESET}" if UNICODE else f"{PURPLE}*{RESET}"
OK = f"{GREEN}✓{RESET}" if UNICODE else f"{GREEN}+{RESET}"
WARN = f"{YELLOW}!{RESET}"
FAIL = f"{RED}✗{RESET}" if UNICODE else f"{RED}x{RESET}"
CLOUD = f"{YELLOW}☁{RESET}" if UNICODE else f"{YELLOW}~{RESET}"
DOT_ON = f"{GREEN}●{RESET}" if UNICODE else f"{GREEN}o{RESET}"
DOT_OFF = f"{YELLOW}●{RESET}" if UNICODE else f"{YELLOW}o{RESET}"

# Human-readable activity words, cycled while Kilo works so the indicator conveys a live
# operator at work rather than a frozen "step 1". None of these expose model reasoning;
# they describe the phase, which is an observable state.
ACTIVITY_WORDS = (
    "thinking",
    "reasoning",
    "planning",
    "working",
    "considering",
    "analysing",
    "composing",
)

_ANSI = re.compile(r"\033\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Rendered width, ignoring escape sequences, so padding and borders line up."""
    return len(_ANSI.sub("", text))
