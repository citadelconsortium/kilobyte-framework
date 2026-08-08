"""Streaming-friendly Markdown rendering for assistant output.

A terminal assistant that prints raw Markdown asterisks and backticks looks unfinished.
This turns the common Markdown a model produces -- headings, bold, italic, inline code,
fenced code blocks, list bullets -- into styled terminal text.

It is built for streaming: text arrives token by token, so the renderer keeps a small
amount of state (whether a code fence is open) and formats whole lines as they complete,
leaving a partial trailing line untouched until its newline arrives. Inline styling is
applied per completed line, never across a boundary, so a bold span split across two
tokens still renders once the line is whole.
"""

from __future__ import annotations

import re

from .theme import BOLD, CYAN, DIM, GREEN, GREY, RESET


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_FENCE = re.compile(r"^\s*```(\w*)\s*$")


def _inline(text: str) -> str:
    """Apply inline styles to one complete line of ordinary prose."""
    text = _CODE.sub(lambda m: f"{CYAN}{m.group(1)}{RESET}", text)
    text = _BOLD.sub(lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    text = _ITALIC.sub(lambda m: f"{DIM}{m.group(1)}{RESET}", text)
    return text


class MarkdownStream:
    """Formats assistant output line by line as it streams.

    ``feed`` returns the renderable text for any lines that completed with the chunk just
    received; ``flush`` returns whatever partial line remains at the end.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_code = False

    def _format_line(self, line: str) -> str:
        fence = _FENCE.match(line)
        if fence is not None:
            if not self._in_code:
                self._in_code = True
                lang = fence.group(1)
                return f"{GREY}┄┄ {lang or 'code'} " + "┄" * max(0, 40 - len(lang)) + RESET
            self._in_code = False
            return f"{GREY}{'┄' * 46}{RESET}"
        if self._in_code:
            # Code is shown verbatim in a distinct colour, never Markdown-processed.
            return f"{CYAN}{line}{RESET}"
        heading = _HEADING.match(line)
        if heading is not None:
            return f"{BOLD}{GREEN}{heading.group(2)}{RESET}"
        bullet = _BULLET.match(line)
        if bullet is not None:
            return f"{bullet.group(1)}{GREEN}•{RESET} {_inline(bullet.group(2))}"
        ordered = _ORDERED.match(line)
        if ordered is not None:
            return f"{ordered.group(1)}{GREEN}{ordered.group(2)}.{RESET} {_inline(ordered.group(3))}"
        if line.strip() in {"---", "***", "___"}:
            return f"{GREY}{'─' * 46}{RESET}"
        return _inline(line)

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            out.append(self._format_line(line))
        return "\n".join(out) + ("\n" if out else "")

    def flush(self) -> str:
        if not self._buffer:
            return ""
        # A trailing partial line: format what is there. In a code block it is shown raw.
        line = self._buffer
        self._buffer = ""
        return self._format_line(line) if not self._in_code else f"{CYAN}{line}{RESET}"
