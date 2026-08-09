"""Small, safe Markdown-to-Telegram-HTML renderer for model replies."""

from __future__ import annotations

import html
import re

_INLINE = re.compile(
    r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)|\*\*([^*\n]+)\*\*|`([^`\n]+)`"
)
_TOOL_BLOCK = re.compile(
    r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL
)
_HTML_TOKEN = re.compile(r"</?(?:b|i|u|s|code|pre|a)(?:\s[^>]*)?>|&(?:#\d+|#x[0-9a-f]+|\w+);", re.IGNORECASE)
_OPEN_TAG = re.compile(r"<((?:b|i|u|s|code|pre|a))(?:\s[^>]*)?>", re.IGNORECASE)
_CLOSE_TAG = re.compile(r"</((?:b|i|u|s|code|pre|a))>", re.IGNORECASE)


def _inline(text: str) -> str:
    pieces: list[str] = []
    position = 0
    for match in _INLINE.finditer(text):
        pieces.append(html.escape(text[position : match.start()]))
        label, url, bold, code = match.groups()
        if url is not None:
            pieces.append(
                f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'
            )
        elif bold is not None:
            pieces.append(f"<b>{html.escape(bold)}</b>")
        else:
            pieces.append(f"<code>{html.escape(code or '')}</code>")
        position = match.end()
    pieces.append(html.escape(text[position:]))
    return "".join(pieces)


def telegram_html(text: str) -> str:
    """Render common model Markdown cleanly using Telegram's restricted HTML subset."""
    text = _TOOL_BLOCK.sub("", text).strip()
    rendered: list[str] = []
    code_lines: list[str] = []
    in_code = False
    for raw_line in text.splitlines():
        if raw_line.strip().startswith("```"):
            if in_code:
                rendered.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(raw_line)
            continue
        if not raw_line.strip():
            # Provider replies often pad tool transitions with many blank lines. Keep
            # one section break, never a screenful of empty Telegram message space.
            if rendered and rendered[-1] != "":
                rendered.append("")
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", raw_line)
        bullet = re.match(r"^(\s*)[-+*]\s+(.+)$", raw_line)
        quote = re.match(r"^\s*>\s?(.*)$", raw_line)
        if heading:
            rendered.append(f"<b>{_inline(heading.group(1))}</b>")
        elif bullet:
            rendered.append(f"{bullet.group(1)}• {_inline(bullet.group(2))}")
        elif quote:
            rendered.append("<i>│ " + _inline(quote.group(1)) + "</i>")
        elif re.fullmatch(r"\s*[-_*]{3,}\s*", raw_line):
            rendered.append("────────")
        else:
            rendered.append(_inline(raw_line))
    if in_code:
        rendered.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>")
    return "\n".join(rendered).strip()


def telegram_html_chunks(text: str, limit: int = 3900) -> list[str]:
    """Split HTML without cutting an entity or leaving Telegram formatting unbalanced."""
    if not text:
        return ["(empty response)"]
    chunks: list[str] = []
    current = ""
    opened: list[tuple[str, str]] = []

    def closures() -> str:
        return "".join(f"</{name}>" for name, _tag in reversed(opened))

    def finish() -> None:
        nonlocal current
        if current:
            chunks.append(current + closures())
        current = "".join(tag for _name, tag in opened)

    position = 0
    tokens: list[str] = []
    for match in _HTML_TOKEN.finditer(text):
        if match.start() > position:
            tokens.append(text[position : match.start()])
        tokens.append(match.group(0))
        position = match.end()
    if position < len(text):
        tokens.append(text[position:])

    for token in tokens:
        is_markup = bool(_HTML_TOKEN.fullmatch(token))
        if is_markup:
            if len(current) + len(token) + len(closures()) > limit:
                finish()
            current += token
            if closing := _CLOSE_TAG.fullmatch(token):
                name = closing.group(1).lower()
                for index in range(len(opened) - 1, -1, -1):
                    if opened[index][0] == name:
                        opened.pop(index)
                        break
            elif opening := _OPEN_TAG.fullmatch(token):
                opened.append((opening.group(1).lower(), token))
            continue
        remaining = token
        while remaining:
            room = limit - len(current) - len(closures())
            if room <= 0:
                finish()
                continue
            if len(remaining) <= room:
                current += remaining
                break
            split = max(remaining.rfind("\n", 0, room + 1), remaining.rfind(" ", 0, room + 1))
            if split < 1:
                split = room
            current += remaining[:split]
            remaining = remaining[split:].lstrip("\n")
            finish()
    finish()
    return chunks or ["(empty response)"]
