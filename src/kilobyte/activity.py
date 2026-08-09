"""Safe, compact rendering for live machine activity in every interface."""

from __future__ import annotations

import json
import re
from typing import Any


_SECRET_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
}
_TELEGRAM_TOKEN = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|password|secret|token)\s*([:=])\s*([^\s,;]+)"
)


def redact_text(value: Any, limit: int = 600) -> str:
    """Remove common credentials before activity is mirrored to a UI or Telegram."""
    text = str(value).replace("\x00", "")
    text = _TELEGRAM_TOKEN.sub("[redacted-token]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    text = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    if len(text) > limit:
        return text[: max(0, limit - 1)] + "…"
    return text


def _is_secret_key(key: str) -> bool:
    normal = key.strip().lower().replace("-", "_")
    return normal in _SECRET_KEYS or normal.endswith("_password") or normal.endswith("_secret")


def format_arguments(arguments: dict[str, Any] | None, limit: int = 600) -> str:
    """Show every tool argument that fits, redacting credential-shaped fields."""
    parts: list[str] = []
    for key, value in (arguments or {}).items():
        if _is_secret_key(str(key)):
            rendered = "[redacted]"
        else:
            try:
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                rendered = str(value)
            rendered = redact_text(rendered, limit)
        parts.append(f"{key}={rendered}")
    return redact_text(" · ".join(parts), limit)


def format_summary(summary: Any, limit: int = 600) -> str:
    return redact_text(summary, limit)
