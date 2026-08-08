"""Deterministic compaction of tool results before they reach the model.

Tool output is the largest and least predictable thing that enters the context. A
single ``ls -la /usr/lib`` tokenises to roughly 33k tokens -- several times the whole
context window -- so an unbounded result does not merely slow inference down, it
displaces the conversation and the system prompt.

Bounding this in the framework rather than asking the model to cope is deliberate: the
model's job is to reason, and every token it does not have to read is prompt-processing
time it does not have to spend. Truncation is middle-out so the shape of a result and
its ending (where errors and totals live) both survive, and the model is told what was
removed so it can ask for a narrower slice instead of guessing.
"""

from __future__ import annotations

import json
from typing import Any


# Dense output -- paths, punctuation, code -- tokenises at roughly two characters per
# token, well below the ~4 that prose averages. Budgeting with the pessimistic figure
# keeps a result inside its token allowance for any content.
CHARS_PER_TOKEN = 2

# Fields whose value is a large blob of text worth shortening in place, so the rest of
# the structured result (exit codes, paths, flags) stays intact and machine-readable.
_TEXT_FIELDS = ("stdout", "stderr", "content", "matches")

# Reserved when budgeting a field, so the notice itself cannot push a result over.
_TRUNCATION_MARK = "\n… [0000000 characters removed from the middle] …\n"


def shorten(text: str, budget_chars: int) -> tuple[str, bool]:
    """Trim ``text`` to ``budget_chars``, keeping the head and the tail."""
    if len(text) <= budget_chars or budget_chars <= 0:
        return text, False
    if budget_chars < 200:
        return text[:budget_chars], True
    head = budget_chars * 2 // 3
    tail = budget_chars - head
    removed = len(text) - budget_chars
    return f"{text[:head]}\n… [{removed} characters removed from the middle] …\n{text[-tail:]}", True


def compact(result: Any, max_tokens: int) -> tuple[Any, bool]:
    """Return ``result`` reduced to roughly ``max_tokens``, plus whether it was cut.

    Large text fields are shortened individually first so that a result stays valid
    JSON with its structure readable. Anything still over budget after that is
    serialised and cut as a whole.
    """
    budget = max(0, max_tokens) * CHARS_PER_TOKEN
    truncated = False

    if isinstance(result, dict):
        result = dict(result)
        entries = result.get("entries")
        if isinstance(entries, list) and entries:
            # Keep as many entries as the budget actually affords rather than a fixed
            # count: a directory of long paths costs far more per entry than short ones.
            per_entry = max(1, len(json.dumps(entries, ensure_ascii=False)) // len(entries))
            affordable = max(1, (budget * 3 // 4) // per_entry)
            if len(entries) > affordable:
                result["entries"] = entries[:affordable]
                result["entries_omitted"] = len(entries) - affordable
                truncated = True

        # Shortening the large fields has to leave room for the rest of the structure,
        # otherwise the encoded result still exceeds the budget and gets flattened into
        # a string, losing the exit codes and paths the model needs. The budget is
        # measured against the encoded form, because JSON escaping (newlines, non-ASCII)
        # makes the serialised field longer than the raw text it came from.
        blobs = [key for key in _TEXT_FIELDS if isinstance(result.get(key), str) and result[key]]
        if blobs:
            original = {key: result[key] for key in blobs}
            skeleton = dict(result)
            for key in blobs:
                skeleton[key] = ""
            overhead = len(json.dumps(skeleton, ensure_ascii=False))
            share = max(120, (budget - overhead) // len(blobs))
            for _ in range(8):
                for key in blobs:
                    result[key], cut = shorten(original[key], share)
                    truncated = truncated or cut
                if len(json.dumps(result, ensure_ascii=False)) <= budget or share <= 120:
                    break
                share = max(120, share * 3 // 4)

    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded) > budget:
        encoded, cut = shorten(encoded, budget)
        truncated = truncated or cut
        return encoded, truncated
    if truncated and isinstance(result, dict):
        result["truncated"] = True
    return result, truncated


def as_tool_message(result: Any, max_tokens: int) -> str:
    """Serialise a tool result for the model, bounded and labelled when shortened."""
    reduced, truncated = compact(result, max_tokens)
    payload = reduced if isinstance(reduced, str) else json.dumps(reduced, ensure_ascii=False)
    if truncated:
        payload += (
            "\n[This result was shortened to fit the context. Request a narrower path, "
            "a search, or a specific range if you need more.]"
        )
    return payload
