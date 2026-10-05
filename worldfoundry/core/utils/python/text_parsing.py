"""Dependency-free helpers for parsing model text outputs.

Responsibility
    Extract a final ``yes``/``no`` from judge / VLM text so benchmarks share
    one parser instead of forking strip logic.

Boundaries
    No torch, no tokenizer, no generation. Regexes stay here; adding a new
    eval parser belongs in this module only if it stays dependency-free.

Public surface
    :func:`extract_yes_no_answer`.
"""

from __future__ import annotations

import re
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Yes/no extractors — structured spans win over free-form reasoning
# ──────────────────────────────────────────────────────────────────────────

_YES_NO_TOKEN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_STRUCTURED_YES_NO_PATTERNS = (
    re.compile(r"\\boxed\s*\{([^{}]*)\}", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<(?:answer|final)>\s*(.*?)\s*</(?:answer|final)>",
        re.IGNORECASE | re.DOTALL,
    ),
    # Some generation backends truncate before emitting the closing tag.
    re.compile(r"<(?:answer|final)>\s*(yes|no)\b", re.IGNORECASE),
    re.compile(
        r"[\"']?(?:answer|final)[\"']?\s*[:=]\s*[\"']?\s*(yes|no)\b",
        re.IGNORECASE,
    ),
)


def extract_yes_no_answer(value: Any, *, default: str = "no") -> str:
    """Return the final explicit ``yes``/``no`` answer in a model response.

    Structured answers (JSON, ``Answer:``, XML tags, or LaTeX ``\\boxed``)
    take precedence over reasoning text. The fallback uses whole-word matches,
    so strings such as ``yesterday`` and ``nobody`` are not misclassified.
    """

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)

    structured: list[tuple[int, str]] = []
    for pattern in _STRUCTURED_YES_NO_PATTERNS:
        structured.extend((match.start(), match.group(1)) for match in pattern.finditer(text))

    if structured:
        _, candidate = max(structured, key=lambda item: item[0])
        matches = _YES_NO_TOKEN.findall(candidate)
        if matches:
            return matches[-1].lower()

    matches = _YES_NO_TOKEN.findall(text)
    return matches[-1].lower() if matches else default


__all__ = ["extract_yes_no_answer"]
