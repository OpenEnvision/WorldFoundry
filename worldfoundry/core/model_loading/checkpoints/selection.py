"""Small, dependency-free helpers for selecting a profiled checkpoint variant.

Studio and eval profiles list several checkpoint records.
:func:`normalize_checkpoint_selector` folds punctuation so human
spellings compare equal. :func:`select_profile_checkpoint` picks
exactly one record after optional aliases; unknown or ambiguous
selectors raise. :func:`selected_checkpoint_options` translates the
chosen record into the shared runtime's path / repo / revision fields.

Matching is identity-only and does not download or inspect files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Identity matching — punctuation-folded aliases, never filesystem inspection
# ──────────────────────────────────────────────────────────────────────────

_SELECTOR_FIELDS = (
    "id",
    "variant",
    "variant_id",
    "role",
    "repo_id",
    "checkpoint_ref",
    "local_dir",
    "path",
    "checkpoint_path",
)


def normalize_checkpoint_selector(value: Any) -> str:
    """Normalize human-facing variant spellings without changing filesystem paths."""

    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _record_selectors(record: Mapping[str, Any]) -> set[str]:
    """Collect folded ids plus the last path component so ``role`` and a repo stem both match."""

    selectors: set[str] = set()
    for field in _SELECTOR_FIELDS:
        value = record.get(field)
        if value in (None, ""):
            continue
        text = str(value)
        selectors.add(normalize_checkpoint_selector(text))
        selectors.add(normalize_checkpoint_selector(text.rstrip("/").rsplit("/", 1)[-1]))
    return selectors


def select_profile_checkpoint(
    checkpoints: Sequence[Mapping[str, Any]],
    selector: Any,
    *,
    aliases: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Select exactly one profile checkpoint, rejecting unknown or ambiguous variants.

    ``aliases`` maps user-facing spellings to a stable value already present in a
    checkpoint record (normally its ``role``). Matching is exact after removing
    punctuation, so similar benchmark names cannot silently select one another.
    """

    if not checkpoints:
        raise ValueError("no checkpoint records are available")
    normalized = normalize_checkpoint_selector(selector)
    if not normalized:
        return checkpoints[0]
    normalized_aliases = {
        normalize_checkpoint_selector(alias): normalize_checkpoint_selector(target)
        for alias, target in dict(aliases or {}).items()
    }
    target = normalized_aliases.get(normalized, normalized)
    matches = [record for record in checkpoints if target in _record_selectors(record)]
    if len(matches) == 1:
        return matches[0]
    available = sorted(
        str(record.get("role") or record.get("variant_id") or record.get("repo_id") or "checkpoint")
        for record in checkpoints
    )
    if not matches:
        raise ValueError(
            f"unknown checkpoint variant {selector!r}; available checkpoint roles: {', '.join(available)}"
        )
    raise ValueError(
        f"ambiguous checkpoint variant {selector!r}; matched {len(matches)} records: {', '.join(available)}"
    )


def selected_checkpoint_options(record: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one profile record into the shared runtime's explicit options."""

    options: dict[str, Any] = {}
    for field in ("local_dir", "path", "checkpoint_path"):
        if record.get(field) not in (None, ""):
            options["checkpoint_path"] = record[field]
            break
    for field in ("repo_id", "checkpoint_ref", "checkpoint_repo_id", "hf_repo_id"):
        if record.get(field) not in (None, ""):
            options["checkpoint_ref"] = record[field]
            break
    if record.get("revision") not in (None, ""):
        options["revision"] = record["revision"]
    return options


__all__ = [
    "normalize_checkpoint_selector",
    "select_profile_checkpoint",
    "selected_checkpoint_options",
]
