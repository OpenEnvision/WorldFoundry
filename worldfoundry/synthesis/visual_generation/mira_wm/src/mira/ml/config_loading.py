"""Helpers for loading model configs saved alongside checkpoints.

Checkpoints save their architecture config as a hydra-instantiated YAML. Two kinds of cruft creep in
that the strict (``extra="forbid"``) Pydantic schemas reject:

* ``_target_`` keys hydra writes at every level it instantiated (e.g. the top-level config and a
  nested ``video`` image config), which carry no information the schema needs.
* Fields that have since been removed from the schema but were still written, at their no-op value,
  by older training runs.

:func:`strip_hydra_targets` removes the former everywhere; :func:`drop_removed_fields` removes the
latter only when present at its documented no-op value, and raises otherwise so a checkpoint that
genuinely exercised a since-removed feature still fails loudly instead of loading silently wrong.
"""

from __future__ import annotations

from typing import Mapping


def strip_hydra_targets(node: object) -> object:
    """Return ``node`` with every ``_target_`` key recursively removed from nested dicts/lists."""
    if isinstance(node, dict):
        return {k: strip_hydra_targets(v) for k, v in node.items() if k != "_target_"}
    if isinstance(node, list):
        return [strip_hydra_targets(v) for v in node]
    return node


def drop_removed_fields(node: object, removed: Mapping[str, object]) -> object:
    """Recursively drop known removed fields, tolerating them only at their no-op value.

    Args:
        node: A plain dict/list config tree (resolve OmegaConf to a container first).
        removed: Maps a removed field name to the no-op value it is allowed to carry. A matching key
            at that value is dropped; at any other value a :class:`ValueError` is raised.

    Returns:
        A new tree with the removed fields filtered out.
    """
    if isinstance(node, dict):
        cleaned: dict = {}
        for key, value in node.items():
            if key in removed:
                noop = removed[key]
                if value != noop:
                    raise ValueError(
                        f"Config sets dropped field {key!r}={value!r}, which is only tolerated at "
                        f"its no-op value {noop!r}. This checkpoint used a feature that has since "
                        f"been removed and cannot be loaded."
                    )
                continue
            cleaned[key] = drop_removed_fields(value, removed)
        return cleaned
    if isinstance(node, list):
        return [drop_removed_fields(v, removed) for v in node]
    return node
