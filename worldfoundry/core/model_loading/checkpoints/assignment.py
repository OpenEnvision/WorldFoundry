"""Strict, allocation-free state-dict assignment for inference models.

Released weights are often assigned onto a meta-device skeleton and
only later moved to the chosen compute dtype.
:func:`validate_state_dict_compatibility` rejects missing, unexpected,
or shape-mismatched keys without consulting dtype.
:func:`assign_state_dict_strict` then loads with ``strict=True`` and
``assign=True`` so tensors are bound in place rather than copied.

This is not a remapper and does not unwrap nested checkpoint wrappers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Compatibility gate — dtype-agnostic so meta skeletons can take FP32 weights
# ──────────────────────────────────────────────────────────────────────────


def validate_state_dict_compatibility(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    label: str = "checkpoint",
) -> None:
    """Reject missing, unexpected, or shape-incompatible checkpoint tensors.

    This validation is intentionally independent of tensor dtype: released
    FP32 weights may be assigned to a meta-device model and converted to the
    selected inference dtype during final placement.
    """

    expected = module.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    mismatched = sorted(
        (
            key,
            tuple(getattr(state_dict[key], "shape", ())),
            tuple(getattr(expected[key], "shape", ())),
        )
        for key in expected_keys & actual_keys
        if tuple(getattr(state_dict[key], "shape", ()))
        != tuple(getattr(expected[key], "shape", ()))
    )
    if not (missing or unexpected or mismatched):
        return

    def preview(values: list[Any], limit: int = 12) -> str:
        """Cap the error payload so a 7B key dump cannot drown the real mismatch."""

        suffix = f" ... (+{len(values) - limit})" if len(values) > limit else ""
        return f"{values[:limit]}{suffix}"

    details = []
    if missing:
        details.append(f"missing={preview(missing)}")
    if unexpected:
        details.append(f"unexpected={preview(unexpected)}")
    if mismatched:
        details.append(f"shape_mismatches={preview(mismatched)}")
    raise RuntimeError(f"{label} is incompatible with {type(module).__name__}: " + "; ".join(details))


# ──────────────────────────────────────────────────────────────────────────
# In-place bind — assign=True so meta parameters are replaced, not copied
# ──────────────────────────────────────────────────────────────────────────


def assign_state_dict_strict(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    label: str = "checkpoint",
) -> Any:
    """Validate and assign tensors, including into a meta-device module."""

    validate_state_dict_compatibility(module, state_dict, label=label)
    return module.load_state_dict(dict(state_dict), strict=True, assign=True)


__all__ = ["assign_state_dict_strict", "validate_state_dict_compatibility"]
