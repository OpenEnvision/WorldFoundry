"""Policy validation for in-tree model targets.

Ensures that integrated, runnable model entries keep their ``runner_target``
and ``pipeline_target`` imports inside the ``worldfoundry`` package, preventing
out-of-tree coupling that would break reproducibility or CI gating.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .schema import ModelVariantSpec, ModelZooEntry
from .zoo_registry import load_model_zoo_registry

# ── Constants ────────────────────────────────────────────────

IN_TREE_MODULE_PREFIX = "worldfoundry."


# ── Policy issue dataclass ────────────────────────────────────


@dataclass(frozen=True)
class ModelPolicyIssue:
    """Represents a policy compliance issue for a model or variant.

    Attributes:
        model_id: The model identifier flagged by the check.
        field: The manifest field that violated policy (e.g. ``"runner_target"``).
        value: The offending value found in that field.
        reason: Human-readable explanation of why the value is disallowed.
        variant_id: Optional variant identifier if the issue is variant-level.
    """

    model_id: str
    field: str
    value: str
    reason: str
    variant_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the policy issue to a plain dictionary representation."""
        return {
            "model_id": self.model_id,
            "variant_id": self.variant_id,
            "field": self.field,
            "value": self.value,
            "reason": self.reason,
        }


# ── In-tree target checks ────────────────────────────────────


def is_in_tree_target(value: str | None) -> bool:
    """Return true when a ``module:Class`` target resolves under the worldfoundry package."""

    if value in (None, ""):
        return True
    module_name = str(value).partition(":")[0].strip()
    return module_name == "worldfoundry" or module_name.startswith(IN_TREE_MODULE_PREFIX)


def _issue(
    entry: ModelZooEntry,
    *,
    field: str,
    value: str | None,
    variant: ModelVariantSpec | None = None,
) -> ModelPolicyIssue | None:
    """Create a policy issue if the given runner or pipeline import target is not in-tree."""
    if is_in_tree_target(value):
        return None
    return ModelPolicyIssue(
        model_id=entry.model_id,
        variant_id=None if variant is None else variant.variant_id,
        field=field,
        value=str(value),
        reason="integrated runnable model targets must import from the in-tree worldfoundry package",
    )


def validate_in_tree_model_entry(entry: ModelZooEntry | Mapping[str, object]) -> tuple[ModelPolicyIssue, ...]:
    """Validate the open-source policy for one model-zoo entry.

    Integrated runnable models may reference upstream sources and checkpoint repos in metadata, but their
    runner and pipeline import targets must live in this repository.
    """

    model = entry if isinstance(entry, ModelZooEntry) else ModelZooEntry.from_dict(entry)
    issues: list[ModelPolicyIssue] = []
    if model.is_runnable_runner_entry or model.integration_status == "integrated":
        for field, value in (
            ("runner_target", model.runner_target),
            ("pipeline_target", model.pipeline_target),
        ):
            item = _issue(model, field=field, value=value)
            if item is not None:
                issues.append(item)
    for variant in model.variants:
        if variant.is_runnable_runner_entry or variant.integration_status == "integrated":
            for field, value in (
                ("variants.runner_target", variant.runner_target),
                ("variants.pipeline_target", variant.pipeline_target),
            ):
                item = _issue(model, field=field, value=value, variant=variant)
                if item is not None:
                    issues.append(item)
    return tuple(issues)


# ── Registry-level validation ─────────────────────────────────


def validate_in_tree_model_registry(path: str | Path | None = None) -> tuple[ModelPolicyIssue, ...]:
    """Validate policy compliance for all model-zoo entries in a directory."""
    issues: list[ModelPolicyIssue] = []
    for entry in load_model_zoo_registry(path).list():
        issues.extend(validate_in_tree_model_entry(entry))
    return tuple(issues)


def summarize_model_policy_issues(issues: Iterable[ModelPolicyIssue]) -> dict[str, object]:
    """Summarize a collection of model policy issues into a report dictionary."""
    rows = [issue.to_dict() for issue in issues]
    return {
        "schema_version": "worldfoundry-model-policy-summary",
        "ok": not rows,
        "issue_count": len(rows),
        "issues": rows,
    }
