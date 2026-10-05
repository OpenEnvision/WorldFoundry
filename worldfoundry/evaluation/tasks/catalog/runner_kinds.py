"""Closed vocabulary for benchmark runner runtime kinds.

Benchmark manifests historically coined increasingly specific ``runtime.kind``
values.  Keep those spellings as compatibility aliases, but normalize them to
the small set of execution categories consumed by catalog and orchestration
code.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

EXTERNAL_RUNNER_KINDS: frozenset[str] = frozenset(
    {
        "external_official_repo",
        "external_official_results_runner",
    }
)

CANONICAL_IN_TREE_RESULT_NORMALIZER_KINDS: frozenset[str] = frozenset(
    {
        "in_tree_artifact_metric_importer",
        "in_tree_metric_aggregator",
        "in_tree_result_importer",
    }
)

CANONICAL_IN_TREE_EXECUTION_KINDS: frozenset[str] = frozenset(
    {
        "in_tree_artifact_evaluator",
        "in_tree_judge_runtime",
        "in_tree_official_judge_runtime",
        "in_tree_official_runner",
        "in_tree_official_runtime",
        "in_tree_official_source",
        "native_closed_loop_simulator",
    }
)

CANONICAL_IN_TREE_RUNNER_KINDS: frozenset[str] = frozenset(
    CANONICAL_IN_TREE_RESULT_NORMALIZER_KINDS | CANONICAL_IN_TREE_EXECUTION_KINDS
)
CANONICAL_RUNNER_KINDS: frozenset[str] = frozenset(CANONICAL_IN_TREE_RUNNER_KINDS | EXTERNAL_RUNNER_KINDS)

# Compatibility spellings already published in checked-in manifests.  The
# right-hand side is the orchestration category; details such as bounded
# metrics or hosted judges remain available in the rest of the runtime spec.
RUNNER_KIND_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "in_tree_composite_evaluator": "in_tree_artifact_evaluator",
        "in_tree_model_backed": "in_tree_artifact_evaluator",
        "in_tree_model_backed_evaluator": "in_tree_artifact_evaluator",
        "in_tree_multi_track_evaluator": "in_tree_artifact_evaluator",
        "in_tree_official_runtime_bounded_clip_metrics": "in_tree_official_runtime",
        "in_tree_official_runtime_hosted_judges": "in_tree_official_judge_runtime",
        "in_tree_protocol_runtime_external_assets": "in_tree_official_runtime",
        "in_tree_result_importer_and_bounded_caption_qa": "in_tree_artifact_metric_importer",
        "in_tree_result_importer_and_bounded_overall_aggregator": "in_tree_metric_aggregator",
    }
)

# Public accepted vocabulary.  Consumers that need execution behavior should
# use the derived semantic groups below or normalize the value first.
RUNNER_KINDS: frozenset[str] = frozenset(CANONICAL_RUNNER_KINDS | RUNNER_KIND_ALIASES.keys())
IN_TREE_RUNTIME_KINDS: frozenset[str] = frozenset(
    kind
    for kind in RUNNER_KINDS
    if RUNNER_KIND_ALIASES.get(kind, kind) in CANONICAL_IN_TREE_RUNNER_KINDS
)
IN_TREE_RESULT_NORMALIZER_KINDS: frozenset[str] = frozenset(
    kind
    for kind in RUNNER_KINDS
    if RUNNER_KIND_ALIASES.get(kind, kind) in CANONICAL_IN_TREE_RESULT_NORMALIZER_KINDS
)
IN_TREE_EXECUTION_KINDS: frozenset[str] = frozenset(IN_TREE_RUNTIME_KINDS - IN_TREE_RESULT_NORMALIZER_KINDS)


def normalize_runner_kind(value: object, *, context: str = "runner.runtime.kind") -> str:
    """Validate and return the canonical runner kind for ``value``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    kind = value.strip()
    if kind not in RUNNER_KINDS:
        allowed_values = ", ".join(sorted(RUNNER_KINDS))
        raise ValueError(f"{context} must be one of: {allowed_values}. Got {kind!r}.")
    return RUNNER_KIND_ALIASES.get(kind, kind)
