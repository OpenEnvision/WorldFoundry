from __future__ import annotations

import pytest

from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.metrics import (
    BuiltinExistingResultsMetric,
    DuplicateMetricRegistryKeyError,
    MetricRegistry,
    MetricRegistryEntry,
    UnknownMetricRegistryKeyError,
    default_metric_registry,
    validate_metric_ids,
)


def test_metric_registry_lists_and_resolves_builtin_metrics() -> None:
    registry = default_metric_registry()

    names = {entry.id for entry in registry.list()}
    has_artifact = registry.get("has_artifact:video")
    has_artifact_alias = registry.get("has-artifact:video")
    numeric_value = registry.get("numeric:quality")

    assert {"artifact_count", "required_artifacts_present", "has_artifact", "numeric", "numeric_value"} <= names
    assert has_artifact.id == "has_artifact"
    assert has_artifact_alias.id == "has_artifact"
    assert has_artifact.parameterized_prefix == "has_artifact:"
    assert numeric_value.id == "numeric_value"


def test_builtin_existing_results_metric_preserves_existing_outputs() -> None:
    metric = BuiltinExistingResultsMetric(
        metrics=("artifact-count", "has-artifact:video", "numeric:quality", "numeric"),
        required_artifacts=("video", "depth"),
    )
    request = GenerationRequest(sample_id="sample-a", task_name="metric-test")
    result = GenerationResult(
        sample_id="sample-a",
        artifacts={"video": ArtifactRef(uri="memory://video.mp4", kind="video")},
        metadata={
            "metrics": {"quality": 0.75},
            "scores": {"alignment": 0.5},
            "extra": {"extra_score": 0.25},
        },
        timings={"latency": 1.5},
    )

    output = metric(request, result)

    assert output["metrics"]["artifact_count"] == 1.0
    assert output["metrics"]["has_artifact:video"] == 1.0
    assert output["metrics"]["has_artifact:depth"] == 0.0
    assert output["metrics"]["required_artifacts_present"] == 0.0
    assert output["metrics"]["quality"] == 0.75
    assert output["metrics"]["alignment"] == 0.5
    assert output["metrics"]["extra_score"] == 0.25
    assert output["metrics"]["timing:latency"] == 1.5
    assert output["metrics"]["generation_success"] == 1.0


def test_metric_registry_validation_reports_unknown_metric_ids() -> None:
    payload = validate_metric_ids(["artifact-count", "has-artifact:video", "has_artifact:", "not_real"])

    assert payload["ok"] is False
    assert [item["metric_id"] for item in payload["metrics"]] == ["artifact-count", "has-artifact:video"]
    assert [item["canonical_metric_id"] for item in payload["metrics"]] == ["artifact_count", "has_artifact:video"]
    assert payload["unknown_metrics"] == ["has_artifact:", "not_real"]

    with pytest.raises(UnknownMetricRegistryKeyError):
        BuiltinExistingResultsMetric(metrics=("not_real",))


def test_metric_registry_replace_removes_old_aliases() -> None:
    registry = MetricRegistry(include_builtins=False)
    registry.register(MetricRegistryEntry(id="quality", aliases=("old-quality",)))

    registry.register(MetricRegistryEntry(id="quality", aliases=("new-quality",)), replace=True)

    assert registry.get("quality").aliases == ("new-quality",)
    assert registry.get("new_quality").id == "quality"
    with pytest.raises(UnknownMetricRegistryKeyError):
        registry.get("old-quality")


def test_metric_registry_replace_does_not_steal_other_entry_alias() -> None:
    registry = MetricRegistry(include_builtins=False)
    registry.register(MetricRegistryEntry(id="latency", aliases=("shared",)))
    registry.register(MetricRegistryEntry(id="quality", aliases=("quality-score",)))

    with pytest.raises(DuplicateMetricRegistryKeyError):
        registry.register(MetricRegistryEntry(id="quality", aliases=("shared",)), replace=True)

    assert registry.get("shared").id == "latency"
    assert registry.get("quality-score").id == "quality"
