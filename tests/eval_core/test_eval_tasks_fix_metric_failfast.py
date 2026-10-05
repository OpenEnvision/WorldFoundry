"""CPU-only regression tests for review fix ET-11.

Registered metric ids without an offline implementation branch (fid, cmmd,
vqa_score, ...) must be rejected at validation time instead of silently
producing no output.
"""

from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.metrics.registry import (
    BuiltinExistingResultsMetric,
    MetricRegistryError,
    UnknownMetricRegistryKeyError,
    create_existing_results_metric,
    default_metric_registry,
    is_offline_computable_metric_id,
)


def test_registered_but_uncomputable_metric_ids_fail_fast() -> None:
    for metric_id in ("fid", "cmmd", "vqa_score", "clip_score"):
        # These ids resolve in the registry (they are real metric packages) ...
        assert default_metric_registry().get(metric_id) is not None
        # ... but the offline evaluator cannot compute them, so it must refuse.
        with pytest.raises(MetricRegistryError, match=metric_id):
            BuiltinExistingResultsMetric(metrics=(metric_id,))


def test_error_message_lists_supported_offline_forms() -> None:
    with pytest.raises(MetricRegistryError) as excinfo:
        create_existing_results_metric(metrics=("fid",))
    message = str(excinfo.value)
    for supported in ("artifact_count", "required_artifacts_present", "numeric", "has_artifact:", "numeric:"):
        assert supported in message


def test_offline_computable_ids_still_work() -> None:
    metric = BuiltinExistingResultsMetric(
        metrics=("artifact_count", "required_artifacts_present", "numeric", "has_artifact:video", "numeric:foo"),
    )
    assert metric.metrics == (
        "artifact_count",
        "required_artifacts_present",
        "numeric",
        "has_artifact:video",
        "numeric:foo",
    )


def test_unknown_ids_keep_raising_unknown_key_error() -> None:
    with pytest.raises(UnknownMetricRegistryKeyError):
        BuiltinExistingResultsMetric(metrics=("not_real",))


def test_spec_implementation_field_is_honest() -> None:
    registry = default_metric_registry()
    offline_impl = "worldfoundry.evaluation.tasks.metrics.registry:BuiltinExistingResultsMetric"
    assert registry.get("artifact_count").spec.implementation == offline_impl
    assert registry.get("fid").spec.implementation is None
    assert registry.get("vqa_score").spec.implementation is None


def test_is_offline_computable_metric_id_matrix() -> None:
    assert is_offline_computable_metric_id("artifact_count")
    assert is_offline_computable_metric_id("has_artifact:video")
    assert is_offline_computable_metric_id("numeric:quality")
    assert not is_offline_computable_metric_id("fid")
    assert not is_offline_computable_metric_id("cmmd")


def test_evaluate_metric_callable_reports_benchmark_supported_metrics() -> None:
    from worldfoundry.evaluation.tasks.execution.framework.in_tree_registry import target_benchmark_metrics
    from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import _metric_callable

    benchmarks = target_benchmark_metrics()
    assert benchmarks, "at least one in-tree benchmark is expected"
    benchmark_id = sorted(benchmarks)[0]

    with pytest.raises(MetricRegistryError) as excinfo:
        _metric_callable(["fid"], (), benchmark_id)
    message = str(excinfo.value)
    assert benchmark_id in message
    assert benchmarks[benchmark_id][0] in message
