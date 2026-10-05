from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.metrics import (
    ExternalMetricEvaluationRequest,
    ExternalMetricEvaluatorEntry,
    ExternalMetricEvaluatorRegistry,
    get_external_metric_evaluator,
)


NEW_VIDEO_WORLD_CONTRACT_IDS = (
    "aigcbench",
    "mirabench",
    "devil-dynamics",
    "genai-bench",
    "phygenbench",
    "videophy",
    "videophy2",
    "physics-iq",
    "t2v-safety-bench",
    "ipv-bench",
    "videoscience-bench",
    "phyeduvideo",
    "worldarena",
    "world-in-world",
    "phyground",
    "ewmbench",
)


def test_external_contract_metric_evaluators_are_discoverable() -> None:
    contract = get_external_benchmark_contract("vbench")

    evaluator = get_external_metric_evaluator("vbench", "overall_quality")

    assert evaluator.benchmark_id == contract.benchmark_id
    assert evaluator.metric_id == "overall_quality"
    assert evaluator.evaluation_kind == "local_deterministic"
    assert evaluator.local_evaluator == "vbench_final_score"
    assert evaluator.requires_upstream_runtime is False


def test_benchmark_contract_package_exports_canonical_contracts() -> None:
    import worldfoundry.evaluation.tasks.contracts as canonical_contracts
    import worldfoundry.evaluation.tasks.contracts.external as canonical_external
    import worldfoundry.evaluation.tasks.contracts.registry as canonical_registry

    assert canonical_contracts.ExternalBenchmarkContract is canonical_registry.ExternalBenchmarkContract
    assert canonical_contracts.ExternalBenchmarkContractRegistry is canonical_registry.ExternalBenchmarkContractRegistry
    assert canonical_contracts.VBenchContract is canonical_external.VBenchContract
    assert canonical_contracts.get_external_benchmark_contract is canonical_external.get_external_benchmark_contract
    assert canonical_contracts.get_external_benchmark_contract("vbench") is canonical_external.VBenchContract


def test_external_metric_entries_report_blocked_runtime_for_non_local_metrics() -> None:
    blocked = get_external_metric_evaluator("camerabench", "camera_caption_score")

    result = blocked.evaluate(
        ExternalMetricEvaluationRequest(
            benchmark_id="camerabench",
            metric_id="camera_caption_score",
            generated_artifact_manifest={"generated_files": []},
        )
    )

    assert result.valid is False
    assert result.skip_reason == "external_runtime_required"
    assert result.diagnostics["requires_upstream_runtime"] is True


def test_external_contract_registry_uses_source_aware_registration_primitive() -> None:
    from worldfoundry.evaluation.tasks.contracts.registry import (
        DuplicateExternalBenchmarkContractError,
        ExternalBenchmarkContractRegistry,
    )
    from worldfoundry.evaluation.tasks.contracts.registry import ExternalBenchmarkContract

    builtin = ExternalBenchmarkContract(
        benchmark_id="toy-benchmark",
        display_name="Toy Benchmark",
        input_keys=("generated_video",),
        output_keys=("scorecard",),
        metric_ids=("toy_score",),
        requires_upstream_runtime=False,
    )
    extension = ExternalBenchmarkContract(
        benchmark_id="TOY-BENCHMARK",
        display_name="Extension Toy Benchmark",
        input_keys=("different_input",),
        output_keys=("different_output",),
        metric_ids=("different_score",),
        requires_upstream_runtime=True,
    )
    registry = ExternalBenchmarkContractRegistry()

    assert registry.register(builtin, source="builtin") is builtin
    assert registry.register(extension, source="extension") is builtin
    assert registry.get("toy-benchmark") is builtin
    assert registry.list() == (builtin,)

    duplicate_registry = ExternalBenchmarkContractRegistry((builtin,))
    try:
        duplicate_registry.register(extension, source="builtin")
    except DuplicateExternalBenchmarkContractError as exc:
        assert "duplicate built-in external benchmark contract" in str(exc)
    else:  # pragma: no cover - explicit failure keeps the assertion message clear.
        raise AssertionError("duplicate built-in benchmark contract did not raise")


def test_metric_package_exports_canonical_metrics_registry() -> None:
    import worldfoundry.evaluation.tasks.metrics.artifacts as canonical_artifacts
    import worldfoundry.evaluation.tasks.metrics.bindings as canonical_bindings
    import worldfoundry.evaluation.tasks.metrics.evaluators as canonical
    import worldfoundry.evaluation.tasks.metrics.local_evaluators as canonical_local_evaluators
    from worldfoundry.evaluation.tasks import metrics

    assert metrics.ExternalMetricEvaluationRequest is canonical.ExternalMetricEvaluationRequest
    assert metrics.ExternalMetricEvaluatorEntry is canonical.ExternalMetricEvaluatorEntry
    assert metrics.ExternalMetricEvaluatorRegistry is canonical.ExternalMetricEvaluatorRegistry
    assert metrics.evaluate_external_metric is canonical.evaluate_external_metric
    assert metrics.ExternalMetricEvaluatorRegistry is canonical.ExternalMetricEvaluatorRegistry
    assert metrics.normalize_artifact_records is canonical_artifacts.normalize_artifact_records
    assert metrics.missing_artifacts is canonical_artifacts.missing_artifacts
    assert metrics.FORMULA_EVALUATOR_BINDINGS is canonical_bindings.FORMULA_EVALUATOR_BINDINGS
    assert metrics.success_metric_bindings is canonical_bindings.success_metric_bindings
    assert metrics.LOCAL_EVALUATORS is canonical_local_evaluators.LOCAL_EVALUATORS


def test_new_video_world_contract_metric_evaluators_are_discoverable() -> None:
    for benchmark_id in NEW_VIDEO_WORLD_CONTRACT_IDS:
        contract = get_external_benchmark_contract(benchmark_id)
        for metric_id in contract.metric_ids:
            evaluator = get_external_metric_evaluator(benchmark_id, metric_id)
            assert evaluator.benchmark_id == benchmark_id
            assert evaluator.metric_id == metric_id


def test_dummy_local_metric_counts_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.mp4"
    artifact.write_bytes(b"video")
    registry = ExternalMetricEvaluatorRegistry(
        entries=(
            ExternalMetricEvaluatorEntry(
                benchmark_id="dummy-benchmark",
                metric_id="dummy_artifact_count",
                evaluation_kind="local_deterministic",
                local_evaluator="artifact_count",
                requires_upstream_runtime=False,
            ),
        ),
        include_external_contracts=False,
    )

    result = registry.evaluate(
        ExternalMetricEvaluationRequest(
            benchmark_id="dummy-benchmark",
            metric_id="dummy_artifact_count",
            generated_artifact_manifest=[str(artifact)],
            sample_id="sample-1",
        )
    )

    assert result.sample_id == "sample-1"
    assert result.metric_id == "dummy_artifact_count"
    assert result.valid is True
    assert result.normalized_value == 1.0


def test_judge_required_metric_returns_blocked_result() -> None:
    registry = ExternalMetricEvaluatorRegistry(
        entries=(
            ExternalMetricEvaluatorEntry(
                benchmark_id="judge-benchmark",
                metric_id="judge_score",
                evaluation_kind="blocked",
                requires_judge=True,
                requires_api=True,
                blocked_reason="judge_api_required",
            ),
        ),
        include_external_contracts=False,
    )

    result = registry.evaluate(
        ExternalMetricEvaluationRequest(
            benchmark_id="judge-benchmark",
            metric_id="judge_score",
            generated_artifact_manifest={"generated_files": []},
        )
    )

    assert result.valid is False
    assert result.skip_reason == "judge_api_required"
    assert result.diagnostics["requires_judge"] is True
    assert result.diagnostics["requires_api"] is True


def test_missing_artifact_returns_failure_result(tmp_path: Path) -> None:
    missing_artifact = tmp_path / "missing.mp4"
    registry = ExternalMetricEvaluatorRegistry(
        entries=(
            ExternalMetricEvaluatorEntry(
                benchmark_id="artifact-benchmark",
                metric_id="artifacts_ready",
                evaluation_kind="local_deterministic",
                local_evaluator="required_artifacts_present",
                required_artifacts=("generated_video",),
                requires_upstream_runtime=False,
            ),
        ),
        include_external_contracts=False,
    )

    result = registry.evaluate(
        ExternalMetricEvaluationRequest(
            benchmark_id="artifact-benchmark",
            metric_id="artifacts_ready",
            generated_artifact_manifest=[
                {"uri": str(missing_artifact), "kind": "generated_video"},
            ],
        )
    )

    assert result.valid is False
    assert result.skip_reason == "missing_artifact"
    assert result.diagnostics["missing_artifacts"] == ["generated_video"]
