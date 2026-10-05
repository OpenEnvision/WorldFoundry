from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import load_benchmark_catalog_shard_entries
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import (
    BenchmarkExecutionUnavailableError,
    BenchmarkRunnerRegistry,
    build_benchmark_runner_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"


def _entries_by_id():
    return {entry.benchmark_id: entry for entry in load_benchmark_catalog_shard_entries("video")}


def _has_dataset_contract_or_explicit_reason(entry) -> bool:
    has_dataset_ref = bool(entry.hf_dataset_ids)
    has_not_applicable_reason = entry.dataset.not_applicable and bool(entry.dataset.reason)
    has_blocked_reason = bool(entry.dataset.reason and entry.blockers)
    return has_dataset_ref or has_not_applicable_reason or has_blocked_reason


@pytest.mark.parametrize("entry", list(_entries_by_id().values()), ids=lambda entry: entry.benchmark_id)
def test_manifest_dataset_refs_have_hf_dataset_or_explicit_reason(entry) -> None:
    assert _has_dataset_contract_or_explicit_reason(entry), entry.to_dict()

    for ref in entry.dataset_refs:
        assert ref.hf_dataset_id or (ref.not_applicable and ref.reason), entry.to_dict()


@pytest.mark.parametrize(
    "benchmark_id",
    [entry.benchmark_id for entry in _entries_by_id().values() if entry.integration_status == "integrated"],
)
def test_integrated_runner_materialization_plan_is_serializable_and_complete(benchmark_id: str) -> None:
    registry = build_benchmark_runner_registry(MANIFEST_PATH)
    entry = registry.zoo.get(benchmark_id)
    runner = registry.get_runner(benchmark_id)

    plan = runner.materialization_plan()
    plan_dict = plan.to_dict()

    json.dumps(plan_dict, sort_keys=True)
    assert plan_dict["benchmark_id"] == benchmark_id
    assert set(plan.dataset_ids) == set(entry.hf_dataset_ids)
    assert all(command[:3] == ("hf", "download", command[2]) for command in plan.commands)
    assert {command[2] for command in plan.commands} == set(plan.dataset_ids)

    if not plan.dataset_ids:
        assert entry.dataset.reason
        if entry.dataset.not_applicable:
            assert any("dataset not applicable" in note for note in plan.notes)
        else:
            assert entry.blockers
            assert any("dataset acquisition blocked" in note for note in plan.notes)


def test_manifest_planned_and_blocked_entries_expose_contract_only_runners() -> None:
    registry = build_benchmark_runner_registry(MANIFEST_PATH)

    for entry in registry.list_entries():
        if entry.integration_status == "integrated":
            continue

        assert registry.has_runner(entry.benchmark_id) is True
        assert registry.has_official_runner(entry.benchmark_id) is False
        runner = registry.get_runner(entry.benchmark_id)
        assert runner.benchmark_id == entry.benchmark_id


def test_dataset_availability_does_not_promote_planned_or_blocked_runners() -> None:
    registry = BenchmarkRunnerRegistry(
        [
            {
                "id": "planned-with-verified-dataset",
                "hf_dataset_id": "org/planned",
                "integration": {"status": "planned"},
                "runner": {"verification_status": "verified"},
            },
            {
                "id": "blocked-with-verified-dataset",
                "hf_dataset_id": "org/blocked",
                "integration": {"status": "blocked", "blocked_reasons": ["waiting on runtime validation"]},
                "runner": {"verification_status": "verified"},
            },
            {
                "id": "planned-with-not-applicable-dataset",
                "dataset": {"not_applicable": True, "reason": "generated artifacts supplied by caller"},
                "integration": {"status": "planned"},
                "runner": {"verification_status": "verified"},
            },
        ]
    )

    for benchmark_id in registry.zoo.keys():
        assert registry.has_runner(benchmark_id) is False
        with pytest.raises(BenchmarkExecutionUnavailableError):
            registry.get_runner(benchmark_id)


def test_vbench2_manifest_exposes_real_data_refs_and_contract_artifacts() -> None:
    entry = _entries_by_id()["vbench-2.0"]

    assert set(entry.hf_dataset_ids) == {
        "Vchitect/VBench-2.0_sampled_videos",
        "Vchitect/VBench-2.0_human_annotation",
        "Vchitect/VBench-2.0_human_anomaly",
    }
    assert "vbench2_dataset_manifest.json" in entry.expected_artifacts
    assert "vbench2_video_coverage.json" in entry.expected_artifacts
    assert "vbench2_benchmark_contract.json" in entry.expected_artifacts
