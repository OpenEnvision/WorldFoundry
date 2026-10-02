from pathlib import Path
from typing import Any

import yaml

from worldfoundry.evaluation.tasks.catalog.schema import BenchmarkRunnerSpec, BenchmarkZooEntry
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark_suite import (
    _benchmark_aware_provenance,
    _benchmark_unavailable_reason,
)


def _entry(
    *,
    integration: str,
    verification: str,
    runner_target: str | None,
    run_command: tuple[str, ...] | None = None,
) -> BenchmarkZooEntry:
    return BenchmarkZooEntry(
        benchmark_id="example",
        integration_status=integration,
        runner=BenchmarkRunnerSpec(
            runner_target=runner_target,
            verification_status=verification,
            run_command=run_command,
        ),
    )


def test_integrated_bounded_runner_can_be_attempted() -> None:
    entry = _entry(
        integration="integrated",
        verification="pending",
        runner_target="example:Runner",
        run_command=("python", "run_example.py"),
    )

    assert _benchmark_unavailable_reason(entry, mode="official-run") is None


def test_unintegrated_or_missing_runner_remains_unavailable() -> None:
    planned = _entry(integration="planned", verification="pending", runner_target="example:Runner")
    missing_runner = _entry(integration="integrated", verification="verified", runner_target=None)

    assert _benchmark_unavailable_reason(planned, mode="official-run") is not None
    assert _benchmark_unavailable_reason(missing_runner, mode="official-run") is not None


def test_result_importer_is_not_planned_as_generated_artifact_official_run() -> None:
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

    entry = load_benchmark_zoo_registry().get("aigcbench")

    reason = _benchmark_unavailable_reason(entry, mode="official-run")

    assert reason is not None
    assert "normalizer/importer" in reason
    assert _benchmark_unavailable_reason(entry, mode="normalizer") is None


def test_bounded_catalog_evidence_downgrades_planned_claim() -> None:
    entry = _entry(integration="integrated", verification="pending", runner_target="example:Runner")
    provenance = {
        "claim": {"level": "benchmark_comparable", "leaderboard_candidate": True},
        "reasons": [],
    }

    result = _benchmark_aware_provenance(provenance, entry)

    assert result["claim"] == {"level": "diagnostic", "leaderboard_candidate": False}
    assert result["reasons"] == ["benchmark catalog does not establish full official-suite evidence"]


def test_full_catalog_evidence_preserves_comparable_claim() -> None:
    entry = BenchmarkZooEntry(
        benchmark_id="example",
        integration_status="integrated",
        runner=BenchmarkRunnerSpec(runner_target="example:Runner", verification_status="verified"),
        official_benchmark_verified=True,
        integration_evidence=True,
        leaderboard_valid=True,
    )
    provenance = {
        "claim": {"level": "benchmark_comparable", "leaderboard_candidate": True},
        "reasons": [],
    }

    result = _benchmark_aware_provenance(provenance, entry)

    assert result["claim"] == provenance["claim"]
    assert result["reasons"] == []


def test_top_level_claim_is_not_overridden_by_bounded_nested_evidence() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "benchmark_id": "example",
            "integration": {
                "status": "integrated",
                "evidence": {
                    "official_benchmark_verified": True,
                    "integration_evidence": True,
                },
            },
            "official_benchmark_verified": False,
            "integration_evidence": True,
        }
    )

    assert entry.official_benchmark_verified is False
    assert entry.integration_evidence is True


def _nested_mappings(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _nested_mappings(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nested_mappings(child, (*path, str(index)))


def test_incomplete_suite_blocks_never_claim_full_benchmark_verification() -> None:
    manifest_root = Path("worldfoundry/data/benchmarks")
    manifest_dirs = (
        manifest_root / "catalog",
        manifest_root / "tasks",
        manifest_root / "runtime_profiles",
    )
    violations: list[str] = []

    for manifest_dir in manifest_dirs:
        for path in sorted(manifest_dir.rglob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for field_path, block in _nested_mappings(payload):
                if block.get("full_suite_verified") is False and block.get("official_benchmark_verified") is True:
                    location = ".".join(field_path) or "<root>"
                    violations.append(f"{path}:{location}")

    assert violations == []
