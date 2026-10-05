from __future__ import annotations

import pytest

from worldfoundry.evaluation.tasks.catalog.runner_kinds import (
    CANONICAL_RUNNER_KINDS,
    IN_TREE_RUNTIME_KINDS,
    RUNNER_KIND_ALIASES,
    RUNNER_KINDS,
    normalize_runner_kind,
)
from worldfoundry.evaluation.tasks.catalog.schema import BenchmarkRunnerSpec
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import ManifestBenchmarkRunner


def test_runner_kind_groups_are_derived_from_the_closed_vocabulary() -> None:
    assert set(RUNNER_KIND_ALIASES) < set(RUNNER_KINDS)
    assert IN_TREE_RUNTIME_KINDS <= RUNNER_KINDS
    assert set(RUNNER_KIND_ALIASES.values()) <= CANONICAL_RUNNER_KINDS


def test_benchmark_runner_spec_rejects_an_unknown_runtime_kind() -> None:
    with pytest.raises(ValueError, match="BenchmarkRunnerSpec.runtime.kind must be one of"):
        BenchmarkRunnerSpec(runtime={"kind": "in_tree_one_off_spelling"})


@pytest.mark.parametrize(
    ("alias", "canonical"),
    sorted(RUNNER_KIND_ALIASES.items()),
)
def test_benchmark_runner_spec_normalizes_registered_aliases(alias: str, canonical: str) -> None:
    spec = BenchmarkRunnerSpec(runtime={"kind": alias, "detail": "preserved"})

    assert normalize_runner_kind(alias) == canonical
    assert spec.runtime == {"kind": canonical, "detail": "preserved"}


def test_checked_in_catalog_uses_only_canonical_runner_kinds_after_loading() -> None:
    registry = load_benchmark_zoo_registry()

    assert registry.list()
    for entry in registry.list():
        kind = entry.runner_runtime.get("kind")
        if kind is not None:
            assert kind in CANONICAL_RUNNER_KINDS, entry.benchmark_id


@pytest.mark.parametrize(
    "benchmark_id",
    (
        "apple-pi",
        "evalcrafter",
        "phygenbench",
        "physical-ai-bench",
        "physvidbench",
        "sana-wm-bench",
        "stevo-bench",
        "worldbench",
        "worldreasonbench",
    ),
)
def test_specialized_in_tree_runner_kinds_do_not_require_an_external_runtime(benchmark_id: str) -> None:
    entry = load_benchmark_zoo_registry().get(benchmark_id)

    assert entry.runner_runtime["kind"] in CANONICAL_RUNNER_KINDS
    assert ManifestBenchmarkRunner(entry).report_metadata()["requires_upstream_runtime"] is False
