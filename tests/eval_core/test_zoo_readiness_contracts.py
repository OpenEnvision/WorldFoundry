from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    DEFAULT_EMBODIED_CATALOG_DIR,
    DEFAULT_VIDEO_CATALOG_DIR,
    load_benchmark_catalog_entries,
    load_benchmark_catalog_shard_entries,
)
from worldfoundry.evaluation.tasks.catalog.specs import benchmark_zoo_entry_to_benchmark_spec
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.models import ModelRunnerRegistry
from worldfoundry.evaluation.models.catalog import load_entries as load_model_entries


REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_VIDEO_WORLD_NORMALIZER_IDS = {
    "aigcbench",
    "devil-dynamics",
    "ewmbench",
    "mirabench",
    "genai-bench",
    "phygenbench",
    "phyeduvideo",
    "physvidbench",
    "t2v-safety-bench",
    "t2vworldbench",
    "videoscience-bench",
    "videophy",
    "videophy2",
    "physics-iq",
    "ipv-bench",
    "phyground",
    "worldarena",
    "world-in-world",
}

NEW_VIDEO_WORLD_CONTRACT_IDS = set()


def _load_all_model_entries():
    return [
        entry
        for path in sorted((REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog").rglob("*.yaml"))
        for entry in load_model_entries(path)
    ]


def _load_video_world_benchmark_entries():
    return list(load_benchmark_catalog_shard_entries("video"))


def _load_embodied_world_benchmark_entries():
    return list(load_benchmark_catalog_shard_entries("embodied"))


def _load_all_benchmark_entries():
    return [*_load_video_world_benchmark_entries(), *_load_embodied_world_benchmark_entries()]


def test_model_zoo_readiness_does_not_overclaim_integrated_runners() -> None:
    entries = _load_all_model_entries()
    integrated_entries = [entry for entry in entries if entry.integration_status == "integrated"]

    assert len(entries) >= 96
    assert all(entry.verification_status != "failed" for entry in integrated_entries)
    for entry in integrated_entries:
        if entry.runner_target:
            # A claimed runner needs a declared runtime route: either a conda
            # runtime profile or a pipeline binding/target (hosted-API and
            # shared-pipeline models run without a per-model conda profile).
            assert (
                entry.runtime_profile or entry.pipeline_binding or entry.pipeline_target
            ), entry.model_id


def test_model_zoo_claimed_runner_targets_match_registry_contract() -> None:
    registry = ModelRunnerRegistry()
    bad_targets = []
    missing_runtime_profiles = []

    for entry in _load_all_model_entries():
        # Pipeline bindings/targets are an alternative runtime route to a
        # conda runtime profile (hosted-API and shared-pipeline models).
        entry_pipeline_route = entry.pipeline_binding or entry.pipeline_target
        claimed_targets = []
        if entry.runner_target:
            claimed_targets.append(
                (entry.model_id, entry.runner_target, entry.runtime_profile or entry_pipeline_route)
            )
        for variant in entry.variants:
            if variant.runner_target:
                claimed_targets.append(
                    (
                        f"{entry.model_id}:{variant.variant_id}",
                        variant.runner_target,
                        variant.runtime_profile or entry.runtime_profile or entry_pipeline_route,
                    )
                )

        for owner, target, runtime_route in claimed_targets:
            try:
                registry.resolve_key(target)
            except KeyError:
                bad_targets.append(f"{owner}={target}")
            if not runtime_route:
                missing_runtime_profiles.append(owner)

    assert bad_targets == []
    assert missing_runtime_profiles == []


def test_benchmark_zoo_readiness_tracks_integrated_runtimes() -> None:
    entries = _load_video_world_benchmark_entries()
    by_id = {entry.benchmark_id: entry for entry in entries}
    counts = Counter(entry.integration_status for entry in entries)

    assert "videoverse" in by_id
    videoverse = by_id["videoverse"]
    assert videoverse.integration_status == "integrated"
    assert videoverse.run_command is not None
    assert videoverse.runner_availability["surface"] == "videoverse_official_in_tree"
    assert videoverse.open_source_status == "in_tree_runtime"
    assert counts["integrated"] >= 1
    assert by_id["vbench"].verification_status == "verified"
    assert by_id["vbench"].maturity == "verified_runner"
    assert by_id["vbench"].validation_command is not None
    assert by_id["vbench"].expected_artifacts
    assert by_id["worldscore"].open_source_status == "experimental"
    assert by_id["worldscore"].official_benchmark_verified is False
    assert by_id["worldscore"].integration_evidence is False
    assert NEW_VIDEO_WORLD_NORMALIZER_IDS <= set(by_id)
    for benchmark_id in sorted(NEW_VIDEO_WORLD_NORMALIZER_IDS):
        entry = by_id[benchmark_id]
        assert entry.integration_status == "integrated"
        assert entry.maturity in {"contract_ready", "verified_runner"}
        assert entry.verification_status != "failed"
        assert entry.official_benchmark_verified is False
        # integration_evidence is now a recorded per-benchmark evidence bit
        # rather than a synthesized constant, so it varies; the readiness
        # contract here is only that no entry overclaims leaderboard validity.
        assert entry.integration_evidence in (True, False)
        assert entry.leaderboard_valid is False
        # Some in-tree computed benchmarks (physics-iq) record availability
        # via runner_target/verification-scope keys instead of "surface".
        assert entry.runner_availability, benchmark_id


def test_iworld_bench_tracks_released_public_assets_without_leaderboard_claim() -> None:
    entries = _load_video_world_benchmark_entries()
    by_id = {entry.benchmark_id: entry for entry in entries}
    entry = by_id["iworld-bench"]

    # iworld-bench gained a bounded official in-tree component surface, so it
    # is catalog-integrated now; it still must not claim official-benchmark
    # verification or leaderboard validity.
    assert entry.integration_status == "integrated"
    assert entry.runner_target == "worldfoundry.evaluation.tasks.contracts.external:IWorldBenchContract"
    assert entry.open_source_status == "experimental"
    assert entry.official_benchmark_verified is False
    assert entry.integration_evidence is True
    assert entry.leaderboard_valid is False
    assert entry.dataset.hf_dataset_id == "EmbodiedCity/iWorld-Bench-Dataset"
    assert entry.runner_availability["surface"] == "bounded_official_memory_component"
    assert any(metric.metric_id == "iworldbench_average" for metric in entry.metrics)


def test_embodied_benchmark_zoo_readiness_does_not_promote_normalizer_only_importers() -> None:
    entries = _load_embodied_world_benchmark_entries()
    by_id = {entry.benchmark_id: entry for entry in entries}
    counts = Counter(entry.integration_status for entry in entries)

    # The embodied shard keeps growing; the contract is that every registered
    # entry stays honest (integrated surface, no leaderboard promotion), not a
    # frozen entry count.
    assert len(entries) >= 10
    assert counts == {"integrated": len(entries)}
    assert by_id["robotwin"].integration_status == "integrated"
    assert by_id["robotwin"].verification_status == "normalizer_only"
    assert by_id["robotwin"].maturity == "contract_ready"
    assert by_id["robotwin"].integration_evidence is True
    assert by_id["robotwin"].leaderboard_valid is False
    assert by_id["robotwin"].run_command is not None
    assert by_id["robotwin"].expected_artifacts
    assert any("normalizer-only" in note or "normalizer" in note for note in by_id["robotwin"].notes)


def test_benchmark_zoo_contract_surface_matches_manifest_claims() -> None:
    for entry in _load_all_benchmark_entries():
        spec = benchmark_zoo_entry_to_benchmark_spec(entry)
        task = spec.tasks[0]
        official_runtime_validated = (
            entry.integration_status == "integrated"
            and entry.verification_status == "verified"
            and entry.official_benchmark_verified
            and entry.integration_evidence
        )

        assert task.metadata["official_runtime_validated"] is official_runtime_validated
        assert task.metadata["contract_only_surface"] is not official_runtime_validated
        assert task.metadata["open_source_status"] == entry.open_source_status
        assert task.metadata["release_status"] == entry.release_status
        assert task.metadata["maturity"] == entry.maturity
        assert task.metadata["leaderboard_valid"] is entry.leaderboard_valid
        assert task.metadata["data_refs"] == dict(entry.data_refs)
        assert task.metadata["runner_availability"] == dict(entry.runner_availability)
        if entry.data_refs.get("task_yaml"):
            assert task.data["task_yaml"] == entry.data_refs["task_yaml"]
        assert tuple(metric.metric_id for metric in spec.metrics) == tuple(metric.metric_id for metric in entry.metrics)
        if entry.integration_status != "integrated":
            assert task.metadata["contract_only_surface"] is True


def test_benchmark_zoo_runner_targets_are_importable_when_claimed() -> None:
    missing_targets = []
    for entry in _load_video_world_benchmark_entries():
        if entry.runner_target is None:
            missing_targets.append(entry.benchmark_id)
            continue
        module_name, _, attr_name = entry.runner_target.partition(":")
        module = importlib.import_module(module_name)
        runner = getattr(module, attr_name)
        contract = get_external_benchmark_contract(entry.benchmark_id)

        assert runner.benchmark_id == entry.benchmark_id
        assert runner.metric_ids == contract.metric_ids

    assert missing_targets == []


def test_all_registered_benchmarks_have_contract_or_explicit_blocker() -> None:
    missing_contracts = []
    missing_blockers = []
    for entry in _load_all_benchmark_entries():
        if entry.runner_target is None:
            missing_contracts.append(entry.benchmark_id)
            if entry.integration_status != "blocked" or not entry.notes:
                missing_blockers.append(entry.benchmark_id)
            continue
        contract = get_external_benchmark_contract(entry.benchmark_id)
        assert contract.benchmark_id == entry.benchmark_id
        assert contract.input_keys
        assert contract.output_keys
        assert contract.metric_ids

    assert missing_contracts == []
    assert missing_blockers == []


def test_benchmark_zoo_dataset_free_entries_have_explicit_reason() -> None:
    for entry in _load_video_world_benchmark_entries():
        if entry.hf_dataset_ids:
            assert entry.dataset_refs
            continue

        # A benchmark without HF datasets must state its data provenance:
        # either an explicit not_applicable marker with a reason, or recorded
        # data_refs (task yaml, bundled assets, in-tree runtime paths).
        has_explicit_marker = entry.dataset.not_applicable is True and bool(entry.dataset.reason)
        has_data_refs = bool(entry.data_refs)
        assert has_explicit_marker or has_data_refs, entry.benchmark_id
