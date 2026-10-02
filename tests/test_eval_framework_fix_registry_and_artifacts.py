"""CPU-only regression tests for evaluation-framework review fixes.

Covers EF-01/EF-02 (api registry duplicate/self-collision handling),
EF-06/EF-07 (ArtifactRef hash knob + actionable coercion errors),
EF-18/EF-19 (catalog error context + duplicate model_id warning),
EF-25 (resolver did-you-mean), EF-29 (pipeline timings), and
EF-37 (run_report public helper API).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.api.artifacts import (
    ArtifactRef,
    coerce_artifact_refs,
    enrich_artifact_ref,
)
from worldfoundry.evaluation.api.metrics import MetricSpec
from worldfoundry.evaluation.api.registry import (
    DuplicateRegistryKeyError,
    MetricSpecRegistry,
    ModelManifestRegistry,
)
from worldfoundry.evaluation.api.world_model_manifest import WorldModelManifest


# ── EF-01: duplicate MetricSpec must raise the documented error ──────────


def test_duplicate_metric_registration_raises_duplicate_key_error() -> None:
    registry = MetricSpecRegistry()
    registry.register(MetricSpec(metric_id="fvd"))
    with pytest.raises(DuplicateRegistryKeyError, match="fvd"):
        registry.register(MetricSpec(metric_id="FVD"))


# ── EF-02: name case-variants and repeated aliases must not fail ─────────


def test_manifest_name_differing_only_in_case_registers() -> None:
    registry = ModelManifestRegistry()
    manifest = WorldModelManifest(
        model_id="cosmos-predict-2",
        name="Cosmos-Predict-2",
        aliases=("cp2", "CP2"),
    )
    registered = registry.register(manifest)
    assert registered is manifest
    assert registry.get("cosmos-predict-2").model_id == "cosmos-predict-2"
    assert registry.get("Cosmos-Predict-2").model_id == "cosmos-predict-2"
    assert registry.get("cp2").model_id == "cosmos-predict-2"
    assert registry.get("CP2").model_id == "cosmos-predict-2"


def test_alias_duplicating_own_canonical_key_raises_self_collision_error() -> None:
    # Duplicate source keys are deliberately left for the registry to
    # reject; the message must name the self-collision, not a phantom
    # cross-item conflict.
    registry = ModelManifestRegistry()
    manifest = WorldModelManifest(model_id="alpha", name="alpha", aliases=("ALPHA",))
    with pytest.raises(DuplicateRegistryKeyError, match="own canonical key"):
        registry.register(manifest)


def test_alias_conflict_with_other_item_still_raises() -> None:
    registry = ModelManifestRegistry()
    registry.register(WorldModelManifest(model_id="model-a", name="Model A", aliases=("shared",)))
    with pytest.raises(DuplicateRegistryKeyError, match="shared"):
        registry.register(WorldModelManifest(model_id="model-b", name="Model B", aliases=("shared",)))


# ── EF-07: coerce_artifact_refs actionable error ─────────────────────────


def test_coerce_artifact_refs_rejects_bare_path_with_actionable_error() -> None:
    with pytest.raises(TypeError, match="artifact 'video'.*got str"):
        coerce_artifact_refs({"video": "/tmp/out.mp4"})


def test_coerce_artifact_refs_accepts_refs_and_mappings(tmp_path: Path) -> None:
    ref = ArtifactRef(uri="memory://x", kind="video")
    result = coerce_artifact_refs({"a": ref, "b": {"uri": "memory://y", "kind": "image"}})
    assert result["a"] is ref
    assert result["b"].kind == "image"


# ── EF-06: compute_hash knob ─────────────────────────────────────────────


def test_from_uri_compute_hash_false_skips_sha256_keeps_size(tmp_path: Path) -> None:
    payload = tmp_path / "artifact.bin"
    payload.write_bytes(b"x" * 2048)

    hashed = ArtifactRef.from_uri(payload, kind="video")
    assert hashed.sha256 is not None
    assert hashed.size_bytes == 2048

    unhashed = ArtifactRef.from_uri(payload, kind="video", compute_hash=False)
    assert unhashed.sha256 is None
    assert unhashed.size_bytes == 2048


def test_enrich_without_hash_preserves_existing_sha256(tmp_path: Path) -> None:
    payload = tmp_path / "artifact.bin"
    payload.write_bytes(b"y" * 100)
    pre = ArtifactRef(uri=str(payload), kind="video", sha256="deadbeef")
    enriched = enrich_artifact_ref(pre, compute_hash=False)
    assert enriched.sha256 == "deadbeef"
    assert enriched.size_bytes == 100


# ── EF-18: broken zoo YAML / schema errors carry the file path ───────────


def test_zoo_yaml_parse_error_contains_path(tmp_path: Path) -> None:
    from worldfoundry.evaluation.models.catalog.zoo_registry import ModelZooRegistry

    broken = tmp_path / "broken.yaml"
    broken.write_text("model_id: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="broken.yaml"):
        ModelZooRegistry.from_paths([broken])


def test_zoo_schema_error_contains_path(tmp_path: Path) -> None:
    from worldfoundry.evaluation.models.catalog.zoo_registry import ModelZooRegistry

    bad_entry = tmp_path / "bad_entry.yaml"
    bad_entry.write_text("models:\n  - name: entry-without-model-id\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="bad_entry.yaml"):
        ModelZooRegistry.from_paths([bad_entry])


# ── EF-19: equal-priority duplicate model_id warns with both paths ───────


def test_duplicate_model_id_across_files_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from worldfoundry.evaluation.models.catalog.zoo_registry import ModelZooRegistry

    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("model_id: dup-model\n", encoding="utf-8")
    second.write_text("model_id: dup-model\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.models.catalog.zoo_registry"):
        registry = ModelZooRegistry.from_paths([first, second])

    assert len(registry) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("dup-model" in message and "a.yaml" in message and "b.yaml" in message for message in messages)


# ── EF-25: unknown model id resolves to ModelResolutionError + hint ──────


def test_unknown_zoo_model_raises_resolution_error_with_suggestions(tmp_path: Path) -> None:
    from worldfoundry.evaluation.models.runners.resolver import (
        ModelResolutionError,
        resolve_model_zoo_config,
    )

    manifest = tmp_path / "zoo.yaml"
    manifest.write_text(
        "models:\n"
        "  - model_id: cosmos-predict-2\n"
        "  - model_id: cosmos-transfer-1\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelResolutionError, match="cosmos-predict-2"):
        resolve_model_zoo_config("cosmos-predic2", manifest_dir=tmp_path)


# ── EF-29: pipeline results carry generate timings ───────────────────────


def test_generation_result_from_pipeline_forwards_timings(tmp_path: Path) -> None:
    from worldfoundry.evaluation.api import GenerationRequest
    from worldfoundry.evaluation.models.pipelines.invocation import build_pipeline_invocation
    from worldfoundry.evaluation.models.pipelines.results import (
        PipelineResultContext,
        generation_result_from_pipeline,
    )

    request = GenerationRequest(sample_id="s1", task_name="unit")
    invocation = build_pipeline_invocation(
        request=request,
        output_dir=tmp_path,
        artifact_filename="out.mp4",
        generation_kwargs={},
    )
    context = PipelineResultContext(
        model_id="m", artifact_kind="video", task_family="t2v", pipeline_target="pkg.mod:Cls"
    )

    result = generation_result_from_pipeline(
        invocation=invocation,
        result={"status": "succeeded"},
        context=context,
        timings={"generate_seconds": 1.5},
    )
    assert dict(result.timings) == {"generate_seconds": 1.5}
    assert result.to_dict()["timings"] == {"generate_seconds": 1.5}

    without = generation_result_from_pipeline(
        invocation=invocation,
        result={"status": "succeeded"},
        context=context,
    )
    assert dict(without.timings) == {}


# ── EF-37: run_report exposes the shared helpers publicly ────────────────


def test_run_report_public_helper_api() -> None:
    from worldfoundry.evaluation.reporting import run_report

    for name in (
        "dedupe_labels",
        "find_run_summary_candidate",
        "load_run_summary",
        "normalise_roots",
        "number_or_none",
        "resolve_run_summary_path",
        "row_from_summary",
    ):
        assert callable(getattr(run_report, name)), name
        assert name in run_report.__all__, name

    rows = run_report.dedupe_labels([{"label": "a"}, {"label": "a"}])
    assert [row["label"] for row in rows] == ["a", "a#2"]
    assert run_report.number_or_none(True) is None
    assert run_report.number_or_none(1.5) == 1.5
