"""CPU-only leftovers for evaluation-framework deferred items.

Covers EF-03 (_Registry wraps AliasRegistryStore), EF-20 (cached registries
are frozen), and EF-21 (intentional catalog status literals are registered).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from worldfoundry.evaluation.api.metrics import MetricSpec
from worldfoundry.evaluation.api.registry import (
    AliasRegistryStore,
    DuplicateRegistryKeyError,
    MetricSpecRegistry,
    ModelManifestRegistry,
)
from worldfoundry.evaluation.api.world_model_manifest import WorldModelManifest
from worldfoundry.evaluation.models.catalog import schema as catalog_schema
from worldfoundry.evaluation.models.catalog.registry import (
    clear_model_registry_cache,
    discover_model_registry,
)
from worldfoundry.evaluation.models.catalog.schema import (
    ModelZooEntry,
    _normalize_demo_status,
    _normalize_integration_status,
    _normalize_source_status,
)
from worldfoundry.evaluation.models.catalog.zoo_registry import (
    ModelZooRegistry,
    clear_model_zoo_registry_cache,
    load_model_zoo_registry,
)
from worldfoundry.evaluation.models.pipelines.aliases import (
    PipelineAliasGroup,
    PipelineAliasRegistry,
    clear_pipeline_alias_registry_cache,
    load_pipeline_alias_registry,
)
from worldfoundry.evaluation.models.pipelines.bindings import (
    PipelineBinding,
    PipelineBindingRegistry,
    clear_pipeline_binding_registry_cache,
    load_pipeline_binding_registry,
    merge_pipeline_binding_plugins,
)


@pytest.fixture(autouse=True)
def _reset_status_warning_dedupe() -> None:
    catalog_schema._WARNED_UNKNOWN_STATUSES.clear()
    yield
    catalog_schema._WARNED_UNKNOWN_STATUSES.clear()


# ── EF-03: _Registry is a thin AliasRegistryStore wrapper ─────────────


def test_typed_registry_is_backed_by_alias_store() -> None:
    registry = ModelManifestRegistry()
    assert isinstance(registry._store, AliasRegistryStore)
    manifest = WorldModelManifest(model_id="alpha", name="Alpha", aliases=("a1",))
    assert registry.register(manifest) is manifest
    assert registry.get("ALPHA") is manifest
    assert registry.get("a1") is manifest
    assert registry.keys() == ["alpha"]
    assert "Alpha" in registry


def test_typed_registry_wrapper_keeps_self_and_cross_collision_errors() -> None:
    with pytest.raises(DuplicateRegistryKeyError, match="own canonical key"):
        ModelManifestRegistry().register(
            WorldModelManifest(model_id="alpha", name="alpha", aliases=("ALPHA",))
        )

    registry = ModelManifestRegistry()
    registry.register(WorldModelManifest(model_id="model-a", aliases=("shared",)))
    with pytest.raises(DuplicateRegistryKeyError, match="shared"):
        registry.register(WorldModelManifest(model_id="model-b", aliases=("shared",)))

    metrics = MetricSpecRegistry()
    metrics.register(MetricSpec(metric_id="fvd"))
    with pytest.raises(DuplicateRegistryKeyError, match="fvd"):
        metrics.register(MetricSpec(metric_id="FVD"))


def test_intra_item_casefold_alias_duplicates_are_deduped() -> None:
    # EF-02: repeats inside one alias list collapse; only a collision with
    # the item's own canonical key (or another item) is an error.
    registry = MetricSpecRegistry()
    spec = MetricSpec(id="clip_iqa", aliases=("quality", "QUALITY"))
    assert registry.register(spec) is spec
    assert registry.get("quality") is spec
    assert registry.get("QUALITY") is spec


# ── EF-20: cached load_*_registry instances reject register ───────────


def test_cached_model_zoo_registry_is_frozen(tmp_path: Path) -> None:
    (tmp_path / "demo.yaml").write_text("model_id: demo-model\n", encoding="utf-8")
    clear_model_zoo_registry_cache()
    first = load_model_zoo_registry(tmp_path)
    second = load_model_zoo_registry(tmp_path)
    assert first is second
    assert first.get("demo-model").model_id == "demo-model"
    with pytest.raises(RuntimeError, match="cached ModelZooRegistry"):
        first.register(ModelZooEntry.from_dict({"model_id": "other"}))

    overlay = ModelZooRegistry(first.list())
    overlay.register(ModelZooEntry.from_dict({"model_id": "other"}))
    assert overlay.get("other").model_id == "other"
    assert "other" not in first

    clear_model_zoo_registry_cache()
    assert load_model_zoo_registry(tmp_path) is not first


def test_cached_pipeline_binding_registry_is_frozen(tmp_path: Path) -> None:
    (tmp_path / "dummy.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: dummy",
                "model_id: dummy-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                "  target: pkg.mod:Cls",
                "",
            )
        ),
        encoding="utf-8",
    )
    clear_pipeline_binding_registry_cache()
    cached = load_pipeline_binding_registry(tmp_path)
    assert load_pipeline_binding_registry(tmp_path) is cached
    extra = PipelineBinding(
        binding_id="extra",
        model_id="extra-model",
        runner="worldfoundry.pipeline",
        pipeline_target="pkg.mod:Extra",
        schema_version=2,
    )
    with pytest.raises(RuntimeError, match="cached PipelineBindingRegistry"):
        cached.register(extra)

    merged = merge_pipeline_binding_plugins(cached, {"extra": extra})
    assert merged.get("extra").binding_id == "extra"
    with pytest.raises(KeyError):
        cached.get("extra")
    clear_pipeline_binding_registry_cache()


def test_cached_pipeline_alias_registry_is_frozen(tmp_path: Path) -> None:
    (tmp_path / "aliases.yaml").write_text(
        "schema_version: 2\naliases:\n  short: dummy\n",
        encoding="utf-8",
    )
    clear_pipeline_alias_registry_cache()
    cached = load_pipeline_alias_registry(tmp_path)
    assert cached.canonical_id("short") == "dummy"
    with pytest.raises(RuntimeError, match="cached PipelineAliasRegistry"):
        cached.register(PipelineAliasGroup(canonical_id="other", aliases=("o",), domain="video"))
    fresh = PipelineAliasRegistry(cached.list())
    fresh.register(PipelineAliasGroup(canonical_id="other", aliases=("o",), domain="video"))
    assert fresh.canonical_id("o") == "other"
    clear_pipeline_alias_registry_cache()


def test_clear_model_registry_cache_empties_lru() -> None:
    clear_model_registry_cache()
    assert discover_model_registry.cache_info().currsize == 0


# ── EF-21: intentional catalog statuses are registered ────────────────


def test_registered_catalog_statuses_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.models.catalog.schema"):
        assert _normalize_integration_status("runtime_ported") == "integrated"
        assert _normalize_integration_status("verified") == "integrated"
        assert _normalize_integration_status("route_ready") == "integrated"
        assert _normalize_demo_status("verified") == "verified"
        assert _normalize_demo_status("passed") == "verified"
        assert _normalize_source_status("open_weights") == "open_source"
        assert _normalize_source_status("paper_only") == "unknown"
    assert not caplog.records


def test_unknown_status_still_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.models.catalog.schema"):
        assert _normalize_integration_status("typo_integarted") == "planned"
    assert any("typo_integarted" in record.getMessage() for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.models.catalog.schema"):
        assert _normalize_integration_status("typo_integarted") == "planned"
    assert not caplog.records


def test_unknown_source_status_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="worldfoundry.evaluation.models.catalog.schema"):
        assert _normalize_source_status("totally_made_up_source") == "unknown"
    assert any("totally_made_up_source" in record.getMessage() for record in caplog.records)
