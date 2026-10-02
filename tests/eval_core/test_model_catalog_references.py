from __future__ import annotations

import pytest

from worldfoundry.evaluation.models.catalog import ModelVariantSpec, ModelZooEntry, default_model_zoo_dir
from worldfoundry.evaluation.models.pipelines import build_pipeline_runner_spec, import_pipeline_target
from worldfoundry.evaluation.models.pipelines.bindings import PipelineBindingRegistry
from worldfoundry.evaluation.models.runners.registry import ModelRunnerRegistry
from worldfoundry.evaluation.models.runners.resolver import resolve_model_zoo_config
from worldfoundry.evaluation.models.runtime import validate_catalog_references


HOSTED_API_PIPELINES = (
    ("hailuo-2p3", "worldfoundry.pipelines.minimax.pipeline_hailuo_2p3:Hailuo2p3Pipeline"),
    ("kling-api", "worldfoundry.pipelines.kling.pipeline_kling_api:KlingApiPipeline"),
    ("luma-ray2", "worldfoundry.pipelines.luma.pipeline_luma_ray2:LumaRay2Pipeline"),
    ("runway-gen4p5", "worldfoundry.pipelines.runway.pipeline_runway_gen4p5:RunwayGen4p5Pipeline"),
    ("sora2", "worldfoundry.pipelines.sora.pipeline_sora2:Sora2Pipeline"),
    ("veo3", "worldfoundry.pipelines.veo.pipeline_veo3:Veo3Pipeline"),
    ("wan-2p5", "worldfoundry.pipelines.wan.pipeline_wan_2p5:Wan2p5Pipeline"),
    ("wan-2p6", "worldfoundry.pipelines.wan.pipeline_wan_2p6:Wan2p6Pipeline"),
    ("wan-2p7", "worldfoundry.pipelines.wan.pipeline_wan_2p7:Wan2p7Pipeline"),
    ("worldlabs", "worldfoundry.pipelines.worldlabs.pipeline_worldlabs:WorldLabsPipeline"),
)

HOSTED_API_UNIFIED_ADAPTERS = tuple(
    row
    for row in HOSTED_API_PIPELINES
    if row[0] in {"hailuo-2p3", "kling-api", "luma-ray2", "wan-2p5", "wan-2p6", "wan-2p7"}
)


def test_catalog_reference_validator_checks_entries_and_variants() -> None:
    entry = ModelZooEntry(
        model_id="broken-model",
        pipeline_binding="missing-entry-binding",
        runtime_profile="runtime-profile:missing-entry-profile",
        runner_target="missing-entry-runner",
        variants=(
            ModelVariantSpec(
                variant_id="broken-variant",
                pipeline_binding="missing-variant-binding",
                runtime_profile="missing-variant-profile",
                runner_target="missing-variant-runner",
            ),
        ),
    )

    issues = validate_catalog_references(
        catalog_entries=(entry,),
        bindings=PipelineBindingRegistry(),
        runtime_profiles=(),
        runner_registry=ModelRunnerRegistry(include_builtins=False),
    )

    assert [issue.code for issue in issues] == [
        "catalog_pipeline_binding_missing",
        "catalog_runtime_profile_missing",
        "catalog_runner_target_unresolved",
        "catalog_pipeline_binding_missing",
        "catalog_runtime_profile_missing",
        "catalog_runner_target_unresolved",
    ]
    assert [issue.field for issue in issues] == [
        "catalog.broken-model.pipeline_binding",
        "catalog.broken-model.runtime_profile",
        "catalog.broken-model.runner_target",
        "catalog.broken-model.variants.broken-variant.pipeline_binding",
        "catalog.broken-model.variants.broken-variant.runtime_profile",
        "catalog.broken-model.variants.broken-variant.runner_target",
    ]


def test_builtin_model_catalog_references_are_closed() -> None:
    assert validate_catalog_references() == ()


@pytest.mark.parametrize(("model_id", "pipeline_target"), HOSTED_API_PIPELINES)
def test_hosted_api_catalog_entries_resolve_real_pipeline_routes(model_id: str, pipeline_target: str) -> None:
    resolved = resolve_model_zoo_config(model_id, manifest_dir=default_model_zoo_dir())

    spec = build_pipeline_runner_spec(resolved.config)

    assert spec.pipeline_target == pipeline_target
    assert callable(getattr(import_pipeline_target(spec.pipeline_target), "from_pretrained", None))
    assert resolved.config.metadata["pipeline_binding"] == model_id
    assert resolved.config.metadata["pipeline_route_source"] in {"binding", "pipeline_target"}


@pytest.mark.parametrize(("model_id", "pipeline_target"), HOSTED_API_UNIFIED_ADAPTERS)
def test_hosted_api_pipeline_adapters_honor_the_unified_loader_contract(
    model_id: str,
    pipeline_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_cls = import_pipeline_target(pipeline_target)
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_api_init(_cls, **kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(pipeline_cls, "api_init", classmethod(fake_api_init))

    result = pipeline_cls.from_pretrained(
        model_path={
            "model_id": model_id,
            "pipeline_binding": model_id,
            "runtime_profile": model_id,
            "endpoint": "https://provider.example/v1",
            "api_key": "secret",
            "provider_option": "kept",
        },
        required_components={"component_option": 1},
        device="cpu",
        model_id=model_id,
    )

    assert result is sentinel
    assert observed == {
        "endpoint": "https://provider.example/v1",
        "api_key": "secret",
        "logger": None,
        "provider_option": "kept",
        "component_option": 1,
    }
