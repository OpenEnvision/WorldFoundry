from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.pipelines import build_pipeline_runner_spec, import_pipeline_target
from worldfoundry.evaluation.models.pipelines.aliases import build_alias_mapping, load_pipeline_alias_registry
from worldfoundry.evaluation.models.pipelines.bindings import (
    load_pipeline_binding,
    resolve_pipeline_binding,
    resolve_pipeline_route,
)
from worldfoundry.evaluation.models.pipelines.loading import load_pipeline_from_config
from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.pipelines.pipeline_utils import PipelineABC


class DummyPipeline(PipelineABC):
    def __init__(
        self,
        *,
        model_id: str,
        model_path: Mapping[str, Any],
        required_components: Mapping[str, Any] | None,
        device: str,
    ) -> None:
        self.model_id = model_id
        self.model_path = dict(model_path)
        self.required_components = dict(required_components or {})
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Mapping[str, Any],
        required_components: Mapping[str, Any] | None,
        device: str,
        model_id: str,
    ) -> "DummyPipeline":
        return cls(
            model_id=model_id,
            model_path=model_path,
            required_components=required_components,
            device=device,
        )


def _target() -> str:
    return f"{__name__}:DummyPipeline"


def test_pipeline_abc_default_contract_is_usable() -> None:
    class EchoPipeline(PipelineABC):
        pass

    pipe = EchoPipeline.from_pretrained(model_path={"model_id": "echo"}, device="cpu")

    assert pipe.process(prompt="hello") == {"prompt": "hello"}
    assert pipe({"sample": 1}) == {"sample": 1}


def test_alias_mapping_reuses_canonical_handlers_with_validation() -> None:
    handlers = {"canonical": object()}

    mapping = build_alias_mapping(handlers, {"alias": "canonical"})

    assert mapping["alias"] is handlers["canonical"]
    with pytest.raises(KeyError):
        build_alias_mapping(handlers, {"missing": "unknown"})


def test_pipeline_binding_loader_validates_explicit_schema_version(tmp_path: Path) -> None:
    binding_path = tmp_path / "binding.yaml"
    binding_path.write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: dummy",
                "model_id: dummy-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                f"  target: {_target()}",
                "",
            )
        ),
        encoding="utf-8",
    )

    binding = load_pipeline_binding(binding_path)

    assert binding.binding_id == "dummy"
    assert binding.model_id == "dummy-model"
    assert binding.pipeline_target == _target()


def test_pipeline_alias_registry_loads_data_backed_aliases(tmp_path: Path) -> None:
    alias_root = tmp_path / "aliases"
    alias_root.mkdir()
    (alias_root / "aliases.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "aliases:",
                "  short: dummy",
                "",
            )
        ),
        encoding="utf-8",
    )

    aliases = load_pipeline_alias_registry(alias_root)

    assert aliases.canonical_id("short") == "dummy"


def test_runner_spec_resolves_pipeline_binding_from_config_metadata(tmp_path: Path) -> None:
    binding_root = tmp_path / "bindings" / "pipelines"
    alias_root = tmp_path / "bindings" / "aliases"
    (binding_root / "video").mkdir(parents=True)
    alias_root.mkdir(parents=True)
    (binding_root / "video" / "dummy.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: dummy",
                "model_id: dummy-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                f"  target: {_target()}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (alias_root / "video.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "aliases:",
                "  alias: dummy",
                "",
            )
        ),
        encoding="utf-8",
    )
    config = WorldModelConfig(
        model_id="dummy-model",
        runner="worldfoundry.pipeline",
        runtime={
            "device": "cpu",
            "pipeline_binding": "alias",
            "pipeline_bindings_root": str(binding_root),
        },
    )

    binding = resolve_pipeline_binding("alias", root=binding_root)
    spec = build_pipeline_runner_spec(config)

    assert binding.pipeline_target == _target()
    assert spec.pipeline_target == _target()
    assert spec.device == "cpu"


def test_pipeline_route_falls_back_to_model_id_binding_without_catalog_duplication(tmp_path: Path) -> None:
    binding_root = tmp_path / "bindings" / "pipelines"
    binding_root.mkdir(parents=True)
    (binding_root / "dummy-model.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: dummy-model",
                "model_id: dummy-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                f"  target: {_target()}",
                "",
            )
        ),
        encoding="utf-8",
    )
    config = WorldModelConfig(
        model_id="dummy-model",
        runner="worldfoundry.pipeline",
        runtime={
            "device": "cpu",
            "pipeline_bindings_root": str(binding_root),
        },
    )

    route = resolve_pipeline_route(model_id="dummy-model", binding_root=binding_root)
    spec = build_pipeline_runner_spec(config)

    assert route == (_target(), "dummy-model", "model_id.pipeline_binding")
    assert spec.pipeline_target == _target()


def test_pipeline_loader_and_runner_from_config_use_resolved_target() -> None:
    config = WorldModelConfig(
        model_id="dummy-model",
        runner="worldfoundry.pipeline",
        parameters={"required_components": {"checkpoint": "required"}},
        runtime={"device": "cpu"},
        metadata={"pipeline_target": _target()},
    )

    spec, pipeline = load_pipeline_from_config(config)
    runner = WorldFoundryPipelineRunner.from_config(config)

    assert import_pipeline_target(_target()) is DummyPipeline
    assert spec.pipeline_target == _target()
    assert isinstance(pipeline, DummyPipeline)
    assert pipeline.model_id == "dummy-model"
    assert pipeline.required_components == {"checkpoint": "required"}
    assert isinstance(runner, WorldFoundryPipelineRunner)
    assert isinstance(runner.pipeline, DummyPipeline)


def test_pipeline_facade_exports_path_a_only() -> None:
    import worldfoundry.evaluation.models.pipelines as pipelines

    assert pipelines.build_pipeline_runner_spec is build_pipeline_runner_spec
    assert pipelines.import_pipeline_target is import_pipeline_target
    for removed_name in (
        "video" + "_gen_pipe",
        "embodied" + "_action_pipe",
        "three" + "_dim_pipe",
        "NAMED" + "_PIPELINES",
    ):
        assert not hasattr(pipelines, removed_name)
