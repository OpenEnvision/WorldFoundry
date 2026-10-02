from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models import (
    ModelRunnerRegistry,
    model_runner_registry_report,
    model_runner_registry_snapshot,
    resolve_model_zoo_config,
    resolve_world_model_runner,
)
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_RUNNER_TARGET, ContractFixtureRunner

PIPELINE_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"


def _write_model_zoo_manifest(manifest_dir: Path, models: list[dict[str, object]]) -> None:
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(json.dumps({"models": models}), encoding="utf-8")


def test_model_runner_registry_lists_builtin_runtime_aliases() -> None:
    registry = ModelRunnerRegistry()

    names = {entry.name for entry in registry.list()}
    pipeline_entry = registry.get("worldfoundry.pipeline")
    pipeline_colon_entry = registry.get("worldfoundry:pipeline")
    pipeline_dash_entry = registry.get("worldfoundry-pipeline")

    assert "worldfoundry.pipeline" in names
    assert CONTRACT_FIXTURE_RUNNER_TARGET not in names
    assert "worldfoundry:jsonl-subprocess" not in names
    assert pipeline_colon_entry is pipeline_entry
    assert pipeline_dash_entry is pipeline_entry
    assert pipeline_entry.runner_class.__name__ == "WorldFoundryPipelineRunner"
    with pytest.raises(KeyError):
        registry.get("worldfoundry:jsonl-subprocess")
    with pytest.raises(KeyError):
        registry.get("worldfoundry:" + "smoke")
    with pytest.raises(KeyError):
        registry.get("vla_va_wam:contract")


def test_model_runner_registry_lazily_resolves_module_targets_and_creates_runner() -> None:
    registry = ModelRunnerRegistry(include_builtins=False)
    target = CONTRACT_FIXTURE_RUNNER_TARGET

    entry = registry.resolve_key(target)

    assert entry.name == target
    assert entry.runner_target == target
    assert entry.source == "module_target"
    assert entry.runner_class is None

    runner = registry.create(
        WorldModelConfig(
            model_id="configured-contract",
            runner=target,
            parameters={"output_artifacts": ["generated_video"]},
        )
    )

    assert runner.__class__ is ContractFixtureRunner
    assert runner.model_id == "configured-contract"
    assert runner.output_artifacts == ("generated_video",)


def test_model_runner_registry_rejects_malformed_unregistered_targets() -> None:
    registry = ModelRunnerRegistry(include_builtins=False)

    with pytest.raises(KeyError):
        registry.resolve_key("not-a-module-target")
    with pytest.raises(KeyError):
        registry.resolve_key("missing_attr:")


def test_model_runner_registry_replace_drops_stale_aliases() -> None:
    registry = ModelRunnerRegistry(include_builtins=False)
    registry.register_runner(
        "replaceable",
        runner_target="worldfoundry.old:Runner",
        aliases=("old-alias",),
    )

    registry.register_runner(
        "replaceable",
        runner_target="worldfoundry.new:Runner",
        aliases=("new-alias",),
        replace=True,
    )

    assert registry.get("replaceable").runner_target == "worldfoundry.new:Runner"
    assert registry.get("new-alias").runner_target == "worldfoundry.new:Runner"
    with pytest.raises(KeyError):
        registry.get("old-alias")


def test_model_runner_registry_discovers_env_plugins_without_importing_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_RUNNERS", "plugin-runner=missing_plugin_module:PluginRunner")

    registry = model_runner_registry_snapshot()
    entry = registry.get("plugin-runner")

    assert entry.runner_target == "missing_plugin_module:PluginRunner"
    assert entry.source == "plugin"
    assert entry.origin == "WORLDFOUNDRY_MODEL_RUNNERS entry 'plugin-runner=missing_plugin_module:PluginRunner'"
    assert entry.runner_class is None
    with pytest.raises(ModuleNotFoundError):
        registry.resolve_runner_class("plugin-runner")


def test_model_runner_registry_reports_plugin_collisions_without_shadowing_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_RUNNERS", "worldfoundry:pipeline=missing_plugin_module:PluginRunner")

    report = model_runner_registry_report()
    entry = next(item for item in report.entries if item.name == "worldfoundry.pipeline")

    assert entry.source == "builtin"
    assert entry.runner_class.__name__ == "WorldFoundryPipelineRunner"
    assert {issue.code for issue in report.issues} == {"plugin_runner_collision"}
    assert report.issues[0].severity == "warning"


def test_world_model_resolver_uses_model_runner_registry_for_builtins() -> None:
    resolved = resolve_world_model_runner(
        "contract-model",
        runner=CONTRACT_FIXTURE_RUNNER_TARGET,
        parameters={"output_artifacts": ["generated_video"]},
    )

    assert isinstance(resolved.runner, ContractFixtureRunner)
    assert resolved.runner_target == CONTRACT_FIXTURE_RUNNER_TARGET
    assert resolved.runner.output_artifacts == ("generated_video",)


def test_model_zoo_config_resolution_selects_variant_without_instantiating_runner(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    _write_model_zoo_manifest(
        manifest_dir,
        [
            {
                "model_id": "lazy-zoo-model",
                "source": {"status": "open_source"},
                "integration_status": "planned",
                "variants": [
                    {
                        "variant_id": "ready",
                        "integration_status": "integrated",
                        "runner_target": "missing_runner_module:MissingRunner",
                        "runtime_profile": "contract-fixture",
                    }
                ],
            }
        ],
    )

    resolved = resolve_model_zoo_config("lazy-zoo-model", manifest_dir=manifest_dir)

    assert resolved.model_id == "lazy-zoo-model"
    assert resolved.runner_target == "missing_runner_module:MissingRunner"
    assert resolved.config.runner == "missing_runner_module:MissingRunner"
    assert resolved.config.variant == "ready"
    assert resolved.config.metadata["runtime_profile"] == "contract-fixture"
    assert resolved.diagnostics["variant_id"] == "ready"
    assert resolved.diagnostics["runtime_profile"] == "contract-fixture"


def test_model_zoo_config_resolution_uses_model_id_pipeline_binding(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    binding_root = tmp_path / "bindings" / "pipelines"
    binding_root.mkdir(parents=True)
    (binding_root / "lazy-pipeline-model.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "binding_id: lazy-pipeline-model",
                "model_id: lazy-pipeline-model",
                "runner: worldfoundry.pipeline",
                "pipeline:",
                "  target: missing_pipeline_module:MissingPipeline",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_model_zoo_manifest(
        manifest_dir,
        [
            {
                "model_id": "lazy-pipeline-model",
                "source": {"status": "open_source"},
                "integration_status": "integrated",
                "runner_target": PIPELINE_RUNNER_TARGET,
            }
        ],
    )

    resolved = resolve_model_zoo_config(
        "lazy-pipeline-model",
        manifest_dir=manifest_dir,
        runtime={"pipeline_bindings_root": str(binding_root)},
    )

    assert resolved.config.metadata["pipeline_target"] == "missing_pipeline_module:MissingPipeline"
    assert resolved.config.metadata["pipeline_binding"] == "lazy-pipeline-model"
    assert resolved.config.metadata["pipeline_route_source"] == "model_id.pipeline_binding"
    assert resolved.diagnostics["pipeline_target"] == "missing_pipeline_module:MissingPipeline"
    assert resolved.diagnostics["pipeline_binding"] == "lazy-pipeline-model"
    assert resolved.diagnostics["pipeline_route_source"] == "model_id.pipeline_binding"
