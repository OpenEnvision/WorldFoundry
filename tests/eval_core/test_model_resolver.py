from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.evaluation.api import GenerationRequest, WorldModelConfig
from worldfoundry.evaluation.models import (
    ModelResolutionError,
    resolve_model_zoo_runner,
    resolve_world_model_runner,
)
from tests.eval_core.contract_fixture import (
    CONTRACT_FIXTURE_MODEL_ID,
    CONTRACT_FIXTURE_RUNNER_TARGET,
    ContractFixtureRunner,
)

CONTRACT_RUNNER_TARGET = CONTRACT_FIXTURE_RUNNER_TARGET


def test_contract_world_model_runner_resolves_and_runs_contract() -> None:
    resolved = resolve_world_model_runner(
        CONTRACT_FIXTURE_MODEL_ID,
        runner=CONTRACT_RUNNER_TARGET,
        parameters={"artifact_kind": "generated_video"},
    )

    assert resolved.runner.__class__ is ContractFixtureRunner
    assert resolved.source == "runner_target"
    assert resolved.runner.model_id == CONTRACT_FIXTURE_MODEL_ID

    results = resolved.runner.generate([GenerationRequest(sample_id="sample-a", task_name="resolver_t2v")])

    assert results[0].status == "succeeded"
    assert results[0].artifacts["generated_video"].uri == "memory://sample-a/generated_video.json"


def test_resolver_imports_runner_target_from_config() -> None:
    resolved = resolve_world_model_runner(
        config=WorldModelConfig(
            model_id="configured-contract",
            runner=CONTRACT_RUNNER_TARGET,
            parameters={"artifact_kind": "video"},
        )
    )

    assert resolved.runner.__class__ is ContractFixtureRunner
    assert resolved.source == "runner_target"
    assert resolved.runner.model_id == "configured-contract"
    assert resolved.runner.output_artifacts == ("video",)


def test_model_zoo_resolver_blocks_listed_only_entries_without_runner_target(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "planned-model",
                        "source": {"status": "open_source"},
                        "integration_status": "planned",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelResolutionError, match="listed_only"):
        resolve_model_zoo_runner("planned-model", manifest_dir=manifest_dir)


def test_model_zoo_resolver_resolves_runner_candidates_with_diagnostics(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "candidate-model",
                        "source": {"status": "open_source"},
                        "integration_status": "planned",
                        "runner_target": CONTRACT_RUNNER_TARGET,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_zoo_runner("candidate-model", manifest_dir=manifest_dir)

    assert resolved.source == "model_zoo"
    assert resolved.diagnostics["entry_runner_entry_kind"] == "runner_candidate"


def test_model_zoo_resolver_constructs_runner_from_entry_metadata(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "zoo-model",
                        "name": "Zoo Model",
                        "source": {"status": "open_source"},
                        "integration_status": "integrated",
                        "runner_target": CONTRACT_RUNNER_TARGET,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_zoo_runner(
        "zoo-model",
        manifest_dir=manifest_dir,
        parameters={"output_artifacts": ["video"]},
    )

    assert resolved.source == "model_zoo"
    assert resolved.runner_target == CONTRACT_RUNNER_TARGET
    assert resolved.runner.model_id == "zoo-model"
    assert resolved.runner.output_artifacts == ("video",)
    assert resolved.diagnostics["entry_id"] == "zoo-model"


def test_model_zoo_resolver_selects_integrated_variant_runner(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "variant-model",
                        "source": {"status": "open_source"},
                        "integration_status": "planned",
                        "variants": [
                            {
                                "variant_id": "planned",
                                "integration_status": "planned",
                                "runner_target": CONTRACT_RUNNER_TARGET,
                            },
                            {
                                "variant_id": "ready",
                                "integration_status": "integrated",
                                "runner_target": CONTRACT_RUNNER_TARGET,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_zoo_runner("variant-model", manifest_dir=manifest_dir)

    assert resolved.source == "model_zoo"
    assert resolved.diagnostics["variant_id"] == "ready"
    assert resolved.diagnostics["variant_integration_status"] == "integrated"


def test_model_zoo_resolver_falls_back_to_planned_variant_runner_target(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "model_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "models.yaml").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "variant-fallback-model",
                        "source": {"status": "open_source"},
                        "integration_status": "planned",
                        "variants": [
                            {
                                "variant_id": "planned-only",
                                "integration_status": "planned",
                                "runner_target": CONTRACT_RUNNER_TARGET,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_zoo_runner("variant-fallback-model", manifest_dir=manifest_dir)

    assert resolved.source == "model_zoo"
    assert resolved.diagnostics["variant_id"] == "planned-only"
