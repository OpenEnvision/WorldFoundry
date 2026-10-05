from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models import WorldFoundryPipelineRunner, resolve_model_zoo_runner
from worldfoundry.pipelines.scope.pipeline_scope import SCOPEPipeline
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_scope_catalog_profile_and_pipeline_target_are_registered() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get("scope")
    profile = load_runtime_profile("scope")

    assert entry.integration_status == "planned"
    assert entry.runner_entry_kind == "runner_candidate"
    assert entry.pipeline_target == "worldfoundry.pipelines.scope.pipeline_scope:SCOPEPipeline"
    assert profile.integration_status == "planned"
    assert profile.backend_stage == "in_tree_official_cli"
    assert profile.artifact_kind == "generated_video"


def test_scope_runtime_logic_lives_under_base_models() -> None:
    synthesis_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/scope/scope_synthesis.py"
    )
    base_runtime_path = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/scope/worldfoundry_runtime.py"
    )

    synthesis_text = synthesis_path.read_text(encoding="utf-8")
    runtime_text = base_runtime_path.read_text(encoding="utf-8")

    assert "worldfoundry.synthesis.visual_generation.scope.worldfoundry_runtime" in synthesis_text
    assert "class SCOPERuntime" in runtime_text
    assert "subprocess.run" in runtime_text
    assert "def _subprocess_env" in runtime_text
    assert "subprocess.run" not in synthesis_text
    assert "def _subprocess_env" not in synthesis_text
    assert "def _argv" not in synthesis_text


def test_scope_pipeline_default_call_rejects_request_plan(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    action_path = tmp_path / "actions.parquet"
    image_path.write_bytes(b"not-a-real-image")
    action_path.write_bytes(b"not-a-real-parquet")
    pipeline = SCOPEPipeline.from_pretrained(
        model_path={"model_dir": tmp_path / "weights"},
        device="cpu",
    )

    plan_path = tmp_path / "scope.request_plan.json"
    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="first-person test scene",
            images=image_path,
            interactions=action_path,
            output_path=tmp_path / "scope.mp4",
            return_dict=True,
        )

    assert not plan_path.exists()


def test_scope_model_zoo_runner_fails_without_execute(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    action_path = tmp_path / "actions.parquet"
    image_path.write_bytes(b"not-a-real-image")
    action_path.write_bytes(b"not-a-real-parquet")
    resolved = resolve_model_zoo_runner(
        "scope",
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )

    assert isinstance(resolved.runner, WorldFoundryPipelineRunner)
    result = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="scope-smoke",
                task_name="scope:smoke",
                inputs={"prompt": "walk forward", "image": str(image_path)},
                controls={"sample_controls": {"actions": str(action_path)}},
                generation_kwargs={"output_dir": str(tmp_path / "runner")},
                output_schema={"generated_video": {"kind": "video"}},
            )
        ]
    )[0]

    plan_path = tmp_path / "runner" / "scope-smoke_scope.request_plan.json"

    assert result.status == "failed"
    assert result.artifacts == {}
    assert result.error == "RuntimeError: SCOPE requires execute=True; request-plan artifacts are no longer emitted."
    assert not plan_path.exists()
