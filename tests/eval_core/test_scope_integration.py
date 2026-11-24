from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evals.api import GenerationRequest
from worldfoundry.evals.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.evals.models.runners.resolver import resolve_model_zoo_runner
from worldfoundry.pipelines.scope.pipeline_scope import SCOPEPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


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
