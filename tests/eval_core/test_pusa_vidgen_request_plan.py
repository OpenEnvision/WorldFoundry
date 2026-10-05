from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models import resolve_model_zoo_runner
from worldfoundry.pipelines.pusa_vidgen.pipeline_pusa_vidgen import PusaVidGenPipeline
from worldfoundry.synthesis.visual_generation.pusa_vidgen.adapter import (
    PusaVidGenRuntimePlan,
)
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_pusa_catalog_and_runtime_profile_document_generated_video_runner() -> None:
    """Verify Pusa metadata records the generated-video public runner validation.

    Args:
        None.
    """
    entry = load_model_zoo_registry(MODEL_MANIFEST_DIR).get("pusa-vidgen")
    profile = load_runtime_profile("pusa-vidgen")

    assert entry.integration_status == "integrated"
    assert entry.runner_entry_kind == "runnable_runner"
    assert entry.output_artifacts == ("generated_video",)
    assert profile.artifact_kind == "generated_video"
    assert profile.artifact_filename == "pusa_vidgen_public_runner.mp4"
    assert profile.runtime_status == "in_tree_pusa_vidgen_public_runner_gpu_validation_verified"
    assert profile.backend_stage == "in_tree_runtime"
    assert any(
        checkpoint.get("repo_id") == "RaphaelLiu/Pusa-V0.5"
        and checkpoint.get("role") == "official_pusa_v0p5_mochi_checkpoint"
        and checkpoint.get("status") == "local_checkpoint_staged_public_runner_gpu_validation_verified"
        for checkpoint in profile.checkpoints
    )
    assert any(
        checkpoint.get("repo_id") == "Wan-AI/Wan2.2-T2V-A14B"
        and checkpoint.get("status") == "local_checkpoint_staged_optional_not_used_in_verified_pusa_v0p5_smoke"
        for checkpoint in profile.checkpoints
    )
    assert any(
        checkpoint.get("repo_id") == "lightx2v/Wan2.2-Lightning"
        and checkpoint.get("status") == "local_checkpoint_staged_optional_not_used_in_verified_pusa_v0p5_smoke"
        for checkpoint in profile.checkpoints
    )
    assert any("pusa_vidgen_public_runner_execute_20260521_193000" in note for note in profile.notes)


def test_pusa_pipeline_default_call_rejects_request_plan(tmp_path: Path) -> None:
    """Verify Pusa no longer emits request-plan artifacts.

    Args:
        tmp_path: Temporary directory for local path placeholders.
    """
    pipeline = PusaVidGenPipeline.from_pretrained(
        model_path={
            "checkpoint_root": tmp_path / "Pusa",
            "base_model_root": tmp_path / "Wan",
            "lightx2v_root": tmp_path / "Lightning",
        },
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="A robot walking through a forest",
            images="memory://frame.png",
            output_path=tmp_path / "pusa_request_plan.json",
            return_dict=True,
            num_frames=8,
        )

    assert not (tmp_path / "pusa_request_plan.json").exists()


def test_pusa_pipeline_execute_path_records_generated_video(tmp_path: Path) -> None:
    """Verify execute=True returns a generated-video artifact through the public pipeline.

    Args:
        tmp_path: Temporary directory for mocked runner artifacts.
    """
    pipeline = PusaVidGenPipeline.from_pretrained(
        model_path={
            "checkpoint_root": tmp_path / "Pusa",
            "base_model_root": tmp_path / "Wan",
            "python_executable": tmp_path / "env" / "bin" / "python",
        },
        device="cuda",
    )
    generated = tmp_path / "outputs" / "pusa_vidgen_public_runner.mp4"
    generated.parent.mkdir()
    generated.write_bytes(b"mp4")
    plan = PusaVidGenRuntimePlan(
        command=("python", "-m", "worldfoundry.synthesis.visual_generation.pusa_vidgen.official_runner"),
        env={},
        workdir=str(tmp_path),
        checkpoint_root=str(tmp_path / "Pusa"),
        output_dir=str(generated.parent),
        output_path=str(generated),
    )
    pipeline.runtime.build_plan = lambda **kwargs: plan
    pipeline.runtime.run_plan = lambda runtime_plan, **kwargs: {
        "ok": True,
        "status": "success",
        "returncode": 0,
        "duration_seconds": 0.1,
        "generated_count": 1,
        "generated_files": [str(generated)],
        "metadata_path": str(generated.with_suffix(".json")),
        "stdout_path": str(tmp_path / "stdout.log"),
        "stderr_path": str(tmp_path / "stderr.log"),
        "runtime_plan": runtime_plan.to_dict(),
    }

    report_path = tmp_path / "pusa_report.json"
    result = pipeline(
        prompt="A robot walking through a forest",
        output_path=report_path,
        return_dict=True,
        execute=True,
        output_dir=generated.parent,
        num_frames=1,
        height=64,
        width=64,
        num_inference_steps=1,
    )

    assert result["status"] == "success"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"] == str(generated)
    assert result["backend_quality"] == "public_runner_official_runner"
    assert report_path.is_file()


def test_pusa_runner_smoke_fails_without_execute() -> None:
    """Verify the model runner no longer exposes request-plan results.

    Args:
        None.
    """
    resolved = resolve_model_zoo_runner(
        "pusa-vidgen",
        manifest_dir=MODEL_MANIFEST_DIR,
        runtime={"device": "cpu"},
    )
    results = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="pusa-validation",
                task_name="pusa:validation",
                inputs={"prompt": "A robot walking through a forest"},
                output_schema={"generated_video": {"kind": "video"}},
            )
        ]
    )

    assert results[0].status == "failed"
    assert not results[0].artifacts
    assert results[0].error == "RuntimeError: Pusa VidGen requires execute=True; request-plan artifacts are no longer emitted."
