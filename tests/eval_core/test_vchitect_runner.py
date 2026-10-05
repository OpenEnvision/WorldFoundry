from __future__ import annotations

from pathlib import Path

import pytest

# worldfoundry.pipelines.vchitect imports ftfy (optional dependency) at module
# load time; skip in environments without it.
pytest.importorskip("ftfy")

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models import resolve_model_zoo_runner
from worldfoundry.evaluation.runner import ModelBenchmarkSuiteRequest, run_model_benchmark_suite
from worldfoundry.pipelines.vchitect.pipeline_vchitect_2_t2v import Vchitect2T2VPipeline
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
BENCHMARK_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"


def test_vchitect_model_zoo_entry_exposes_generated_video_benchmark_contract() -> None:
    entry = load_model_zoo_registry(MODEL_MANIFEST_DIR).get("vchitect-2-t2v")
    profile = load_runtime_profile("vchitect-2-t2v")

    assert entry.integration_status == "integrated"
    assert entry.runner_entry_kind == "runnable_runner"
    assert entry.output_artifacts == ("generated_video",)
    assert entry.pipeline_target == "worldfoundry.pipelines.vchitect.pipeline_vchitect_2_t2v:Vchitect2T2VPipeline"
    assert profile.artifact_kind == "generated_video"
    assert profile.artifact_filename == "vchitect_2_t2v_public_runner.mp4"
    assert profile.runtime_status == "in_tree_vchitect_2_t2v_public_runner_gpu_validation_verified"
    assert any("vchitect2_public_runner_execute_20260521_200412" in note for note in profile.notes)


def test_vchitect_suite_plan_routes_generated_video_to_external_benchmarks(tmp_path: Path) -> None:
    result = run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=tmp_path / "suite",
            benchmark_manifest_dir=BENCHMARK_MANIFEST_DIR,
            model_manifest_dir=MODEL_MANIFEST_DIR,
            model_ids=("vchitect-2-t2v",),
            benchmark_ids=("vbench-2.0", "video-bench"),
            execute=False,
        )
    )

    cells = {(cell["model_id"], cell["benchmark_id"]): cell for cell in result.cells}

    assert result.status == "planned"
    assert cells[("vchitect-2-t2v", "vbench-2.0")]["output_artifact"] == "generated_video"
    assert cells[("vchitect-2-t2v", "vbench-2.0")]["required_artifacts"] == ["generated_video"]
    assert cells[("vchitect-2-t2v", "vbench-2.0")]["compatibility"] == "compatible"
    assert cells[("vchitect-2-t2v", "vbench-2.0")]["reason"] is None
    assert cells[("vchitect-2-t2v", "video-bench")]["output_artifact"] == "generated_video"
    assert cells[("vchitect-2-t2v", "video-bench")]["required_artifacts"] == ["generated_video"]
    assert cells[("vchitect-2-t2v", "video-bench")]["compatibility"] == "compatible"
    assert cells[("vchitect-2-t2v", "video-bench")]["reason"] is None


def test_vchitect_default_runner_uses_real_runtime_path() -> None:
    resolved = resolve_model_zoo_runner("vchitect-2-t2v", manifest_dir=MODEL_MANIFEST_DIR)

    synthesis_model = resolved.runner.pipeline.get_synthesis_model()
    assert "plan_only" not in synthesis_model.runtime_kwargs
    assert resolved.runner.pipeline_target == "worldfoundry.pipelines.vchitect.pipeline_vchitect_2_t2v:Vchitect2T2VPipeline"


def test_vchitect_rejects_plan_only_runtime_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no longer supports plan_only"):
        Vchitect2T2VPipeline.from_pretrained(
            model_path=str(tmp_path / "checkpoints" / "hfd" / "Vchitect--Vchitect-2.0-2B"),
            model_id="vchitect-2-t2v",
            plan_only=True,
        )


def test_vchitect_runner_rejects_plan_only_parameter() -> None:
    with pytest.raises(ValueError, match="no longer supports plan_only"):
        resolve_model_zoo_runner(
            "vchitect-2-t2v",
            manifest_dir=MODEL_MANIFEST_DIR,
            parameters={"plan_only": True},
        )
