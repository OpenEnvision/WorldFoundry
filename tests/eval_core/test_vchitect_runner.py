# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Vchitect catalog, native loading delegation and weight-free planning contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from worldfoundry.evaluation.models import resolve_model_zoo_runner
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.evaluation.runner import ModelBenchmarkSuiteRequest, run_model_benchmark_suite
from worldfoundry.pipelines.vchitect.pipeline_vchitect_2_t2v import Vchitect2T2VPipeline

pytestmark = pytest.mark.unit

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
    assert profile.runtime_status == "integrated"
    assert profile.checkpoints[0]["status"] == "official_demo_parity_pending"
    assert any("native Vchitect recipe" in note for note in profile.notes)


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


def test_vchitect_default_runner_loads_canonical_native_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.vchitect import pipeline_vchitect_2_t2v as module

    native = object()
    calls: list[tuple[str, dict[str, Any]]] = []

    def load(model_id: str, **kwargs: Any) -> Any:
        calls.append((model_id, kwargs))
        return native

    monkeypatch.setattr(module.NativeDiffusionPipeline, "from_pretrained", load)
    resolved = resolve_model_zoo_runner("vchitect-2-t2v", manifest_dir=MODEL_MANIFEST_DIR)
    assert isinstance(resolved.runner.pipeline, Vchitect2T2VPipeline)
    assert resolved.runner.pipeline.get_synthesis_model() is native
    assert (
        resolved.runner.pipeline_target
        == "worldfoundry.pipelines.vchitect.pipeline_vchitect_2_t2v:Vchitect2T2VPipeline"
    )
    assert len(calls) == 1 and calls[0][0] == "vchitect-2-t2v"
    assert set(calls[0][1]) == {"policy", "checkpoint_overrides"}


def test_vchitect_rejects_plan_only_before_loading_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.vchitect import pipeline_vchitect_2_t2v as module

    monkeypatch.setattr(
        module.NativeDiffusionPipeline, "from_pretrained", lambda *args, **kwargs: pytest.fail("weights")
    )
    with pytest.raises(ValueError, match="no longer supports plan_only"):
        Vchitect2T2VPipeline.from_pretrained(
            model_path=str(tmp_path / "checkpoints"),
            model_id="vchitect-2-t2v",
            plan_only=True,
        )


def test_vchitect_runner_rejects_plan_only_parameter_before_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.vchitect import pipeline_vchitect_2_t2v as module

    monkeypatch.setattr(
        module.NativeDiffusionPipeline, "from_pretrained", lambda *args, **kwargs: pytest.fail("weights")
    )
    with pytest.raises(ValueError, match="no longer supports plan_only"):
        resolve_model_zoo_runner(
            "vchitect-2-t2v",
            manifest_dir=MODEL_MANIFEST_DIR,
            parameters={"plan_only": True},
        )


def test_vchitect_call_rejects_plan_only_without_inference() -> None:
    def native(request: Any) -> Any:
        pytest.fail("sampling")

    pipeline = Vchitect2T2VPipeline(native_pipeline=native, device="cpu")
    with pytest.raises(ValueError, match="no longer supports plan_only"):
        pipeline("scene", plan_only=True)


def test_vchitect_plan_is_weight_free_and_retains_recipe_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.vchitect import pipeline_vchitect_2_t2v as module

    monkeypatch.setattr(
        module.NativeDiffusionPipeline, "from_pretrained", lambda *args, **kwargs: pytest.fail("weights")
    )
    result = Vchitect2T2VPipeline.plan("local-vchitect")
    assert result == {
        "model_id": "vchitect-2-t2v",
        "checkpoint": "local-vchitect",
        "backend": "worldfoundry-native-diffusion",
        "native_inference": True,
        "blocked": False,
    }
    assert "video_chunk" not in result and "artifact_path" not in result
