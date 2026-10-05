"""Public discovery and invocation contracts, without weights or Torch."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world_v15 import InspatioWorldV15Pipeline
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_runtime import (
    CHECKPOINT_FILENAME,
    InspatioWorldV15Runtime,
)


def test_v15_discovery_is_lazy():
    script = """
import sys
from worldfoundry.synthesis.visual_generation.inspatio_world import InspatioWorldV15Synthesis
from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world_v15 import InspatioWorldV15Pipeline
pipeline = InspatioWorldV15Pipeline.from_pretrained('not-materialized')
assert 'torch' not in sys.modules
assert pipeline.synthesis_model.runtime.rollout is None
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_v15_catalog_has_distinct_checkpoint_and_route():
    from worldfoundry.evaluation.models.runtime import validate_catalog_references
    from worldfoundry.studio.inference.catalog import find_entry

    old = find_entry("inspatio-world")
    new = find_entry("inspatio-world-1p5")
    assert old.module_path != new.module_path
    assert new.default_model_ref.endswith("world-1.5")
    assert {"images", "videos", "scene_dir", "traj_txt_path", "seed"} <= set(new.call_params)
    assert not [issue for issue in validate_catalog_references() if "inspatio-world-1p5" in issue.field]


def test_v15_plan_does_not_select_older_weights(tmp_path):
    (tmp_path / "InSpatio-World-1.3B.safetensors").touch()
    runtime = InspatioWorldV15Runtime(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="1.5-1.3B"):
        runtime._checkpoint()
    (tmp_path / CHECKPOINT_FILENAME).touch()
    assert runtime._checkpoint() == tmp_path / CHECKPOINT_FILENAME
    runtime = InspatioWorldV15Runtime(str(tmp_path / "InSpatio-World-1.3B.safetensors"))
    with pytest.raises(ValueError, match="older weights"):
        runtime._checkpoint()


def test_v15_plan_requires_components_without_loading(tmp_path):
    (tmp_path / CHECKPOINT_FILENAME).touch()
    runtime = InspatioWorldV15Runtime(str(tmp_path), wan_model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="config.json"):
        runtime.plan()
    for name in ("config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                 "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/spiece.model"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert runtime.plan()["frames_per_block"] == 3
    assert runtime.rollout is None


def test_v15_checkpoint_symlink_retains_safetensors_format(tmp_path):
    blob = tmp_path / "blobs/released-weight-hash"
    blob.parent.mkdir()
    blob.write_bytes(b"released-weights")
    snapshot = tmp_path / "snapshots/revision"
    snapshot.mkdir(parents=True)
    checkpoint = snapshot / CHECKPOINT_FILENAME
    checkpoint.symlink_to(blob)
    resolved = InspatioWorldV15Runtime(str(checkpoint))._checkpoint()
    assert resolved == checkpoint and resolved.name == CHECKPOINT_FILENAME
    assert resolved.read_bytes() == blob.read_bytes()


def test_v15_unified_invocation_preserves_camera_and_artifact(tmp_path):
    seen = {}
    class Backend:
        def predict(self, **kwargs):
            seen.update(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"encoded-result")
            return {"artifact_path": kwargs["output_path"]}
    pipeline = InspatioWorldV15Pipeline(synthesis_model=Backend())
    invocation = SimpleNamespace(image="a.png", video=None, prompt="room",
                                 output_path=tmp_path / "result.mp4",
                                 pipeline_kwargs={"scene_dir": "/prepared", "traj_txt_path": "camera.txt", "seed": 17})
    result = pipeline.run_pipeline_invocation(invocation)
    assert result["status"] == "succeeded"
    assert seen["scene_dir"] == "/prepared" and seen["traj_txt_path"] == "camera.txt"
    assert seen["images"] is None and seen["videos"] is None and seen["seed"] == 17
    assert Path(result["artifact_path"]).is_file()


def test_v15_has_replay_recipes_for_each_source_type():
    matrix = Path(__file__).resolve().parents[1] / "manual/geometry_regression_cases.json"
    cases = [case for case in json.loads(matrix.read_text()).values()
             if case.get("model_id") == "inspatio-world-1p5"]
    assert len(cases) == 3
    assert {case["call"]["scene_dir"].rsplit("/", 1)[-1] for case in cases} == {
        "image_example_00", "multiview_example_00", "video_example_00",
    }
    assert all(case["deterministic"] and case["call"]["return_latents"]
               and case["required_outputs"] == ["result.latents", "result.video"] for case in cases)
