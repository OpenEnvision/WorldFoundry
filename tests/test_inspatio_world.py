import os
from pathlib import Path

import pytest

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    _model_name_from_repo_id,
)
from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.synthesis.visual_generation.inspatio_world.worldfoundry_runtime import (
    InspatioWorldRuntime,
)


def test_inspatio_causal_writer_uses_shared_torchvision_pyav_fallback():
    script = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/inspatio_world/"
        "inspatio_world_runtime/inference_causal.py"
    ).read_text(encoding="utf-8")

    assert "from worldfoundry.core.media.codecs.video import write_video_torchvision" in script
    assert "write_video_torchvision(filename, video_array, fps=fps)" in script


def test_inspatio_workspace_video_list_accepts_resume_directory(tmp_path):
    source_dir = tmp_path / "previous-input"
    source_dir.mkdir()
    source_video = source_dir / "movie.mp4"
    source_video.write_bytes(b"video")
    (source_dir / "new_vggt").mkdir()

    runtime = object.__new__(InspatioWorldRuntime)
    input_dir, staged_videos, resolved_source = runtime._stage_input_dir(
        [str(source_dir)],
        tmp_path / "rerun",
        fps=24,
    )

    assert resolved_source == source_dir.resolve()
    assert input_dir == tmp_path / "rerun" / "input"
    assert staged_videos == [input_dir / "movie.mp4"]
    assert staged_videos[0].is_symlink()


def test_depth_anything_v3_resolves_nested_model_from_local_hfd_export(tmp_path):
    model_root = tmp_path / "depth-anything--DA3NESTED-GIANT-LARGE-1.1"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        '{"model_name": "da3nested-giant-large"}\n',
        encoding="utf-8",
    )

    assert _model_name_from_repo_id(str(model_root)) == "da3nested-giant-large"
    assert (
        _model_name_from_repo_id("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        == "da3nested-giant-large"
    )


def test_inspatio_world_workspace_defaults_use_official_repo_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub"))

    assert studio_catalog._inspatio_world_default_ref() == "inspatio/world"
    assert (
        studio_catalog._depth_anything_v3_default_ref()
        == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    )


def test_inspatio_world_workspace_defaults_resolve_complete_local_exports(monkeypatch, tmp_path):
    hfd_root = tmp_path / "hfd"
    main_root = hfd_root / "inspatio--world"
    da3_root = hfd_root / "depth-anything--DA3NESTED-GIANT-LARGE-1.1"
    main_root.mkdir(parents=True)
    da3_root.mkdir(parents=True)
    (main_root / "InSpatio-World-1.3B.safetensors").touch()
    (da3_root / "config.json").touch()
    (da3_root / "model.safetensors").touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub"))

    assert studio_catalog._inspatio_world_default_ref() == str(main_root.resolve())
    assert studio_catalog._depth_anything_v3_default_ref() == str(da3_root.resolve())


def test_inspatio_world_gpu_integration(tmp_path):
    if os.environ.get("INSPATIO_WORLD_RUN_INTEGRATION") != "1":
        pytest.skip("Set INSPATIO_WORLD_RUN_INTEGRATION=1 with local checkpoints to run this GPU integration test.")

    device = os.environ.get("INSPATIO_WORLD_DEVICE", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("InSpatio-World GPU integration test requested but CUDA is unavailable.")

    from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world import InspatioWorldPipeline

    video_path = os.environ.get(
        "INSPATIO_WORLD_INPUT_VIDEO",
        str(worldfoundry_data_path("test_cases", "longcat_video", "motorcycle.mp4")),
    )
    pipeline = InspatioWorldPipeline.from_pretrained(
        model_path=os.environ.get("INSPATIO_WORLD_MODEL_PATH", "inspatio/world"),
        required_components={
            "wan_model_path": os.environ.get("INSPATIO_WORLD_WAN_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B"),
            "da3_model_path": os.environ.get(
                "INSPATIO_WORLD_DA3_MODEL_PATH",
                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            ),
            "florence_model_path": os.environ.get("INSPATIO_WORLD_FLORENCE_MODEL_PATH", "microsoft/Florence-2-large"),
        },
        device=device,
    )
    result = pipeline(
        videos=video_path,
        traj_txt_path=os.environ.get("INSPATIO_WORLD_TRAJ", "x_y_circle_cycle.txt"),
        prompt="A motorcycle moves through a natural outdoor scene.",
        output_dir=str(tmp_path / "inspatio_world_output"),
        return_dict=True,
    )
    assert result["generated_video_paths"]
