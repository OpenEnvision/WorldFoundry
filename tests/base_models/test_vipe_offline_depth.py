from __future__ import annotations

from pathlib import Path

import pytest


def test_unidepth_prefers_local_snapshot_and_fails_offline(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.base_models.three_dimensions.depth.unidepth import resolve_unidepth_source

    snapshot = tmp_path / "unidepth"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("WORLDFOUNDRY_UNIDEPTH_MODEL", str(snapshot))
    assert resolve_unidepth_source("l") == str(snapshot)

    monkeypatch.setenv("WORLDFOUNDRY_UNIDEPTH_MODEL", str(tmp_path / "missing"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        resolve_unidepth_source("l")


def test_priorda_resolves_local_dir_and_fails_offline(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.base_models.three_dimensions.depth.priorda.priorda import resolve_priorda_file

    weight = tmp_path / "depth_anything_v2_vitb.pth"
    weight.write_bytes(b"w")
    monkeypatch.setenv("WORLDFOUNDRY_PRIORDA_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert resolve_priorda_file("depth_anything_v2_vitb.pth") == str(weight)

    monkeypatch.setenv("WORLDFOUNDRY_PRIORDA_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("WORLDFOUNDRY_CKPT_DIR", raising=False)
    monkeypatch.delenv("WORLDARENA_CHECKPOINT_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="offline"):
        resolve_priorda_file("prior_depth_anything_vitb.pth")


def test_vda_checkpoint_discovery_and_offline_fail(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.base_models.three_dimensions.depth.videodepthanything.paths import (
        resolve_video_depth_checkpoint,
        small_checkpoint_path,
    )

    weight = tmp_path / "Video-Depth-Anything-Small" / "video_depth_anything_vits.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"vda")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    monkeypatch.delenv("WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT", raising=False)
    assert small_checkpoint_path() == weight
    assert resolve_video_depth_checkpoint("vits") == weight

    import worldfoundry.base_models.three_dimensions.depth.videodepthanything.paths as vda_paths

    monkeypatch.delenv("WORLDFOUNDRY_CKPT_DIR", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT", raising=False)
    monkeypatch.setattr(vda_paths, "_checkpoint_roots", lambda: [tmp_path / "missing"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        resolve_video_depth_checkpoint("vits")


def test_slam_keeps_weight_holders_across_videos() -> None:
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.slam.constants import (
        SLAM_PER_VIDEO_COMPONENTS,
    )

    assert "metric_depth" not in SLAM_PER_VIDEO_COMPONENTS
    assert "droid_net" not in SLAM_PER_VIDEO_COMPONENTS
    assert "buffer" in SLAM_PER_VIDEO_COMPONENTS
    assert "frontend" in SLAM_PER_VIDEO_COMPONENTS


def test_adaptive_depth_models_are_reusable_weight_holders() -> None:
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.adaptive_models import (
        AdaptiveDepthModels,
    )

    shared = AdaptiveDepthModels(
        video_depth_model="vda",
        depth_model="unidepth",
        prompt_model="priorda",
    )
    again = AdaptiveDepthModels(
        video_depth_model=shared.video_depth_model,
        depth_model=shared.depth_model,
        prompt_model=shared.prompt_model,
    )
    assert again == shared


def test_memory_depth_pipeline_config_is_complete() -> None:
    pytest.importorskip("loguru")
    from worldfoundry.base_models.three_dimensions.general_3d.vipe import get_config_path
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import parse_typed_config

    config = parse_typed_config(
        "default",
        [
            "pipeline=default",
            "streams.base_path=/tmp/input.mp4",
            "pipeline.init.instance=null",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_viz=false",
        ],
        config_dir=get_config_path(),
    )
    assert config.pipeline.init.instance is None
    assert config.pipeline.slam.keyframe_depth == "unidepth-l"
    assert config.pipeline.post.depth_align_model == "adaptive_unidepth-l_svda"
    assert config.pipeline.output.save_artifacts is True
    assert config.pipeline.output.save_viz is False
