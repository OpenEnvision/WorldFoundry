from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import discover_catalog


@pytest.mark.parametrize(
    ("model_id", "width", "height", "num_frames", "fps"),
    (
        ("wan21-fun-14b-cam", 832, 480, 49, 16),
        ("wan22-fun-5b-cam", 1280, 704, 121, 24),
        ("wan22-fun-a14b-cam", 832, 480, 81, 16),
    ),
)
def test_wan_fun_camera_catalog_preserves_full_upstream_demo_contract(
    model_id: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> None:
    entry = {entry.model_id: entry for entry in discover_catalog()}[model_id]

    assert Path(entry.default_input_path).is_file()
    assert Path(entry.default_call_kwargs["pose_txt"]).is_file()
    assert entry.default_call_kwargs["width"] == width
    assert entry.default_call_kwargs["height"] == height
    assert entry.default_call_kwargs["num_frames"] == num_frames
    assert entry.default_call_kwargs["fps"] == fps
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert "pose_txt" in entry.call_params
