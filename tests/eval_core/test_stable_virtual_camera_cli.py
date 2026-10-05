from __future__ import annotations

from worldfoundry.base_models.three_dimensions.general_3d.stable_virtual_camera.stable_virtual_camera_runtime.cli import (
    parse_main_args,
)


def test_stable_virtual_camera_argparse_fallback_matches_workspace_command() -> None:
    parsed = parse_main_args(
        [
            "--data_path",
            "/tmp/images",
            "--version",
            "1.1",
            "--task",
            "img2trajvid_s-prob",
            "--save_subdir",
            "run",
            "--seed",
            "1234",
            "--pretrained_model_name_or_path",
            "/ckpts/stable-virtual-camera",
            "--weight_name",
            "modelv1.1.safetensors",
            "--num_steps",
            "25",
        ]
    )

    assert parsed["data_path"] == "/tmp/images"
    assert parsed["task"] == "img2trajvid_s-prob"
    assert parsed["seed"] == 1234
    assert parsed["pretrained_model_name_or_path"] == "/ckpts/stable-virtual-camera"
    assert parsed["weight_name"] == "modelv1.1.safetensors"
    assert parsed["num_steps"] == 25
