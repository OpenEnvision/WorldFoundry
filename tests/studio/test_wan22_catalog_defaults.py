from __future__ import annotations

import json
from pathlib import Path

import av
import torch
from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.open_oasis.utils import (
    load_actions,
    load_prompt,
    write_video_compat,
)
from worldfoundry.synthesis.visual_generation.open_oasis.worldfoundry_runtime import (
    build_command as build_oasis_command,
)


def test_wan22_ti2v_catalog_uses_local_unified_checkpoint() -> None:
    entry = find_entry("wan2.2-ti2v-5b")

    assert Path(entry.default_model_ref).name == "Wan-AI--Wan2.2-TI2V-5B"
    assert entry.default_load_kwargs["offload_mode"] == "resident"
    assert entry.default_load_kwargs == {
        "offload_mode": "resident",
        "fuse_qkv": False,
        "qkv_strategy": "auto",
        "qkv_split_threshold": 8192,
        "inplace_residual": False,
        "static_cross_kv": True,
        "fused_rope": False,
        "rope_precision": "fp32",
        "rms_norm_precision": "input",
    }
    for option in (
        "fuse_qkv",
        "qkv_strategy",
        "qkv_split_threshold",
        "inplace_residual",
        "static_cross_kv",
        "fused_rope",
        "rope_precision",
        "rms_norm_precision",
    ):
        assert option in entry.load_params
    assert entry.default_call_kwargs["num_frames"] == 121
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert "wan2.2" in entry.aliases


def test_wan21_i2v_720p_catalog_uses_local_unified_checkpoint() -> None:
    entry = find_entry("wan2.1-i2v-14b-720p")

    assert Path(entry.default_model_ref).name == "Wan-AI--Wan2.1-I2V-14B-720P"
    assert entry.default_load_kwargs["offload_mode"] == "block"
    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/studio_demo/00/image.jpg")
    assert entry.default_call_kwargs["height"] == 720
    assert entry.default_call_kwargs["width"] == 1280
    assert entry.default_call_kwargs["shift"] == 5.0


def test_oasis_catalog_uses_local_checkpoint_and_auditable_actions() -> None:
    entry = find_entry("oasis-500m")

    assert Path(entry.default_model_ref).name == "oasis500m.safetensors"
    assert Path(entry.default_input_path).name == "thumb.png"
    assert Path(entry.default_load_kwargs["vae_ckpt"]).name == "vit-l-20.safetensors"
    actions_path = Path(entry.default_load_kwargs["actions_path"])
    actions = load_actions(str(actions_path))
    assert actions.shape == (1, 5, 25)
    assert actions[0, 0].count_nonzero() == 0
    assert actions[0, 1:, 11].eq(1).all()


def test_oasis_command_uses_call_time_smoke_overrides(tmp_path: Path) -> None:
    plan_path = tmp_path / "oasis.runtime_plan.json"
    plan_path.write_text(
        json.dumps({"fps": 16, "extra": {"num_frames": 5, "ddim_steps": 1}}),
        encoding="utf-8",
    )
    command = build_oasis_command(
        {
            "python": "/unified/python",
            "entrypoint": "/runtime/generate.py",
            "output_path": "/tmp/oasis.mp4",
            "plan_path": str(plan_path),
            "options": {
                "checkpoint_path": "/ckpt/oasis.safetensors",
                "vae_ckpt": "/ckpt/vae.safetensors",
                "prompt_path": "artifacts/data/prompt.png",
                "actions_path": "/data/actions.json",
            },
        }
    )

    assert command[command.index("--num-frames") + 1] == "5"
    assert command[command.index("--fps") + 1] == "16"
    assert command[command.index("--ddim-steps") + 1] == "1"
    assert Path(command[command.index("--prompt-path") + 1]).is_absolute()


def test_oasis_pyav_writer_is_compatible_with_unified_environment(tmp_path: Path) -> None:
    output = tmp_path / "oasis.mp4"
    frames = torch.zeros((2, 8, 8, 3), dtype=torch.uint8)
    frames[1, :, :, 1] = 255

    write_video_compat(output, frames, fps=16)

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
        assert stream.width == 8
        assert stream.height == 8
        assert float(stream.average_rate) == 16.0
        assert len(decoded) == 2


def test_oasis_prompt_loader_normalizes_grayscale_to_rgb(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.png"
    Image.new("L", (64, 36), color=127).save(prompt_path)

    prompt = load_prompt(str(prompt_path))

    assert prompt.shape == (1, 1, 3, 360, 640)
    assert prompt[:, :, 0].equal(prompt[:, :, 1])
    assert prompt[:, :, 1].equal(prompt[:, :, 2])
