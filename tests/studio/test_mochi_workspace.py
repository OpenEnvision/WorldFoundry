from __future__ import annotations

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime


def test_mochi_workspace_uses_official_full_quality_defaults() -> None:
    entry = find_entry("mochi-1-preview-t2v")

    assert "chameleon's eye" in entry.default_prompt
    assert entry.default_call_kwargs == {
        "height": 480,
        "width": 848,
        "num_frames": 84,
        "num_inference_steps": 64,
        "guidance_scale": 4.5,
        "fps": 30,
        "seed": 12345,
    }


def test_mochi_runtime_uses_official_high_quality_precision_recipe() -> None:
    runtime = OfficialVideoRuntime.from_model_id("mochi-1-preview-t2v")

    assert runtime.runtime["torch_dtype"] == "float32"
    assert runtime.runtime["autocast_dtype"] == "bfloat16"
    assert runtime.runtime["autocast_cache_enabled"] is False
    assert runtime.runtime["call_kwargs"] == {
        "height": 480,
        "width": 848,
        "num_frames": 84,
        "num_inference_steps": 64,
        "guidance_scale": 4.5,
        "seed": 12345,
    }
