from __future__ import annotations

from pathlib import Path

from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_stable_video_infinity_resolves_complete_local_checkpoint_pair() -> None:
    entry = find_entry("stable-video-infinity")

    lora_path = Path(entry.default_model_ref)
    load_kwargs = entry.default_load_kwargs
    wan_model_dir = Path(load_kwargs["wan_model_dir"])

    assert lora_path.is_file()
    assert lora_path.name == "SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors"
    assert load_kwargs["svi_lora_path"] == str(lora_path)
    assert wan_model_dir.is_dir()
    assert (wan_model_dir / "Wan2.1_VAE.pth").is_file()
    assert (wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
    assert len(tuple(wan_model_dir.glob("diffusion_pytorch_model-*.safetensors"))) == 7


def test_stable_video_infinity_exposes_minimal_smoke_overrides() -> None:
    entry = find_entry("stable-video-infinity")
    spec = get_model_inference_spec("stable-video-infinity")

    assert entry.default_call_kwargs["num_frames"] == 81
    assert entry.default_call_kwargs["num_inference_steps"] == 50
    assert entry.default_call_kwargs["num_motion_frames"] == 5
    assert {
        "num_clips",
        "num_frames",
        "num_motion_frames",
        "num_inference_steps",
        "height",
        "width",
        "cfg_scale_text",
        "fps",
    }.issubset(entry.call_params)
    assert spec is not None
    assert {"height", "width"}.issubset(field.field_id for field in spec.tasks[0].inputs)
