from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime.utils.camera_io import (
    read_w2cs_from_txt,
)


def test_spatia_workspace_defaults_bind_complete_local_demo() -> None:
    entry = find_entry("spatia")
    checkpoint = Path(entry.default_model_ref)
    load_kwargs = entry.default_load_kwargs
    call_kwargs = entry.default_call_kwargs

    assert (checkpoint / "control_weight_8500.safetensors").is_file()
    assert (checkpoint / "lora_weights_10000.safetensors").is_file()
    assert (Path(load_kwargs["base_model_path"]) / "Wan2.2_VAE.pth").is_file()
    assert (Path(load_kwargs["map_model_path"]) / "model.safetensors").is_file()
    assert Path(entry.default_input_path).is_file()

    trajectory = read_w2cs_from_txt(call_kwargs["w2c_trajectory_file"])
    assert trajectory.shape == (121, 3, 4)
    assert call_kwargs["num_frames"] == trajectory.shape[0]
    assert len(call_kwargs["intrinsics"]) == 1
