from pathlib import Path

from worldfoundry.runtime.inference_catalog import get_model_inference_spec


def test_rolling_forcing_workspace_defaults_use_downloaded_flat_hfd_assets() -> None:
    spec = get_model_inference_spec("rolling-forcing")
    variant = spec.variant()
    checkpoints = variant.checkpoint_map()

    assert Path(checkpoints["primary"]).is_file()
    assert Path(checkpoints["primary"]).name == "rolling_forcing_dmd.pt"
    assert Path(checkpoints["base"]).is_dir()
    assert Path(checkpoints["base"]).name == "Wan-AI--Wan2.1-T2V-1.3B"
    assert variant.load_kwargs["checkpoint_path"] == checkpoints["primary"]
    assert Path(variant.load_kwargs["wan_models_root"]) == Path(checkpoints["base"]).parent
