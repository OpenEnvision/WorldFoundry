from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry


def test_forcing_catalog_entries_resolve_flat_local_checkpoints() -> None:
    expected_names = {
        "self-forcing": "self_forcing_dmd.pt",
        "causal-forcing": "causal_forcing.pt",
    }

    for model_id, expected_name in expected_names.items():
        entry = find_entry(model_id)
        checkpoint = Path(entry.default_model_ref)
        components = entry.default_load_kwargs["required_components"]
        wan_model = Path(components["wan_models_root"])

        assert checkpoint.is_file()
        assert checkpoint.name == expected_name
        assert components["checkpoint_path"] == str(checkpoint)
        assert wan_model.name == "Wan-AI--Wan2.1-T2V-1.3B"
        assert (wan_model / "diffusion_pytorch_model.safetensors").is_file()
        assert (wan_model / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
        assert (wan_model / "Wan2.1_VAE.pth").is_file()


def test_forcing_catalog_entries_expose_bounded_frame_overrides() -> None:
    for model_id in ("self-forcing", "causal-forcing"):
        entry = find_entry(model_id)

        assert entry.default_call_kwargs["num_output_frames"] == 21
        assert "num_output_frames" in entry.call_params
        assert "seed" in entry.call_params
