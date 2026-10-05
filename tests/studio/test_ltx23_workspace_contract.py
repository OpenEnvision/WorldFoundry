from __future__ import annotations

from pathlib import Path

from worldfoundry.cli.model_run import load_model_run_schema
from worldfoundry.studio.inference.catalog import find_entry


def test_ltx23_exact_i2v_id_keeps_the_full_workspace_contract() -> None:
    entry = find_entry("ltx-2.3-i2v")

    assert entry.default_input_path.endswith("studio_demo/00/image.jpg")
    assert entry.default_call_kwargs == {
        "num_frames": 121,
        "fps": 24,
        "height": 512,
        "width": 768,
        "num_inference_steps": 11,
        "guidance_scale": 1.0,
        "seed": 0,
    }
    assert {
        "images",
        "num_frames",
        "num_inference_steps",
        "height",
        "width",
        "guidance_scale",
        "seed",
    } <= set(entry.call_params)
    required = entry.default_load_kwargs["required_components"]
    assert Path(required["spatial_upsampler_path"]).name == (
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    )
    assert Path(required["gemma_root"]).name in {
        "gemma-3-12b-it-qat-q4_0-unquantized",
        "google--gemma-3-12b-it-qat-q4_0-unquantized",
    }


def test_ltx23_t2v_declares_the_same_required_runtime_components() -> None:
    i2v = find_entry("ltx-2.3-i2v")
    t2v = find_entry("ltx-2.3-t2v")

    assert t2v.default_load_kwargs == i2v.default_load_kwargs


def test_ltx23_i2v_cli_exposes_typed_generation_overrides() -> None:
    load_model_run_schema.cache_clear()
    schema = load_model_run_schema("ltx-2.3-i2v")
    fields = {field.option: field for field in schema.fields}

    assert fields["--pipeline.frames"].default == 121
    assert fields["--pipeline.steps"].default == 11
    assert fields["--pipeline.height"].default == 512
    assert fields["--pipeline.width"].default == 768
    assert fields["--pipeline.guidance-scale"].default == 1.0
    assert fields["--pipeline.seed"].default == 0
