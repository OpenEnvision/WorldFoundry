from __future__ import annotations

from pathlib import Path

from worldfoundry.runtime.conda import load_runtime_conda_env_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_ltx25_default_variant_maps_the_complete_official_runtime() -> None:
    entry = find_entry("ltx-2.5")

    assert entry.module_path == "worldfoundry.pipelines.ltx25.pipeline_ltx25"
    assert entry.class_name == "LTX25Pipeline"
    assert entry.default_task_type == "t2v"
    assert Path(entry.default_model_ref).name == "Lightricks--LTX-2.5"
    assert entry.default_call_kwargs == {
        "num_frames": 121,
        "fps": 24,
        "height": 512,
        "width": 768,
        "seed": 42,
        "execute": True,
        "timeout_seconds": 7200,
        "return_dict": True,
    }

    required = entry.default_load_kwargs["required_components"]
    assert Path(required["source_root"]).name == "Lightricks--LTX-2"
    assert Path(required["python_executable"]).parts[-4:] == (
        "envs",
        "ltx-2.5",
        "bin",
        "python",
    )
    assert Path(required["transformer_path"]).name == (
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    assert Path(required["text_encoder_path"]).name == (
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    )
    assert Path(required["video_vae_path"]).name == "ltx-2.5-video-vae-bf16.safetensors"
    assert Path(required["audio_vae_path"]).name == "ltx-2.5-audio-vae-bf16.safetensors"
    assert Path(required["spatial_upsampler_path"]).name == (
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )


def test_ltx25_workspace_call_contract_requires_real_execution() -> None:
    entry = find_entry("ltx-2.5")

    assert entry.default_call_kwargs["execute"] is True
    assert {"execute", "num_frames", "height", "width", "fps", "seed"} <= set(
        entry.call_params
    )


def test_ltx25_runtime_pins_cuda_12_8_torch_wheels() -> None:
    spec = load_runtime_conda_env_spec("ltx-2.5")

    assert spec is not None
    assert spec.cuda_profile == "cu128"
    assert "torch==2.11.0+cu128" in spec.pip_packages
    assert "torchaudio==2.11.0+cu128" in spec.pip_packages
    assert "torchvision==0.26.0+cu128" in spec.pip_packages
