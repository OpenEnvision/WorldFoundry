from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import find_entry


@pytest.mark.parametrize(
    ("model_id", "dit_file", "codec_files"),
    (
        (
            "sana-streaming-2b-720p",
            "dit/sana_streaming_ar.pth",
            ("ltx-2-19b-dev.safetensors",),
        ),
        (
            "sana-streaming-bidirectional-2b-720p",
            "dit/sana_bidirectional_short.pth",
            ("vae/config.json", "vae/diffusion_pytorch_model.safetensors"),
        ),
    ),
)
def test_sana_streaming_workspace_defaults_bind_every_native_checkpoint(
    model_id: str,
    dit_file: str,
    codec_files: tuple[str, ...],
) -> None:
    entry = find_entry(model_id)
    overrides = entry.default_load_kwargs["checkpoint_overrides"]

    assert Path(entry.default_model_ref).is_dir()
    assert (Path(overrides["dit"]) / dit_file).is_file()
    assert (Path(overrides["text-encoder"]) / "gemma-2-2b-it.safetensors").is_file()
    assert (Path(overrides["tokenizer"]) / "tokenizer.model").is_file()
    assert all((Path(overrides["codec"]) / filename).is_file() for filename in codec_files)
    assert Path(entry.default_input_path).is_file()
    assert entry.default_load_kwargs["model_id"] == model_id
