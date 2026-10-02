from __future__ import annotations

import json

import torch
from safetensors.torch import save_file
from torch import nn

from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config


class _TinyNativeModel(nn.Module, NativeConfigMixin):
    @register_to_config
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)


def test_native_from_config_filters_diffusers_metadata() -> None:
    loaded = _TinyNativeModel.from_config(
        {"_class_name": "IgnoredDiffusersMetadata", "width": 3}
    )

    assert loaded.config.width == 3
    assert loaded.proj.in_features == 3


def test_native_config_mixin_loads_diffusers_checkpoint_directory(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"_class_name": "IgnoredDiffusersMetadata", "width": 3}),
        encoding="utf-8",
    )
    expected = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    save_file({"proj.weight": expected}, tmp_path / "diffusion_pytorch_model.safetensors")

    loaded = _TinyNativeModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)

    assert loaded.config.width == 3
    assert loaded.proj.weight.dtype == torch.bfloat16
    torch.testing.assert_close(loaded.proj.weight.float(), expected)


def test_native_config_mixin_honors_diffusers_subfolder(tmp_path) -> None:
    subfolder = tmp_path / "low_noise_model"
    subfolder.mkdir()
    (subfolder / "config.json").write_text(json.dumps({"width": 3}), encoding="utf-8")
    expected = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    save_file({"proj.weight": expected}, subfolder / "diffusion_pytorch_model.safetensors")

    loaded = _TinyNativeModel.from_pretrained(tmp_path, subfolder="low_noise_model")

    torch.testing.assert_close(loaded.proj.weight, expected)
