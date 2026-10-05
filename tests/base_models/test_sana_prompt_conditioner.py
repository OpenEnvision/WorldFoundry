
import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")
from types import SimpleNamespace

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.sana import (
    _module_map,
    _prepare_sana_execution_tensors,
)
from worldfoundry.base_models.diffusion_model.models.encoders.sana.component import (
    SanaPromptConditioner,
    _sana_vram_module_map,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.normalization import RMSNorm
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana import Sana
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import (
    SizeEmbedder,
    TimestepEmbedder,
)
from worldfoundry.core.vram import AutoWrappedModule


class _Tokenizer:
    padding_side = "left"

    def encode(self, _text: str) -> list[int]:
        return [1, 2]

    def __call__(self, texts, **_kwargs):
        batch_size = len(texts)
        return SimpleNamespace(
            input_ids=torch.ones((batch_size, 4), dtype=torch.long),
            attention_mask=torch.ones((batch_size, 4), dtype=torch.long),
        )


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offloaded_weight = torch.nn.Parameter(torch.ones(1))
        self.input_device: torch.device | None = None

    def forward(self, *, input_ids, attention_mask, **_kwargs):
        self.input_device = input_ids.device
        assert attention_mask.device == input_ids.device
        hidden = torch.empty((*input_ids.shape, 2), device=input_ids.device)
        return SimpleNamespace(last_hidden_state=hidden)


def test_sana_prompt_branch_uses_requested_runtime_device() -> None:
    model = _RecordingModel()
    encoder = SimpleNamespace(model=model)
    conditioner = SanaPromptConditioner(encoder, _Tokenizer(), max_length=4, enhance_prompt=False)

    result = conditioner._branch(["a lighthouse"], enhance=False, device=torch.device("meta"))

    assert model.offloaded_weight.device.type == "cpu"
    assert model.input_device == torch.device("meta")
    assert result["context"].device.type == "meta"


def test_sana_vram_map_wraps_transformers_gemma_rms_norm() -> None:
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

    assert _sana_vram_module_map()[Gemma2RMSNorm] is AutoWrappedModule


def test_sana_denoiser_vram_map_wraps_native_rms_norm() -> None:
    assert _module_map()[RMSNorm] is AutoWrappedModule


def test_sana_graph_prefers_framework_execution_dtype_over_storage_parameter() -> None:
    model = object.__new__(Sana)
    torch.nn.Module.__init__(model)
    model.register_parameter("storage_weight", torch.nn.Parameter(torch.ones(1, dtype=torch.float32)))

    assert model.dtype is torch.float32
    model._worldfoundry_execution_dtype = torch.bfloat16
    assert model.dtype is torch.bfloat16


def test_sana_nested_embedders_prefer_framework_execution_dtype() -> None:
    for module in (TimestepEmbedder(hidden_size=4), SizeEmbedder(hidden_size=4)):
        assert module.dtype is torch.float32
        module._worldfoundry_execution_dtype = torch.bfloat16
        assert module.dtype is torch.bfloat16


def test_sana_execution_setup_moves_only_unwrapped_direct_tensors() -> None:
    root = torch.nn.Module()
    root.register_parameter("direct", torch.nn.Parameter(torch.ones(1, dtype=torch.float32)))
    root.child = torch.nn.Module()
    root.child.register_parameter("direct", torch.nn.Parameter(torch.ones(1, dtype=torch.float32)))
    root.wrapped = AutoWrappedModule(
        torch.nn.LayerNorm(2, dtype=torch.float32),
        offload_dtype=torch.float32,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device="cpu",
        preparing_dtype=torch.bfloat16,
        preparing_device="cpu",
        computation_dtype=torch.bfloat16,
        computation_device="cpu",
    )

    _prepare_sana_execution_tensors(root, device=torch.device("cpu"), dtype=torch.bfloat16)

    assert root.direct.dtype is torch.bfloat16
    assert root.child.direct.dtype is torch.bfloat16
    assert root.wrapped.module.weight.dtype is torch.float32
