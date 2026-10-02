"""Request-scoped attention policy wiring through the native loader."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.core.model_loading.policy import AttentionBackend, RuntimePolicy

safetensors = pytest.importorskip("safetensors.torch")


class _AttentionWrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = SelfAttention(128, 4)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.attention(x, freqs)


def _checkpoint(tmp_path) -> CheckpointSpec:
    safetensors.save_file(_AttentionWrap().state_dict(), str(tmp_path / "attention.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("attention.safetensors",))


def test_explicit_backend_is_pipeline_scoped_and_audited(tmp_path) -> None:
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_AttentionWrap),
        _checkpoint(tmp_path),
        RuntimePolicy(attention=AttentionBackend.TORCH),
    )

    assert loaded.attention.attn.attention_backend == "torch"
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["attention"] == "torch"
    assert snapshot.effective["attention"] == "torch"
    assert snapshot.effective["attention_modules"] == 1
    assert snapshot.quality_tier == "exact"


def test_unavailable_external_backend_falls_back_honestly_on_cpu(tmp_path) -> None:
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_AttentionWrap),
        _checkpoint(tmp_path),
        RuntimePolicy(device="cpu", attention=AttentionBackend.FLASH_ATTENTION_3),
    )

    assert loaded.attention.attn.attention_backend == "torch"
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["attention"] == "flash_attention_3"
    assert snapshot.effective["attention"] == "torch"
    assert any("flash_attention_3" in fallback for fallback in snapshot.fallbacks)


def test_unavailable_flash4_falls_back_honestly_on_cpu(tmp_path) -> None:
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_AttentionWrap),
        _checkpoint(tmp_path),
        RuntimePolicy(device="cpu", attention=AttentionBackend.FLASH_ATTENTION_4),
    )

    assert loaded.attention.attn.attention_backend == "torch"
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["attention"] == "flash_attention_4"
    assert snapshot.effective["attention"] == "torch"
    assert any("flash_attention_4" in fallback for fallback in snapshot.fallbacks)
