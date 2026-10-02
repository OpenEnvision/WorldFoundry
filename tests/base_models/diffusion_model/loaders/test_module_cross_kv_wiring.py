"""End-to-end guard that options["static_cross_kv"] installs the K/V cache.

Loads a real checkpoint wrapping Wan CrossAttention through the full loader and
checks the processor is wrapped, the cache handle is attached for runner reset,
and the choice is recorded in the audit snapshot. CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import StaticCrossKVProcessor
from worldfoundry.core.model_loading.policy import RuntimePolicy

safetensors = pytest.importorskip("safetensors.torch")


class _Wrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ca = CrossAttention(256, 8)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.ca(x, ctx)


def _checkpoint(tmp_path) -> CheckpointSpec:
    safetensors.save_file(_Wrap().state_dict(), str(tmp_path / "m.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("m.safetensors",))


def test_static_cross_kv_option_installs_cache(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy(options={"static_cross_kv": True}))
    assert isinstance(loaded.ca.get_processor(), StaticCrossKVProcessor)
    assert hasattr(loaded, "_worldfoundry_static_cross_kv")
    assert loaded._worldfoundry_static_cross_kv.wrapped_blocks == 1


def test_static_cross_kv_recorded_in_audit(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy(options={"static_cross_kv": True}))
    snap = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snap.requested["static_cross_kv"] is True
    assert snap.effective["static_cross_kv_blocks"] == 1


def test_off_by_default(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy())
    assert not isinstance(loaded.ca.get_processor(), StaticCrossKVProcessor)
    assert not hasattr(loaded, "_worldfoundry_static_cross_kv")
