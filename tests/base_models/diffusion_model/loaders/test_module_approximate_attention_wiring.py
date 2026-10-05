"""End-to-end guard that options["approximate_attention"] installs the lossy lane.

Loads a real checkpoint wrapping Wan SelfAttention through the full loader and
checks: the self-attention processor is swapped, the state handle is attached,
the audit snapshot marks quality_tier="approximate", and (on CPU, no kernel) the
kernel fallback is recorded honestly. CPU-only — the sparse kernel is never
required for this wiring test.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.base_models.diffusion_model.optimizations import (
    approximate_attention as approximate_module,
)
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateSelfAttentionProcessor,
)
from worldfoundry.core.model_loading.policy import RuntimePolicy

safetensors = pytest.importorskip("safetensors.torch")


class _Wrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sa = SelfAttention(256, 8)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.sa(x, freqs)


def _checkpoint(tmp_path) -> CheckpointSpec:
    safetensors.save_file(_Wrap().state_dict(), str(tmp_path / "m.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("m.safetensors",))


def test_option_installs_approximate_lane(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap, supports_approximate_attention=True)
    policy = RuntimePolicy(options={"approximate_attention": {"kind": "vsa", "sparsity": 0.9}})
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    assert isinstance(loaded.sa.get_processor(), ApproximateSelfAttentionProcessor)
    assert hasattr(loaded, "_worldfoundry_approximate_attention")
    assert loaded._worldfoundry_approximate_attention.wrapped_blocks == 1


def test_recorded_as_approximate_in_audit(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap, supports_approximate_attention=True)
    policy = RuntimePolicy(options={"approximate_attention": "sta"})
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    snap = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snap.quality_tier == "approximate"
    assert snap.requested["approximate_attention"] == "sta"
    assert snap.effective["approximate_attention_blocks"] == 1
    # CPU has no fastvideo_kernel -> honest fallback string + recorded fallback.
    assert snap.effective["approximate_attention_kernel"].startswith("exact")
    assert any("approximate_attention" in f for f in snap.fallbacks)


def test_off_by_default(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap, supports_approximate_attention=True)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy())
    assert not isinstance(loaded.sa.get_processor(), ApproximateSelfAttentionProcessor)
    assert not hasattr(loaded, "_worldfoundry_approximate_attention")
    snap = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snap.quality_tier == "exact"


def test_shared_policy_skips_component_without_approximate_attention_capability(
    tmp_path,
) -> None:
    policy = RuntimePolicy(options={"approximate_attention": "dynamic_sparse"})
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_Wrap),
        _checkpoint(tmp_path),
        policy,
    )

    assert not isinstance(
        loaded.sa.get_processor(),
        ApproximateSelfAttentionProcessor,
    )
    assert not hasattr(loaded, "_worldfoundry_approximate_attention")
    snapshot = (
        loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    )
    assert snapshot.requested["approximate_attention"] is False


def test_loader_runs_lightx2v_preparation_after_install(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def prepare(model, state, *, fallback_device):
        calls.append((model, state, fallback_device))

    monkeypatch.setattr(
        approximate_module,
        "prepare_lightx2v_providers",
        prepare,
    )
    policy = RuntimePolicy(
        options={"approximate_attention": "dynamic_sparse"}
    )

    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=_Wrap,
            supports_approximate_attention=True,
        ),
        _checkpoint(tmp_path),
        policy,
    )

    assert calls == [
        (
            loaded,
            loaded._worldfoundry_approximate_attention,
            policy.device,
        )
    ]
