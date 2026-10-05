"""End-to-end guard that options["fuse_qkv"] drives merged-QKV in the loader.

Loads a real (tiny) checkpoint wrapping Wan's SelfAttention through the full
``NativeModuleLoader.load`` path and checks that fusion happens in-load, before
quantization, and composes with FP8 (the fused ``qkv`` becomes a Float8Linear).
CPU-only: FP8 uses its dense fallback while the replacement wiring is exercised.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy, RuntimePolicy

safetensors = pytest.importorskip("safetensors.torch")


class _Wrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = SelfAttention(1024, 8)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.attn(x, freqs)


def _checkpoint(tmp_path) -> CheckpointSpec:
    safetensors.save_file(_Wrap().state_dict(), str(tmp_path / "model.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("model.safetensors",))


def test_fuse_qkv_option_merges_in_load(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    policy = RuntimePolicy(options={"fuse_qkv": True})
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    assert hasattr(loaded.attn, "qkv")
    assert not hasattr(loaded.attn, "q")


def test_no_fuse_leaves_separate_projections(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy())
    assert hasattr(loaded.attn, "q")
    assert not hasattr(loaded.attn, "qkv")


def test_fuse_then_fp8_composes(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_Wrap)
    policy = RuntimePolicy(
        quantization=QuantizationPolicy(mode=QuantizationMode.FP8, options={"min_features": 512}),
        options={"fuse_qkv": True},
    )
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    # Fused qkv (1024x3072) and o (1024x1024) both clear min_features -> FP8.
    assert type(loaded.attn.qkv).__name__ == "Float8Linear"
    assert type(loaded.attn.o).__name__ == "Float8Linear"


def test_fused_qkv_then_sequence_parallel_keeps_both_paths(
    tmp_path,
    monkeypatch,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations import (
        sequence_parallel as sequence_parallel_module,
    )
    from worldfoundry.base_models.diffusion_model.optimizations.sequence_parallel import (
        SequenceParallelSelfAttentionProcessor,
    )

    monkeypatch.setattr(
        sequence_parallel_module,
        "require_sequence_parallel_runtime",
        lambda degree: None,
    )
    policy = RuntimePolicy(
        options={"fuse_qkv": True, "sequence_parallel": 2},
    )
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_Wrap),
        _checkpoint(tmp_path),
        policy,
    )
    assert hasattr(loaded.attn, "qkv")
    assert isinstance(
        loaded.attn.get_processor(),
        SequenceParallelSelfAttentionProcessor,
    )
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["sequence_parallel"] == 2
    assert snapshot.effective["sequence_parallel_backend"] == "native-ulysses"


def test_sequence_parallel_rejects_stateful_or_shape_changing_options(tmp_path) -> None:
    policy = RuntimePolicy(
        options={"sequence_parallel": 2, "teacache": True},
    )
    with pytest.raises(ValueError, match="teacache"):
        NativeModuleLoader().load(
            ModuleLoadSpec(module_class=_Wrap),
            _checkpoint(tmp_path),
            policy,
        )


def test_sequence_parallel_accepts_fused_rope_offset_aware_kernel(
    tmp_path,
    monkeypatch,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations import (
        sequence_parallel as sequence_parallel_module,
    )

    monkeypatch.setattr(
        sequence_parallel_module,
        "require_sequence_parallel_runtime",
        lambda degree: None,
    )
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_Wrap),
        _checkpoint(tmp_path),
        RuntimePolicy(options={"sequence_parallel": 2, "fused_rope": True}),
    )
    assert isinstance(
        loaded.attn.get_processor(),
        sequence_parallel_module.SequenceParallelSelfAttentionProcessor,
    )


@pytest.mark.parametrize(
    ("option", "value", "missing_mechanism"),
    [
        ("approximate_attention", "vsa", "metadata/routing"),
        ("cuda_graph", True, "rank-coordinated graph lifecycle"),
        ("taylorseer", True, "history must remain shard-consistent"),
    ],
)
def test_sequence_parallel_conflicts_explain_missing_correctness_mechanism(
    tmp_path,
    option: str,
    value: object,
    missing_mechanism: str,
) -> None:
    policy = RuntimePolicy(options={"sequence_parallel": 2, option: value})
    with pytest.raises(ValueError, match=missing_mechanism):
        NativeModuleLoader().load(
            ModuleLoadSpec(module_class=_Wrap),
            _checkpoint(tmp_path),
            policy,
        )
