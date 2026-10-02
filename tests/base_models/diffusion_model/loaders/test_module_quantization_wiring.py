"""End-to-end guard that RuntimePolicy.quantization drives FP8 replacement.

Previously the loader raised NotImplementedError for any non-NONE quantization.
Now it runs ``apply_quantization_policy`` after weight materialization and
before offload hooks. These tests load a real (tiny) checkpoint through the full
``NativeModuleLoader.load`` path on CPU — FP8's dense fallback keeps CPU
correct, while the module-replacement wiring is exactly what runs on GPU.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.core.model_loading.policy import (
    OffloadMode,
    OffloadPolicy,
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)

safetensors = pytest.importorskip("safetensors.torch")


class _TinyDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.big1 = torch.nn.Linear(1024, 1024)
        self.big2 = torch.nn.Linear(1024, 1024)
        self.small = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.big2(self.big1(x))


def _checkpoint(tmp_path) -> CheckpointSpec:
    model = _TinyDiT()
    safetensors.save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("model.safetensors",))


def _types(module: torch.nn.Module) -> dict[str, str]:
    return {n: type(m).__name__ for n, m in module.named_modules() if n in ("big1", "big2", "small")}


def test_fp8_policy_replaces_eligible_linears(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_TinyDiT)
    policy = RuntimePolicy(
        quantization=QuantizationPolicy(mode=QuantizationMode.FP8, options={"min_features": 512})
    )
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    types = _types(loaded)
    assert types["big1"] == "Float8Linear"
    assert types["big2"] == "Float8Linear"
    assert types["small"] == "Linear"  # 16 < min_features
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.effective["quantization"] == "fp8-wrapper-installed (runtime-pending)"
    assert snapshot.quality_tier == "numerically-approximate"


def test_none_policy_leaves_dense(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_TinyDiT)
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), RuntimePolicy())
    assert set(_types(loaded).values()) == {"Linear"}


def test_nvfp4_policy_replaces_aligned_linears_with_dense_fallback(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_TinyDiT)
    policy = RuntimePolicy(
        quantization=QuantizationPolicy(
            mode=QuantizationMode.NVFP4,
            options={"min_features": 512, "keep_dense_fallback": True},
        )
    )
    loaded = NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
    types = _types(loaded)
    assert types["big1"] == "NVFP4Linear"
    assert types["big2"] == "NVFP4Linear"
    assert types["small"] == "Linear"
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["quantization"] == "nvfp4"
    assert snapshot.effective["quantization"] == "nvfp4-wrapper-installed (runtime-pending)"
    assert snapshot.effective["quantization_replaced"] == 2
    assert snapshot.effective["quantization_storage"] == "nvfp4-1x16"
    assert snapshot.quality_tier == "numerically-approximate"


@pytest.mark.parametrize(
    ("mode", "expected_type", "expected_storage"),
    [
        (QuantizationMode.INT8, "WeightOnlyLinear", "groupwise-int8"),
        (QuantizationMode.INT4, "WeightOnlyLinear", "groupwise-int4"),
    ],
)
def test_integer_weight_only_policy_is_consumed_by_loader(
    tmp_path,
    mode,
    expected_type,
    expected_storage,
) -> None:
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_TinyDiT),
        _checkpoint(tmp_path),
        RuntimePolicy(
            quantization=QuantizationPolicy(
                mode=mode,
                options={"min_features": 512, "group_size": 64},
            )
        ),
    )
    assert _types(loaded)["big1"] == expected_type
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["quantization"] == mode.value
    assert snapshot.effective["quantization_storage"] == expected_storage
    assert "runtime-pending" in snapshot.effective["quantization"]


def test_disk_offload_plus_quant_rejected(tmp_path) -> None:
    spec = ModuleLoadSpec(module_class=_TinyDiT)
    policy = RuntimePolicy(
        offload=OffloadPolicy(mode=OffloadMode.DISK, target="disk"),
        quantization=QuantizationPolicy(mode=QuantizationMode.FP8),
    )
    with pytest.raises(NotImplementedError, match="disk offload"):
        NativeModuleLoader().load(spec, _checkpoint(tmp_path), policy)
