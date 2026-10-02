from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import _build_wan_denoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy

_TINY_WAN_CONFIG = {
    "dim": 12,
    "in_dim": 2,
    "ffn_dim": 24,
    "out_dim": 2,
    "text_dim": 8,
    "freq_dim": 4,
    "eps": 1e-6,
    "patch_size": (1, 1, 1),
    "num_heads": 1,
    "num_layers": 0,
    "has_image_input": False,
}


@pytest.mark.parametrize(
    ("runtime_options", "component_options", "expected"),
    (
        ({}, {}, torch.float32),
        ({"dit_weight_dtype": "bf16"}, {}, torch.bfloat16),
        (
            {"dit_weight_dtype": "bf16"},
            {"weight_dtype": torch.float16},
            torch.float16,
        ),
    ),
)
def test_wan_resident_weight_dtype_is_optional_and_component_override_wins(
    monkeypatch,
    runtime_options,
    component_options,
    expected,
) -> None:
    loaded_dtypes: list[torch.dtype] = []

    def fake_load(self, spec, checkpoint, policy):
        del self, spec, checkpoint
        loaded_dtypes.append(policy.dtype)
        return WanModel(**_TINY_WAN_CONFIG)

    monkeypatch.setattr(NativeModuleLoader, "load", fake_load)
    context = ComponentBuildContext(
        model_id="wan-test",
        key="denoiser:main",
        policy=RuntimePolicy(dtype=torch.bfloat16, options=runtime_options),
        checkpoints={"weights": CheckpointSpec(source="unused")},
        component_options=component_options,
    )

    denoiser = _build_wan_denoiser(context, config=_TINY_WAN_CONFIG)

    assert loaded_dtypes == [expected]
    assert denoiser.model._worldfoundry_dit_weight_dtype is expected
    assert denoiser.compute_dtype is torch.bfloat16


@pytest.mark.parametrize("options,compile_enabled,error", [
    ({"fused_residual_adaln": "yes"}, False, TypeError),
    ({"fused_residual_adaln": True, "inplace_residual": True}, False, ValueError),
    ({"fused_residual_adaln": True, "cuda_graph": True}, False, ValueError),
    ({"fused_residual_adaln": True}, True, ValueError),
])
def test_unimplemented_residual_adaln_fails_before_loading(monkeypatch, options, compile_enabled, error):
    def no_load(*args, **kwargs):
        pytest.fail("invalid fusion policy must fail before checkpoint loading")

    monkeypatch.setattr(NativeModuleLoader, "load", no_load)
    with pytest.raises(error, match="fused_residual_adaln"):
        context = ComponentBuildContext(
            model_id="wan-test", key="denoiser:main",
            policy=RuntimePolicy(compile=compile_enabled, options=options),
            checkpoints={"weights": CheckpointSpec(source="unused")},
        )
        _build_wan_denoiser(context, config=_TINY_WAN_CONFIG)


def test_disabled_residual_adaln_does_not_claim_installation(monkeypatch):
    config = dict(_TINY_WAN_CONFIG, num_layers=1)
    monkeypatch.setattr(NativeModuleLoader, "load", lambda *a, **kw: WanModel(**config))
    context = ComponentBuildContext(
        model_id="wan-test", key="denoiser:main",
        policy=RuntimePolicy(options={"fused_residual_adaln": False}),
        checkpoints={"weights": CheckpointSpec(source="unused")},
    )
    denoiser = _build_wan_denoiser(context, config=config)
    assert not hasattr(denoiser.model, "_worldfoundry_residual_adaln_runtime")
    report = denoiser.runtime_optimization_report()
    assert "fused_residual_adaln" not in report["runtime"]
