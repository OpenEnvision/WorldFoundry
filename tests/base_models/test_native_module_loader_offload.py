from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
    parse_offload_policy,
)


@pytest.mark.parametrize("value", ["none", "resident", "fast"])
def test_resident_offload_presets_resolve_to_no_offload(value: str) -> None:
    assert parse_offload_policy(value).mode is OffloadMode.NONE


@pytest.mark.parametrize("value", ["block", "async-block", "async_block"])
def test_async_block_presets_resolve_to_block_offload(value: str) -> None:
    policy = parse_offload_policy(value)

    assert policy.mode is OffloadMode.BLOCK
    assert policy.pin_memory is True


def test_block_offload_restores_on_cpu_before_installing_cuda_hooks(monkeypatch) -> None:
    import worldfoundry.core.model_loading as model_loading
    import worldfoundry.core.vram as vram
    from worldfoundry.base_models.diffusion_model.loaders import module as loader_module

    events: list[tuple[str, object]] = []

    class TrackingModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        def to(self, *args, **kwargs):
            events.append(("move_skeleton", kwargs))
            return self

    module = TrackingModule()
    load_kwargs_seen: dict[str, object] = {}

    def fake_load_model(model_class, source, **kwargs):
        del model_class, source
        load_kwargs_seen.update(kwargs)
        events.append(("load", kwargs["device"]))
        return module

    handle = SimpleNamespace(enabled=True, reason="")

    def fake_enable_layerwise_cpu_offload(model, **kwargs):
        assert model is module
        events.append(("enable", kwargs["device"]))
        return handle

    monkeypatch.setattr(
        loader_module.NativeCheckpointResolver,
        "materialize",
        lambda self, checkpoint: SimpleNamespace(paths=("checkpoint.safetensors",)),
    )
    monkeypatch.setattr(model_loading, "load_model", fake_load_model)
    monkeypatch.setattr(vram, "enable_layerwise_cpu_offload", fake_enable_layerwise_cpu_offload)

    result = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=TrackingModule,
            layer_container="layers",
            # A declared wrapper map must not divert BLOCK back to the legacy
            # attribute-triggered copy path when a layer container is known.
            vram_module_map={torch.nn.Linear: torch.nn.Linear},
        ),
        object(),
        RuntimePolicy(
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
            offload=OffloadPolicy(mode=OffloadMode.BLOCK, target="cpu", pin_memory=True),
        ),
    )

    assert result is module
    assert events[0] == ("load", torch.device("cpu"))
    assert load_kwargs_seen["module_map"] is None
    assert load_kwargs_seen["vram_config"] is None
    assert events[1] == ("enable", torch.device("cuda:0"))
    assert events[2] == (
        "move_skeleton",
        {"dtype": torch.bfloat16, "device": torch.device("cuda:0")},
    )
    assert result._worldfoundry_layerwise_cpu_offload_handle is handle
    snapshot = result._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["offload"] == "block"
    assert snapshot.effective["offload"] == (
        "async-double-buffer-installed (runtime-pending)"
    )
    assert not any(str(value).startswith("offload ") for value in snapshot.fallbacks)


def test_block_offload_rejects_quantization_before_wrapper_install(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.loaders import module as loader_module
    from worldfoundry.base_models.diffusion_model.optimizations import (
        QuantizationMode,
        QuantizationPolicy,
    )

    monkeypatch.setattr(
        loader_module.NativeCheckpointResolver,
        "materialize",
        lambda self, checkpoint: SimpleNamespace(paths=("checkpoint.safetensors",)),
    )

    with pytest.raises(NotImplementedError, match="quantization with block offload"):
        NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=torch.nn.Linear,
                config={"in_features": 2, "out_features": 2},
                layer_container="layers",
            ),
            object(),
            RuntimePolicy(
                device=torch.device("cuda:0"),
                dtype=torch.bfloat16,
                offload=OffloadPolicy(
                    mode=OffloadMode.BLOCK,
                    target="cpu",
                    pin_memory=True,
                ),
                quantization=QuantizationPolicy(mode=QuantizationMode.FP8),
            ),
        )
