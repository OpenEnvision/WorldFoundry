from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.policy import parse_device_map
from worldfoundry.core.vram.device_map import balanced_layer_devices, enable_balanced_device_map
from worldfoundry.pipelines.cosmos.cosmos3_runtime import resolve_cosmos3_runtime_options


def test_parse_device_map_aliases() -> None:
    assert parse_device_map(None) is None
    assert parse_device_map("none") is None
    assert parse_device_map("balanced") == "balanced"
    assert parse_device_map("auto") == "balanced"


def test_balanced_layer_devices_are_contiguous() -> None:
    devices = tuple(torch.device(f"cuda:{index}") for index in range(8))
    placement = balanced_layer_devices(64, devices)
    assert len(placement) == 64
    assert placement[0] == devices[0]
    assert placement[7] == devices[0]
    assert placement[8] == devices[1]
    assert placement[-1] == devices[7]
    assert [placement.count(device) for device in devices] == [8] * 8


def test_enable_balanced_device_map_keeps_layers_and_home_modules() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([Block() for _ in range(4)])
            self.norm = nn.LayerNorm(4)

    model = Toy()
    handle = enable_balanced_device_map(
        model,
        layer_container="layers",
        home_device="cpu",
        devices=(torch.device("cpu"), torch.device("cpu")),
    )
    assert handle.enabled is True
    assert handle.layer_count == 4
    assert model.embed.weight.device.type == "cpu"
    assert model.layers[0].linear.weight.device.type == "cpu"
    assert getattr(model, "_worldfoundry_device_map") == "balanced"


def test_super_defaults_to_resident_device_map_on_multi_gpu() -> None:
    offload_mode, device_map = resolve_cosmos3_runtime_options({}, model_id="cosmos3-super", visible_gpus=8)
    assert offload_mode == "none"
    assert device_map == "balanced"


def test_super_keeps_block_offload_on_one_gpu() -> None:
    offload_mode, device_map = resolve_cosmos3_runtime_options({}, model_id="cosmos3-super", visible_gpus=1)
    assert offload_mode == "block"
    assert device_map is None


def test_nano_does_not_enable_device_map() -> None:
    offload_mode, device_map = resolve_cosmos3_runtime_options({}, model_id="cosmos3-nano", visible_gpus=8)
    assert offload_mode == "block"
    assert device_map is None


def test_device_map_rejects_cpu_offload() -> None:
    try:
        resolve_cosmos3_runtime_options(
            {"device_map": "balanced", "offload_mode": "block"},
            model_id="cosmos3-super",
            visible_gpus=8,
        )
    except ValueError as exc:
        assert "offload_mode=none" in str(exc)
    else:
        raise AssertionError("expected device_map + block offload to fail")


if __name__ == "__main__":
    test_parse_device_map_aliases()
    test_balanced_layer_devices_are_contiguous()
    test_enable_balanced_device_map_keeps_layers_and_home_modules()
    test_super_defaults_to_resident_device_map_on_multi_gpu()
    test_super_keeps_block_offload_on_one_gpu()
    test_nano_does_not_enable_device_map()
    test_device_map_rejects_cpu_offload()
    print("worldfoundry device_map ok")
