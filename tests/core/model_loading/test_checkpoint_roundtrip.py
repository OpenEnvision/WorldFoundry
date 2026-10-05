"""Real tensor checkpoints must restore inference and fail before partial mutation."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.core.model_loading.checkpoints import (
    assign_state_dict_strict,
    load_tensor_state_dict,
    remap_checkpoint_keys,
    submodule_state_dict,
)


class _InferenceProbe(torch.nn.Module):
    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.projection = torch.nn.Linear(4, 3, device=device, dtype=dtype)
        self.register_buffer("scale", torch.ones(3, device=device, dtype=dtype))

    def forward(self, value):
        return self.projection(value) * self.scale


def _known_model(dtype):
    model = _InferenceProbe(dtype=dtype).eval()
    with torch.no_grad():
        model.projection.weight.copy_(torch.arange(12, dtype=dtype).reshape(3, 4) / 8)
        model.projection.bias.copy_(torch.tensor([-1, 0, 1], dtype=dtype))
        model.scale.copy_(torch.tensor([0.5, 1, 2], dtype=dtype))
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("format_name", ["safetensors", "plain", "state_dict", "model", "module", "model_state_dict"])
def test_checkpoint_roundtrip_preserves_forward_and_buffers(tmp_path, dtype, device, format_name):
    reference = _known_model(dtype)
    tensors = OrderedDict((f"backbone.{key}", value.clone()) for key, value in reference.state_dict().items())
    if format_name == "safetensors":
        path = tmp_path / "model.safetensors"
        save_file(tensors, path)
    else:
        path = tmp_path / "model.pt"
        payload = tensors if format_name == "plain" else {format_name: tensors, "epoch": 12}
        torch.save(payload, path)
    restored = load_tensor_state_dict(path)
    converted = remap_checkpoint_keys(restored, {r"^backbone\.": ""})
    target = _InferenceProbe(device=device, dtype=torch.float64).eval().requires_grad_(False)
    incompatible = assign_state_dict_strict(target, converted)

    assert incompatible.missing_keys == incompatible.unexpected_keys == []
    for key, expected in reference.state_dict().items():
        actual = target.state_dict()[key]
        assert actual.device.type == "cpu"
        assert actual.dtype == dtype
        assert actual.data_ptr() == converted[key].data_ptr()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not any(parameter.requires_grad for parameter in target.parameters())
    sample = torch.arange(24, dtype=dtype).reshape(2, 3, 4) / 8
    with torch.inference_mode():
        torch.testing.assert_close(target(sample), reference(sample), atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_invalid_checkpoint_is_rejected_before_any_parameter_changes(device, defect):
    model = _InferenceProbe(device=device)
    before = {name: value for name, value in model.named_parameters()}
    before.update(model.named_buffers())
    snapshots = {name: value.detach().clone() for name, value in before.items() if not value.is_meta}
    candidate = dict(_known_model(torch.float32).state_dict())
    if defect == "missing":
        candidate.pop("scale")
    elif defect == "unexpected":
        candidate["unknown.weight"] = torch.ones(1)
    else:
        candidate["projection.bias"] = torch.zeros(4)
    with pytest.raises(RuntimeError, match="incompatible"):
        assign_state_dict_strict(model, candidate, label="probe checkpoint")
    after = dict(model.named_parameters())
    after.update(model.named_buffers())
    assert after.keys() == before.keys()
    for name in before:
        assert after[name] is before[name]
        if name in snapshots:
            torch.testing.assert_close(after[name], snapshots[name], atol=0, rtol=0)


@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize("existing_destination", [False, True])
def test_remap_key_collision_cannot_silently_replace_a_released_weight(order, existing_destination):
    pairs = [("encoder.weight", torch.ones(2))]
    other_key = "weight" if existing_destination else "decoder.weight"
    pairs.append((other_key, torch.zeros(2)))
    if order:
        pairs.reverse()
    source = OrderedDict(pairs)
    with pytest.raises(ValueError, match="collision.*weight"):
        remap_checkpoint_keys(source, {r"^(encoder|decoder)\.": ""})
    assert list(source) == [key for key, _ in pairs]
    assert all(source[key] is value for key, value in pairs)


def test_remapping_uses_first_matching_rule_without_changing_tensor_storage():
    tensor = torch.arange(6).reshape(2, 3)
    source = {"encoder.blocks.2.weight": tensor, "untouched": tensor}
    converted = remap_checkpoint_keys(source, {r"^encoder\.": "", r"^encoder\.blocks\.": "wrong."})
    assert set(converted) == {"blocks.2.weight", "untouched"}
    assert all(value is tensor for value in converted.values())
    assert set(source) == {"encoder.blocks.2.weight", "untouched"}


def test_submodule_selection_preserves_order_aliases_and_exact_prefix():
    source = OrderedDict(
        (name, torch.tensor([index]))
        for index, name in enumerate(["encoder.weight", "encoder.bias", "encoder_extra.weight", "decoder.weight"])
    )
    selected = submodule_state_dict(source, "encoder.")
    assert list(selected) == ["weight", "bias"]
    assert selected["weight"] is source["encoder.weight"]
    assert selected["bias"] is source["encoder.bias"]
    assert not submodule_state_dict(source, "absent.")
