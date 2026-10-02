"""Regression tests for the deduplicated model-adapter helpers (review TR-6)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.training.models._shared import (
    component_module,
    freeze_module,
    merge_without_overwrite,
    module_device_dtype,
)


def test_component_module_accepts_none_and_attribute_lookup() -> None:
    module = nn.Linear(2, 2)

    assert component_module(module) is module
    assert component_module(None, "model") is None  # hunyuan-style optional component
    holder = type("Holder", (), {"model": module})()
    assert component_module(holder, "missing", "model") is module
    assert component_module(object(), "model") is None


def test_module_device_dtype_fallbacks() -> None:
    linear = nn.Linear(2, 2)
    device, dtype = module_device_dtype(linear)
    assert device == torch.device("cpu")
    assert dtype == torch.float32

    class _BufferOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("table", torch.ones(2, dtype=torch.bfloat16))

    device, dtype = module_device_dtype(_BufferOnly())
    assert dtype == torch.bfloat16

    class _IntBufferOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("ids", torch.ones(2, dtype=torch.int64))

    device, dtype = module_device_dtype(_IntBufferOnly())
    assert dtype == torch.float32

    device, dtype = module_device_dtype(nn.Identity())
    assert device == torch.device("cpu")
    assert dtype == torch.float32


def test_freeze_module_disables_grads_and_accepts_none() -> None:
    module = nn.Linear(2, 2)
    freeze_module(module)
    assert not any(parameter.requires_grad for parameter in module.parameters())
    assert not module.training
    freeze_module(None)  # no-op


@pytest.mark.parametrize("family", ("Wan", "SANA", "Cosmos"))
def test_merge_without_overwrite_keeps_per_family_messages(family: str) -> None:
    destination = {"context": 1}
    merge_without_overwrite(destination, {"extra": 2}, source_name="conditioner.shared", family=family)
    assert destination == {"context": 1, "extra": 2}

    with pytest.raises(
        ValueError,
        match=rf"conditioner\.shared collides with encoded {family} conditioning keys: \['context'\]",
    ):
        merge_without_overwrite(
            destination,
            {"context": 3},
            source_name="conditioner.shared",
            family=family,
        )
