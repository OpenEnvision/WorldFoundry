from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _gate_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "training" / "validate_anyflow_roundtrip.py"
    spec = importlib.util.spec_from_file_location("validate_anyflow_roundtrip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_anyflow_gate_binds_pinned_checkpoint_without_loading_weights() -> None:
    gate = _gate_module()

    identity = gate._checkpoint_identity()

    assert identity == (
        "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
        "@4c2ec05c7fa4dbafbca131ad32430905c7ff2974"
    )
    assert gate._far_checkpoint_identity() == (
        "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
        "@915af337434035df8545797ecc910d79fa78cf29"
    )
    assert gate._torch_dtype("bfloat16") is torch.bfloat16


def test_real_anyflow_gate_compares_nested_state_directly() -> None:
    gate = _gate_module()
    expected = gate.snapshot_state(
        {
            "tensor": torch.tensor([1.0, 2.0]),
            "optimizer_step": torch.tensor(1.0),
            "nested": {"values": [1, 2]},
        }
    )
    actual = {
        "tensor": torch.tensor([1.0, 2.0]),
        "optimizer_step": torch.tensor(1.0),
        "nested": {"values": [1, 2]},
    }

    gate.assert_state_equal(expected, actual, path="state")

    actual["tensor"][1] = 3.0
    with pytest.raises(AssertionError, match="tensor values differ"):
        gate.assert_state_equal(expected, actual, path="state")
