from __future__ import annotations

import pytest
import torch

from worldfoundry.training.state_comparison import (
    assert_state_equal,
    file_size_inventory,
    snapshot_state,
    state_changed,
)


def test_nested_state_comparison_owns_and_compares_tensor_values() -> None:
    source = {"model": [torch.tensor([1.0, 2.0])], "step": 3}
    snapshot = snapshot_state(source)
    source["model"][0].add_(1.0)

    assert state_changed(snapshot, source)
    assert assert_state_equal(snapshot, {"model": [torch.tensor([1.0, 2.0])], "step": 3}) == 2


def test_nested_state_comparison_reports_structure_changes() -> None:
    with pytest.raises(AssertionError, match="keys differ"):
        assert_state_equal({"left": 1}, {"right": 1})


def test_file_size_inventory_lists_relative_payloads(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "weights.bin").write_bytes(b"weights")

    assert file_size_inventory(tmp_path) == {"nested/weights.bin": 7}
