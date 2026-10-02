from __future__ import annotations

import pytest
import torch

from worldfoundry.core.nn.transformers_compat import (
    find_pruneable_heads_and_indices,
    prepare_head_mask,
)


def test_attention_head_pruning_compatibility():
    heads, indices = find_pruneable_heads_and_indices([1, 3], 4, 2, set())
    assert heads == {1, 3}
    assert indices.tolist() == [0, 1, 4, 5]


def test_attention_head_pruning_accounts_for_already_pruned_heads():
    heads, indices = find_pruneable_heads_and_indices([2], 3, 2, {0})
    assert heads == {2}
    assert indices.tolist() == [0, 1, 4, 5]


def test_prepare_head_mask_defaults_to_per_layer_none():
    assert prepare_head_mask(None, 3) == [None, None, None]


def test_prepare_head_mask_expands_one_dimensional_mask():
    result = prepare_head_mask(torch.tensor([1.0, 0.0]), 3)
    assert result.shape == (3, 1, 2, 1, 1)
    assert result.dtype == torch.float32


def test_prepare_head_mask_rejects_unsupported_dimensions():
    with pytest.raises(ValueError, match="dimension 1 or 2"):
        prepare_head_mask(torch.ones(1, 1, 1), 1)
