from __future__ import annotations

import pickle

import pytest
import torch

from worldfoundry.core.checkpoint.safe_loading import load_tensor_state_dict


class _UnsafeCheckpointMetadata:
    pass


def test_tensor_state_dict_allows_set_metadata_with_weights_only(tmp_path) -> None:
    checkpoint = tmp_path / "set_metadata.pt"
    expected = torch.arange(4)
    torch.save(
        {
            "state_dict": {"weight": expected},
            "metadata": {"tags": {"wan", "wow"}},
        },
        checkpoint,
    )

    loaded = load_tensor_state_dict(checkpoint)

    assert loaded.keys() == {"weight"}
    assert torch.equal(loaded["weight"], expected)


def test_tensor_state_dict_still_rejects_arbitrary_classes(tmp_path) -> None:
    checkpoint = tmp_path / "unsafe_metadata.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "metadata": _UnsafeCheckpointMetadata(),
        },
        checkpoint,
    )

    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        load_tensor_state_dict(checkpoint)
