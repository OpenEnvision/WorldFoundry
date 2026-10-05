from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.core.model_loading.checkpoints import load_weights_only


def test_load_weights_only_supports_safetensors_without_pickle(tmp_path) -> None:
    path = tmp_path / "weights.safetensors"
    expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    save_file(expected, path)

    loaded = load_weights_only(path)

    assert isinstance(loaded, dict)
    torch.testing.assert_close(loaded["weight"], expected["weight"])


def test_safetensors_loader_rejects_torch_mmap_option(tmp_path) -> None:
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(1)}, path)

    with pytest.raises(ValueError, match="mmap"):
        load_weights_only(path, mmap=True)
