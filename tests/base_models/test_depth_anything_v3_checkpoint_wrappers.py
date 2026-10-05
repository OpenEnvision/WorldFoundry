from __future__ import annotations

from pathlib import Path

import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    _load_state_dict_file,
)


def test_load_state_dict_file_unwraps_deepspeed_module_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    expected = {"model.layer.weight": torch.arange(4)}
    torch.save({"module": expected, "optimizer": {"step": 1}}, checkpoint)

    loaded = _load_state_dict_file(checkpoint)

    assert loaded.keys() == expected.keys()
    assert torch.equal(loaded["model.layer.weight"], expected["model.layer.weight"])
