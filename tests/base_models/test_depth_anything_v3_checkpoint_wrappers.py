from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    DepthAnything3,
    _load_state_dict_file,
    logger,
)


def test_load_state_dict_file_unwraps_deepspeed_module_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    expected = {"model.layer.weight": torch.arange(4)}
    torch.save({"module": expected, "optimizer": {"step": 1}}, checkpoint)

    loaded = _load_state_dict_file(checkpoint)

    assert loaded.keys() == expected.keys()
    assert torch.equal(loaded["model.layer.weight"], expected["model.layer.weight"])



def test_tied_checkpoint_aliases_do_not_inflate_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    class SharedHeadModel(DepthAnything3):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            normalization = torch.nn.LayerNorm(2)
            self.model = torch.nn.ModuleDict(
                {"primary": normalization, "alias": normalization, "separate": torch.nn.Linear(2, 1, bias=False)}
            )

    messages = []
    monkeypatch.setattr(logger, "warn", messages.append)
    model = SharedHeadModel()
    state = {"model.primary.weight": torch.tensor([2.0, 3.0]), "model.primary.bias": torch.tensor([4.0, 5.0])}

    model._load_flexible_state_dict(state)

    assert messages == ["Depth Anything 3 checkpoint did not populate 1 parameters/buffers."]
    assert torch.equal(model.model["alias"].weight, state["model.primary.weight"])
    assert torch.equal(model.model["alias"].bias, state["model.primary.bias"])

    messages.clear()
    model._load_flexible_state_dict({**state, "model.separate.weight": torch.tensor([[6.0, 7.0]])})
    assert messages == []
