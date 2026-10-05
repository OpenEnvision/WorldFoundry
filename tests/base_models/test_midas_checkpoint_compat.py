from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.three_dimensions.depth.midas.base_model import BaseModel
from worldfoundry.base_models.three_dimensions.depth.midas.backbones.beit import block_forward


class _TinyMidasModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


class _TinyBeitBlock:
    def __init__(self, *, modern_drop_paths: bool, layer_scale: bool) -> None:
        self.gamma_1 = torch.tensor(0.5) if layer_scale else None
        self.gamma_2 = torch.tensor(0.25) if layer_scale else None
        self.norm1 = lambda value: value
        self.norm2 = lambda value: value
        self.attn = lambda value, resolution, shared_rel_pos_bias=None: value
        self.mlp = lambda value: value
        if modern_drop_paths:
            self.drop_path1 = lambda value: value
            self.drop_path2 = lambda value: value
        else:
            self.drop_path = lambda value: value


def test_midas_load_ignores_recomputed_relative_position_indices(tmp_path: Path) -> None:
    checkpoint = tmp_path / "midas.pt"
    torch.save(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "pretrained.model.blocks.0.attn.relative_position_index": torch.arange(4),
        },
        checkpoint,
    )

    model = _TinyMidasModel()
    model.load(checkpoint)

    assert torch.equal(model.weight, torch.tensor([1.0, 2.0]))


def test_midas_load_remains_strict_for_unrelated_unexpected_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "midas.pt"
    torch.save(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "unexpected.weight": torch.ones(1),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _TinyMidasModel().load(checkpoint)


@pytest.mark.parametrize("modern_drop_paths", [False, True])
@pytest.mark.parametrize("layer_scale", [False, True])
def test_midas_beit_forward_supports_timm_drop_path_layouts(
    modern_drop_paths: bool,
    layer_scale: bool,
) -> None:
    block = _TinyBeitBlock(modern_drop_paths=modern_drop_paths, layer_scale=layer_scale)

    output = block_forward(block, torch.ones(1), (512, 512))

    expected = torch.tensor([1.875 if layer_scale else 4.0])
    assert torch.equal(output, expected)
