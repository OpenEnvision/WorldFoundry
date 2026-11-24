from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.three_dimensions.depth.midas.backbones.beit import block_forward
from worldfoundry.base_models.three_dimensions.depth.midas.base_model import BaseModel


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


def test_next_vit_preserves_midas_flat_feature_hooks() -> None:
    from worldfoundry.base_models.three_dimensions.depth.midas.backbones.next_vit import (
        _make_pretrained_next_vit_large_6m,
        forward_next_vit,
    )

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(0)
        backbone = _make_pretrained_next_vit_large_6m().eval()
        features = forward_next_vit(backbone, torch.randn(1, 3, 64, 64))
    assert [tuple(value.shape) for value in features] == [
        (1, 96, 16, 16),
        (1, 256, 8, 8),
        (1, 512, 4, 4),
        (1, 1024, 2, 2),
    ]
    keys = backbone.state_dict()
    assert "model.features.39.e_mhsa.q.weight" in keys
    assert "model.proj_head.0.weight" in keys
    assert not any(key.startswith("model.stages.") for key in keys)


@pytest.mark.parametrize("use_pretrained", [False, True])
def test_midas_wsl_uses_torchvision_and_original_checkpoint(
    monkeypatch: pytest.MonkeyPatch, use_pretrained: bool
) -> None:
    from types import SimpleNamespace

    from torchvision import models

    from worldfoundry.base_models.three_dimensions.depth.midas.blocks import _make_pretrained_resnext101_wsl

    checkpoint = {"weight": torch.ones(1)}
    loaded = []
    resnet = SimpleNamespace(
        conv1=torch.nn.Identity(),
        bn1=torch.nn.Identity(),
        relu=torch.nn.Identity(),
        maxpool=torch.nn.Identity(),
        layer1=torch.nn.Identity(),
        layer2=torch.nn.Identity(),
        layer3=torch.nn.Identity(),
        layer4=torch.nn.Identity(),
        load_state_dict=loaded.append,
    )

    def build(*, weights: object) -> SimpleNamespace:
        assert weights is None
        return resnet

    def load(url: str, *, progress: bool) -> dict[str, torch.Tensor]:
        assert url == "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth"
        assert progress
        return checkpoint

    monkeypatch.setattr(models, "resnext101_32x8d", build)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load)
    backbone = _make_pretrained_resnext101_wsl(use_pretrained)
    assert loaded == [checkpoint]
    assert backbone.layer2 is resnet.layer2
