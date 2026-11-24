"""DINO v1 backbone constructors (facebookresearch/dino hub API).

Upstream downloads pretrained weights from a URL. These constructors never
touch the network: ``pretrained=True`` loads the same upstream checkpoint file
from ``path`` with the same strict state-dict match.
"""

from __future__ import annotations

from pathlib import Path

from .. import models as vits


def _load_pretrained(model, path: str | Path | None, *, strict: bool = True):
    import torch

    if path is None:
        raise RuntimeError("The in-tree DINO hub does not download weights; pass path=<upstream DINO checkpoint .pth>.")
    state_dict = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    return model


def dino_vits16(pretrained: bool = False, path: str | Path | None = None, **kwargs):
    model = vits.vit_small(patch_size=16, num_classes=0, **kwargs)
    return _load_pretrained(model, path) if pretrained else model


def dino_vits8(pretrained: bool = False, path: str | Path | None = None, **kwargs):
    model = vits.vit_small(patch_size=8, num_classes=0, **kwargs)
    return _load_pretrained(model, path) if pretrained else model


def dino_vitb16(pretrained: bool = False, path: str | Path | None = None, **kwargs):
    model = vits.vit_base(patch_size=16, num_classes=0, **kwargs)
    return _load_pretrained(model, path) if pretrained else model


def dino_vitb8(pretrained: bool = False, path: str | Path | None = None, **kwargs):
    model = vits.vit_base(patch_size=8, num_classes=0, **kwargs)
    return _load_pretrained(model, path) if pretrained else model


def dino_resnet50(pretrained: bool = False, path: str | Path | None = None, **kwargs):
    """DINO ResNet-50 backbone with identity classifier head."""
    import torch
    from torchvision.models.resnet import resnet50

    model = resnet50(weights=None, **kwargs)
    model.fc = torch.nn.Identity()
    # Upstream loads this checkpoint non-strictly: it has no ``fc`` entries.
    return _load_pretrained(model, path, strict=False) if pretrained else model
