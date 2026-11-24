# SPDX-License-Identifier: Apache-2.0
"""``torch.hub`` entry points for the in-tree DINO v1 backbones.

Mirrors facebookresearch/dino ``hubconf.py``: same entry-point names, same
``pretrained=True`` default, same architectures and checkpoints. The one
difference is that weights come from a local file passed as ``path`` instead
of being downloaded. VBench uses it as::

    torch.hub.load(repo_or_dir=<this directory>, source="local", model="dino_vitb16", path=<checkpoint>)

Extra keyword arguments (VBench passes ``read_frame`` and ``resolution``) are
ignored by the model constructors, as upstream.
"""

from worldfoundry.base_models.perception_core.general_perception.dino.hub import backbones

dependencies = ["torch", "torchvision"]


def dino_vits16(pretrained=True, **kwargs):
    return backbones.dino_vits16(pretrained=pretrained, **kwargs)


def dino_vits8(pretrained=True, **kwargs):
    return backbones.dino_vits8(pretrained=pretrained, **kwargs)


def dino_vitb16(pretrained=True, **kwargs):
    return backbones.dino_vitb16(pretrained=pretrained, **kwargs)


def dino_vitb8(pretrained=True, **kwargs):
    return backbones.dino_vitb8(pretrained=pretrained, **kwargs)


def dino_resnet50(pretrained=True, **kwargs):
    return backbones.dino_resnet50(pretrained=pretrained, **kwargs)
