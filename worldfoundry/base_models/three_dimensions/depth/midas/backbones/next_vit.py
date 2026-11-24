import torch.nn as nn

from .next_vit_model import NextViT
from .utils import activations, forward_default, get_activation


def forward_next_vit(pretrained, x):
    return forward_default(pretrained, x, "forward")


def _make_next_vit_backbone(
    model,
    hooks=[2, 6, 36, 39],  # noqa: B006 - Keeps the upstream default object and signature.
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.features[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.features[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.features[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.features[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    return pretrained


def _make_pretrained_next_vit_large_6m(hooks=None):
    model = NextViT(stem_chs=(64, 32, 64), depths=(3, 4, 30, 3), path_dropout=0.2)

    hooks = [2, 6, 36, 39] if hooks == None else hooks  # noqa: E711 - Keeps `==` for operands that may compare elementwise.
    return _make_next_vit_backbone(
        model,
        hooks=hooks,
    )
