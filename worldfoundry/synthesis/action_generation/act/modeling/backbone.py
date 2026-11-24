# Inference-only ACT source retained in-tree.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

from .position_encoding import build_position_encoding


def build_backbone(args):
    """Build the ResNet backbone and position encoder used by ACT.

    Args:
        args: Namespace-like object with backbone, dilation, masks, and lr_backbone.
    """
    import torch
    import torchvision
    from torch import nn
    from torchvision.models._utils import IntermediateLayerGetter

    class FrozenBatchNorm2d(torch.nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.register_buffer("weight", torch.ones(n))
            self.register_buffer("bias", torch.zeros(n))
            self.register_buffer("running_mean", torch.zeros(n))
            self.register_buffer("running_var", torch.ones(n))

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ) -> None:
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key in state_dict:
                del state_dict[num_batches_tracked_key]
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def forward(self, x):
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            scale = w * (rv + 1e-5).rsqrt()
            bias = b - rm * scale
            return x * scale + bias

    class BackboneBase(nn.Module):
        def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool) -> None:
            super().__init__()
            return_layers = (
                {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                if return_interm_layers
                else {"layer4": "0"}
            )
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self.num_channels = num_channels

        def forward(self, tensor):
            return self.body(tensor)

    class Backbone(BackboneBase):
        def __init__(self, name: str, return_interm_layers: bool, dilation: bool) -> None:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=None,
                norm_layer=FrozenBatchNorm2d,
            )
            num_channels = 512 if name in ("resnet18", "resnet34") else 2048
            super().__init__(backbone, num_channels, return_interm_layers)

    class Joiner(nn.Sequential):
        def forward(self, tensor):
            xs = self[0](tensor)
            out = []
            pos = []
            for item in xs.values():
                out.append(item)
                pos.append(self[1](item).to(item.dtype))
            return out, pos

    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone, args.masks, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
