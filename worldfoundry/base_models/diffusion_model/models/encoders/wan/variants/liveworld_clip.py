# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""LiveWorld CLIP bindings: weight-dtype norms and unmaterialized direct embeddings."""

from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    AttentionBlock as CanonicalAttentionBlock,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    AttentionPool as CanonicalAttentionPool,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    SelfAttention,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    VisionTransformer as CanonicalVisionTransformer,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    XLMRobertaCLIP as CanonicalXLMRobertaCLIP,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.geometry_attention import flash_attention


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.to(self.weight.dtype))


class AttentionBlock(CanonicalAttentionBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, norm_layer=LayerNorm, attention_layer=SelfAttention, **kwargs)


class AttentionPool(CanonicalAttentionPool):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The pool fixes version=2, retaining LiveWorld's FA2-only call contract.
        super().__init__(*args, norm_layer=LayerNorm, attention_fn=flash_attention, **kwargs)


class VisionTransformer(CanonicalVisionTransformer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            norm_layer=LayerNorm,
            block_layer=AttentionBlock,
            pool_layer=AttentionPool,
            materialize_embeddings=False,
            **kwargs,
        )


class XLMRobertaCLIP(CanonicalXLMRobertaCLIP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, vision_layer=VisionTransformer, **kwargs)
