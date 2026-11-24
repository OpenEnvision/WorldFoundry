# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Linear-projection CLIP variant used by some Wan world models.

Same XLM-RoBERTa + ViT CLIP family as :mod:`~..clip`, but
the projection head matches linear-attention / world-model
checkpoints.  :class:`CLIPModel` is the public surface.

Token layout remains CLIP visual / text embeddings for
``Conditioning.shared``.  Default I2V should keep
:mod:`~..clip`.

Wan CLIP variant, sibling of :mod:`.action_clip`.
"""

import logging
from typing import Any as Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import AttentionBlock as CanonicalAttentionBlock
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import AttentionPool as CanonicalAttentionPool
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import SelfAttention as CanonicalSelfAttention
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    VisionTransformer as CanonicalVisionTransformer,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import XLMRobertaCLIP as CanonicalXLMRobertaCLIP
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.linear_attention import flash_attention
from worldfoundry.core.model_loading import load_state_dict as load_core_state_dict
from worldfoundry.core.nn.blocks.layers import InputFP32LayerNorm as LayerNorm

__all__ = [
    "CLIPModel",
    "XLMRobertaCLIP",
    "clip_xlm_roberta_vit_h_14",
]


class SelfAttention(CanonicalSelfAttention):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, attention_fn=flash_attention, **kwargs)


class AttentionBlock(CanonicalAttentionBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, norm_layer=LayerNorm, attention_layer=SelfAttention, **kwargs)


class AttentionPool(CanonicalAttentionPool):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


def _clip(
    pretrained=False,
    pretrained_name=None,
    model_cls=XLMRobertaCLIP,
    return_transforms=False,
    return_tokenizer=False,
    tokenizer_padding="eos",
    dtype=torch.float32,
    device="cpu",
    **kwargs,
):
    # init a model on device
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if "siglip" in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = T.Compose(
            [
                T.Resize((model.image_size, model.image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        output += (transforms,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(pretrained=False, pretrained_name="open-clip-xlm-roberta-large-vit-huge-14", **kwargs):
    cfg = {
        "embed_dim": 1024,
        "image_size": 224,
        "patch_size": 14,
        "vision_dim": 1280,
        "vision_mlp_ratio": 4,
        "vision_heads": 16,
        "vision_layers": 32,
        "vision_pool": "token",
        "activation": "gelu",
        "vocab_size": 250002,
        "max_text_len": 514,
        "type_size": 1,
        "pad_id": 1,
        "text_dim": 1024,
        "text_heads": 16,
        "text_layers": 24,
        "text_post_norm": True,
        "text_dropout": 0.1,
        "attn_dropout": 0.0,
        "proj_dropout": 0.0,
        "embedding_dropout": 0.0,
    }
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class CLIPModel:
    def __init__(self, dtype, device, checkpoint_path, tokenizer_path):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False, return_transforms=True, return_tokenizer=False, dtype=dtype, device=device
        )
        self.model = self.model.eval().requires_grad_(False)
        logging.info(f"loading {checkpoint_path}")
        self.model.load_state_dict(load_core_state_dict(checkpoint_path, device="cpu"))

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=self.model.max_text_len - 2, clean="whitespace"
        )

    def visual(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        videos = F.interpolate(videos, size=size, mode="bicubic", align_corners=False)
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out
