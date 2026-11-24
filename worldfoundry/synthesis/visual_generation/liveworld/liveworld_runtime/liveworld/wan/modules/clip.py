# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.liveworld_clip import (
    XLMRobertaCLIP as XLMRobertaCLIP,
)

__all__ = [
    "CLIPModel",
    "XLMRobertaCLIP",
    "clip_xlm_roberta_vit_h_14",
]


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


class CLIPModel(nn.Module):
    def __init__(self, dtype, device, checkpoint_path):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        # self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False, return_transforms=True, return_tokenizer=False, dtype=dtype, device=device
        )
        self.model = self.model.eval().requires_grad_(False)
        logging.info(f"loading {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

        del self.model.log_scale, self.model.textual

    def visual(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        input_device = videos[0].device
        videos = torch.cat(
            [F.interpolate(u.transpose(0, 1), size=size, mode="bicubic", align_corners=False) for u in videos]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))
        # Ensure videos are on the same device as input (transforms may move to CPU)
        videos = videos.to(input_device)

        # forward
        out = self.model.visual(videos, use_31_block=True)
        out = out.unsqueeze(0)
        return out
