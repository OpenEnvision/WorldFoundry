# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import WanVAE_
from worldfoundry.core import autocast_context
from worldfoundry.core.model_loading import load_torch_state_dict

__all__ = [
    "WanVAE",
]


def _video_vae(pretrained_path=None, z_dim=None, device="cpu", **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = {
        "dim": 96,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
    }
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(load_torch_state_dict(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:
    def __init__(self, z_dim=16, vae_pth="cache/vae_step_411000.pth", dtype=torch.float, device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with autocast_context(self.device, dtype=self.dtype):
            return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos]

    def decode(self, zs):
        with autocast_context(self.device, dtype=self.dtype):
            return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]

    def encode_batch(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with autocast_context(self.device, dtype=self.dtype):
            return self.model.encode(videos, self.scale).float()

    def decode_batch(self, zs):
        with autocast_context(self.device, dtype=self.dtype):
            return self.model.decode(zs, self.scale).float().clamp_(-1, 1)
