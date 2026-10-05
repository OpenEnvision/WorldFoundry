# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lyra_2._src.tokenizers.interface import VideoTokenizerInterface

from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention
from worldfoundry.core.configuration.lazy_config import LazyCall as L
from worldfoundry.core.configuration.lazy_config import LazyDict
from worldfoundry.core.distributed.runtime.logging import log
from worldfoundry.core.distributed.runtime.torch_process_group import broadcast, get_rank, sync_model_states
from worldfoundry.data.io import easy_io

__all__ = [
    "WanVAE",
]

CACHE_T = 2


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import CausalConv3d


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import Resample


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import ResidualBlock


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import AttentionBlock


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import Encoder3d


from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import Decoder3d


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        temporal_window=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.temporal_window = temporal_window
        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        batch_size = x.shape[0]

        if batch_size >= 8:
            chunk_size = 4
            chunks = []
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunks.append(self._encode_single_batch(x[start_idx:end_idx], scale))
            return torch.cat(chunks, dim=0)
        else:
            return self._encode_single_batch(x, scale)

    def _encode_single_batch(self, x, scale):
        """Encode a single batch."""
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // self.temporal_window
        # Split x along T into chunks: [1, temporal_window, temporal_window, ...]
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self._i0_encode(x)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + self.temporal_window * (i - 1) : 1 + self.temporal_window * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        if (t - 1) % self.temporal_window:
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1 + self.temporal_window * (iter_ - 1) :, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    @torch.compiler.disable
    def _i0_encode(self, x):
        """
        If enabled torch.compile uses significantly more memory for this step, so we disable it
        """
        out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        return out

    def decode(self, z, scale):
        batch_size = z.shape[0]

        if batch_size >= 8:
            chunk_size = 4
            log.info(f"Decoding with chunking, batch size: {batch_size}, chunk size: {chunk_size}")
            chunks = []
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunks.append(self._decode_single_batch(z[start_idx:end_idx], scale))
            return torch.cat(chunks, dim=0)
        else:
            return self._decode_single_batch(z, scale)

    def _decode_single_batch(self, z, scale):
        """Decode a single batch."""
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path=None,
    z_dim=None,
    device="cpu",
    load_mean_std=False,
    mean_std_path=None,
    image_mean_std_path: str = "./checkpoints/vae/images_mean_std.pt",
    video_mean_std_path: str = "./checkpoints/vae/video_mean_std.pt",
    **kwargs,
):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    if mean_std_path is not None:
        image_mean_std_path = mean_std_path.replace("mean_std.pt", "images_mean_std.pt")
        video_mean_std_path = mean_std_path.replace("mean_std.pt", "video_mean_std.pt")

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    if pretrained_path is None:
        model.to_empty(device=device)
        if load_mean_std:
            img_mean, img_std = torch.randn(1, 16, 1, 1, 1, device=device), torch.randn(1, 16, 1, 1, 1, device=device)
            video_mean, video_std = (
                torch.randn(1, 16, 32, 1, 1, device=device),
                torch.randn(1, 16, 32, 1, 1, device=device),
            )
    else:
        if get_rank() == 0:
            ckpt = easy_io.load(
                pretrained_path,
                map_location=device,
            )
            if load_mean_std:
                img_mean, img_std = easy_io.load(image_mean_std_path, map_location=device)
                video_mean, video_std = easy_io.load(video_mean_std_path, map_location=device)
                img_mean = img_mean.reshape(1, 16, 1, 1, 1)
                img_std = img_std.reshape(1, 16, 1, 1, 1)
                video_mean = video_mean.reshape(1, 16, 32, 1, 1)
                video_std = video_std.reshape(1, 16, 32, 1, 1)

            # load checkpoint
            log.info(f"loading {pretrained_path}")
            model.load_state_dict(ckpt, assign=True)
        else:
            model.to_empty(device=device)
            if load_mean_std:
                img_mean, img_std = (
                    torch.randn(1, 16, 1, 1, 1, device=device),
                    torch.randn(1, 16, 1, 1, 1, device=device),
                )
                video_mean, video_std = (
                    torch.randn(1, 16, 32, 1, 1, device=device),
                    torch.randn(1, 16, 32, 1, 1, device=device),
                )
    sync_model_states(model)

    if load_mean_std:
        log.info("broadcast mean and std for wan2pt1")
        broadcast(img_mean, 0)
        broadcast(img_std, 0)
        broadcast(video_mean, 0)
        broadcast(video_std, 0)
        return model, img_mean, img_std, video_mean, video_std

    return (
        model,
        torch.zeros(1, 1, 1, 1, 1, device=device),
        torch.ones(1, 1, 1, 1, 1, device=device),
        torch.zeros(1, 1, 50, 1, 1, device=device),
        torch.ones(1, 1, 50, 1, 1, device=device),
    )


class WanVAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth="./checkpoints/vae/vae.pth",
        load_mean_std=False,
        mean_std_path=None,
        image_mean_std_path: str = "./checkpoints/vae/images_mean_std.pt",
        video_mean_std_path: str = "./checkpoints/vae/video_mean_std.pt",
        dtype=torch.float,
        device="cuda",
        is_amp=True,
        benchmark: bool = False,
        temporal_window: int = 4,
    ):
        self.dtype = dtype
        self.device = device
        self.temporal_window = temporal_window

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
        self.model, self.img_mean, self.img_std, self.video_mean, self.video_std = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            load_mean_std=load_mean_std,
            mean_std_path=mean_std_path,
            image_mean_std_path=image_mean_std_path,
            video_mean_std_path=video_mean_std_path,
            device=device,
            temporal_window=temporal_window,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """

        in_dtype = videos.dtype
        with self.context:
            if not self.is_amp:
                videos = videos.to(self.dtype)
            latent = self.model.encode(videos, self.scale)
        latent = latent.to(in_dtype)
        return latent

    @torch.no_grad()
    def decode(self, zs):
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            video_recon = self.model.decode(zs, self.scale)
        video_recon = video_recon.to(in_dtype)
        return video_recon


class Wan2pt1VAEInterface(VideoTokenizerInterface):
    def __init__(self, chunk_duration: int = 81, load_mean_std=False, **kwargs):
        self.model = WanVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            load_mean_std=load_mean_std,
            vae_pth=kwargs.get(
                "vae_pth",
                "./checkpoints/vae/vae.pth",
            ),
            mean_std_path=kwargs.get("mean_std_path"),
            image_mean_std_path=kwargs.get(
                "image_mean_std_path",
                "./checkpoints/vae/images_mean_std.pt",
            ),
            video_mean_std_path=kwargs.get(
                "video_mean_std_path",
                "./checkpoints/vae/video_mean_std.pt",
            ),
            temporal_window=kwargs.get("temporal_window", 4),
        )
        if kwargs.get("compile_encode", False) and hasattr(torch, "compile"):
            torch_compile_available = True
            try:
                # PyTorch >= 2.7
                torch._dynamo.config.recompile_limit = 32
            except AttributeError:
                try:
                    torch._dynamo.config.cache_size_limit = 32
                except AttributeError:
                    log.warning(
                        "`compile_encode=True` requested, but Torch Dynamo is unavailable – skipping compilation."
                    )
                    torch_compile_available = False
            if torch_compile_available:
                log.warning(
                    "The 'model.config.tokenizer.compile_encode' config option is deprecated. Please switch to using CompileTokenizer callback."
                )
                self.encode = torch.compile(self.encode, dynamic=False)
        del kwargs
        self.chunk_duration = chunk_duration

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        latents = self.model.encode(state)
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(latents)
        else:
            return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[
                :, :, :num_frames
            ].type_as(latents)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        num_frames = latent.shape[2]
        if num_frames == 1:
            return self.model.decode(
                (latent * self.model.img_std.type_as(latent)) + self.model.img_mean.type_as(latent)
            )
        else:
            return self.model.decode(
                (latent * self.model.video_std[:, :, :num_frames].type_as(latent))
                + self.model.video_mean[:, :, :num_frames].type_as(latent)
            )

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "wan2pt1_tokenizer"


Wan2pt1VAEConfig: LazyDict = L(Wan2pt1VAEInterface)(name="wan2pt1_tokenizer", compile_encode=False)
