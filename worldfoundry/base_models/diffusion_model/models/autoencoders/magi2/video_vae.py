# Ported from SandAI MAGI-2-preview (Apache-2.0): Wan2.2 causal-3D video encode VAE.
"""Wan2_2_VAE (z_dim=48) video encoder.

encode(video [B,3,T,H,W]) -> latent [B,48,1+(T-1)/4,H/16,W/16].
Spatial /16, temporal /4, deterministic (returns normalized mu).

Faithful single-GPU port of ``<SG>/model/vae2_2.py``. The math/shapes/dtype are
preserved verbatim; only the upstream ``@torch.compile`` decorators are dropped
so the module runs eagerly (they were transparent wrappers that did not change
the wrapped computation, and error out on some torch builds).
"""

import itertools
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import AvgDown3D as AvgDown3D
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import CausalConv3d as CausalConv3d
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import DupUp3D as DupUp3D
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import RMS_norm as RMSNorm
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import patchify as patchify
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import Resample as Resample

from .config import WAN22_LATENTS_MEAN, WAN22_LATENTS_STD

__all__ = ["Wan2_2_VAE", "get_vae2_2"]

CACHE_T = 2


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x,
        feat_cache=None,
        feat_idx=[0],  # noqa: B006 - Keeps the upstream default object and signature.
    ):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        x = x + h
        return x


class AttentionBlock(nn.Module):
    """Causal self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        x = x + identity
        return x


class DownResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class UpResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self,
        x,
        feat_cache=None,
        feat_idx=[0],  # noqa: B006 - Keeps the upstream default object and signature.
        first_chunk=False,
    ):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temporal_downsample=[True, True, False],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            t_down_flag = temporal_downsample[i] if i < len(temporal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temporal_upsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        # scale = 1.0 / 2 ** (len(dim_mult) - 2)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            t_up_flag = temporal_upsample[i] if i < len(temporal_upsample) else False
            upsamples.append(
                UpResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temporal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(
        self,
        x,
        feat_cache=None,
        feat_idx=[0],  # noqa: B006 - Keeps the upstream default object and signature.
        first_chunk=False,
    ):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx, first_chunk) if feat_cache is not None else layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temporal_downsample=[True, True, False],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temporal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):  # noqa: B006 - Keeps the upstream default object and signature.
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
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


def _video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", **kwargs):
    # params
    cfg = {
        "dim": dim,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temporal_downsample": [True, True, True],
        "dropout": 0.0,
    }
    cfg.update(**kwargs)

    # init model on meta device (weights assigned at load time)
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    if pretrained_path is not None:
        logging.info(f"loading {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device), assign=True)

    return model


class Wan2_2_VAE:
    """Wan2.2 causal-3D video encode VAE (z_dim=48).

    encode(video [B,3,T,H,W]) -> latent [B,48,1+(T-1)/4,H/16,W/16], normalized mu.
    """

    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        temporal_downsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dtype=torch.float,
        device=None,
        **legacy_kwargs,
    ):
        if "temperal_downsample" in legacy_kwargs:
            warnings.warn(
                "temperal_downsample is deprecated; use temporal_downsample instead",
                DeprecationWarning,
                stacklevel=2,
            )
            temporal_downsample = legacy_kwargs.pop("temperal_downsample")
        if legacy_kwargs:
            unexpected = next(iter(legacy_kwargs))
            raise TypeError(f"unexpected keyword argument {unexpected!r}")
        self.dtype = dtype
        self.device = device

        self.mean = torch.tensor(WAN22_LATENTS_MEAN, dtype=dtype, device=device)
        self.std = torch.tensor(WAN22_LATENTS_STD, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.vae = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temporal_downsample=temporal_downsample,
            )
            .eval()
            .requires_grad_(False)
        )
        if device is not None:
            self.vae = self.vae.to(device)

    def encode(self, video):
        return self.vae.encode(video, self.scale).float()

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        self.scale = [self.mean, 1.0 / self.std]
        self.vae = self.vae.to(*args, **kwargs)
        return self


# Alias matching the destination package naming convention.
Magi2VideoVAE = Wan2_2_VAE


def get_vae2_2(model_path, device="cuda", weight_dtype=torch.float32) -> Wan2_2_VAE:
    import gc

    vae = Wan2_2_VAE(vae_pth=model_path).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vae
