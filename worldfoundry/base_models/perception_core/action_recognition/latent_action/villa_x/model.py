"""Villa-X Igor encoder used for latent-action extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .base import PretrainedConfig, PretrainedModel
from .embed import PatchEmbed
from .st import STBlock
from .transformer import (
    MAPBlock,
    RMSNorm,
    get_1D_position_embeddings,
    get_2D_position_embeddings,
)
from .utils import hwc2chw, normalize_images, resize
from .vq import VectorQuantizer


def _select_last_frames(values: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    if len(values) != len(lengths):
        raise ValueError("Batch size and length count must match")
    return [values[index, -length:] for index, length in enumerate(lengths)]


class IgorConfig(PretrainedConfig):
    resolution: int = 224
    patch_size: int = 14
    in_channels: int = 3
    d_t: int = 8
    mlp_ratio: float = 4.0
    use_xformers: bool | None = None
    encoder_depth: int = 12
    encoder_embed_dim: int = 768
    encoder_n_heads: int = 8
    action_latent_dim: int = 128
    st_use_qk_norm: bool = True
    num_learned_tokens: int = 4
    map_heads: int = 24
    n_codes: int = 32
    grid_size: int | None = None
    embed_tokens: int | None = None

    def model_post_init(self, __context: object) -> None:
        if self.resolution % self.patch_size:
            raise ValueError("Image resolution must be divisible by patch size")
        self.grid_size = self.resolution // self.patch_size
        self.embed_tokens = self.grid_size**2


class IgorPretrainedModel(PretrainedModel):
    config_class = IgorConfig

    @staticmethod
    def transformer_initializer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def _init_weights(self, module: nn.Module) -> None:
        self.transformer_initializer(module)

    def preprocess(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.shape[-1] == 3:
            clips = hwc2chw(clips)
        clips = resize(clips, size=self.config.resolution)
        return normalize_images(clips)


class IgorEncoder(IgorPretrainedModel):
    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.embed = PatchEmbed(
            config.resolution,
            config.patch_size,
            config.encoder_embed_dim,
            in_channels=config.in_channels,
        )
        self.pe_spatial = nn.Parameter(
            torch.from_numpy(get_2D_position_embeddings(config.encoder_embed_dim, config.grid_size))
            .float()
            .unsqueeze(0),
            requires_grad=False,
        )
        self.pe_temporal = nn.Parameter(
            torch.from_numpy(get_1D_position_embeddings(config.encoder_embed_dim, config.d_t)).float().unsqueeze(0),
            requires_grad=False,
        )
        self.layers = nn.ModuleList(
            [
                STBlock(
                    config.encoder_embed_dim,
                    config.encoder_n_heads,
                    d_s=config.embed_tokens,
                    d_t=config.d_t,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=0.0,
                    use_xformers=config.use_xformers,
                    enable_layernorm_kernel=False,
                    st_use_qk_norm=config.st_use_qk_norm,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = RMSNorm(config.encoder_embed_dim)
        self.map_block = MAPBlock(
            n_latents=config.num_learned_tokens,
            embed_dim=config.encoder_embed_dim,
            n_heads=config.map_heads,
            mlp_ratio=config.mlp_ratio,
            output_dim=config.action_latent_dim,
            do_rms_norm=True,
            do_swish_glu=True,
            qk_norm=False,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if module is self.embed.proj:
            nn.init.xavier_uniform_(module.weight.data.view(module.weight.shape[0], -1))
        super()._init_weights(module)

    def forward(self, clips: torch.Tensor, clip_lengths: list[int]) -> torch.Tensor:
        embedding = self.embed(clips) + self.pe_spatial
        embeddings = _select_last_frames(embedding, clip_lengths)
        values = torch.cat(embeddings, dim=0)

        for index, layer in enumerate(self.layers):
            values = layer(
                values,
                clip_lengths,
                tpe=self.pe_temporal if index == 0 else None,
            )

        values = self.norm(values)
        split_values = torch.split(values, clip_lengths, dim=0)
        transitions = torch.cat(
            [(value[1:] + value[:-1]) / 2.0 for value in split_values],
            dim=0,
        )
        actions = self.map_block(transitions)
        return rearrange(actions, "b n d -> b 1 (n d)")

    @torch.inference_mode()
    def idm(self, clips: torch.Tensor) -> list[torch.Tensor]:
        if clips.ndim == 4:
            clips = clips.unsqueeze(0)
        if clips.ndim != 5:
            raise ValueError(f"Expected BTCHW or BTHWC clips, got {clips.shape}")

        clips = self.preprocess(clips)
        batch_size, frame_count = clips.shape[:2]
        valid_lengths = torch.full((batch_size,), frame_count, device=clips.device, dtype=torch.long)

        if frame_count < self.config.d_t:
            pad_count = self.config.d_t - frame_count
            padding = clips.new_zeros(batch_size, pad_count, *clips.shape[2:])
            clips = torch.cat((padding, clips), dim=1)
            frame_count = self.config.d_t

        actions: list[torch.Tensor] | None = None
        for index in range(frame_count - self.config.d_t + 1):
            window = clips[:, index : index + self.config.d_t]
            window_lengths = valid_lengths.clamp(max=self.config.d_t)
            action = self.forward(window, window_lengths.tolist())
            split_sizes = (window_lengths - 1).tolist()
            new_actions = list(torch.split(action, split_sizes, dim=0))
            if actions is None:
                actions = new_actions
            else:
                actions = [
                    torch.cat((current, new[-1:]), dim=0) for current, new in zip(actions, new_actions, strict=False)
                ]

        if actions is None:
            raise RuntimeError("No Villa-X action window was produced")
        return actions


class IgorModel(IgorPretrainedModel):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.encoder = IgorEncoder(config)
        self.vq = VectorQuantizer(
            config.n_codes,
            config.action_latent_dim,
            beta=0.25,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if module is self.vq.embedding:
            nn.init.uniform_(module.weight, -1.0 / self.vq.n_e, 1.0 / self.vq.n_e)
        super()._init_weights(module)

    @torch.inference_mode()
    def idm(
        self,
        clips: torch.Tensor,
        *,
        return_dict: bool = False,
        return_vq_tokens: bool = True,
    ):
        tokens = self.encoder.idm(clips)
        vq_tokens, indices = self.vq(torch.stack(tokens).contiguous())
        if not return_dict:
            return vq_tokens if return_vq_tokens else tokens
        return {
            "tokens": tokens,
            "vq_tokens": vq_tokens,
            "indices": indices,
        }
