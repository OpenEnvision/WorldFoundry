"""LAQ encoders used for latent-action extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..backbones import (
    get_dino_reps,
    get_dino_tokenizer,
    get_dinov3_reps,
    get_dinov3_tokenizer,
    get_magvit2_tokenizer,
    get_reps_magvit2,
    get_siglip2_reps,
    get_siglip2_tokenizer,
)
from .attention import ContinuousPositionBias, Transformer
from .nsvq import NSVQ


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    result = (value, value) if isinstance(value, int) else value
    if len(result) != 2:
        raise ValueError(f"Expected one or two dimensions, got {result!r}")
    return result


class _LatentActionQuantizationBase(nn.Module):
    """Shared encoder and vector quantizer for LAQ backbone variants."""

    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 3,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        code_seq_len: int = 1,
    ) -> None:
        super().__init__()
        del channels

        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        self.dim = dim
        patch_height, patch_width = self.patch_size
        image_height, image_width = self.image_size
        if image_height % patch_height or image_width % patch_width:
            raise ValueError(f"Image size {self.image_size} must be divisible by patch size {self.patch_size}")

        transformer_kwargs = {
            "dim": dim,
            "dim_head": dim_head,
            "heads": heads,
            "attn_dropout": attn_dropout,
            "ff_dropout": ff_dropout,
            "peg": True,
            "peg_causal": True,
        }
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            device="cpu",
            code_seq_len=code_seq_len,
            patch_size=patch_height,
            image_size=image_height,
        )

    @property
    def patch_height_width(self) -> tuple[int, int]:
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = False, assign: bool = False):
        del strict
        try:
            return super().load_state_dict(state_dict, strict=False, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=False)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path))
        state_dict = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        self.load_state_dict(state_dict)

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        raise NotImplementedError

    def _encode(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = tokens.shape[0]
        height, width = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attention_bias = self.spatial_rel_pos_bias(height, width, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attention_bias, video_shape=video_shape)
        tokens = rearrange(
            tokens,
            "(b t) (h w) d -> b t h w d",
            b=batch_size,
            h=height,
            w=width,
        )
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(
            tokens,
            "(b h w) t d -> b t h w d",
            b=batch_size,
            h=height,
            w=width,
        )
        return tokens[:, :1], tokens[:, 1:]

    def forward(
        self,
        video: Tensor,
        *,
        mask: Tensor | None = None,
        return_only_codebook_ids: bool = False,
        **_: Any,
    ) -> tuple[Tensor, Tensor]:
        if not return_only_codebook_ids:
            raise ValueError("The in-tree LAQ integration only supports latent-action extraction")
        if mask is not None:
            raise ValueError("Masked training is not part of latent-action extraction")
        if video.ndim == 4:
            video = rearrange(video, "b c h w -> b c 1 h w")
        if video.ndim != 5:
            raise ValueError(f"Expected BCHW or BCTHW input, got {video.shape}")
        if tuple(video.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {tuple(video.shape[-2:])}")
        if video.shape[2] != 2:
            raise ValueError(f"LAQ extraction expects frame pairs, got {video.shape[2]} frames")

        first_frame, last_frame = video[:, :, :1], video[:, :, 1:]
        first_tokens, last_tokens = self._encode(self._tokenize_video(first_frame, last_frame))
        batch_size = video.shape[0]
        first_tokens = first_tokens.reshape(batch_size, -1, first_tokens.shape[-1])
        last_tokens = last_tokens.reshape(batch_size, -1, last_tokens.shape[-1])
        actions, indices = self.vq(first_tokens, last_tokens)
        return actions, indices


class LatentActionQuantization(_LatentActionQuantizationBase):
    def __init__(self, **kwargs: Any) -> None:
        channels = kwargs.get("channels", 3)
        super().__init__(**kwargs)
        patch_height, patch_width = self.patch_size
        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange(
                "b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(
                channels * patch_width * patch_height,
                self.dim,
            ),
            nn.LayerNorm(self.dim),
        )

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        return torch.cat(
            (
                self.to_patch_emb_first_frame(first_frame),
                self.to_patch_emb_first_frame(last_frame),
            ),
            dim=1,
        )


class LatentActionQuantizationDinov2Feature(_LatentActionQuantizationBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dino_tokenizer = get_dino_tokenizer(device="cpu")

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        return torch.cat(
            (
                get_dino_reps(first_frame, self.dino_tokenizer),
                get_dino_reps(last_frame, self.dino_tokenizer),
            ),
            dim=1,
        )


class LatentActionQuantizationDinov3Feature(_LatentActionQuantizationBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dino_tokenizer = get_dinov3_tokenizer(device="cpu")

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        return torch.cat(
            (
                get_dinov3_reps(first_frame, self.dino_tokenizer),
                get_dinov3_reps(last_frame, self.dino_tokenizer),
            ),
            dim=1,
        )


class LatentActionQuantizationMagvit2(_LatentActionQuantizationBase):
    def __init__(self, *, model_type: str = "en18", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.magvit2_tokenizer = get_magvit2_tokenizer(model_type=model_type, device="cpu")

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        first_tokens = get_reps_magvit2(first_frame, self.magvit2_tokenizer)
        last_tokens = get_reps_magvit2(last_frame, self.magvit2_tokenizer)
        return torch.cat(
            (
                rearrange(first_tokens, "b d h w -> b 1 h w d"),
                rearrange(last_tokens, "b d h w -> b 1 h w d"),
            ),
            dim=1,
        )


class LatentActionQuantizationSiglipv2Feature(_LatentActionQuantizationBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder = get_siglip2_tokenizer(device="cpu")

    def _tokenize_video(self, first_frame: Tensor, last_frame: Tensor) -> Tensor:
        return torch.cat(
            (
                get_siglip2_reps(first_frame, self.encoder),
                get_siglip2_reps(last_frame, self.encoder),
            ),
            dim=1,
        )
