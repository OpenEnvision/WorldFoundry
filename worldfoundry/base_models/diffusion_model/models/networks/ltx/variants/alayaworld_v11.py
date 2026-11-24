# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Alaya's action extension and checkpoint names on the native video-only LTX graph.

Adapted from AlayaLab/AlayaWorld and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from typing import Any, cast

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel, LTXModelType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.text_projection import PixArtAlphaTextProjection


class ActionAdaLNEmbedder(nn.Module):
    """Embed scaled latent-rate delta-6D camera actions into the transformer hidden dim."""

    def __init__(
        self,
        dim: int,
        action_dim: int = 6,
        subframes: int = 8,
        freq_dim_per_axis: int = 32,
        freq_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.subframes = subframes
        self.freq_dim_per_axis = freq_dim_per_axis
        self.freq_scale = float(freq_scale)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim * freq_dim_per_axis, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
        orig_shape = position.shape
        pos_flat = position.flatten().to(torch.float64)
        freqs = torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(max(dim // 2, 1)),
        )
        sinusoid = torch.outer(pos_flat, freqs)
        emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
        return emb.view(*orig_shape, dim).to(position.dtype)

    def forward(self, action_vectors: torch.Tensor) -> torch.Tensor:
        if action_vectors.dim() == 2:
            action_vectors = action_vectors.unsqueeze(0)
        if action_vectors.dim() != 3:
            raise ValueError(f"ActionAdaLNEmbedder expected [B,T,6], got {tuple(action_vectors.shape)}")
        if action_vectors.shape[-1] != self.action_dim:
            raise ValueError(f"ActionAdaLNEmbedder expected {self.action_dim} dims, got {action_vectors.shape[-1]}")
        embeddings = [
            self._sinusoidal_embedding(self.freq_dim_per_axis, action_vectors[..., axis] * self.freq_scale)
            for axis in range(self.action_dim)
        ]
        combined = torch.cat(embeddings, dim=-1)
        return self.mlp(combined.to(self.mlp[0].weight.dtype))


class AlayaWorldLTXModel(LTXModel):
    """Reuse native stems, heads and blocks with Alaya's optional action AdaLN branch."""

    def __init__(
        self,
        *,
        caption_channels: int = 3840,
        caption_proj_before_connector: bool = False,
        enable_action_control: bool = False,
        action_freq_dim: int = 32,
        action_freq_scale: float = 1000.0,
        action_subframes: int = 8,
        **kwargs: Any,
    ) -> None:
        self.caption_channels = caption_channels
        self.caption_proj_before_connector = caption_proj_before_connector
        self.enable_action_control = enable_action_control
        self.action_freq_dim = action_freq_dim
        self.action_freq_scale = action_freq_scale
        self.action_subframes = action_subframes
        super().__init__(model_type=LTXModelType.VideoOnly, **kwargs)

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        """Keep native access while registering only the released ``blocks.*`` keys."""
        return cast(nn.ModuleList, self._modules["blocks"])

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: nn.Module | None = None,
    ) -> None:
        super()._init_video(in_channels, out_channels, norm_eps, caption_projection)
        self.caption_projection = (
            None
            if self.caption_proj_before_connector
            else PixArtAlphaTextProjection(in_features=self.caption_channels, hidden_size=self.inner_dim)
        )
        # Preserve the module traversal order used by Alaya's init_weights.
        for name in ("norm_out", "proj_out"):
            self.add_module(name, self._modules.pop(name))

    def _init_transformer_blocks(self, *args: Any, **kwargs: Any) -> None:
        if self.enable_action_control:
            self.action_adaln_embedder = ActionAdaLNEmbedder(
                dim=self.inner_dim,
                action_dim=6,
                subframes=self.action_subframes,
                freq_dim_per_axis=self.action_freq_dim,
                freq_scale=self.action_freq_scale,
            )
            self.action_adaln_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(self.inner_dim, self._adaln_embedding_coefficient * self.inner_dim, bias=True)
            )
            nn.init.normal_(self.action_adaln_projection[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.action_adaln_projection[-1].bias)
            print(
                f"[LTX23-ActionAdaLN] Enabled: action_dim=6, subframes={self.action_subframes}, "
                f"freq_dim_per_axis={self.action_freq_dim}, freq_scale={self.action_freq_scale}, "
                f"adaln_coeff={self._adaln_embedding_coefficient}"
            )
        super()._init_transformer_blocks(*args, **kwargs)
        self.blocks = cast(nn.ModuleList, self._modules.pop("transformer_blocks"))
