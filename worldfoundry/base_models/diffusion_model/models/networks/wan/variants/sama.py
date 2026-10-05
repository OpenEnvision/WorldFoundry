"""SAMA (Wan 2.x): joint video + SigLIP semantic-token denoising.

The host :class:`WanModel` stem is unchanged.  Optional source-video
tokens and projected SigLIP semantic tokens are prepended in
:meth:`prepare_token_sequence`, tagged with a 3-way segment embedding
(source=0, semantic=1, target=2).  Semantic tokens use zero RoPE.
:meth:`finalize_token_sequence` strips prefixes and emits a semantic
diffusion head prediction.

No action, camera, VACE, causal, or linear-attention hooks live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..model import WanModel


class SigLIPFeatureProjection(nn.Module):
    """Project the released 1152-wide semantic state into Wan token space."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class SemanticDiffusionHead(nn.Module):
    """Map semantic hidden tokens back to their diffusion-state width."""

    def __init__(self, dim: int, out_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x.to(dtype=self.norm.weight.dtype)))


@dataclass(slots=True)
class _SamaTokenState:
    """Counts used to split concatenated source / semantic / target tokens."""

    target_tokens: int
    source_tokens: int
    semantic_tokens: int


class SamaWanModel(WanModel):
    """Wan variant jointly denoising video and SAMA semantic tokens."""

    def __init__(
        self,
        *args,
        semantic_dim: int = 1152,
        semantic_hidden_dim: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_dim = int(semantic_dim)
        self.siglip_feat_mlp = SigLIPFeatureProjection(
            self.semantic_dim,
            self.dim,
            hidden_dim=semantic_hidden_dim,
        )
        self.semantic_head = SemanticDiffusionHead(self.dim, self.semantic_dim)
        self.segment_embedding = nn.Embedding(3, self.dim)

    def prepare_token_sequence(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        t: torch.Tensor,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ):
        """Concat source (optional), semantic (optional), then target video tokens."""
        del grid_size
        source_latents = kwargs.get("source_latents")
        semantic_latents = kwargs.get("semantic_latents")
        segments: list[torch.Tensor] = []
        frequency_segments: list[torch.Tensor] = []
        type_segments: list[torch.Tensor] = []
        batch = x.shape[0]

        source_count = 0
        if source_latents is not None:
            source_latents = source_latents.to(device=x.device, dtype=x.dtype)
            source, source_grid = self.patchify(source_latents)
            source_count = source.shape[1]
            segments.append(source)
            frequency_segments.append(self.rotary_frequencies(source_grid, device=x.device))
            type_segments.append(torch.zeros((batch, source_count), dtype=torch.long, device=x.device))

        semantic_count = 0
        if semantic_latents is not None:
            semantic = self.siglip_feat_mlp(semantic_latents.to(device=x.device, dtype=x.dtype))
            semantic_count = semantic.shape[1]
            segments.append(semantic)
            frequency_segments.append(
                torch.zeros(
                    (semantic_count, 1, freqs.shape[-1]),
                    device=freqs.device,
                    dtype=freqs.dtype,
                )
            )
            type_segments.append(torch.ones((batch, semantic_count), dtype=torch.long, device=x.device))

        target_count = x.shape[1]
        segments.append(x)
        frequency_segments.append(freqs)
        type_segments.append(torch.full((batch, target_count), 2, dtype=torch.long, device=x.device))
        tokens = torch.cat(segments, dim=1)
        tokens = tokens + self.segment_embedding(torch.cat(type_segments, dim=1))
        return (
            tokens,
            torch.cat(frequency_segments, dim=0),
            t_mod,
            t,
            _SamaTokenState(target_count, source_count, semantic_count),
        )

    def finalize_token_sequence(self, x: torch.Tensor, token_state: _SamaTokenState, **kwargs: Any):
        """Return target tokens plus an optional semantic-head prediction."""
        del kwargs
        semantic_prediction = None
        if token_state.semantic_tokens:
            begin = token_state.source_tokens
            end = begin + token_state.semantic_tokens
            semantic_prediction = self.semantic_head(x[:, begin:end])
        return x[:, -token_state.target_tokens :], semantic_prediction


__all__ = [
    "SamaWanModel",
    "SemanticDiffusionHead",
    "SigLIPFeatureProjection",
]
