"""HelixWorld's checkpoint-visible additions to the shared LTX AV graph.

Adapted from NoizAI/HelixWorld, revision 3fa5f166834b08ea20ddc925c01f85d936ac03b5.
Code and model terms are recorded in THIRD-PARTY-NOTICES.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from worldfoundry.core.attention.cache.context import prepend_history_mask
from worldfoundry.core.attention.rotary.projective_rope import (
    apply_token_projective_matrix,
    invert_k,
    invert_se3,
    lift_k,
)
from worldfoundry.core.nn.diffusion.timestep import TimestepEmbedding, get_timestep_embedding

from .attention import Attention
from .model import LTXModel
from .transformer import BasicAVTransformerBlock


@dataclass(frozen=True)
class VideoControlCondition:
    """Normalized per-token camera matrices and action IDs for the video stream."""

    camera_intrinsics: torch.Tensor
    camera_w2c: torch.Tensor
    camera_valid_mask: torch.Tensor
    action_ids: torch.Tensor
    action_valid_mask: torch.Tensor
    camera_projection: torch.Tensor | None = None
    camera_projection_inverse: torch.Tensor | None = None

    @property
    def has_action(self) -> bool:
        return True

    def token_slice(self, start: int, end: int) -> VideoControlCondition:
        return replace(self, **{
            item.name: getattr(self, item.name)[:, start:end]
            for item in fields(self) if getattr(self, item.name) is not None
        })

    def split(self, sizes: list[int]) -> list[VideoControlCondition]:
        chunks = {
            item.name: getattr(self, item.name).split(sizes)
            for item in fields(self) if getattr(self, item.name) is not None
        }
        return [replace(self, **{name: values[index] for name, values in chunks.items()}) for index in range(len(sizes))]

    def with_projective_matrices(self) -> VideoControlCondition:
        valid = self.camera_valid_mask[..., None, None]
        intrinsics = torch.zeros_like(self.camera_intrinsics, dtype=torch.float32)
        intrinsics[..., 0, 0] = self.camera_intrinsics[..., 0, 0]
        intrinsics[..., 1, 1] = self.camera_intrinsics[..., 1, 1]
        intrinsics[..., 2, 2] = 1
        intrinsics = torch.where(valid, intrinsics, torch.eye(3, device=intrinsics.device))
        w2c = torch.where(valid, self.camera_w2c.float(), torch.eye(4, device=intrinsics.device))
        return replace(
            self,
            camera_projection=lift_k(intrinsics) @ w2c,
            camera_projection_inverse=invert_se3(w2c) @ lift_k(invert_k(intrinsics)),
        )


class ActionTimestepEmbedder(torch.nn.Module):
    """Checkpoint-compatible 256-dimensional action sinusoid and SiLU MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = TimestepEmbedding(256, hidden_dim, hidden_dim)

    def forward(self, action_ids, action_valid_mask, *, hidden_dtype):
        projected = get_timestep_embedding(
            action_ids.flatten(), 256, flip_sin_to_cos=True, downscale_freq_shift=0,
        )
        embedded = self.mlp(projected.to(hidden_dtype)).view(*action_ids.shape, -1)
        return embedded * action_valid_mask.to(embedded.dtype).unsqueeze(-1)


class HelixWorldAttention(Attention):
    """LTX self-attention plus the released full-rank camera PRoPE residual."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.camera_to_out = torch.nn.Linear(self.heads * self.dim_head, self.to_out[0].out_features)

    def _camera_attention(self, q, k, v, x, control, mask, perturbation_mask, cache):
        if control.camera_projection is None:
            control = control.with_projective_matrices()
        projection, inverse = control.camera_projection, control.camera_projection_inverse
        q = apply_token_projective_matrix(q, projection.transpose(-1, -2), self.heads)
        k = apply_token_projective_matrix(k, inverse, self.heads)
        v = apply_token_projective_matrix(v, inverse, self.heads)
        key_mask = control.camera_valid_mask
        if cache is not None:
            k, v, key_mask, history_tokens = cache.combine((self, "camera"), k, v, key_mask)
            mask = prepend_history_mask(mask, history_tokens)
        key_bias = q.new_zeros((q.shape[0], 1, 1, k.shape[1]))
        key_bias.masked_fill_(~key_mask[:, None, None], torch.finfo(q.dtype).min)
        out = self.masked_attention_function(q, k, v, self.heads, key_bias if mask is None else mask + key_bias)
        out = apply_token_projective_matrix(out, projection, self.heads)
        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)
        out = self.camera_to_out(out) * control.camera_valid_mask.to(out.dtype).unsqueeze(-1)
        return out if perturbation_mask is None else out * perturbation_mask


class HelixWorldTransformerBlock(BasicAVTransformerBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, video_attention_class=HelixWorldAttention, **kwargs)


class HelixWorldModel(LTXModel):
    """Reuse every LTX stem and block operation with the released control layers."""

    TRANSFORMER_BLOCK_CLS = HelixWorldTransformerBlock

    def _init_video(self, *args, **kwargs) -> None:
        super()._init_video(*args, **kwargs)
        self.video_action_embedder = ActionTimestepEmbedder(self.inner_dim)
