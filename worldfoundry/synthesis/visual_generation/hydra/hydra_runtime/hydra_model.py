"""HyDRA attention adapters for the native Wan video DiT.

HyDRA fine-tunes a regular Wan 2.1 DiT with three small additions per block:
camera encoders, a memory tokenizer, and dynamic retrieval attention.  Keeping
those additions as adapters lets HyDRA reuse the canonical Wan implementation
and checkpoint loader instead of carrying a private runtime fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.networks.wan import AttentionModule, WanModel
from worldfoundry.core.attention import (
    apply_complex_rotary_embedding as rope_apply,
    packed_sequence_attention as flash_attention,
)
from worldfoundry.core.kernels import layer_norm_scale_shift


@dataclass(frozen=True)
class HyDRAAttentionConfig:
    """Token-grid layout used by a HyDRA checkpoint."""

    num_frames: int = 40
    frame_height: int = 30
    frame_width: int = 52
    context_frames: int = 20
    top_k: int = 10
    window_size: int = 5
    frame_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.frame_height <= 0 or self.frame_width <= 0:
            raise ValueError("HyDRA token-grid dimensions must be positive")
        if not 0 < self.context_frames < self.num_frames:
            raise ValueError("HyDRA context_frames must be between zero and num_frames")
        if not 0 < self.window_size <= self.num_frames:
            raise ValueError("HyDRA window_size must be between one and num_frames")
        if self.top_k < 0 or self.window_size + self.top_k > self.num_frames:
            raise ValueError("HyDRA top_k must fit outside the local frame window")
        if self.frame_chunk_size is not None and self.frame_chunk_size <= 0:
            raise ValueError("HyDRA frame_chunk_size must be positive when provided")

    @property
    def frame_hw(self) -> int:
        return self.frame_height * self.frame_width

    @property
    def target_frames(self) -> int:
        return self.num_frames - self.context_frames


class MemoryTokenizer(nn.Module):
    """Compress spatiotemporal tokens before frame retrieval."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _precompute_axis_freqs(dim: int, positions: torch.Tensor, theta: float) -> torch.Tensor:
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=positions.device)[: dim // 2].float() / dim)
    )
    phases = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(phases), phases)


def compressed_freqs_cis_3d(
    dim: int,
    frames: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Build the compressed 3-D RoPE frequencies used by HyDRA memory tokens."""

    temporal_positions = torch.arange(frames, dtype=torch.float32, device=device) * 2 + 0.5
    height_positions = torch.arange(height, dtype=torch.float32, device=device) * 2
    width_positions = torch.arange(width, dtype=torch.float32, device=device) * 2

    temporal_dim = dim - 2 * (dim // 3)
    height_dim = dim // 3
    width_dim = dim // 3
    temporal = _precompute_axis_freqs(temporal_dim, temporal_positions, theta)
    vertical = _precompute_axis_freqs(height_dim, height_positions, theta)
    horizontal = _precompute_axis_freqs(width_dim, width_positions, theta)

    temporal = temporal.view(frames, 1, 1, -1).expand(frames, height, width, -1)
    vertical = vertical.view(1, height, 1, -1).expand(frames, height, width, -1)
    horizontal = horizontal.view(1, 1, width, -1).expand(frames, height, width, -1)
    return torch.cat([temporal, vertical, horizontal], dim=-1).reshape(frames * height * width, 1, -1)


class DynamicRetrievalAttention(nn.Module):
    """Attend to a local frame window plus top-k retrieved memory frames."""

    def __init__(
        self,
        num_heads: int,
        num_frames: int = 40,
        frame_hw: int = 30 * 52,
        top_k: int = 10,
        frame_chunk_size: int | None = None,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.frame_hw = frame_hw
        self.top_k = top_k
        self.window_size = window_size
        self.total_selected = window_size + top_k
        self.frame_chunk_size = frame_chunk_size

    def _frame_chunks(self) -> Iterable[range]:
        chunk_size = self.frame_chunk_size or self.num_frames
        for start in range(0, self.num_frames, chunk_size):
            yield range(start, min(start + chunk_size, self.num_frames))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        similarity: torch.Tensor,
        k_comp: torch.Tensor | None = None,
        v_comp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence_length, dim = q.shape
        frames, frame_hw, heads = self.num_frames, self.frame_hw, self.num_heads
        expected_length = frames * frame_hw
        if sequence_length != expected_length:
            raise ValueError(
                f"HyDRA attention expected {expected_length} tokens ({frames}x{frame_hw}), "
                f"got {sequence_length}"
            )

        q_frames = rearrange(q, "b (f hw) d -> b f hw d", f=frames, hw=frame_hw)
        k_frames = rearrange(k, "b (f hw) d -> b f hw d", f=frames, hw=frame_hw)
        v_frames = rearrange(v, "b (f hw) d -> b f hw d", f=frames, hw=frame_hw)
        q_frames_multihead = rearrange(q_frames, "b f hw (h d) -> b f h hw d", h=heads)

        local_indices = self._local_frame_indices(q.device)
        local_indices = local_indices.unsqueeze(0).expand(batch, -1, -1)
        output = torch.zeros(batch, frames, frame_hw, dim, device=q.device, dtype=q.dtype)

        if k_comp is not None and v_comp is not None:
            compressed_frames = int(similarity.shape[-1])
            compressed_frame_hw = k_comp.shape[1] // compressed_frames
            k_comp_frames = rearrange(
                k_comp,
                "b (f hw) d -> b f hw d",
                f=compressed_frames,
                hw=compressed_frame_hw,
            )
            v_comp_frames = rearrange(
                v_comp,
                "b (f hw) d -> b f hw d",
                f=compressed_frames,
                hw=compressed_frame_hw,
            )
            topk_indices = self._retrieved_compressed_indices(similarity, compressed_frames)
            for frame_indices in self._frame_chunks():
                self._process_mixed_frame_chunk(
                    output,
                    q_frames_multihead,
                    k_frames,
                    v_frames,
                    k_comp_frames,
                    v_comp_frames,
                    local_indices,
                    topk_indices,
                    frame_indices,
                    compressed_frame_hw,
                )
        else:
            mask = torch.zeros(batch, frames, frames, device=similarity.device, dtype=torch.bool)
            mask.scatter_(2, local_indices, True)
            retrieved = torch.topk(similarity.masked_fill(mask, -1e9), k=self.top_k, dim=-1).indices
            selected_indices = torch.cat([local_indices, retrieved], dim=-1)
            for frame_indices in self._frame_chunks():
                self._process_frame_chunk(
                    output,
                    q_frames_multihead,
                    k_frames,
                    v_frames,
                    selected_indices,
                    frame_indices,
                )

        return rearrange(output, "b f hw d -> b (f hw) d")

    def _local_frame_indices(self, device: torch.device) -> torch.Tensor:
        radius = self.window_size // 2
        indices = torch.arange(self.num_frames, device=device)
        starts = (indices - radius).clamp(min=0, max=self.num_frames - self.window_size)
        return starts.unsqueeze(1) + torch.arange(self.window_size, device=device).unsqueeze(0)

    def _retrieved_compressed_indices(
        self,
        similarity: torch.Tensor,
        compressed_frames: int,
    ) -> torch.Tensor:
        if self.top_k > compressed_frames:
            raise ValueError(
                f"HyDRA top_k={self.top_k} exceeds the {compressed_frames} compressed memory frames"
            )
        radius = self.window_size // 2
        frame_indices = torch.arange(self.num_frames, device=similarity.device)
        window_starts = (frame_indices - radius).clamp(min=0, max=self.num_frames - self.window_size)
        window_ends = window_starts + self.window_size - 1
        compressed_indices = torch.arange(compressed_frames, device=similarity.device)
        stride = max(1, round((self.num_frames - 2) / (compressed_frames - 1))) if compressed_frames > 1 else 1
        token_starts = compressed_indices * stride
        token_ends = token_starts + 1
        inside_window = (token_starts.unsqueeze(0) >= window_starts.unsqueeze(1)) & (
            token_ends.unsqueeze(0) <= window_ends.unsqueeze(1)
        )
        mask = inside_window.unsqueeze(0).expand(similarity.shape[0], -1, -1)
        return torch.topk(similarity.masked_fill(mask, -1e9), k=self.top_k, dim=-1).indices

    def _process_mixed_frame_chunk(
        self,
        output: torch.Tensor,
        q_frames: torch.Tensor,
        k_frames: torch.Tensor,
        v_frames: torch.Tensor,
        k_comp_frames: torch.Tensor,
        v_comp_frames: torch.Tensor,
        local_indices: torch.Tensor,
        topk_indices: torch.Tensor,
        frame_indices: Iterable[int],
        compressed_frame_hw: int,
    ) -> None:
        del compressed_frame_hw
        batch = output.shape[0]
        batch_indices_local = torch.arange(batch, device=k_frames.device).unsqueeze(1).expand(-1, self.window_size)
        batch_indices_topk = torch.arange(batch, device=k_frames.device).unsqueeze(1).expand(-1, self.top_k)
        for frame_index in frame_indices:
            current_local = local_indices[:, frame_index]
            k_local = k_frames[batch_indices_local, current_local]
            v_local = v_frames[batch_indices_local, current_local]
            current_topk = topk_indices[:, frame_index]
            k_retrieved = k_comp_frames[batch_indices_topk, current_topk]
            v_retrieved = v_comp_frames[batch_indices_topk, current_topk]
            k_all = torch.cat(
                [
                    rearrange(k_local, "b window hw d -> b (window hw) d"),
                    rearrange(k_retrieved, "b topk hw d -> b (topk hw) d"),
                ],
                dim=1,
            )
            v_all = torch.cat(
                [
                    rearrange(v_local, "b window hw d -> b (window hw) d"),
                    rearrange(v_retrieved, "b topk hw d -> b (topk hw) d"),
                ],
                dim=1,
            )
            query = rearrange(q_frames[:, frame_index], "b heads hw d -> b hw (heads d)")
            output[:, frame_index] = flash_attention(
                q=query,
                k=k_all,
                v=v_all,
                num_heads=self.num_heads,
            )

    def _process_frame_chunk(
        self,
        output: torch.Tensor,
        q_frames: torch.Tensor,
        k_frames: torch.Tensor,
        v_frames: torch.Tensor,
        selected_indices: torch.Tensor,
        frame_indices: Iterable[int],
    ) -> None:
        batch = output.shape[0]
        batch_indices = torch.arange(batch, device=k_frames.device).unsqueeze(1).expand(-1, self.total_selected)
        for frame_index in frame_indices:
            selected = selected_indices[:, frame_index]
            k_candidates = k_frames[batch_indices, selected]
            v_candidates = v_frames[batch_indices, selected]
            query = rearrange(q_frames[:, frame_index], "b heads hw d -> b hw (heads d)")
            keys = rearrange(k_candidates, "b selected hw d -> b (selected hw) d")
            values = rearrange(v_candidates, "b selected hw d -> b (selected hw) d")
            output[:, frame_index] = flash_attention(
                q=query,
                k=keys,
                v=values,
                num_heads=self.num_heads,
            )


def _hydra_self_attention_forward(self: nn.Module, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    config: HyDRAAttentionConfig = self.hydra_config
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    if self.hydra_enabled:
        expected_tokens = config.num_frames * config.frame_hw
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f"HyDRA expected a {config.num_frames}x{config.frame_height}x{config.frame_width} "
                f"token grid ({expected_tokens} tokens), got {x.shape[1]}"
            )
        x_3d = rearrange(
            x,
            "b (f h w) c -> b c f h w",
            f=config.num_frames,
            h=config.frame_height,
            w=config.frame_width,
        )
        compressed = self.tokenizer(x_3d)
        compressed_frames, compressed_height, compressed_width = compressed.shape[2:]
        compressed_flat = rearrange(compressed, "b c f h w -> b (f h w) c")
        k_comp = self.norm_k(self.k(compressed_flat))
        v_comp = self.v(compressed_flat)

        # Retrieval uses the unrotated frame descriptors, matching the
        # original HyDRA implementation. RoPE is applied only to attention Q/K.
        q_frame_repr = rearrange(
            q,
            "b (f h w) d -> b f h w d",
            f=config.num_frames,
            h=config.frame_height,
            w=config.frame_width,
        ).mean(dim=(2, 3))
        k_comp_repr = rearrange(
            k_comp,
            "b (f h w) d -> b f h w d",
            f=compressed_frames,
            h=compressed_height,
            w=compressed_width,
        ).mean(dim=(2, 3))
        similarity = torch.matmul(q_frame_repr, k_comp_repr.transpose(-2, -1))

        compressed_freqs = compressed_freqs_cis_3d(
            self.head_dim,
            compressed_frames,
            compressed_height,
            compressed_width,
            device=x.device,
        )
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        k_comp = rope_apply(k_comp, compressed_freqs, self.num_heads)
        output = self.attn(q, k, v, similarity, k_comp=k_comp, v_comp=v_comp)
    else:
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        output = self.attn(q, k, v)
    return self.o(output)


def _camera_condition(
    encoder: nn.Module,
    camera: torch.Tensor,
    *,
    expected_frames: int,
    height: int,
    width: int,
    label: str,
) -> torch.Tensor:
    if camera.ndim != 3 or camera.shape[1:] != (expected_frames, 12):
        raise ValueError(
            f"HyDRA {label} camera must have shape [batch, {expected_frames}, 12], got {tuple(camera.shape)}"
        )
    embedding = encoder(camera)
    return rearrange(embedding, "b f c -> b c f 1 1").expand(-1, -1, -1, height, width)


def _hydra_block_forward(
    self: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    *,
    cam_emb_tgt: torch.Tensor,
    cam_emb_con: torch.Tensor,
) -> torch.Tensor:
    config: HyDRAAttentionConfig = self.hydra_config
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=1)
    normalized = layer_norm_scale_shift(x, scale_msa, shift_msa, eps=self.norm1.eps)
    normalized = rearrange(
        normalized,
        "b (f h w) c -> b c f h w",
        f=config.num_frames,
        h=config.frame_height,
        w=config.frame_width,
    )
    context_camera = _camera_condition(
        self.cam_encoder_con,
        cam_emb_con,
        expected_frames=config.context_frames,
        height=config.frame_height,
        width=config.frame_width,
        label="context",
    )
    target_camera = _camera_condition(
        self.cam_encoder_tgt,
        cam_emb_tgt,
        expected_frames=config.target_frames,
        height=config.frame_height,
        width=config.frame_width,
        label="target",
    )
    normalized = torch.cat(
        [
            normalized[:, :, : config.context_frames] + context_camera,
            normalized[:, :, config.context_frames :] + target_camera,
        ],
        dim=2,
    )
    normalized = rearrange(normalized, "b c f h w -> b (f h w) c").contiguous()
    x = self.gate(x, gate_msa, self.projector(self.self_attn(normalized, freqs)))
    x = x + self.cross_attn(self.norm3(x), context)
    normalized = layer_norm_scale_shift(x, scale_mlp, shift_mlp, eps=self.norm2.eps)
    return self.gate(x, gate_mlp, self.ffn(normalized))


def _new_linear_like(reference: torch.Tensor, in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(
        in_features,
        out_features,
        device=reference.device,
        dtype=reference.dtype,
    )


def configure_hydra_model(
    dit: WanModel,
    *,
    enabled: bool = True,
    config: HyDRAAttentionConfig | None = None,
) -> WanModel:
    """Install HyDRA's trainable adapters and forwards on a shared Wan DiT.

    The adapter names intentionally match the original HyDRA checkpoint, so a
    fine-tuned state dict can still be loaded with ``strict=True``.
    Reconfiguring an already adapted model preserves its trained parameters.
    """

    config = config or HyDRAAttentionConfig()
    if not getattr(dit, "blocks", None):
        raise ValueError("HyDRA requires a Wan DiT with at least one transformer block")

    for block in dit.blocks:
        dim = block.dim
        reference = block.self_attn.q.weight
        if not isinstance(getattr(block.self_attn, "tokenizer", None), MemoryTokenizer):
            block.self_attn.tokenizer = MemoryTokenizer(dim).to(device=reference.device, dtype=reference.dtype)
        if not isinstance(getattr(block, "cam_encoder_con", None), nn.Linear):
            block.cam_encoder_con = _new_linear_like(reference, 12, dim)
            nn.init.zeros_(block.cam_encoder_con.weight)
            nn.init.zeros_(block.cam_encoder_con.bias)
        if not isinstance(getattr(block, "cam_encoder_tgt", None), nn.Linear):
            block.cam_encoder_tgt = _new_linear_like(reference, 12, dim)
            nn.init.zeros_(block.cam_encoder_tgt.weight)
            nn.init.zeros_(block.cam_encoder_tgt.bias)
        if not isinstance(getattr(block, "projector", None), nn.Linear):
            block.projector = _new_linear_like(reference, dim, dim)
            nn.init.eye_(block.projector.weight)
            nn.init.zeros_(block.projector.bias)

        block.hydra_config = config
        block.self_attn.hydra_config = config
        block.self_attn.hydra_enabled = bool(enabled)
        # Retain the upstream attribute names for checkpoint/runtime tooling.
        block.self_attn.hydra = bool(enabled)
        block.self_attn.change_sparse = bool(enabled)
        block.self_attn.attention_type = "sparse_frame" if enabled else "standard"
        block.self_attn.attn = (
            DynamicRetrievalAttention(
                num_heads=block.self_attn.num_heads,
                num_frames=config.num_frames,
                frame_hw=config.frame_hw,
                top_k=config.top_k,
                frame_chunk_size=config.frame_chunk_size,
                window_size=config.window_size,
            )
            if enabled
            else AttentionModule(block.self_attn.num_heads)
        )
        block.self_attn.forward = MethodType(_hydra_self_attention_forward, block.self_attn)
        block.forward = MethodType(_hydra_block_forward, block)

    dit.hydra_config = config
    dit.hydra_enabled = bool(enabled)
    return dit


__all__ = [
    "DynamicRetrievalAttention",
    "HyDRAAttentionConfig",
    "MemoryTokenizer",
    "compressed_freqs_cis_3d",
    "configure_hydra_model",
]
