"""ABot-World inference-only causal Wan 2.2 backbone.

Causal self-attention, relative / reference RoPE, and an action adapter
on the Wan 2.2 official stem.  Adds action + causal hooks.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from torch import nn

from worldfoundry.core.attention.sequence.varlen import (
    attention,
    flash_attention,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_22 import (
    WanLayerNorm,
    WanRMSNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d,
)


class _AffineWanLayerNorm(WanLayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            value.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).type_as(value)


@torch.amp.autocast("cuda", enabled=False)
def _causal_rope_apply(
    value: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    start_frame: int = 0,
) -> torch.Tensor:
    heads = value.size(2)
    rotary_dim = value.size(3) // 2
    bands = freqs.split(
        [rotary_dim - 2 * (rotary_dim // 3), rotary_dim // 3, rotary_dim // 3],
        dim=1,
    )
    output = []
    for batch_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        sequence_length = frames * height * width
        sample = torch.view_as_complex(
            value[batch_index, :sequence_length]
            .float()
            .reshape(sequence_length, heads, -1, 2)
        )
        sample_freqs = torch.cat(
            [
                bands[0][start_frame : start_frame + frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                bands[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                bands[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(sequence_length, 1, -1)
        sample = torch.view_as_real(sample * sample_freqs).flatten(2)
        output.append(torch.cat([sample, value[batch_index, sequence_length:]]))
    return torch.stack(output).to(value.dtype)


@torch.amp.autocast("cuda", enabled=False)
def _relative_rope_apply(
    value: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    frame_indices: torch.Tensor,
    maximum_frame_index: int,
) -> torch.Tensor:
    heads = value.size(2)
    rotary_dim = value.size(3) // 2
    bands = freqs.split(
        [rotary_dim - 2 * (rotary_dim // 3), rotary_dim // 3, rotary_dim // 3],
        dim=1,
    )
    output = []
    for batch_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        sequence_length = frames * height * width
        sample = torch.view_as_complex(
            value[batch_index, :sequence_length]
            .float()
            .reshape(sequence_length, heads, -1, 2)
        )
        temporal_indices = frame_indices[:frames].to(value.device, torch.long)
        temporal_indices = temporal_indices.clamp(0, maximum_frame_index)
        sample_freqs = torch.cat(
            [
                bands[0][temporal_indices]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                bands[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                bands[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(sequence_length, 1, -1)
        sample = torch.view_as_real(sample * sample_freqs).flatten(2)
        output.append(torch.cat([sample, value[batch_index, sequence_length:]]))
    return torch.stack(output).to(value.dtype)


def _frequencies_at(
    indices: list[int],
    dim: int,
    *,
    device: torch.device,
    theta: float = 10000.0,
) -> torch.Tensor:
    positions = torch.tensor(indices, dtype=torch.float64, device=device)
    phases = torch.outer(
        positions,
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2, dtype=torch.float64, device=device).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(phases), phases)


def _reference_frequencies(
    freqs: torch.Tensor,
    *,
    num_slots: int,
    tokens_per_slot: int,
    grid: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    patch_frames, patch_height, patch_width = grid
    freq_dim = int(freqs.shape[1])
    frame_band = freq_dim - 2 * (freq_dim // 3)
    height_band = width_band = freq_dim // 3
    temporal_step = max(tokens_per_slot, 256)
    temporal_positions = [-(num_slots - index) * temporal_step for index in range(num_slots)]
    temporal = _frequencies_at(
        temporal_positions,
        2 * frame_band,
        device=device,
    )
    bands = freqs.split([frame_band, height_band, width_band], dim=1)
    result = torch.cat(
        [
            temporal[:, None, None, None, :].expand(
                num_slots,
                patch_frames,
                patch_height,
                patch_width,
                frame_band,
            ),
            bands[1][:patch_height]
            .to(device)[None, None, :, None, :]
            .expand(
                num_slots,
                patch_frames,
                patch_height,
                patch_width,
                height_band,
            ),
            bands[2][:patch_width]
            .to(device)[None, None, None, :, :]
            .expand(
                num_slots,
                patch_frames,
                patch_height,
                patch_width,
                width_band,
            ),
        ],
        dim=-1,
    )
    return result.reshape(num_slots * tokens_per_slot, 1, -1).to(torch.complex64)


def _apply_reference_rope(
    value: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    complex_value = torch.view_as_complex(
        value.float().reshape(*value.shape[:3], -1, 2)
    )
    output = torch.view_as_real(
        complex_value * freqs.to(value.device, torch.complex64)
    ).flatten(3)
    return output.to(value.dtype)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(value))) + value


class _ActionAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        downscale_factor: int,
    ) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv = nn.Conv2d(
            in_dim * downscale_factor * downscale_factor,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.residual_blocks = nn.Sequential(_ResidualBlock(out_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            channels,
            height,
            width,
        )
        value = self.residual_blocks(self.conv(self.pixel_unshuffle(value)))
        value = value.view(batch, frames, *value.shape[1:])
        return value.permute(0, 2, 1, 3, 4)


class _CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        local_attn_size: int,
        sink_size: int,
        qk_norm: bool,
        eps: float,
        use_relative_rope: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_relative_rope = use_relative_rope
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.reference_token_len = 0
        self.query_reference_token_len = 0
        self.reference_num_slots = 0
        self.reference_tokens_per_slot = 0
        self.reference_grid: tuple[int, int, int] | None = None

    def set_reference_layout(
        self,
        *,
        token_len: int,
        query_token_len: int,
        num_slots: int,
        tokens_per_slot: int,
        grid: tuple[int, int, int] | None,
    ) -> None:
        self.reference_token_len = token_len
        self.query_reference_token_len = query_token_len
        self.reference_num_slots = num_slots
        self.reference_tokens_per_slot = tokens_per_slot
        self.reference_grid = grid

    def _reference_rope(self, value: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if self.reference_grid is None:
            raise RuntimeError("reference RoPE layout is missing")
        reference_freqs = _reference_frequencies(
            freqs,
            num_slots=self.reference_num_slots,
            tokens_per_slot=self.reference_tokens_per_slot,
            grid=self.reference_grid,
            device=value.device,
        )
        return _apply_reference_rope(value, reference_freqs)

    @staticmethod
    def _cache_index(cache: dict[str, Any], name: str) -> int:
        value = cache[name]
        return int(value.item()) if torch.is_tensor(value) else int(value)

    @staticmethod
    def _set_cache_index(cache: dict[str, Any], name: str, value: int) -> None:
        current = cache[name]
        if torch.is_tensor(current):
            current.fill_(value)
        else:
            cache[name] = value

    def _roll_cache(
        self,
        cache: dict[str, Any],
        *,
        key_name: str,
        value_name: str,
        sink_tokens: int,
        new_tokens: int,
        cache_current_end: int,
    ) -> tuple[int, int]:
        cache_size = int(cache[value_name].shape[1])
        global_end = self._cache_index(cache, "global_end_index")
        local_end = self._cache_index(cache, "local_end_index")
        if (
            self.local_attn_size != -1
            and cache_current_end > global_end
            and new_tokens + local_end > cache_size
        ):
            evicted = new_tokens + local_end - cache_size
            rolled = local_end - evicted - sink_tokens
            if evicted < 0 or rolled < 0:
                raise RuntimeError(
                    f"invalid ABot-World KV roll: evicted={evicted}, rolled={rolled}"
                )
            cache[key_name][:, sink_tokens : sink_tokens + rolled] = cache[key_name][
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
            cache[value_name][:, sink_tokens : sink_tokens + rolled] = cache[value_name][
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
            local_end = local_end + cache_current_end - global_end - evicted
        else:
            local_end = local_end + cache_current_end - global_end
        return local_end - new_tokens, local_end

    def _relative_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        cache: dict[str, Any],
        current_start: int,
        frame_sequence_length: int,
    ) -> torch.Tensor:
        reference_len = self.reference_token_len
        query_reference_len = self.query_reference_token_len
        video_token_len = q.shape[1] - query_reference_len
        num_video_frames = video_token_len // frame_sequence_length
        video_grid = grid_sizes.clone()
        video_grid[:, 0] = num_video_frames
        sink_tokens = reference_len + self.sink_size * frame_sequence_length
        cache_current_end = reference_len + current_start + video_token_len
        if "k_raw" not in cache or cache["k_raw"].shape != cache["v"].shape:
            cache["k_raw"] = torch.empty_like(cache["v"])
        local_start, local_end = self._roll_cache(
            cache,
            key_name="k_raw",
            value_name="v",
            sink_tokens=sink_tokens,
            new_tokens=video_token_len,
            cache_current_end=cache_current_end,
        )
        if local_start < sink_tokens or local_end > cache["k_raw"].shape[1]:
            raise RuntimeError(
                f"ABot-World relative KV write [{local_start}:{local_end}] is out of range"
            )
        if query_reference_len:
            cache["k_raw"][:, :reference_len] = k[:, :query_reference_len]
            cache["v"][:, :reference_len] = v[:, :query_reference_len]
        cache["k_raw"][:, local_start:local_end] = k[:, query_reference_len:]
        cache["v"][:, local_start:local_end] = v[:, query_reference_len:]

        maximum_video_tokens = (
            local_end - sink_tokens
            if self.local_attn_size == -1
            else self.local_attn_size * frame_sequence_length
        )
        recent_start = max(sink_tokens, local_end - maximum_video_tokens)
        visible_tokens = local_end - recent_start
        remainder = visible_tokens % frame_sequence_length
        if remainder:
            recent_start += remainder
            visible_tokens -= remainder

        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        query_parts: list[torch.Tensor] = []
        if reference_len:
            key_parts.append(self._reference_rope(cache["k_raw"][:, :reference_len], freqs))
            value_parts.append(cache["v"][:, :reference_len])
            if query_reference_len:
                query_parts.append(self._reference_rope(q[:, :query_reference_len], freqs))

        protected_video_tokens = max(0, sink_tokens - reference_len)
        raw_video_parts = []
        video_value_parts = []
        if protected_video_tokens:
            raw_video_parts.append(cache["k_raw"][:, reference_len:sink_tokens])
            video_value_parts.append(cache["v"][:, reference_len:sink_tokens])
        if visible_tokens:
            raw_video_parts.append(cache["k_raw"][:, recent_start:local_end])
            video_value_parts.append(cache["v"][:, recent_start:local_end])
        if not raw_video_parts:
            raise RuntimeError("ABot-World attention has no visible video tokens")
        raw_video = torch.cat(raw_video_parts, dim=1)
        video_values = torch.cat(video_value_parts, dim=1)
        visible_frames = raw_video.shape[1] // frame_sequence_length
        visible_grid = grid_sizes.clone()
        visible_grid[:, 0] = visible_frames
        key_indices = torch.arange(visible_frames, device=q.device)
        maximum_index = max(0, self.local_attn_size - 1) if self.local_attn_size != -1 else 1023
        key_parts.append(
            _relative_rope_apply(
                raw_video,
                visible_grid,
                freqs,
                frame_indices=key_indices,
                maximum_frame_index=maximum_index,
            )
        )
        value_parts.append(video_values)
        if num_video_frames <= visible_frames:
            query_indices = key_indices[-num_video_frames:]
        else:
            query_indices = torch.arange(num_video_frames, device=q.device)
        query_parts.append(
            _relative_rope_apply(
                q[:, query_reference_len:],
                video_grid,
                freqs,
                frame_indices=query_indices,
                maximum_frame_index=maximum_index,
            )
        )
        self._set_cache_index(cache, "global_end_index", cache_current_end)
        self._set_cache_index(cache, "local_end_index", local_end)
        return attention(
            torch.cat(query_parts, dim=1),
            torch.cat(key_parts, dim=1),
            torch.cat(value_parts, dim=1),
        )

    def _absolute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        cache: dict[str, Any],
        current_start: int,
        frame_sequence_length: int,
    ) -> torch.Tensor:
        reference_len = self.reference_token_len
        query_reference_len = self.query_reference_token_len
        video_token_len = q.shape[1] - query_reference_len
        num_video_frames = video_token_len // frame_sequence_length
        video_grid = grid_sizes.clone()
        video_grid[:, 0] = num_video_frames
        start_frame = current_start // frame_sequence_length
        if query_reference_len:
            roped_q = torch.cat(
                [
                    self._reference_rope(q[:, :query_reference_len], freqs),
                    _causal_rope_apply(
                        q[:, query_reference_len:],
                        video_grid,
                        freqs,
                        start_frame=start_frame,
                    ),
                ],
                dim=1,
            )
            roped_k = torch.cat(
                [
                    self._reference_rope(k[:, :query_reference_len], freqs),
                    _causal_rope_apply(
                        k[:, query_reference_len:],
                        video_grid,
                        freqs,
                        start_frame=start_frame,
                    ),
                ],
                dim=1,
            )
        else:
            roped_q = _causal_rope_apply(q, video_grid, freqs, start_frame=start_frame)
            roped_k = _causal_rope_apply(k, video_grid, freqs, start_frame=start_frame)
        sink_tokens = reference_len + self.sink_size * frame_sequence_length
        cache_current_end = reference_len + current_start + video_token_len
        local_start, local_end = self._roll_cache(
            cache,
            key_name="k",
            value_name="v",
            sink_tokens=sink_tokens,
            new_tokens=roped_q.shape[1],
            cache_current_end=cache_current_end,
        )
        cache["k"][:, local_start:local_end] = roped_k
        cache["v"][:, local_start:local_end] = v
        maximum_tokens = (
            local_end if self.local_attn_size == -1 else self.local_attn_size * frame_sequence_length
        )
        recent_start = max(sink_tokens, local_end - maximum_tokens)
        if sink_tokens and recent_start > sink_tokens:
            visible_k = torch.cat(
                [cache["k"][:, :sink_tokens], cache["k"][:, recent_start:local_end]],
                dim=1,
            )
            visible_v = torch.cat(
                [cache["v"][:, :sink_tokens], cache["v"][:, recent_start:local_end]],
                dim=1,
            )
        else:
            visible_k = cache["k"][:, :local_end]
            visible_v = cache["v"][:, :local_end]
        self._set_cache_index(cache, "global_end_index", cache_current_end)
        self._set_cache_index(cache, "local_end_index", local_end)
        return attention(roped_q, visible_k, visible_v)

    def forward(
        self,
        value: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        *,
        kv_cache: dict[str, Any] | None,
        current_start: int,
    ) -> torch.Tensor:
        del seq_lens
        if kv_cache is None:
            raise RuntimeError("ABot-World is integrated for KV-cache inference only")
        batch, sequence_length = value.shape[:2]
        q = self.norm_q(self.q(value)).view(
            batch,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        k = self.norm_k(self.k(value)).view_as(q)
        v = self.v(value).view_as(q)
        frame_sequence_length = math.prod(grid_sizes[0][1:].tolist())
        if self.use_relative_rope:
            output = self._relative_attention(
                q,
                k,
                v,
                grid_sizes=grid_sizes,
                freqs=freqs,
                cache=kv_cache,
                current_start=current_start,
                frame_sequence_length=frame_sequence_length,
            )
        else:
            output = self._absolute_attention(
                q,
                k,
                v,
                grid_sizes=grid_sizes,
                freqs=freqs,
                cache=kv_cache,
                current_start=current_start,
                frame_sequence_length=frame_sequence_length,
            )
        return self.o(output.flatten(2))


class _CrossAttention(WanSelfAttention):
    def forward(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        *,
        cache: dict[str, Any] | None,
    ) -> torch.Tensor:
        batch = value.shape[0]
        q = self.norm_q(self.q(value)).view(
            batch,
            -1,
            self.num_heads,
            self.head_dim,
        )
        initialized = False
        if cache is not None:
            marker = cache.get("is_init", False)
            initialized = bool(marker.item()) if torch.is_tensor(marker) else bool(marker)
        if cache is not None and initialized:
            k, v = cache["k"], cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(
                batch,
                -1,
                self.num_heads,
                self.head_dim,
            )
            v = self.v(context).view_as(k)
            if cache is not None:
                cache["k"], cache["v"] = k, v
                marker = cache.get("is_init", False)
                if torch.is_tensor(marker):
                    marker.fill_(True)
                else:
                    cache["is_init"] = True
        return self.o(flash_attention(q, k, v, k_lens=context_lens).flatten(2))


class _CausalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        *,
        local_attn_size: int,
        sink_size: int,
        qk_norm: bool,
        cross_attn_norm: bool,
        eps: float,
        use_relative_rope: bool,
    ) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = _CausalSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            use_relative_rope=use_relative_rope,
        )
        self.norm3 = (
            _AffineWanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = _CrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        value: torch.Tensor,
        modulation: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        kv_cache: dict[str, Any],
        crossattn_cache: dict[str, Any] | None,
        current_start: int,
    ) -> torch.Tensor:
        token_level = modulation.shape[1] == value.shape[1]
        frames = modulation.shape[1]
        frame_sequence_length = 1 if token_level else value.shape[1] // frames
        chunks = (self.modulation.unsqueeze(0) + modulation).chunk(6, dim=2)

        def modulate(norm: nn.Module, tensor: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            tensor = norm(tensor)
            if token_level:
                return tensor * (1 + scale.squeeze(2)) + shift.squeeze(2)
            tensor = tensor.unflatten(1, (frames, frame_sequence_length))
            return (tensor * (1 + scale) + shift).flatten(1, 2)

        def gate(tensor: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
            if token_level:
                return tensor * factor.squeeze(2)
            tensor = tensor.unflatten(1, (frames, frame_sequence_length))
            return (tensor * factor).flatten(1, 2)

        residual = self.self_attn(
            modulate(self.norm1, value, chunks[1], chunks[0]),
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache=kv_cache,
            current_start=current_start,
        )
        value = value + gate(residual, chunks[2])
        value = value + self.cross_attn(
            self.norm3(value),
            context,
            context_lens,
            cache=crossattn_cache,
        )
        residual = self.ffn(
            modulate(self.norm2, value, chunks[4], chunks[3])
        )
        return value + gate(residual, chunks[5])


class _CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float) -> None:
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        frames = embedding.shape[1]
        frame_sequence_length = value.shape[1] // frames
        shift, scale = (self.modulation.unsqueeze(1) + embedding).chunk(2, dim=2)
        value = self.norm(value).unflatten(1, (frames, frame_sequence_length))
        return self.head(value * (1 + scale) + shift)


class ABotWorldModel(nn.Module, NativeConfigMixin):
    """ABot-World causal DiT with action and reference-view conditioning."""

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["_CausalBlock"]

    @register_to_config
    def __init__(
        self,
        model_type: str = "ci2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 48,
        dim: int = 3072,
        ffn_dim: int = 14336,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 48,
        num_heads: int = 24,
        num_layers: int = 30,
        window_size: tuple[int, int] = (-1, -1),
        local_attn_size: int = 21,
        num_frame_per_block: int = 3,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        act_control_in_dim: int = 32,
        use_relative_rope: bool = True,
        downscale_factor_control_adapter: int = 16,
    ) -> None:
        super().__init__()
        if model_type != "ci2v":
            raise ValueError("The in-tree ABot-World checkpoint supports model_type='ci2v' only")
        self.model_type = model_type
        self.patch_size = tuple(patch_size)
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        self.sink_size = sink_size
        self.use_relative_rope = use_relative_rope

        self.patch_embedding = nn.Conv3d(
            in_dim,
            dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                _CausalBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    use_relative_rope=use_relative_rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = _CausalHead(dim, out_dim, self.patch_size, eps)
        self.act_control_adapter = _ActionAdapter(
            act_control_in_dim,
            dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
            downscale_factor=downscale_factor_control_adapter,
        )
        head_dim = dim // num_heads
        # Keep RoPE frequencies outside the module buffers: Module.to(dtype=...)
        # casts complex buffers to a real dtype and discards their phase.
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

    def _apply_actions(
        self,
        features: list[torch.Tensor],
        actions: torch.Tensor | None,
        scale: float,
    ) -> list[torch.Tensor]:
        if actions is None:
            return features
        action_features = [
            self.act_control_adapter(sample.unsqueeze(0)) for sample in actions
        ]
        output = []
        for feature, action in zip(features, action_features):
            if feature.shape[2] > action.shape[2]:
                offset = feature.shape[2] - action.shape[2]
                feature = torch.cat(
                    [feature[:, :, :offset], feature[:, :, offset:] + action * scale],
                    dim=2,
                )
            else:
                feature = feature + action * scale
            output.append(feature)
        return output

    def _reference_tokens(
        self,
        reference_latents: torch.Tensor | None,
        reference_mask: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Any] | None:
        if reference_latents is None:
            return None
        reference_latents = reference_latents.detach().to(device=device, dtype=dtype)
        if reference_latents.ndim == 4:
            reference_latents = reference_latents.unsqueeze(0).unsqueeze(3)
        elif reference_latents.ndim == 5 and reference_latents.shape[2] == self.in_dim:
            reference_latents = reference_latents.unsqueeze(3)
        elif reference_latents.ndim == 5:
            reference_latents = reference_latents.unsqueeze(0)
        if reference_latents.ndim != 6:
            raise ValueError("reference_latents must have [B,K,C,T,H,W] layout")
        batch, slots, channels, frames, height, width = reference_latents.shape
        if batch == 1 and batch_size != 1:
            reference_latents = reference_latents.expand(batch_size, -1, -1, -1, -1, -1)
            batch = batch_size
        if batch != batch_size:
            raise ValueError("reference latent batch does not match video batch")
        if reference_mask is None:
            reference_mask = reference_latents.new_ones(batch, slots)
        else:
            reference_mask = reference_mask.detach().to(device=device, dtype=dtype)
            if reference_mask.ndim == 1:
                reference_mask = reference_mask.unsqueeze(0)
            if reference_mask.shape[0] == 1 and batch != 1:
                reference_mask = reference_mask.expand(batch, -1)
        flattened = reference_latents.reshape(
            batch * slots,
            channels,
            frames,
            height,
            width,
        )
        if channels < self.in_dim:
            flattened = torch.cat(
                [
                    flattened,
                    flattened.new_zeros(
                        batch * slots,
                        self.in_dim - channels,
                        frames,
                        height,
                        width,
                    ),
                ],
                dim=1,
            )
        elif channels > self.in_dim:
            flattened = flattened[:, : self.in_dim]
        features = self.patch_embedding(flattened)
        grid = tuple(int(value) for value in features.shape[2:])
        tokens_per_slot = math.prod(grid)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, slots, tokens_per_slot, self.dim)
        tokens = tokens * reference_mask[:, :, None, None]
        return {
            "tokens": tokens.reshape(batch, slots * tokens_per_slot, self.dim),
            "num_slots": slots,
            "tokens_per_slot": tokens_per_slot,
            "token_len": slots * tokens_per_slot,
            "grid": grid,
        }

    def _zero_reference_modulation(
        self,
        batch_size: int,
        token_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        timestep = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype)
        )
        embedding = self.time_projection(embedding).unflatten(1, (6, self.dim))
        embedding = embedding.unflatten(0, timestep.shape)
        return embedding.expand(-1, token_len, -1, -1)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: list[torch.Tensor] | torch.Tensor,
        seq_len: int | None = None,
        *,
        kv_cache: list[dict[str, Any]] | None,
        crossattn_cache: list[dict[str, Any]] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        act_context: torch.Tensor | None = None,
        act_context_scale: float = 1.0,
        ref_latents: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del cache_start
        if kv_cache is None:
            raise RuntimeError("ABot-World supports inference with a KV cache only")
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        features = [self.patch_embedding(sample.unsqueeze(0)) for sample in x]
        features = self._apply_actions(features, act_context, act_context_scale)
        grid_sizes = torch.stack(
            [
                torch.tensor(feature.shape[2:], dtype=torch.long, device=device)
                for feature in features
            ]
        )
        tokens = [feature.flatten(2).transpose(1, 2) for feature in features]
        seq_lens = torch.tensor(
            [value.shape[1] for value in tokens],
            dtype=torch.long,
            device=device,
        )
        if seq_len is not None and int(seq_lens.max()) > int(seq_len):
            raise ValueError("ABot-World sequence exceeds configured seq_len")
        tokens_tensor = torch.cat(tokens, dim=0)

        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(tokens_tensor.dtype)
        )
        modulation = self.time_projection(time_embedding).unflatten(1, (6, self.dim))
        modulation = modulation.unflatten(0, t.shape)
        if torch.is_tensor(context):
            context_values = list(context)
        else:
            context_values = context
        padded_context = []
        for value in context_values:
            value = value[: self.text_len]
            if value.shape[0] < self.text_len:
                value = torch.cat(
                    [
                        value,
                        value.new_zeros(self.text_len - value.shape[0], value.shape[1]),
                    ]
                )
            padded_context.append(value)
        context_tensor = self.text_embedding(torch.stack(padded_context))

        reference = self._reference_tokens(
            ref_latents,
            ref_mask,
            batch_size=tokens_tensor.shape[0],
            dtype=tokens_tensor.dtype,
            device=device,
        )
        reference_len = reference["token_len"] if reference else 0
        query_reference_len = reference_len if reference and current_start == 0 else 0
        if query_reference_len:
            frame_sequence_length = math.prod(grid_sizes[0][1:].tolist())
            modulation = modulation.repeat_interleave(frame_sequence_length, dim=1)
            reference_modulation = self._zero_reference_modulation(
                tokens_tensor.shape[0],
                query_reference_len,
                dtype=tokens_tensor.dtype,
                device=device,
            )
            tokens_tensor = torch.cat([reference["tokens"], tokens_tensor], dim=1)
            modulation = torch.cat([reference_modulation, modulation], dim=1)

        for block in self.blocks:
            block.self_attn.set_reference_layout(
                token_len=reference_len,
                query_token_len=query_reference_len,
                num_slots=reference["num_slots"] if reference else 0,
                tokens_per_slot=reference["tokens_per_slot"] if reference else 0,
                grid=reference["grid"] if reference else None,
            )
        for index, block in enumerate(self.blocks):
            tokens_tensor = block(
                tokens_tensor,
                modulation,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context_tensor,
                context_lens=None,
                kv_cache=kv_cache[index],
                crossattn_cache=(
                    crossattn_cache[index] if crossattn_cache is not None else None
                ),
                current_start=current_start,
            )
        if query_reference_len:
            tokens_tensor = tokens_tensor[:, query_reference_len:]
        output = self.head(
            tokens_tensor,
            time_embedding.unflatten(0, t.shape).unsqueeze(2),
        )
        return torch.stack(self._unpatchify(output, grid_sizes))

    def _unpatchify(
        self,
        value: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        output = []
        for sample, grid in zip(value, grid_sizes.tolist()):
            sample = sample[: math.prod(grid)].view(
                *grid,
                *self.patch_size,
                self.out_dim,
            )
            sample = torch.einsum("fhwpqrc->cfphqwr", sample)
            sample = sample.reshape(
                self.out_dim,
                *[left * right for left, right in zip(grid, self.patch_size)],
            )
            output.append(sample)
        return output


CausalWanModel = ABotWorldModel

__all__ = ["ABotWorldModel", "CausalWanModel"]
