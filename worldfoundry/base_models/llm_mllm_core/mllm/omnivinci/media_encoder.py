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


from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from einops import repeat
from torch import nn

from worldfoundry.core.attention.rotary.interleaved import rotate_pairs as rotate_half


def pool(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """Pool a tensor axis, padding a partial group with its scalar mean."""
    if x.shape[dim] % size != 0:
        remainder = x.shape[dim] % size
        pad_len = size - remainder
        last_elements = x.narrow(dim, x.shape[dim] - remainder, remainder)
        mean_value = last_elements.mean()
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        padding = torch.ones(pad_shape, device=x.device, dtype=x.dtype) * mean_value
        x = torch.cat([x, padding], dim=dim)
    shape_before = x.shape[:dim]
    shape_after = x.shape[dim + 1 :]
    new_shape = (*shape_before, -1, size, *shape_after)
    return x.view(new_shape).mean(dim + 1)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Preserve OmniVinci's FP64 temporal rotation and interleaved feature pairs."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ori_dtype = t.dtype
        t = t.to(torch.float64)
        if t.ndim == 3:
            seq_len = t.shape[-2]
            freqs = freqs[-seq_len:].to(t)
        rot_dim = freqs.shape[-1]
        assert rot_dim <= t.shape[-1]
        t_left, t, t_right = t[..., :0], t[..., :rot_dim], t[..., rot_dim:]
        t = (t * freqs.cos() * 1.0) + (rotate_half(t) * freqs.sin() * 1.0)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


class RotaryEmbedding(nn.Module):
    """Language-frequency axial RoPE used by the released temporal encoders."""

    def __init__(self, dim: int, theta: float = 10000) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("cached_freqs", None, persistent=False)

    def forward(self, t: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.cached_freqs is not None and seq_len <= self.cached_freqs.shape[0]:
            return self.cached_freqs[:seq_len].detach()
        freqs = torch.einsum("..., f -> ... f", t.type(self.freqs.dtype), self.freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        self.cached_freqs = freqs.detach()
        return freqs

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        all_freqs = []
        for index, dim in enumerate(dims):
            positions = torch.arange(dim, device=self.freqs.device)
            freqs = self.forward(positions, seq_len=dim)
            axes = [None] * len(dims)
            axes[index] = slice(None)
            all_freqs.append(freqs[(Ellipsis, *axes, slice(None))])
        return torch.cat(torch.broadcast_tensors(*all_freqs), dim=-1)


class BaseEncoder(nn.Module):
    def __init__(self, parent: nn.Module, start_tokens: str | None = None, end_tokens: str | None = "\n") -> None:
        super().__init__()
        # The parent is not a child module: encoders must not register its weights again.
        self._parent = [parent]
        self.start_tokens = start_tokens
        self.end_tokens = None if end_tokens == "None" else end_tokens

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]

    def embed_tokens(self, tokens: str | None) -> torch.Tensor | None:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm_model_embed_tokens(token_ids)


class BasicImageEncoder(BaseEncoder):
    def _process_features(
        self, features: torch.Tensor, start_token_embeds: torch.Tensor | None, end_token_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, images: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        features = self.parent.encode_images(torch.stack(images, dim=0), block_sizes=config.get("block_sizes"))
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(feature) for feature in features]


class BasicSoundEncoder(BaseEncoder):
    def __init__(
        self,
        parent: nn.Module,
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
        embed_time: str = "True",
        trope_theta: float = 50000,
        trope_dim: int = 128,
        time_embed_type: str = "lang",
    ) -> None:
        super().__init__(parent, start_tokens, end_tokens)
        if embed_time != "True" or time_embed_type != "lang":
            raise ValueError("OmniVinci requires the released language-frequency temporal sound encoder")
        # The released sound path uses RotaryEmbedding's 10000 default, not the video theta.
        self.pos_emb = RotaryEmbedding(trope_dim)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        times: torch.Tensor,
    ) -> torch.Tensor:
        features = features.to(self.parent.device)
        device = features.device
        times = times.unsqueeze(0)
        pos_emb = self.pos_emb.to(device)
        angle = (-times * 2 * np.pi).to(device)
        freqs = pos_emb.get_axial_freqs(times.shape[0], features.shape[-2]).to(device)
        angle_expanded = angle.unsqueeze(2).expand(times.shape[0], features.shape[-2], freqs.shape[-1])
        freqs = (freqs * angle_expanded).squeeze(0)
        features = apply_rotary_emb(freqs, features)
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, sounds: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        features = self.parent.encode_sound(sounds)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        new_features = []
        audio_index = 0
        for audio_info in mm_info["audio_info"]:
            if audio_info is not None:
                for info in audio_info:
                    feature = features[audio_index]
                    sec_per_embed = info["new_audio_chunk_length"] / feature.shape[0]
                    times = [
                        info["audio_start_sec"] + i * sec_per_embed + sec_per_embed / 2 for i in range(feature.shape[0])
                    ]
                    times = torch.tensor(times).to(features[0].device)
                    new_features.append(process_features(feature, times=times))
                    audio_index += 1
        assert audio_index == len(features)
        return new_features


class TSPVideoEncoder(BaseEncoder):
    def __init__(
        self,
        parent: nn.Module,
        pool_sizes: list[tuple[int, int, int]],
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
        sep_tokens: str | None = None,
        embed_time: str = "True",
        trope_theta: float = 50000,
        trope_dim: int = 128,
        time_embed_type: str = "lang",
    ) -> None:
        super().__init__(parent, start_tokens, end_tokens)
        if embed_time != "True" or time_embed_type != "lang":
            raise ValueError("OmniVinci requires the released language-frequency temporal video encoder")
        self.pool_sizes = pool_sizes
        self.sep_tokens = sep_tokens
        self.pos_emb = RotaryEmbedding(trope_dim, theta=trope_theta)

    def _process_features(
        self,
        inputs: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        sep_token_embeds: torch.Tensor | None,
        times: torch.Tensor,
    ) -> torch.Tensor:
        nt, ns = inputs.shape[:2]
        nl = int(ns**0.5)
        outputs = []
        for pool_size in self.pool_sizes:
            features = inputs.view(nt, nl, nl, -1)
            for dim, size in enumerate(pool_size):
                features = pool(features, size, dim)
            features = features.flatten(1, 2)
            temporal_pool_size = pool_size[0]
            if temporal_pool_size != 1:
                if len(times) % temporal_pool_size != 0:
                    remainder = len(times) % temporal_pool_size
                    pad_len = temporal_pool_size - remainder
                    last_window_mean_times = times[-remainder:].mean()
                    times = torch.cat([times, torch.ones(pad_len).to(times.device) * last_window_mean_times])
                new_times = pool(times, temporal_pool_size, 0)
            else:
                new_times = times
            pos_emb = self.pos_emb.to(features.device)
            angle = (-new_times * 2 * np.pi).to(features.device)
            freqs = pos_emb.get_axial_freqs(new_times.shape[0], features.shape[-2]).to(features.device)
            angle_expanded = angle.unsqueeze(1).unsqueeze(2)
            angle_expanded = angle_expanded.expand(new_times.shape[0], features.shape[-2], freqs.shape[-1])
            features = apply_rotary_emb(freqs * angle_expanded, features)
            if start_token_embeds is not None:
                start_embeds = torch.stack([start_token_embeds] * features.shape[0], dim=0)
                features = torch.cat([start_embeds, features], dim=1)
            if end_token_embeds is not None:
                end_embeds = torch.stack([end_token_embeds] * features.shape[0], dim=0)
                features = torch.cat([features, end_embeds], dim=1)
            features = features.flatten(0, 1)
            if sep_token_embeds is not None:
                features = torch.cat([features, sep_token_embeds], dim=0)
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def forward(self, videos: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        num_frames = [video.shape[0] for video in videos]
        features = self.parent.encode_video(videos)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
            sep_token_embeds=self.embed_tokens(self.sep_tokens),
        )
        new_features = []
        video_index = 0
        for video_info in mm_info["video_info"]:
            if video_info is not None:
                for info in video_info:
                    times = torch.tensor(info["video_frame_times"]).to(features[0].device)
                    new_features.append(process_features(features[video_index], times=times))
                    video_index += 1
        assert video_index == len(features)
        return new_features
