# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License.
"""Inference-only episodic memory bank used by Mem-0."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


class PositionEmbedder(nn.Module):
    """Embed relative positions where one denotes the newest observation."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def position_embedding(
        positions: torch.Tensor,
        dimension: int,
        max_period: int = 30,
    ) -> torch.Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=positions.device) / half
        )
        angles = positions[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dimension % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        parameter = next(self.mlp.parameters())
        positions = positions.to(parameter.device)
        frequencies = self.position_embedding(positions, self.frequency_embedding_size)
        return self.mlp(frequencies.to(parameter.dtype))


class MemoryBank(nn.Module):
    """Two-stage recent-window and initial-anchor memory fusion."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        window_size: int,
        initial_anchor_size: int,
        num_heads: int,
        dropout: float,
        memory_accumulation: int,
        frequency_embedding_size: int | None = None,
    ) -> None:
        super().__init__()
        if initial_anchor_size != 1:
            raise ValueError("Mem-0 supports exactly one initial anchor")
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.initial_anchor_size = initial_anchor_size
        self.num_heads = num_heads
        self.memory_accumulation = memory_accumulation
        frequency_dim = frequency_embedding_size or hidden_dim // 4
        self.window_position_encoder = PositionEmbedder(hidden_dim, frequency_dim)
        self.cross_attn1 = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.cross_attn2 = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.bank: dict[Any, dict[str, list[torch.Tensor]]] = {}
        self.end_signal_count: dict[Any, int] = {}

    @staticmethod
    def _episode_key(episode_id: Any) -> Any:
        if isinstance(episode_id, torch.Tensor) and episode_id.numel() == 1:
            return int(episode_id.item())
        if isinstance(episode_id, np.generic):
            return int(episode_id)
        return episode_id

    def reset(self, episode_ids: list[Any] | None = None) -> None:
        if episode_ids is None:
            self.bank.clear()
            self.end_signal_count.clear()
            return
        for episode_id in episode_ids:
            key = self._episode_key(episode_id)
            self.bank.pop(key, None)
            self.end_signal_count.pop(key, None)

    def clear_episode(self, episode_id: Any) -> None:
        self.reset([episode_id])

    def _get_memory_tensor(self, episode_id: Any) -> tuple[torch.Tensor, int, int] | None:
        entry = self.bank.get(self._episode_key(episode_id))
        if not entry:
            return None
        anchors = entry.get("anchors", [])
        window = entry.get("window", [])
        tensors = []
        if anchors:
            tensors.append(torch.stack(anchors, dim=0).squeeze(1))
        if window:
            tensors.append(torch.stack(window, dim=0).squeeze(1))
        if not tensors:
            return None
        return torch.cat(tensors, dim=0), len(anchors), len(window)

    def _upd_to_window(self, episode_id: Any, vector: torch.Tensor) -> None:
        key = self._episode_key(episode_id)
        entry = self.bank.setdefault(key, {"anchors": [], "window": []})
        entry["window"].append(vector.detach().clone())
        if len(entry["window"]) > self.window_size:
            entry["window"].pop(0)

    def _upd_to_anchor(self, episode_id: Any, vector: torch.Tensor) -> None:
        key = self._episode_key(episode_id)
        entry = self.bank.setdefault(key, {"anchors": [], "window": []})
        if not entry["anchors"]:
            entry["anchors"] = [vector.detach().clone()]

    def _fuse(
        self,
        current_vector: torch.Tensor,
        memory: tuple[torch.Tensor, int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = current_vector.unsqueeze(1)
        if memory is None:
            return query, query
        memory_tensor, anchor_length, window_length = memory
        memory_tensor = memory_tensor.to(
            device=current_vector.device,
            dtype=current_vector.dtype,
        )
        if window_length:
            window = memory_tensor[anchor_length:]
            query_norm = self.norm1(query.squeeze(1)).unsqueeze(1)
            positions = torch.arange(
                window_length,
                0,
                -1,
                device=current_vector.device,
            )
            position_embedding = self.window_position_encoder(positions).to(window.dtype)
            window_norm = self.norm1(window + position_embedding)
            attended, _ = self.cross_attn1(
                query_norm,
                window_norm.unsqueeze(1),
                window_norm.unsqueeze(1),
                need_weights=False,
            )
            fused_window = attended + query
        else:
            fused_window = query
        if anchor_length:
            anchors = memory_tensor[:anchor_length]
            query_norm = self.norm2(query.squeeze(1)).unsqueeze(1)
            anchors_norm = self.norm2(anchors)
            attended, _ = self.cross_attn2(
                query_norm,
                anchors_norm.unsqueeze(1),
                anchors_norm.unsqueeze(1),
                need_weights=False,
            )
            fused_anchor = attended + query
        else:
            fused_anchor = query
        return fused_window, fused_anchor

    @torch.inference_mode()
    def update_on_eval(
        self,
        new_vector: torch.Tensor,
        text_vector: torch.Tensor,
        classifier: Any,
        episode_id: Any,
        *,
        classifier_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if new_vector.shape != (1, 1, self.hidden_dim):
            raise ValueError(
                f"Mem-0 memory input must have shape (1, 1, {self.hidden_dim}), received {tuple(new_vector.shape)}"
            )
        key = self._episode_key(episode_id)
        current_vector = new_vector[0]
        fused_window, fused_anchor = self._fuse(
            current_vector,
            self._get_memory_tensor(key),
        )
        self._upd_to_window(key, current_vector)
        self._upd_to_anchor(key, current_vector)
        summary = torch.cat([fused_window, fused_anchor, text_vector], dim=2)
        probability = classifier.predict(summary)["prob"]
        subtask_ended = bool((probability >= classifier_threshold).item())
        self.end_signal_count.setdefault(key, 0)
        if subtask_ended:
            self.end_signal_count[key] += 1
        return fused_window, fused_anchor, subtask_ended

    def get_memory_size(self, episode_id: Any) -> int:
        entry = self.bank.get(self._episode_key(episode_id), {})
        return len(entry.get("anchors", [])) + len(entry.get("window", []))


__all__ = ["MemoryBank", "PositionEmbedder"]
