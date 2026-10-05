# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CausalKVCache:
    num_layers: int
    action_history_frames: int
    local_attn_size: int = -1
    sink_size: int = 0
    frames_per_block: int = 4
    self_k: list[torch.Tensor] | None = None
    self_v: list[torch.Tensor] | None = None
    cross_k: list[torch.Tensor] | None = None
    cross_v: list[torch.Tensor] | None = None
    positions: torch.Tensor | None = None
    block_ids: torch.Tensor | None = None
    action_history: torch.Tensor | None = None
    active_start: int | None = None
    reserved_length: int | None = None
    first_block_id: int | None = None
    pinned_block_id: int = -1
    pending_pin_block_id: int = -1

    def __post_init__(self) -> None:
        if self.local_attn_size == -1:
            if self.sink_size != 0:
                raise ValueError("sink_size requires a local attention window")
            return
        local_blocks, sink_blocks = self._window_geometry()
        if self.local_attn_size < 1 or self.sink_size < 0 or local_blocks - sink_blocks < 2:
            raise ValueError("local attention window is invalid")

    @property
    def is_cold(self) -> bool:
        return self.positions is None

    def _window_geometry(self) -> tuple[int, int]:
        local_blocks = (self.local_attn_size + self.frames_per_block - 1) // self.frames_per_block
        sink_blocks = (self.sink_size + self.frames_per_block - 1) // self.frames_per_block
        return local_blocks, sink_blocks

    def _visible_indices(self, query_block_id: int, local_end: int) -> torch.Tensor | None:
        if self.local_attn_size == -1 or local_end == 0:
            return None
        local_blocks, sink_blocks = self._window_geometry()
        capacity_without_pin = local_blocks - sink_blocks
        pin_is_distinct = self.pinned_block_id >= 0 and query_block_id - self.pinned_block_id >= capacity_without_pin
        local_budget = capacity_without_pin - int(pin_is_distinct)
        cached = self.block_ids[:local_end]
        first = self.first_block_id if self.first_block_id is not None else int(cached[0])
        keep = (
            (cached - first < sink_blocks) | (cached == self.pinned_block_id) | (query_block_id - cached < local_budget)
        )
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        return None if int(indices.numel()) == local_end else indices

    def _prune_for_query(self, query_block_id: int) -> None:
        if self.positions is None or self.active_start is not None:
            return
        local_end = int(self.positions.shape[0])
        indices = self._visible_indices(query_block_id, local_end)
        if indices is None:
            return
        self._release_unused_cuda_memory()
        for collection in (self.self_k, self.self_v):
            for index, value in enumerate(collection):
                collection[index] = value[:, indices].contiguous()
        self.positions = self.positions[indices].contiguous()
        self.block_ids = self.block_ids[indices].contiguous()

    def _release_unused_cuda_memory(self) -> None:
        if self.self_k is None or not self.self_k[0].is_cuda:
            return
        free, total = torch.cuda.mem_get_info(self.self_k[0].device)
        if free < total // 10:
            torch.cuda.empty_cache()

    def history(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.self_k is None:
            return None, None
        end = self.active_start if self.active_start is not None else int(self.positions.shape[0])
        return self.self_k[layer][:, :end], self.self_v[layer][:, :end]

    def cross(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cross_k is None:
            return None, None
        return self.cross_k[layer], self.cross_v[layer]

    def reset_cross(self) -> None:
        self.cross_k = None
        self.cross_v = None

    def reserve(self, new_length: int) -> None:
        if self.local_attn_size != -1 or self.self_k is None or self.active_start is not None:
            return
        old_length = int(self.positions.shape[0])
        if self.reserved_length is not None:
            if self.reserved_length != old_length + new_length:
                raise ValueError("cache reservation length changed")
            return
        total = old_length + int(new_length)
        for collection in (self.self_k, self.self_v):
            for index, tensor in enumerate(collection):
                expanded = tensor.new_empty((tensor.shape[0], total, tensor.shape[2], tensor.shape[3]))
                expanded[:, :old_length].copy_(tensor[:, :old_length])
                collection[index] = expanded
        self.reserved_length = total

    def prepare(
        self,
        grid_sizes: torch.Tensor,
        device: torch.device,
        action: torch.Tensor | None,
        prompt_switch: bool,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        if prompt_switch:
            self.reset_cross()
        frames, height, width = (int(value) for value in grid_sizes[0].tolist())
        sequence = frames * height * width
        action_window = action
        if action is not None and self.action_history is not None and self.action_history_frames:
            action_window = torch.cat((self.action_history[:, -self.action_history_frames :], action), dim=1)
        if self.active_start is not None:
            stop = int(self.positions.shape[0])
            if stop - self.active_start != sequence:
                raise ValueError("active cache block shape changed")
            block_id = int(self.block_ids[self.active_start])
            return self.positions[self.active_start : stop], block_id, action_window
        start_frame = 0 if self.positions is None else int(self.positions[:, 0].max()) + 1
        start_block = 0 if self.block_ids is None else int(self.block_ids.max()) + 1
        if self.first_block_id is None:
            self.first_block_id = start_block
        if self.local_attn_size != -1:
            local_blocks, sink_blocks = self._window_geometry()
            if prompt_switch:
                self.pending_pin_block_id = start_block if start_block - self.first_block_id >= sink_blocks else -1
            threshold = local_blocks - sink_blocks - int(self.pinned_block_id >= 0)
            if self.pending_pin_block_id >= 0 and start_block - self.pending_pin_block_id >= threshold:
                self.pinned_block_id = self.pending_pin_block_id
                self.pending_pin_block_id = -1
            self._prune_for_query(start_block)
        temporal = (torch.arange(frames, device=device) + start_frame).view(frames, 1, 1).expand(frames, height, width)
        vertical = torch.arange(height, device=device).view(1, height, 1).expand(frames, height, width)
        horizontal = torch.arange(width, device=device).view(1, 1, width).expand(frames, height, width)
        positions = torch.stack((temporal.reshape(-1), vertical.reshape(-1), horizontal.reshape(-1)), dim=1).long()
        return positions, start_block, action_window

    def attention_positions(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positions is None:
            return current, current
        end = self.active_start if self.active_start is not None else int(self.positions.shape[0])
        return current, torch.cat((self.positions[:end], current), dim=0)

    def update(
        self,
        new_self: list[tuple[torch.Tensor, torch.Tensor]],
        new_cross: list[tuple[torch.Tensor, torch.Tensor] | None],
        positions: torch.Tensor,
        block_id: int,
        mode: str | None,
        action_current: torch.Tensor | None,
    ) -> None:
        if mode not in (None, "active", "final"):
            raise ValueError("cache mode must be active, final, or None")
        if self.cross_k is None and new_cross and new_cross[0] is not None:
            self.cross_k = [pair[0] for pair in new_cross]
            self.cross_v = [pair[1] for pair in new_cross]
        if mode is None:
            return
        length = int(positions.shape[0])
        current_block_ids = torch.full((length,), block_id, device=positions.device, dtype=torch.int32)
        if self.active_start is not None:
            stop = int(self.positions.shape[0])
            if stop - self.active_start != length or not torch.equal(
                self.block_ids[self.active_start : stop], current_block_ids
            ):
                raise ValueError("active cache block changed")
            for index, (key, value) in enumerate(new_self):
                self.self_k[index][:, self.active_start : stop].copy_(key)
                self.self_v[index][:, self.active_start : stop].copy_(value)
        elif self.positions is None:
            self.self_k = [key for key, _ in new_self]
            self.self_v = [value for _, value in new_self]
            self.positions = positions
            self.block_ids = current_block_ids
            self.first_block_id = block_id
        else:
            old_length = int(self.positions.shape[0])
            self.positions = torch.cat((self.positions, positions), dim=0)
            self.block_ids = torch.cat((self.block_ids, current_block_ids), dim=0)
            if self.reserved_length is not None:
                if self.reserved_length != old_length + length:
                    raise ValueError("cache reservation does not match the block")
                for index, (key, value) in enumerate(new_self):
                    self.self_k[index][:, old_length : old_length + length].copy_(key)
                    self.self_v[index][:, old_length : old_length + length].copy_(value)
                self.reserved_length = None
            else:
                self._release_unused_cuda_memory()
                for index, (key, _) in enumerate(new_self):
                    self.self_k[index] = torch.cat((self.self_k[index][:, :old_length], key), dim=1)
                for index, (_, value) in enumerate(new_self):
                    self.self_v[index] = torch.cat((self.self_v[index][:, :old_length], value), dim=1)
        if mode == "active":
            self.active_start = int(self.positions.shape[0]) - length
        else:
            self.active_start = None
            if action_current is not None:
                current = action_current.detach()
                self.action_history = (
                    current if self.action_history is None else torch.cat((self.action_history, current), dim=1)
                )
