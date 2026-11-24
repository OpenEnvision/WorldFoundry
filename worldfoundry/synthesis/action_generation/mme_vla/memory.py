"""Online history buffers used by MME-VLA inference."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from copy import deepcopy

import einops
import jax
import jax.numpy as jnp
import numpy as np

from ..openpi import image_tensor as image_tools
from .memory_utils import (
    even_sampling_indices,
    left_padding_token_emb,
    pool_tokens_to_size,
    right_padding_token_emb,
)
from .modeling.positional import PosEmb3D


class MemoryBuffer:
    """Incrementally encode observations for perceptual-memory inference."""

    def __init__(
        self,
        *,
        vision_enc_fn: Callable,
        num_views: int = 1,
        img_emb_dim: int = 2048,
        pos_emb_dim: int = 768,
        state_emb_dim: int = 8,
        compute_token_drop_score: bool = False,
        token_drop_keptsize: int = 2048,
        token_drop_stride: int = 8,
        pool_type: str = "mean",
    ) -> None:
        if num_views < 1:
            raise ValueError("num_views must be positive")
        if token_drop_stride < 1:
            raise ValueError("token_drop_stride must be positive")
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim
        self.pool_type = pool_type
        self.token_drop_keptsize = token_drop_keptsize
        self.token_drop_stride = token_drop_stride
        self.token_drop_last_frame = -1
        self.scored_token_heap: list[tuple[float, int, int, int]] = []
        self.compute_token_drop_score = compute_token_drop_score
        self._history_feats: dict[int, dict[str, np.ndarray]] = {}
        self.vision_enc = vision_enc_fn
        # Position encodings are generated only for frames actually added. The
        # upstream eager 4096-frame tables consumed gigabytes during every reset.
        self._position_encoder = PosEmb3D(dim=pos_emb_dim)

    @property
    def has_history(self) -> bool:
        return bool(self._history_feats)

    def _validate_inputs(
        self,
        images: np.ndarray,
        states: np.ndarray,
        step_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        images = np.asarray(images)
        states = np.asarray(states, dtype=np.float32)
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"Expected history images [time, view, height, width, 3], got {images.shape}")
        if images.shape[1] != self.num_views:
            raise ValueError(f"Expected {self.num_views} history views, got {images.shape[1]}")
        if states.ndim != 2 or states.shape[0] != images.shape[0]:
            raise ValueError(f"History states must be [time, dim], got {states.shape}")
        if states.shape[-1] != self.state_emb_dim:
            raise ValueError(f"Expected state width {self.state_emb_dim}, got {states.shape[-1]}")
        if len(step_indices) != images.shape[0]:
            raise ValueError("step_indices must contain one index per history frame")
        if any(index < 0 for index in step_indices):
            raise ValueError("History step indices must be non-negative")
        if np.issubdtype(images.dtype, np.floating):
            if not np.isfinite(images).all():
                raise ValueError("History images contain non-finite values")
            images = (images + 1.0) * 127.5 if images.min() < 0 else images * 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)
        return images, states

    def _encode_images(self, images: np.ndarray) -> np.ndarray:
        time_steps, views = images.shape[:2]
        image_jax = jnp.asarray(images, dtype=jnp.float32) / 127.5 - 1.0
        image_jax = einops.rearrange(image_jax, "t v h w c -> (t v) h w c")
        image_jax = image_tools.resize_with_pad(image_jax, 224, 224)
        image_jax = einops.rearrange(
            image_jax,
            "(t v) h w c -> t v h w c",
            t=time_steps,
            v=views,
        )
        return self.vision_enc(image_jax)

    def _positions(self, step_indices: Sequence[int], spatial_size: int) -> np.ndarray:
        repeated = np.repeat(np.asarray(step_indices, dtype=np.int32), self.num_views)
        encoded = self._position_encoder(jnp.asarray(repeated), spatial_size)
        return np.asarray(jax.device_get(encoded)).reshape(
            len(step_indices),
            self.num_views,
            spatial_size * spatial_size,
            self.pos_emb_dim,
        )

    def add_buffer(self, images, states, step_idx_list: Sequence[int]) -> None:
        images, states = self._validate_inputs(images, states, step_idx_list)
        output = self._encode_images(images)
        image_by_size = {
            size: np.asarray(jax.device_get(pool_tokens_to_size(output, size * size, self.pool_type)))
            for size in (8, 4, 2)
        }
        positions = {size: self._positions(step_idx_list, size) for size in (8, 4, 2)}

        for offset, step_idx in enumerate(step_idx_list):
            if step_idx in self._history_feats:
                raise ValueError(f"History step {step_idx} is already buffered")
            self._history_feats[step_idx] = {
                "image_pixels": images[offset].copy(),
                "image_emb_8x8": image_by_size[8][offset],
                "image_emb_4x4": image_by_size[4][offset],
                "image_emb_2x2": image_by_size[2][offset],
                "pos_emb_8x8": positions[8][offset],
                "pos_emb_4x4": positions[4][offset],
                "pos_emb_2x2": positions[2][offset],
                "state_emb": states[offset],
            }
            if self.compute_token_drop_score:
                self._process_token_drop_score(step_idx)

    def _process_token_drop_score(self, step_idx: int) -> None:
        if step_idx == 0:
            for patch_idx in range(64):
                for view_idx in range(self.num_views):
                    heapq.heappush(self.scored_token_heap, (1000.0, 0, view_idx, patch_idx))

        if step_idx != self.token_drop_last_frame + self.token_drop_stride:
            return
        previous_step = max(0, self.token_drop_last_frame)
        previous = self._history_feats[previous_step]["image_pixels"].astype(np.float32) / 127.5 - 1.0
        current = self._history_feats[step_idx]["image_pixels"].astype(np.float32) / 127.5 - 1.0
        previous = einops.rearrange(previous, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
        current = einops.rearrange(current, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
        for view_idx in range(self.num_views):
            differences = np.abs(previous[view_idx] - current[view_idx]).mean(axis=-1)
            for patch_idx, score in enumerate(differences):
                if score < 1e-4:
                    continue
                heapq.heappush(
                    self.scored_token_heap,
                    (float(score), step_idx, view_idx, patch_idx),
                )
                if len(self.scored_token_heap) > self.token_drop_keptsize:
                    heapq.heappop(self.scored_token_heap)
        self.token_drop_last_frame += self.token_drop_stride

    def clear(self) -> None:
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self._history_feats.clear()

    def get_token_dropping_indices(self) -> list[tuple[int, int, int]]:
        selected = deepcopy(self.scored_token_heap)
        indices = []
        while selected:
            _, step_idx, view_idx, patch_idx = heapq.heappop(selected)
            indices.append((step_idx, view_idx, patch_idx))
        return indices

    @staticmethod
    def filter_token_dropping_indices(indices, step_idx, token_budget, is_sorted=True):
        indices = [item for item in indices if item[0] <= step_idx][-token_budget:]
        return sorted(indices) if is_sorted else indices

    def prepare_token_dropping(self, step_idx, token_budget, history_feats_gather_fn):
        kept = self.filter_token_dropping_indices(
            self.get_token_dropping_indices(),
            step_idx,
            token_budget,
        )
        steps = sorted({item[0] for item in kept})
        history = history_feats_gather_fn(steps)
        image = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        position = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        mask = np.zeros(token_budget, dtype=np.bool_)
        for offset, (buffer_idx, view_idx, patch_idx) in enumerate(kept):
            image[offset] = history[buffer_idx]["image_emb_8x8"][view_idx, patch_idx]
            position[offset] = history[buffer_idx]["pos_emb_8x8"][view_idx, patch_idx]
            state[offset] = history[buffer_idx]["state_emb"]
            mask[offset] = True
        return image, position, state, mask

    def prepare_frame_sampling(
        self,
        step_idx,
        token_budget,
        token_per_image,
        history_feats_gather_fn,
    ):
        grid_size = math.isqrt(token_per_image)
        if grid_size * grid_size != token_per_image or grid_size not in {2, 4, 8}:
            raise ValueError(f"Unsupported token_per_image: {token_per_image}")
        max_frames = token_budget // (token_per_image * self.num_views)
        if max_frames < 1:
            raise ValueError("Memory token budget is smaller than one frame")
        indices = even_sampling_indices(step_idx, max_frames)
        history = history_feats_gather_fn(indices)
        spatial_key = f"{grid_size}x{grid_size}"
        image = self._load_emb(history, indices, f"image_emb_{spatial_key}")
        position = self._load_emb(history, indices, f"pos_emb_{spatial_key}")
        state = self._load_emb(history, indices, "state_emb")
        mask = np.ones(image.shape[0], dtype=np.bool_)
        image, position, state, mask = right_padding_token_emb(
            image,
            position,
            state,
            mask,
            max_frames,
        )
        image = image.reshape(-1, self.img_emb_dim)
        position = position.reshape(-1, self.pos_emb_dim)
        state = np.repeat(state, self.num_views * token_per_image, axis=0)
        mask = np.repeat(mask, self.num_views * token_per_image)
        return image, position, state, mask

    @staticmethod
    def _load_emb(history: dict, indices: Sequence[int], key: str) -> np.ndarray:
        return np.stack([history[index][key] for index in indices], axis=0)

    def default_history_feats_gather_fn(self, indices: Sequence[int]) -> dict:
        return {index: self._history_feats[index] for index in indices}


class MemoryBufferRecurrent(MemoryBuffer):
    """Online history buffer for recurrent MME-VLA memory variants."""

    def __init__(
        self,
        *,
        input_obs_horizon: int = 8,
        max_recur_steps: int = 64,
        max_video_steps: int = 40,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if input_obs_horizon < 1 or max_recur_steps < 1 or max_video_steps < 1:
            raise ValueError("Recurrent memory horizons must be positive")
        self.input_obs_horizon = input_obs_horizon
        self.max_recur_steps = max_recur_steps
        self.max_video_steps = max_video_steps

    def add_buffer(self, images, states, step_idx_list: Sequence[int]) -> None:
        images, states = self._validate_inputs(images, states, step_idx_list)
        output = self._encode_images(images)
        image = np.asarray(jax.device_get(pool_tokens_to_size(output, 64, self.pool_type)))
        positions = self._positions(step_idx_list, 8)
        for offset, step_idx in enumerate(step_idx_list):
            if step_idx in self._history_feats:
                raise ValueError(f"History step {step_idx} is already buffered")
            self._history_feats[step_idx] = {
                "image_emb_8x8": image[offset],
                "pos_emb_8x8": positions[offset],
                "state_emb": states[offset],
            }

    def get_token_recurrent_indices(self, step_idx: int, exec_start_idx: int) -> list[int]:
        if step_idx < 0 or step_idx < exec_start_idx:
            raise ValueError("Invalid recurrent history indices")
        horizon = self.input_obs_horizon
        if exec_start_idx == 0:
            indices = [step_idx] if step_idx < horizon else list(range(step_idx % horizon, step_idx + 1, horizon))
        else:
            if exec_start_idx <= horizon * 2:
                video_indices = list(range(0, exec_start_idx, max(1, horizon // 2)))
            elif exec_start_idx <= self.max_video_steps * horizon:
                video_indices = list(range(0, exec_start_idx, horizon))
            else:
                video_indices = np.linspace(
                    0,
                    exec_start_idx - 1,
                    self.max_video_steps,
                    dtype=int,
                ).tolist()
            if step_idx - exec_start_idx < horizon:
                rest_indices = [step_idx]
            else:
                start = (step_idx - exec_start_idx) % horizon + exec_start_idx
                rest_indices = list(range(start, step_idx + 1, horizon))
            indices = video_indices + rest_indices
        indices = indices[-self.max_recur_steps :]
        if not indices:
            raise ValueError("No recurrent history frames were selected")
        return indices

    def prepare_token_recurrent(self, step_idx, exec_start_idx, history_feats_gather_fn):
        indices = self.get_token_recurrent_indices(step_idx, exec_start_idx)
        history = history_feats_gather_fn(indices)
        image = self._load_emb(history, indices, "image_emb_8x8")
        position = self._load_emb(history, indices, "pos_emb_8x8")
        state = self._load_emb(history, indices, "state_emb")
        mask = np.ones(len(indices), dtype=np.bool_)
        return left_padding_token_emb(
            image,
            position,
            state,
            mask,
            self.max_recur_steps,
        )
