"""Action representation consumed by the world model's action encoder.

`ActionTensors` is the batched, time-indexed container the encoder reads. It holds three tensors:

- ``key_presses`` ``(B, T, n_keys)`` int32 multi-hot keyboard state,
- ``mouse_movements`` ``(B, T, 2)`` float32 raw mouse deltas,
- ``game_mouse_sensitivity`` ``(B,)`` float32 (no time axis), NaN where unknown.

The 4-player Rocket League data is keyboard-only, so the loader fills ``mouse_movements`` with
zeros and ``game_mouse_sensitivity`` with NaN; the encoder masks the NaN sensitivity to a learned
token (``torch.nan_to_num`` + ``torch.where``), so all-NaN is the expected "no mouse" signal.

The JSONL-to-tensor parsing lives in :mod:`mira.data.actions` (``tensorize_actions``); this
module only defines the tensor container and its config so the encoder surface stays stable.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel


class ActionConfig(BaseModel):
    """Vocabulary and frame rate of an action stream.

    Attributes:
        valid_keys: Ordered key names defining the multi-hot ``key_presses`` columns. The length
            sets ``n_keys`` and therefore the encoder's keyboard embedding count, so it must match
            the value saved with a checkpoint.
        source_fps: Frame rate of the recorded action stream.
        target_fps: Frame rate the actions are downsampled to.
    """

    valid_keys: list[str]
    source_fps: int = 20  # the 4-player Rocket League recordings are ~20fps
    target_fps: int = 10

    @property
    def downsampling_factor(self) -> int:
        """Integer ratio ``source_fps / target_fps`` used to downsample the action stream."""
        if self.target_fps > self.source_fps:
            raise ValueError(
                f"Upsampling not supported: target_fps ({self.target_fps}) > source_fps ({self.source_fps})"
            )
        if self.source_fps % self.target_fps != 0:
            raise ValueError(
                f"Only integer downsampling is supported: source_fps ({self.source_fps}) must be a "
                f"multiple of target_fps ({self.target_fps})."
            )
        return self.source_fps // self.target_fps


class ActionTensors:
    """A tensor representation of the actions over time in a game, with a batch dimension.

    For training, actions are parsed into per-clip tensors and stacked in the data loader; for
    inference they are provided in a streaming fashion. The encoder reads only ``key_presses``,
    ``mouse_movements`` and ``game_mouse_sensitivity`` plus ``config``.
    """

    def __init__(self, config: ActionConfig, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.key_presses: torch.Tensor = torch.zeros(
            (self.batch_size, 0, len(config.valid_keys)), dtype=torch.int32
        )
        self.mouse_movements: torch.Tensor = torch.zeros((self.batch_size, 0, 2), dtype=torch.float32)

        # Filled in externally from metadata; NaN means "unknown", which the encoder masks to a
        # learned token rather than reading the NaN.
        self.game_mouse_sensitivity: torch.Tensor = torch.full(
            (self.batch_size,), float("nan"), dtype=torch.float32
        )

    @property
    def n_steps(self) -> int:
        """Number of time steps ``T`` (asserts key/mouse time dimensions agree)."""
        assert self.key_presses.shape[1] == self.mouse_movements.shape[1]
        return self.key_presses.shape[1]

    def slice_time(self, start: int | None, end: int | None) -> ActionTensors:
        """Slice along the time dimension to ``[start, end)``. Views the tensors, does not copy."""
        sliced = ActionTensors(config=self.config, batch_size=self.batch_size)
        sliced.key_presses = self.key_presses[:, start:end, :]
        sliced.mouse_movements = self.mouse_movements[:, start:end, :]
        sliced.game_mouse_sensitivity = self.game_mouse_sensitivity  # No time dimension
        return sliced

    def slice_batch(self, start: int, end: int) -> ActionTensors:
        """Slice along the batch dimension to ``[start, end)``."""
        sliced = ActionTensors(config=self.config, batch_size=end - start)
        sliced.key_presses = self.key_presses[start:end, :, :]
        sliced.mouse_movements = self.mouse_movements[start:end, :, :]
        sliced.game_mouse_sensitivity = self.game_mouse_sensitivity[start:end]
        return sliced

    def cat_time(self, other: ActionTensors) -> ActionTensors:
        """Concatenate another ``ActionTensors`` along the time dimension.

        Args:
            other: Actions to append; must have the same ``batch_size`` and ``config``.

        Returns:
            A new ``ActionTensors`` with the time steps of both concatenated.
        """
        assert other.batch_size == self.batch_size, "Batch size must match"
        assert other.config == self.config, f"Config must match: {other.config=} != {self.config=}"

        result = ActionTensors(config=self.config, batch_size=self.batch_size)
        result.key_presses = torch.cat([self.key_presses, other.key_presses], dim=1)
        result.mouse_movements = torch.cat([self.mouse_movements, other.mouse_movements], dim=1)

        # game_mouse_sensitivity has no time dimension. If one side is all-NaN we take the other;
        # otherwise the two must agree.
        if torch.isnan(other.game_mouse_sensitivity).all():
            result.game_mouse_sensitivity = self.game_mouse_sensitivity.clone()
        elif torch.isnan(self.game_mouse_sensitivity).all():
            result.game_mouse_sensitivity = other.game_mouse_sensitivity.clone()
        else:
            if not torch.allclose(self.game_mouse_sensitivity, other.game_mouse_sensitivity, equal_nan=True):
                raise ValueError(
                    "Mouse sensitivities do not match: "
                    f"{self.game_mouse_sensitivity} vs {other.game_mouse_sensitivity}"
                )
            result.game_mouse_sensitivity = self.game_mouse_sensitivity.clone()

        return result

    def to(self, *args, **kwargs) -> ActionTensors:
        """Propagate ``torch.Tensor.to`` to every held tensor."""
        moved = ActionTensors(config=self.config, batch_size=self.batch_size)
        moved.key_presses = self.key_presses.to(*args, **kwargs)
        moved.mouse_movements = self.mouse_movements.to(*args, **kwargs)
        moved.game_mouse_sensitivity = self.game_mouse_sensitivity.to(*args, **kwargs)
        return moved

    def pin_memory(self) -> ActionTensors:
        """Propagate ``torch.Tensor.pin_memory`` to every held tensor."""
        pinned = ActionTensors(config=self.config, batch_size=self.batch_size)
        pinned.key_presses = self.key_presses.pin_memory()
        pinned.mouse_movements = self.mouse_movements.pin_memory()
        pinned.game_mouse_sensitivity = self.game_mouse_sensitivity.pin_memory()
        return pinned

    def clone(self) -> ActionTensors:
        """Create a deep copy of this ``ActionTensors``."""
        cloned = ActionTensors(config=self.config.model_copy(deep=True), batch_size=self.batch_size)
        cloned.key_presses = self.key_presses.clone()
        cloned.mouse_movements = self.mouse_movements.clone()
        cloned.game_mouse_sensitivity = self.game_mouse_sensitivity.clone()
        return cloned

    def __repr__(self) -> str:
        """Short description of the contents for debugging."""
        parts = [
            f"b={self.batch_size}",
            f"t={self.key_presses.shape[1]}",
            f"num_keys={self.key_presses.shape[2]}",
        ]
        if self.key_presses.shape[1] > 0:
            batch_mouse_sums = self.mouse_movements.sum(dim=1).tolist()
            batch_keys = []
            for b_idx in range(self.batch_size):
                pressed_indices = self.key_presses[b_idx].any(dim=0).nonzero().flatten().tolist()
                pressed_keys = [self.config.valid_keys[i] for i in pressed_indices]
                batch_keys.append(pressed_keys)

            parts.append(f"mouse_sum={batch_mouse_sums}")
            parts.append(f"keys_sum={batch_keys}")

        parts.append(f"sensitivity={self.game_mouse_sensitivity.flatten()}")
        return f"ActionTensors({', '.join(parts)})"


def stack_action_tensors(action_tensors_list: list[ActionTensors]) -> ActionTensors:
    """Concatenate per-sample ``ActionTensors`` along the batch dimension.

    Args:
        action_tensors_list: Non-empty list of ``ActionTensors`` sharing one ``config``.

    Returns:
        A single ``ActionTensors`` whose ``batch_size`` is the sum of the inputs'.
    """
    assert len(action_tensors_list) > 0

    config = action_tensors_list[0].config
    for at in action_tensors_list:
        assert at.config == config, (
            f"All ActionTensors must have the same config to be stacked, got {at.config=} and {config=}"
        )

    batch_size = sum(at.batch_size for at in action_tensors_list)
    stacked = ActionTensors(config=config, batch_size=batch_size)
    stacked.key_presses = torch.cat([at.key_presses for at in action_tensors_list], dim=0)
    stacked.mouse_movements = torch.cat([at.mouse_movements for at in action_tensors_list], dim=0)
    stacked.game_mouse_sensitivity = torch.cat(
        [at.game_mouse_sensitivity for at in action_tensors_list], dim=0
    )
    return stacked
