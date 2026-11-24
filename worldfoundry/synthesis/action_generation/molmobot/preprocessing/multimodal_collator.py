"""Inference-only NumPy-to-Torch collation for MolmoBot."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .preprocessor_utils import TensorSpec, VariablePaddingSpec


def _collate(
    arrays,
    max_shape=None,
    dtype=None,
    pad=None,
    pad_value=-1,
    allow_truncate=True,
):
    present = [array for array in arrays if array is not None]
    if not present:
        return None
    batch_shape = np.stack([array.shape for array in present], axis=0).max(axis=0)
    if pad == "to_max":
        row_shape = np.asarray(max_shape)
        if np.any(batch_shape[1:] > row_shape[1:]):
            raise ValueError(f"Input shape {batch_shape} exceeds padding shape {row_shape}.")
        if not allow_truncate and batch_shape[0] > row_shape[0]:
            raise ValueError(f"Input length {batch_shape[0]} exceeds bound {row_shape[0]}.")
    elif pad is None:
        row_shape = batch_shape
    else:
        raise ValueError(f"Unsupported padding mode: {pad!r}")

    output = np.full(
        [len(arrays), *row_shape.tolist()],
        pad_value,
        dtype=dtype or present[0].dtype,
    )
    for index, array in enumerate(arrays):
        if array is None:
            continue
        value = array[: row_shape[0]]
        slices = tuple(slice(None, dim) for dim in value.shape)
        output[(index, *slices)] = value
    return torch.from_numpy(output)


class MMCollator:
    """Collate already-preprocessed inference examples."""

    def __init__(
        self,
        special_tokens,
        shapes_to_pad_to: dict[str, VariablePaddingSpec | TensorSpec] | None = None,
        include_metadata: bool = True,
        pad=None,
        skip_padding=None,
        cp_enabled: bool = False,
    ):
        del special_tokens, skip_padding
        if cp_enabled:
            raise ValueError("Context-parallel training collation is not part of MolmoBot inference.")
        if pad and shapes_to_pad_to is None:
            raise ValueError("A padding specification is required when pad is enabled.")
        self.shapes_to_pad_to = shapes_to_pad_to or {}
        self.include_metadata = include_metadata
        self.pad = pad

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty MolmoBot batch.")
        max_sequence_len = self.shapes_to_pad_to["tokens"].shape[0] if self.pad is not None else None
        output: dict[str, Any] = {
            "input_ids": _collate(
                [example["input_tokens"] for example in batch],
                [max_sequence_len],
                np.int64,
                pad=self.pad,
            ),
            "position_ids": _collate(
                [example["position_ids"] for example in batch],
                [max_sequence_len],
                np.int64,
                pad=self.pad,
            ),
        }

        for key, spec in self.shapes_to_pad_to.items():
            if key == "tokens":
                continue
            arrays = [example.get(key) for example in batch]
            if all(array is None for array in arrays):
                continue
            item_pad = None if isinstance(spec, VariablePaddingSpec) else self.pad
            pad_value = 0 if spec.dtype == np.uint8 else -1
            value = _collate(
                arrays,
                spec.shape,
                dtype=spec.dtype,
                pad=item_pad,
                pad_value=pad_value,
                allow_truncate=False,
            )
            if value is not None:
                output[key] = value

        states = [example.get("states") for example in batch]
        if any(state is not None for state in states):
            if not all(state is not None for state in states):
                raise ValueError("Every example in a MolmoBot batch must provide state.")
            state_arrays = [np.asarray(state, dtype=np.float32) for state in states]
            if any(array.shape != state_arrays[0].shape for array in state_arrays[1:]):
                raise ValueError(f"Inconsistent state shapes: {[array.shape for array in state_arrays]}")
            output["states"] = torch.from_numpy(np.stack(state_arrays))

        if self.include_metadata:
            output["metadata"] = [example.get("metadata", {}) for example in batch]
        return output


__all__ = ["MMCollator"]
