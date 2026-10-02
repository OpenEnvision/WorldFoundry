"""Compatibility helpers for loading partially matching model state dictionaries.

Released checkpoints can omit TE extra-state, carry quantization
observers, or disagree on a few tensor shapes.
:func:`load_state_dict_non_strict` (alias :data:`non_strict_load_model`)
drops mismatched keys, skips ``_extra_state``, then calls
``load_state_dict(..., strict=False)``. :class:`IncompatibleKeys`
reports missing, unexpected, and shape-incorrect names.

This is not the fail-closed assigner in
:mod:`worldfoundry.core.model_loading.checkpoints.assignment`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from worldfoundry.core.distributed.runtime.logging import log

# ──────────────────────────────────────────────────────────────────────────
# Non-strict assign — drop shape mismatches and TE extra-state; report leftovers
# ──────────────────────────────────────────────────────────────────────────


class IncompatibleKeys(NamedTuple):
    """Keys that did not load cleanly during a non-strict state-dict load."""

    missing_keys: list[str]
    unexpected_keys: list[str]
    incorrect_shapes: list[tuple[str, tuple[int, ...], tuple[int, ...]]]


def _module_for_key(model: torch.nn.Module, key: str) -> torch.nn.Module:
    """Walk dotted *key* to the owning module (last segment is the tensor name).

    Used to detect FakeQuantize/Observer parents so a shape mismatch on a
    quantization buffer is not treated as a checkpoint error.
    """
    module = model
    for part in key.split(".")[:-1]:
        module = getattr(module, part)
    return module


def load_state_dict_non_strict(model: torch.nn.Module, state_dict: dict) -> IncompatibleKeys:
    """Load compatible tensors, reporting rather than raising on shape mismatches."""

    candidate = dict(state_dict)
    model_state = model.state_dict()
    incorrect_shapes: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    try:
        from torch.ao.quantization import FakeQuantizeBase, ObserverBase

        quantization_exceptions = (ObserverBase, FakeQuantizeBase)
    except ImportError:  # pragma: no cover - depends on the installed torch build.
        quantization_exceptions = ()

    for key in list(candidate):
        if key not in model_state:
            continue
        if "_extra_state" in key:
            log.warning(f"Skipping TransformerEngine extra-state key {key}.")
            candidate.pop(key)
            continue

        model_value = model_state[key]
        if isinstance(model_value, torch.nn.parameter.UninitializedParameter):
            continue
        if not isinstance(model_value, torch.Tensor):
            raise TypeError(f"Model state {key!r} is not a tensor: {type(model_value)!r}")

        checkpoint_value = candidate[key]
        checkpoint_shape = tuple(checkpoint_value.shape)
        model_shape = tuple(model_value.shape)
        if checkpoint_shape == model_shape:
            continue
        if quantization_exceptions and isinstance(_module_for_key(model, key), quantization_exceptions):
            continue
        incorrect_shapes.append((key, checkpoint_shape, model_shape))
        candidate.pop(key)

    incompatible = model.load_state_dict(candidate, strict=False)
    return IncompatibleKeys(
        missing_keys=[key for key in incompatible.missing_keys if "_extra_state" not in key],
        unexpected_keys=[key for key in incompatible.unexpected_keys if "_extra_state" not in key],
        incorrect_shapes=incorrect_shapes,
    )


non_strict_load_model = load_state_dict_non_strict

__all__ = ["IncompatibleKeys", "load_state_dict_non_strict", "non_strict_load_model"]
