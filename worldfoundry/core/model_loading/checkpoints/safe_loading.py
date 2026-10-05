"""Fail-closed helpers for inference-time PyTorch checkpoint loading.

Untrusted or wrapper-shaped checkpoints must not execute pickle or
leak non-tensor metadata into ``load_state_dict``. :func:`load_weights_only`
reads safetensors or ``torch.load(..., weights_only=True)``.
:func:`tensor_state_dict` / :func:`load_tensor_state_dict` unwrap
conventional containers (``state_dict``, ``model``, …) and require a
flat string-to-tensor mapping. :func:`require_tensor` and
:func:`require_mapping` reject unexpected payload shapes.

Not a format dispatcher — use :mod:`worldfoundry.core.model_loading.checkpoints.load`
for URL routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from pathlib import Path
from typing import Any, BinaryIO

import torch

# ──────────────────────────────────────────────────────────────────────────
# Wrapper keys — conventional training dumps nest tensors under one of these
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "module")


def load_weights_only(
    source: str | PathLike[str] | BinaryIO,
    *,
    map_location: Any = "cpu",
    mmap: bool | None = None,
) -> object:
    """Load safetensors or deserialize PyTorch's allowlisted tensor containers."""

    if isinstance(source, (str, PathLike)) and Path(fspath(source)).suffix == ".safetensors":
        if mmap is not None:
            raise ValueError("safetensors loading does not accept the PyTorch mmap option")
        from safetensors.torch import load_file

        device = "cpu" if map_location is None else str(map_location)
        return load_file(fspath(source), device=device)

    kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": True,
    }
    if mmap is not None:
        kwargs["mmap"] = mmap
    # Some official tensor checkpoints contain Python ``set`` values in
    # non-weight metadata.  Keep weights-only deserialization enabled and
    # allow only that primitive container for the duration of this load.
    # Downstream state-dict validation still requires string -> Tensor values.
    with torch.serialization.safe_globals([set]):
        return torch.load(source, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Fail-closed unwrap — reject pickle leftovers before load_state_dict
# ──────────────────────────────────────────────────────────────────────────


def require_tensor(payload: object, *, source: str) -> torch.Tensor:
    """Return one tensor payload or reject the checkpoint shape."""

    if not torch.is_tensor(payload):
        raise TypeError(f"Checkpoint {source!r} must contain a tensor, got {type(payload).__name__}")
    return payload


def require_mapping(payload: object, *, source: str) -> Mapping[str, object]:
    """Return a string-keyed checkpoint mapping or reject its shape."""

    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise TypeError(f"Checkpoint {source!r} must contain a string-keyed mapping")
    return payload


def tensor_state_dict(
    payload: object,
    *,
    source: str,
    wrapper_keys: Sequence[str] = DEFAULT_STATE_DICT_KEYS,
    allow_empty: bool = False,
) -> dict[str, torch.Tensor]:
    """Extract and validate a flat string-to-tensor state dictionary."""

    candidates: list[object] = [payload]
    if isinstance(payload, Mapping):
        candidates.extend(payload[key] for key in wrapper_keys if key in payload)
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or (not candidate and not allow_empty):
            continue
        if all(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items()):
            return dict(candidate)
    raise TypeError(
        f"Checkpoint {source!r} does not contain a "
        "flat string-to-tensor state dictionary"
    )


def load_tensor_state_dict(
    source: str | PathLike[str] | BinaryIO,
    *,
    map_location: Any = "cpu",
    wrapper_keys: Sequence[str] = DEFAULT_STATE_DICT_KEYS,
    mmap: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Weights-only load followed by strict tensor-state validation."""

    payload = load_weights_only(source, map_location=map_location, mmap=mmap)
    return tensor_state_dict(
        payload,
        source=str(getattr(source, "name", source)),
        wrapper_keys=wrapper_keys,
    )


__all__ = [
    "DEFAULT_STATE_DICT_KEYS",
    "load_tensor_state_dict",
    "load_weights_only",
    "require_mapping",
    "require_tensor",
    "tensor_state_dict",
]
