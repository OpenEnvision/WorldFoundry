"""Local, pickle-free policy and NumPy msgpack protocol for DreamZero."""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np


class BasePolicy(ABC):
    """Minimal policy interface required by the DreamZero WebSocket servers."""

    @abstractmethod
    def infer(self, observation: Mapping[str, Any]) -> Any:
        """Return an action for one decoded observation."""

    def reset(self, reset_info: Mapping[str, Any]) -> None:  # noqa: B027 - Optional hook with a no-op default.
        """Reset optional episode-local state."""


def pack_array(value: Any) -> Any:
    """Encode numeric NumPy values without falling back to pickle."""

    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in {"V", "O"}:
        raise ValueError(f"DreamZero protocol does not support dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": contiguous.tobytes(),
            b"dtype": contiguous.dtype.str,
            b"shape": contiguous.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"DreamZero protocol cannot encode {type(value).__name__}")


def _safe_dtype(value: Any) -> np.dtype[Any]:
    dtype = np.dtype(value)
    if dtype.kind in {"V", "O"} or dtype.hasobject:
        raise ValueError(f"DreamZero protocol rejected dtype {dtype}")
    return dtype


def unpack_array(value: dict[Any, Any]) -> Any:
    """Decode a numeric NumPy value after validating shape and byte length."""

    if value.get(b"__ndarray__") is True:
        dtype = _safe_dtype(value[b"dtype"])
        shape_value = value[b"shape"]
        if not isinstance(shape_value, (list, tuple)) or len(shape_value) > 16:
            raise ValueError("DreamZero protocol rejected ndarray shape")
        shape = tuple(int(dimension) for dimension in shape_value)
        if any(dimension < 0 for dimension in shape):
            raise ValueError("DreamZero protocol rejected a negative ndarray dimension")
        data = value[b"data"]
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("DreamZero ndarray payload must contain bytes")
        expected = math.prod(shape) * dtype.itemsize
        if expected != len(data):
            raise ValueError(f"DreamZero ndarray payload length mismatch: expected {expected}, got {len(data)}")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if value.get(b"__npgeneric__") is True:
        dtype = _safe_dtype(value[b"dtype"])
        return dtype.type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


__all__ = ["BasePolicy", "Packer", "Unpacker", "pack_array", "packb", "unpack_array", "unpackb"]
