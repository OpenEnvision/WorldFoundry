"""Unified camera adapter protocol and registry.

An adapter turns a model-agnostic OpenCV ``CameraTrajectory`` into a model-native
``CameraPayload``. Adapters self-register with ``@register(...)`` against one or
more canonical model keys; ``wrbench.adapters`` imports every adapter module on
import so the registry is fully populated.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Protocol

from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key, is_deferred_model
from wrbench.trajectory import CameraTrajectory


class CameraAdapter(Protocol):
    name: str

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        ...


_REGISTRY: dict[str, CameraAdapter] = {}


def register_adapter(model_name_or_group: str | list[str] | tuple[str, ...], adapter: CameraAdapter) -> None:
    names = [model_name_or_group] if isinstance(model_name_or_group, str) else list(model_name_or_group)
    for name in names:
        key = canonical_model_key(name)
        if is_deferred_model(key):
            raise ValueError(f"Cannot register deferred model {key}")
        _REGISTRY[key] = adapter


def register(*model_keys: str) -> Callable[[type], type]:
    """Class decorator: instantiate the adapter and register it for the given keys."""

    def decorator(cls: type) -> type:
        register_adapter(list(model_keys), cls())
        return cls

    return decorator


def adapter_for_model(model_name: str) -> CameraAdapter:
    key = canonical_model_key(model_name)
    if is_deferred_model(key):
        raise ValueError(f"Model {key} is deferred and excluded from unified camera registry")
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"No unified camera adapter registered for {key}") from exc


def registered_model_keys() -> list[str]:
    return sorted(_REGISTRY)


def compile_camera_payload(
    trajectory: CameraTrajectory,
    *,
    model_name: str,
    width: int,
    height: int,
    num_frames: int,
    work_dir: str | Path | None = None,
    device: str | None = None,
    prompt: str = "",
) -> CameraPayload:
    adapter = adapter_for_model(model_name)
    kwargs = {
        "model_name": model_name,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "work_dir": work_dir,
        "device": device,
    }
    if "prompt" in inspect.signature(adapter.compile).parameters:
        kwargs["prompt"] = prompt
    return adapter.compile(trajectory, **kwargs)
