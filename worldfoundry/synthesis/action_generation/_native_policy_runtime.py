"""Shared utilities for native in-process embodied policy runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def option_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def option_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def option_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def runtime_options_cache_key(options: Mapping[str, Any]) -> str:
    """Return a deterministic cache key for a policy runtime option mapping.

    Policy instances retain preprocessing and inference options, so omitting a
    seemingly small flag from a hand-written cache tuple can silently reuse the
    wrong runtime.  JSON with a string fallback handles paths, dtypes, and other
    declarative config values without requiring them to be hashable.
    """

    return json.dumps(dict(options), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def load_json_if_present(path: str | Path) -> dict[str, Any] | None:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_image(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape") or hasattr(value, "convert"):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return load_image(value[0]) if value else None
    if isinstance(value, (str, Path)):
        from PIL import Image

        return Image.open(value).convert("RGB")
    return value


def to_numpy_image(value: Any) -> Any:
    image = load_image(value)
    if image is None:
        return None
    import numpy as np

    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f" and array.max(initial=0) <= 1.0:
        array = (array * 255).astype("uint8")
    return array


def collect_images(observation: Mapping[str, Any], image: Any, keys: Sequence[str]) -> list[Any]:
    images = first_present(observation, "images")
    if isinstance(images, Mapping):
        return [images[key] for key in keys if key in images and images[key] is not None]
    if hasattr(images, "shape") or hasattr(images, "convert"):
        return [images]
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        return [item for item in images if item is not None]

    collected = [observation[key] for key in keys if key in observation and observation[key] is not None]
    if collected:
        return collected
    if image is None:
        return []
    if isinstance(image, Mapping):
        return [image[key] for key in keys if key in image and image[key] is not None]
    if hasattr(image, "shape") or hasattr(image, "convert"):
        return [image]
    if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
        return [item for item in image if item is not None]
    return [image]


def extract_action_values(value: Any) -> Any:
    if isinstance(value, Mapping) and "actions" in value:
        return value["actions"]
    if isinstance(value, list) and value and hasattr(value[0], "actions"):
        return [item.actions for item in value]
    if hasattr(value, "actions"):
        return value.actions
    return value


def completed_action_result(
    *,
    model_id: str,
    instruction: str,
    actions: Any,
    checkpoint_path: str,
    device: str,
    runtime: str,
    raw_output: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "model_id": model_id,
        "instruction": instruction,
        "actions": extract_action_values(actions),
        "checkpoint_path": checkpoint_path,
        "device": device,
        "runtime": runtime,
        "raw_output": raw_output if raw_output is not None else actions,
        "metadata": dict(metadata or {}),
    }
