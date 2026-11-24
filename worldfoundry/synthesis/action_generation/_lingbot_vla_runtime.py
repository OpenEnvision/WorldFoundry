"""Shared helpers for the in-tree LingBot-VLA inference runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation._native_policy_runtime import first_present, to_numpy_image


def find_training_config(checkpoint: Path, explicit: Any = None, fallback: Any = None) -> Path:
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LingBot-VLA training config not found: {path}")
        return path
    candidates = [checkpoint / "lingbotvla_cli.yaml"]
    candidates.extend(parent / "lingbotvla_cli.yaml" for parent in list(checkpoint.parents)[:4])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    nested = sorted(checkpoint.glob("**/lingbotvla_cli.yaml"))
    if len(nested) == 1:
        return nested[0].resolve()
    if fallback:
        path = Path(str(fallback)).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"in-tree LingBot-VLA inference config not found: {path}")
    raise FileNotFoundError(
        "LingBot-VLA inference requires lingbotvla_cli.yaml. Put it in the checkpoint directory "
        "or pass training_config_path explicitly; no in-tree inference fallback was supplied."
    )


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected a YAML mapping in {path}")
    return dict(payload)


def integration_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing in-tree LingBot-VLA asset: {path}")
    return path


def resolve_robot_config(root: Path, robot_name: str, options: Mapping[str, Any]) -> Path:
    explicit = options.get("robot_config_path")
    path = Path(str(explicit)).expanduser().resolve() if explicit else integration_asset(root, f"{robot_name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"LingBot-VLA robot config not found: {path}")
    return path


def resolve_norm_stats(root: Path, default_name: str, data_config: Any, options: Mapping[str, Any]) -> Path:
    explicit = options.get("norm_stats_path") or options.get("robot_norm_path")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
    else:
        configured = getattr(data_config, "norm_stats_file", None)
        configured_path = Path(str(configured)).expanduser() if configured else None
        if configured_path and configured_path.is_file():
            path = configured_path.resolve()
        else:
            path = integration_asset(root, default_name)
    if not path.is_file():
        raise FileNotFoundError(f"LingBot-VLA normalization statistics not found: {path}")
    return path


def unwrap_observation(observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    nested = observation.get("observation")
    result = dict(nested) if isinstance(nested, Mapping) else dict(observation)
    for key, value in observation.items():
        if key not in {"observation", "image", "images", "state", "video"} and key not in result:
            result[key] = value
    state = first_present(result, "state", "robot_state", "proprio", "joint_state")
    if state is not None:
        result.setdefault("observation.state", state)
    result.setdefault("task", instruction)
    return result


def _image_candidates(image: Any, observation: Mapping[str, Any], required_keys: Sequence[str]) -> list[Any]:
    sources = [observation.get("images"), observation.get("image"), image]
    for source in sources:
        if isinstance(source, Mapping):
            values: list[Any] = []
            for key in required_keys:
                short = key.rsplit(".", 1)[-1]
                value = source.get(key, source.get(short))
                if value is not None:
                    values.append(value)
            if values:
                return values
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            return list(source)
        if source is not None:
            return [source]
    return []


def populate_images(
    raw: dict[str, Any],
    image: Any,
    required_keys: Sequence[str],
    *,
    replicate_single: bool,
) -> None:
    missing = [key for key in required_keys if raw.get(key) is None]
    if missing:
        values = _image_candidates(image, raw, required_keys)
        if len(values) == len(required_keys):
            for key, value in zip(required_keys, values, strict=False):
                if key in missing:
                    raw[key] = value
        elif len(values) == 1 and len(missing) > 1 and replicate_single:
            for key in missing:
                raw[key] = values[0]
        elif len(values) == len(missing):
            for key, value in zip(missing, values, strict=False):
                raw[key] = value
    remaining = [key for key in required_keys if raw.get(key) is None]
    if remaining:
        raise ValueError(
            "LingBot-VLA requires all configured camera views. Missing "
            f"{remaining}; pass an image mapping/list, or explicitly set replicate_single_image=true."
        )
    # Normalize every view, including camera keys already present in the
    # observation. This makes CHW tensors/arrays and HWC images equivalent.
    import numpy as np

    for key in required_keys:
        array = np.asarray(to_numpy_image(raw[key]))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"LingBot-VLA camera {key!r} must be an RGB image, got shape {array.shape}")
        raw[key] = np.ascontiguousarray(array)


def action_values(action_dict: Mapping[str, Any]) -> Any:
    values = list(action_dict.values())
    return values[0] if len(values) == 1 else dict(action_dict)
