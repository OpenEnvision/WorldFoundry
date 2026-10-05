"""Cosmos3 load-time placement options shared by the public pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.base_models.diffusion_model.optimizations.policy import parse_device_map


def resolve_cosmos3_runtime_options(
    options: Mapping[str, Any],
    *,
    model_id: str,
    visible_gpus: int,
) -> tuple[str, str | None]:
    """Return ``(offload_mode, device_map)`` for one Cosmos3 load.

    Super on multiple visible GPUs defaults to a resident balanced device map
    so the 64B transformer stays in VRAM instead of layerwise CPU offload.
    """

    explicit_device_map = "device_map" in options
    device_map = parse_device_map(options.get("device_map"), owner="Cosmos3")
    if device_map is None and not explicit_device_map and model_id == "cosmos3-super" and visible_gpus > 1:
        device_map = "balanced"
    if device_map and visible_gpus < 2 and model_id == "cosmos3-super":
        raise ValueError(
            "Cosmos3-Super device_map=balanced needs at least 2 visible GPUs; "
            f"got {visible_gpus}"
        )
    offload_raw = options.get("offload_mode")
    if offload_raw is None:
        offload_mode = "none" if device_map else "block"
    else:
        offload_mode = str(offload_raw).strip().lower() or "block"
    if device_map and offload_mode not in {"none", "false", "0"}:
        raise ValueError("Cosmos3 device_map=balanced requires offload_mode=none")
    return offload_mode, device_map


__all__ = ["resolve_cosmos3_runtime_options"]
