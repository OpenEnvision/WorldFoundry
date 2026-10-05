# Inference-only Wan-VA source retained in-tree.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Build Wan-VA inference objects from bundled ``worldfoundry/data`` YAML."""

from __future__ import annotations

from typing import Any

import torch
import yaml

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.core.configuration.python import EasyDict


_CONFIG_ROOT = resolve_data_path("models", "runtime", "configs", "wan_va")


def _load_yaml_config(name: str) -> dict[str, Any]:
    path = _CONFIG_ROOT / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return dict(payload)


def _inverse_action_channels(action_dim: int, used_action_channel_ids: list[int]) -> list[int]:
    inverse = [len(used_action_channel_ids)] * action_dim
    for index, channel_id in enumerate(used_action_channel_ids):
        inverse[channel_id] = index
    return inverse


def _torch_dtype(name: str) -> torch.dtype:
    if name == "auto":
        # Resolved per rank after torch.distributed binds the CUDA device.
        return torch.float32
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype in Wan-VA YAML: {name}")
    return dtype


def _config(name: str, filename: str, *, base: EasyDict | None = None, inverse: bool = False) -> EasyDict:
    merged = {**dict(base or {}), **_load_yaml_config(filename)}
    result = EasyDict(merged, __name__=f"Config: VA {name}")
    if inverse:
        result.inverse_used_action_channel_ids = _inverse_action_channels(
            int(result.action_dim),
            list(result.used_action_channel_ids),
        )
    return result


_shared = _load_yaml_config("shared.yaml")
_dtype_name = str(_shared.pop("param_dtype", "bfloat16"))
va_shared_cfg = EasyDict(_shared, __name__="Config: VA shared")
va_shared_cfg.param_dtype = _torch_dtype(_dtype_name)
va_shared_cfg.param_dtype_request = _dtype_name

va_robotwin_cfg = _config("robotwin", "robotwin.yaml", base=va_shared_cfg, inverse=True)
va_franka_cfg = _config("franka", "franka.yaml", base=va_shared_cfg, inverse=True)
va_inference_cfg = _config("inference", "demo.yaml", base=va_shared_cfg, inverse=True)
va_libero_cfg = _config("libero", "libero.yaml", base=va_shared_cfg, inverse=True)

va_robotwin_i2va_cfg = _config("robotwin i2va", "robotwin_i2va.yaml", base=va_robotwin_cfg)
va_franka_i2va_cfg = _config("franka i2va", "franka_i2va.yaml", base=va_franka_cfg)
va_inference_i2va_cfg = _config("inference i2va", "demo_i2va.yaml", base=va_inference_cfg)
va_libero_i2va_cfg = _config("libero i2va", "libero_i2va.yaml", base=va_libero_cfg)

VA_CONFIGS = {
    "robotwin": va_robotwin_cfg,
    "franka": va_franka_cfg,
    "demo": va_inference_cfg,
    "libero": va_libero_cfg,
    # Keep the released spelling for checkpoint/deployment compatibility while
    # accepting the correctly ordered suffix as an equivalent runtime alias.
    "robotwin_i2av": va_robotwin_i2va_cfg,
    "franka_i2av": va_franka_i2va_cfg,
    "demo_i2av": va_inference_i2va_cfg,
    "libero_i2av": va_libero_i2va_cfg,
    "robotwin_i2va": va_robotwin_i2va_cfg,
    "franka_i2va": va_franka_i2va_cfg,
    "demo_i2va": va_inference_i2va_cfg,
    "libero_i2va": va_libero_i2va_cfg,
}


__all__ = [
    "VA_CONFIGS",
    "va_franka_cfg",
    "va_franka_i2va_cfg",
    "va_inference_cfg",
    "va_inference_i2va_cfg",
    "va_libero_cfg",
    "va_libero_i2va_cfg",
    "va_robotwin_cfg",
    "va_robotwin_i2va_cfg",
    "va_shared_cfg",
]
