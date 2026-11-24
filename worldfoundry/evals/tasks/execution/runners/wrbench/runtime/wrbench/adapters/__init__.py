"""Camera adapters. Importing this package registers every adapter."""

from __future__ import annotations

import importlib

from wrbench.adapters.base import (
    adapter_for_model,
    compile_camera_payload,
    register,
    register_adapter,
    registered_model_keys,
)

_ADAPTER_MODULES = [
    "wrbench.adapters.wan_fun",
    "wrbench.adapters.hunyuan",
    "wrbench.adapters.recammaster",
    "wrbench.adapters.magicworld",
    "wrbench.adapters.versecrafter",
    "wrbench.adapters.lingbot",
    "wrbench.adapters.liveworld",
    "wrbench.adapters.gen3c",
    "wrbench.adapters.sana_wm",
    "wrbench.adapters.easyanimate",
    "wrbench.adapters.minwm",
    "wrbench.adapters.hydra",
    "wrbench.adapters.spatia_inspatio",
    "wrbench.adapters.ati",
    "wrbench.adapters.cosmos3",
    "wrbench.adapters.ac3d",
    "wrbench.adapters.action_candidate",
]

for _module in _ADAPTER_MODULES:
    importlib.import_module(_module)

__all__ = [
    "adapter_for_model",
    "compile_camera_payload",
    "register",
    "register_adapter",
    "registered_model_keys",
]
