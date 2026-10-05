"""Lazy export surface for declarative recipes and the instance-local registry.

This package does not construct a runner. ``NativeDiffusionRecipe`` /
``NativeDiffusionRegistry`` sit upstream on the pipeline; ``wan21_*_recipe``
and similar factories load their model module only on first attribute access.
Must not materialize every recipe at import time (that would pull optional
dependencies). Unknown attributes raise :exc:`AttributeError`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DuplicateNativeDiffusionRecipeError": ".registry",
    "NativeDiffusionRecipe": ".spec",
    "NativeDiffusionRegistry": ".registry",
    "UnknownNativeDiffusionRecipeError": ".registry",
    "default_native_diffusion_registry": ".registry",
    "gen3c_recipe": ".cosmos1",
    "gamma_world_bidirectional_recipe": ".gamma_world",
    "gamma_world_causal_recipe": ".gamma_world",
    "gamma_world_causal_few_step_recipe": ".gamma_world",
    "cosmos2_2b_video2world_recipe": ".cosmos2",
    "cosmos2_14b_video2world_recipe": ".cosmos2",
    "cosmos3_nano_recipe": ".cosmos3",
    "cosmos3_super_recipe": ".cosmos3",
    "cosmos25_2b_recipe": ".cosmos2p5",
    "cosmos25_14b_recipe": ".cosmos2p5",
    "cosmos25_transfer_2b_recipe": ".cosmos2p5",
    "echo_memory_recipe": ".echo_memory",
    "echo_memory_recipes": ".echo_memory",
    "hunyuan_video15_i2v_recipe": ".hunyuan_video",
    "hunyuan_video15_t2v_recipe": ".hunyuan_video",
    "hunyuan_video_i2v_recipe": ".hunyuan_video",
    "hunyuan_video_recipes": ".hunyuan_video",
    "hunyuan_video_t2v_recipe": ".hunyuan_video",
    "matrix_game_35_first_person_recipe": ".matrix_game",
    "matrix_game_35_third_person_recipe": ".matrix_game",
    "step_video_t2v_recipe": ".step_video",
    "skyreels_v2_recipe": ".skyreels",
    "skyreels_v3_recipe": ".skyreels",
    "ltx2_i2v_recipe": ".ltx",
    "ltx2_t2v_recipe": ".ltx",
    "ltx23_i2v_recipe": ".ltx",
    "ltx_video_i2v_recipe": ".ltx",
    "hello_world_recipe": ".hello_world",
    "fastvideo_causal_wan22_i2v_14b_recipe": ".wan",
    "sana_wm_streaming_recipe": ".sana",
    "t2v_turbo_t2v_recipe": ".t2v_turbo",
    "vchitect_2_t2v_recipe": ".vchitect",
    "wan21_t2v_1p3b_recipe": ".wan",
    "wan21_t2v_14b_recipe": ".wan",
    "wan21_i2v_14b_480p_recipe": ".wan",
    "wan21_i2v_14b_720p_recipe": ".wan",
    "wan22_ti2v_5b_recipe": ".wan",
    "wan22_t2v_a14b_recipe": ".wan",
    "wan22_i2v_a14b_recipe": ".wan",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily load a submodule attribute from ``_EXPORTS`` and cache it in ``globals()``."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
