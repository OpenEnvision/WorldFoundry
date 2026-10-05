"""Compatibility re-exports of canonical recipe types onto the diffusion_model root.

This file is not on the recipe → assembler → strategy → runner runtime path.
It only keeps a stable import surface:
``from worldfoundry.base_models.diffusion_model.registry import ...``
resolves the same symbols as ``from ...diffusion_model.recipes import ...``.

Must not register recipes, construct a runner, or add a second registry implementation.
"""

from .recipes import (
    DuplicateNativeDiffusionRecipeError,
    NativeDiffusionRecipe,
    NativeDiffusionRegistry,
    UnknownNativeDiffusionRecipeError,
    default_native_diffusion_registry,
)

__all__ = [
    "DuplicateNativeDiffusionRecipeError",
    "NativeDiffusionRecipe",
    "NativeDiffusionRegistry",
    "UnknownNativeDiffusionRecipeError",
    "default_native_diffusion_registry",
]
