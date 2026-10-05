"""Checkpoint-compatible Echo-Memory Wan blocks and recipes.

Echo extends Wan DiT tokens ``B (F·H·W) C`` with action injection,
optional temporal SSM / VideoSSM, and spatial-grid memory.  Recipes in
``schema.py`` pin which branches a released checkpoint must contain.
"""

from .schema import (
    ECHO_MEMORY_RECIPES,
    EchoMemoryMechanism,
    EchoMemoryRecipe,
    SpatialInjection,
    get_echo_memory_recipe,
)

__all__ = [
    "ECHO_MEMORY_RECIPES",
    "EchoMemoryMechanism",
    "EchoMemoryRecipe",
    "SpatialInjection",
    "get_echo_memory_recipe",
]
