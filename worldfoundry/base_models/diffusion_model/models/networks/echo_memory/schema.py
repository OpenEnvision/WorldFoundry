"""Immutable architecture settings for native Echo-Memory networks.

A public model identity maps to exactly one :class:`EchoMemoryRecipe`.
Recipes pin the memory mechanism (raw context, spatial grid, block
state-space, or VideoSSM hybrid), context-frame count, spatial injection
mode, and sampler defaults.  They are frozen dataclasses so released
checkpoints stay comparable and new research methods register as sibling
recipes without changing existing behavior.

Lookup is by recipe id only; callers must not infer a recipe from a
checkpoint path.  Use :func:`get_echo_memory_recipe` to resolve ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class EchoMemoryMechanism(str, Enum):
    """Where a released Echo method stores and reads temporal evidence."""

    RAW_CONTEXT = "raw_context"
    SPATIAL_GRID = "spatial_grid"
    BLOCK_STATE_SPACE = "block_state_space"
    VIDEO_SSM_HYBRID = "video_ssm_hybrid"


class SpatialInjection(str, Enum):
    """How spatial memory tokens are exposed to the Wan backbone."""

    NONE = "none"
    CONCAT_TEXT = "concat_text"
    CROSS_ATTENTION_READOUT = "cross_attn_readout"


@dataclass(frozen=True)
class EchoMemoryRecipe:
    """Architecture and sampler semantics pinned by one model identity.

    A recipe is deliberately not a bag of mutable runtime flags.  Public model
    IDs refer to exactly one recipe, which makes comparisons reproducible and
    lets new research methods register a sibling recipe without changing the
    behavior of released models.
    """

    recipe_id: str
    mechanism: EchoMemoryMechanism
    context_frames: int
    spatial_tokens: int = 0
    spatial_injection: SpatialInjection = SpatialInjection.NONE
    action_attention: bool = True
    temporal_memory_every_n_blocks: int | None = None
    ssm_num_blocks_hint: int | None = None
    action_dim: int = 12
    action_temporal_stride: int = 4
    context_position: str = "suffix"
    target_only_cfg: bool = True
    freeze_context_after_step: bool = True
    default_sample_shift: float = 5.0

    def __post_init__(self) -> None:
        if not self.recipe_id.strip():
            raise ValueError("recipe_id must be non-empty")
        if self.context_frames <= 0:
            raise ValueError("context_frames must be positive")
        if self.action_dim != 12:
            raise ValueError("released Echo-Memory checkpoints require 12D RT actions")
        if self.action_temporal_stride <= 0:
            raise ValueError("action_temporal_stride must be positive")
        if self.default_sample_shift <= 0:
            raise ValueError("default_sample_shift must be positive")
        if self.context_position != "suffix":
            raise ValueError("released Echo-Memory recipes use target-then-context suffix layout")
        if self.mechanism is EchoMemoryMechanism.SPATIAL_GRID and self.spatial_tokens <= 0:
            raise ValueError("spatial-grid recipes require spatial_tokens")
        if self.mechanism is not EchoMemoryMechanism.SPATIAL_GRID and self.spatial_tokens:
            raise ValueError("only spatial-grid recipes may set spatial_tokens")
        temporal_mechanisms = {
            EchoMemoryMechanism.BLOCK_STATE_SPACE,
            EchoMemoryMechanism.VIDEO_SSM_HYBRID,
        }
        if self.mechanism in temporal_mechanisms:
            if not self.temporal_memory_every_n_blocks or self.temporal_memory_every_n_blocks <= 0:
                raise ValueError("state-space recipes require a positive block cadence")
        elif self.temporal_memory_every_n_blocks is not None:
            raise ValueError("only state-space recipes may set a block cadence")
        if self.ssm_num_blocks_hint is not None:
            if self.mechanism is not EchoMemoryMechanism.BLOCK_STATE_SPACE:
                raise ValueError("ssm_num_blocks_hint only applies to block-state-space recipes")
            if self.ssm_num_blocks_hint <= 0:
                raise ValueError("ssm_num_blocks_hint must be positive")


_RECIPES = {
    "context-k1": EchoMemoryRecipe(
        recipe_id="context-k1",
        mechanism=EchoMemoryMechanism.RAW_CONTEXT,
        context_frames=1,
    ),
    "context-k20": EchoMemoryRecipe(
        recipe_id="context-k20",
        mechanism=EchoMemoryMechanism.RAW_CONTEXT,
        context_frames=20,
    ),
    "spatial-grid-64": EchoMemoryRecipe(
        recipe_id="spatial-grid-64",
        mechanism=EchoMemoryMechanism.SPATIAL_GRID,
        context_frames=1,
        spatial_tokens=64,
        spatial_injection=SpatialInjection.CONCAT_TEXT,
        default_sample_shift=15.0,
    ),
    "block-ssm-k5": EchoMemoryRecipe(
        recipe_id="block-ssm-k5",
        mechanism=EchoMemoryMechanism.BLOCK_STATE_SPACE,
        context_frames=5,
        temporal_memory_every_n_blocks=4,
        ssm_num_blocks_hint=21,
        default_sample_shift=15.0,
    ),
    "videossm-hybrid-k5": EchoMemoryRecipe(
        recipe_id="videossm-hybrid-k5",
        mechanism=EchoMemoryMechanism.VIDEO_SSM_HYBRID,
        context_frames=5,
        temporal_memory_every_n_blocks=4,
        default_sample_shift=15.0,
    ),
    "spatial-concat-text-64": EchoMemoryRecipe(
        recipe_id="spatial-concat-text-64",
        mechanism=EchoMemoryMechanism.SPATIAL_GRID,
        context_frames=1,
        spatial_tokens=64,
        spatial_injection=SpatialInjection.CONCAT_TEXT,
        default_sample_shift=15.0,
    ),
    "spatial-no-injection-64": EchoMemoryRecipe(
        recipe_id="spatial-no-injection-64",
        mechanism=EchoMemoryMechanism.SPATIAL_GRID,
        context_frames=1,
        spatial_tokens=64,
        spatial_injection=SpatialInjection.NONE,
        default_sample_shift=15.0,
    ),
    "spatial-cross-attn-32": EchoMemoryRecipe(
        recipe_id="spatial-cross-attn-32",
        mechanism=EchoMemoryMechanism.SPATIAL_GRID,
        context_frames=1,
        spatial_tokens=32,
        spatial_injection=SpatialInjection.CROSS_ATTENTION_READOUT,
        default_sample_shift=15.0,
    ),
    "block-ssm-k1-every4-hint21": EchoMemoryRecipe(
        recipe_id="block-ssm-k1-every4-hint21",
        mechanism=EchoMemoryMechanism.BLOCK_STATE_SPACE,
        context_frames=1,
        temporal_memory_every_n_blocks=4,
        ssm_num_blocks_hint=21,
        default_sample_shift=15.0,
    ),
    "block-ssm-k5-every1-hint21": EchoMemoryRecipe(
        recipe_id="block-ssm-k5-every1-hint21",
        mechanism=EchoMemoryMechanism.BLOCK_STATE_SPACE,
        context_frames=5,
        temporal_memory_every_n_blocks=1,
        ssm_num_blocks_hint=21,
        default_sample_shift=15.0,
    ),
    "block-ssm-k5-every4-hint81": EchoMemoryRecipe(
        recipe_id="block-ssm-k5-every4-hint81",
        mechanism=EchoMemoryMechanism.BLOCK_STATE_SPACE,
        context_frames=5,
        temporal_memory_every_n_blocks=4,
        ssm_num_blocks_hint=81,
        default_sample_shift=15.0,
    ),
}

ECHO_MEMORY_RECIPES: Mapping[str, EchoMemoryRecipe] = MappingProxyType(_RECIPES)


def get_echo_memory_recipe(recipe_id: str) -> EchoMemoryRecipe:
    """Return the frozen recipe for ``recipe_id``.

    Raises:
        KeyError: if the id is not in :data:`ECHO_MEMORY_RECIPES`.
    """

    try:
        return ECHO_MEMORY_RECIPES[str(recipe_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(ECHO_MEMORY_RECIPES))
        raise KeyError(f"unknown Echo-Memory recipe {recipe_id!r}; choices: {choices}") from exc


__all__ = [
    "ECHO_MEMORY_RECIPES",
    "EchoMemoryMechanism",
    "EchoMemoryRecipe",
    "SpatialInjection",
    "get_echo_memory_recipe",
]
