"""Checkpoint-visible Echo-Memory architecture contract.

This module describes which optional network branches a released checkpoint
contains.  It deliberately contains no checkpoint I/O or model construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import EchoMemoryMechanism, EchoMemoryRecipe, SpatialInjection


class EchoArchitectureError(ValueError):
    """Raised when checkpoint-visible branches do not match a recipe."""


@dataclass(frozen=True)
class EchoCheckpointArchitecture:
    """Architecture slots declared by checkpoint tensor names."""

    action_blocks: tuple[int, ...]
    action_attention_blocks: tuple[int, ...]
    block_ssm_blocks: tuple[int, ...]
    video_ssm_blocks: tuple[int, ...]
    spatial_grid_shape: tuple[int, int] | None
    has_spatial_readout: bool


def validate_echo_checkpoint_architecture(
    architecture: EchoCheckpointArchitecture,
    recipe: EchoMemoryRecipe,
    *,
    num_blocks: int = 30,
) -> None:
    """Require the checkpoint-visible structure pinned by ``recipe``."""

    expected_actions = tuple(range(num_blocks))
    if architecture.action_blocks != expected_actions:
        missing = sorted(set(expected_actions).difference(architecture.action_blocks))
        extra = sorted(set(architecture.action_blocks).difference(expected_actions))
        raise EchoArchitectureError(
            f"Echo action projection coverage mismatch: missing_blocks={missing}, unexpected_blocks={extra}"
        )
    expected_action_attention = expected_actions if recipe.action_attention else ()
    if architecture.action_attention_blocks != expected_action_attention:
        missing = sorted(set(expected_action_attention).difference(architecture.action_attention_blocks))
        extra = sorted(set(architecture.action_attention_blocks).difference(expected_action_attention))
        raise EchoArchitectureError(
            "Echo action-attention coverage does not match the immutable recipe: "
            f"missing_blocks={missing}, unexpected_blocks={extra}"
        )
    for name, block_ids in (
        ("action-attention", architecture.action_attention_blocks),
        ("block-wise SSM", architecture.block_ssm_blocks),
        ("VideoSSM", architecture.video_ssm_blocks),
    ):
        out_of_range = [index for index in block_ids if index < 0 or index >= num_blocks]
        if out_of_range:
            raise EchoArchitectureError(f"{name} declares out-of-range blocks: {out_of_range}")

    mechanism = recipe.mechanism
    if mechanism is EchoMemoryMechanism.SPATIAL_GRID:
        if architecture.spatial_grid_shape is None:
            raise EchoArchitectureError("spatial recipe requires spatial_memory_module weights")
        _, tokens = architecture.spatial_grid_shape
        if tokens != recipe.spatial_tokens:
            raise EchoArchitectureError(
                f"spatial token count mismatch: recipe={recipe.spatial_tokens}, checkpoint={tokens}"
            )
        grid_cells = architecture.spatial_grid_shape[0]
        grid_size = round(grid_cells**0.5)
        if grid_size * grid_size != grid_cells:
            raise EchoArchitectureError(f"spatial grid must be square, checkpoint has {grid_cells} cells")
        if (
            recipe.spatial_injection is SpatialInjection.CROSS_ATTENTION_READOUT
            and not architecture.has_spatial_readout
        ):
            raise EchoArchitectureError("cross-attention spatial recipe requires spatial_memory_readout_module weights")
    elif architecture.spatial_grid_shape is not None:
        raise EchoArchitectureError("non-spatial recipe received spatial memory tensors")

    if mechanism is EchoMemoryMechanism.BLOCK_STATE_SPACE:
        cadence = int(recipe.temporal_memory_every_n_blocks or 0)
        expected_blocks = tuple(range(0, num_blocks, cadence))
        if architecture.block_ssm_blocks != expected_blocks:
            raise EchoArchitectureError(
                "block-wise SSM coverage does not match the immutable recipe: "
                f"expected_blocks={list(expected_blocks)}, "
                f"checkpoint_blocks={list(architecture.block_ssm_blocks)}"
            )
        if architecture.video_ssm_blocks:
            raise EchoArchitectureError("block-state-space recipe cannot also contain VideoSSM tensors")
    elif architecture.block_ssm_blocks:
        raise EchoArchitectureError("checkpoint contains block-wise SSM tensors for a different recipe")

    if mechanism is EchoMemoryMechanism.VIDEO_SSM_HYBRID:
        cadence = int(recipe.temporal_memory_every_n_blocks or 0)
        expected_blocks = tuple(range(0, num_blocks, cadence))
        if architecture.video_ssm_blocks != expected_blocks:
            raise EchoArchitectureError(
                "VideoSSM coverage does not match the immutable recipe: "
                f"expected_blocks={list(expected_blocks)}, "
                f"checkpoint_blocks={list(architecture.video_ssm_blocks)}"
            )
        if architecture.block_ssm_blocks:
            raise EchoArchitectureError("VideoSSM recipe cannot also contain block-wise SSM tensors")
    elif architecture.video_ssm_blocks:
        raise EchoArchitectureError("checkpoint contains VideoSSM tensors for a different recipe")


__all__ = [
    "EchoArchitectureError",
    "EchoCheckpointArchitecture",
    "validate_echo_checkpoint_architecture",
]
