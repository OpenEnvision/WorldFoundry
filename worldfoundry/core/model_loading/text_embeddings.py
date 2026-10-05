"""Text-embedding configuration types.

Video and world-model runtimes stack encoder hidden states in more
than one way. :class:`EmbeddingConcatStrategy` names the supported
layouts: full concatenation, mean pooling, or pooling every N layers
then concatenating.

This module holds configuration and validation only; encoding and pooling
live with the text encoder that consumes the strategy.
"""

from enum import Enum

DEFAULT_EMBEDDING_CACHE_MAX_BYTES = 512 * 1024 * 1024

# ──────────────────────────────────────────────────────────────────────────
# Layout names only — encoding / pooling live on the consuming text encoder
# ──────────────────────────────────────────────────────────────────────────


class EmbeddingConcatStrategy(str, Enum):
    """How stacked text-encoder hidden states are reduced to one embedding."""

    FULL_CONCAT = "full_concat"
    MEAN_POOLING = "mean_pooling"
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __str__(self) -> str:
        """Return the on-the-wire value so YAML/logs use ``full_concat``, not the member name."""
        return self.value


def validate_prompt_encoder_options(
    *,
    run_on_cpu: bool = False,
    embedding_cache_size: int = 0,
    embedding_cache_max_bytes: int = DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
) -> dict[str, bool | int]:
    """Validate placement and cache bounds before loading encoder weights."""

    if not isinstance(run_on_cpu, bool):
        raise TypeError("run_on_cpu must be a bool")
    for name, value in (
        ("embedding_cache_size", embedding_cache_size),
        ("embedding_cache_max_bytes", embedding_cache_max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    return {
        "run_on_cpu": run_on_cpu,
        "embedding_cache_size": embedding_cache_size,
        "embedding_cache_max_bytes": embedding_cache_max_bytes,
    }


__all__ = ["DEFAULT_EMBEDDING_CACHE_MAX_BYTES", "EmbeddingConcatStrategy", "validate_prompt_encoder_options"]
