"""Validation for cached DreamX prompt-embedding inputs."""

from __future__ import annotations

from typing import Any


def _embedding_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"Prompt embedding values must expose a shape, got {type(value)!r}.")
    return tuple(int(dimension) for dimension in shape)


def validate_prompt_embedding_pair(positive: Any, negative: Any) -> None:
    """Validate tensor or per-sample positive/negative embedding pairs.

    Variable token lengths are valid for list-based cached embeddings; rank and
    hidden width must still agree for every pair.
    """

    positive_batch = isinstance(positive, (list, tuple))
    negative_batch = isinstance(negative, (list, tuple))
    if positive_batch != negative_batch:
        raise ValueError("Positive and negative prompt embeddings must use the same container.")
    if not positive_batch:
        if _embedding_shape(positive) != _embedding_shape(negative):
            raise ValueError("Tensor prompt embeddings must have equal shapes.")
        return
    if len(positive) != len(negative):
        raise ValueError("Positive and negative prompt embedding batches must be equal.")
    for positive_item, negative_item in zip(positive, negative):
        positive_shape = _embedding_shape(positive_item)
        negative_shape = _embedding_shape(negative_item)
        if (
            len(positive_shape) != len(negative_shape)
            or not positive_shape
            or positive_shape[-1] != negative_shape[-1]
        ):
            raise ValueError("Prompt embedding rank and hidden width must match.")


__all__ = ["validate_prompt_embedding_pair"]
