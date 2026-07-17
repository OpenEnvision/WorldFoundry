"""Compatibility exports for the shared TensorFlow CFID implementation."""

from worldfoundry.evaluation.tasks.metrics._shared.conditional_frechet import (
    cfid,
    no_embedding,
    sample_covariance,
    symmetric_matrix_square_root,
    trace_sqrt_product,
)

__all__ = [
    "cfid",
    "no_embedding",
    "sample_covariance",
    "symmetric_matrix_square_root",
    "trace_sqrt_product",
]
