"""Mechanics package initializer for LV-Bench 2.

This file marks `mechanics` as a package and exposes the primary entry points
so that running scripts from the repo root (`python mechanics/integrated_pipeline.py ...`)
can resolve intra-package imports reliably.
"""

# Public re-exports for convenience
# from .object_segmentation import process_mechanics_pipeline  # noqa: F401
# from .integrated_pipeline import run_integrated_mechanics_pipeline  # noqa: F401

"""
__all__ = [
    "process_mechanics_pipeline",
    "run_integrated_mechanics_pipeline",
]
"""

# Version metadata
__version__ = "1.0.1"
