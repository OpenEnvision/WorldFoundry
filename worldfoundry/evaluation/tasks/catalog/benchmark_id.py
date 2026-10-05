"""Canonical benchmark-id normalization."""

from __future__ import annotations


def normalize_benchmark_id(value: str | None) -> str:
    """Strip, casefold, and treat underscores as hyphens."""
    return str(value or "").strip().casefold().replace("_", "-")
