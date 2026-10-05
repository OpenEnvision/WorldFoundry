"""Inference-only adapters for the official LTX-2.5 pipelines."""

from .runtime import LTX25DistilledRuntime, LTX25RuntimePlan

__all__ = ["LTX25DistilledRuntime", "LTX25RuntimePlan"]
