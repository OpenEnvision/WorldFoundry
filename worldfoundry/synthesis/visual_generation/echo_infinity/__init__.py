"""
This module serves as the primary entry point for the EchoInfinity project's synthesis capabilities.

It re-exports the `EchoInfinitySynthesis` class from the `synthesis` submodule,
making it directly accessible when importing from the package root.
"""

from __future__ import annotations

from .synthesis import EchoInfinitySynthesis

__all__ = ["EchoInfinitySynthesis"]
