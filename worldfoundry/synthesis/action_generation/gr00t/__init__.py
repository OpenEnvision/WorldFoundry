"""
Package entry point for the GR00T synthesis module.

This module primarily re-exports the `GR00TSynthesis` class from the `gr00t_synthesis`
submodule, making it directly accessible when importing the package.
"""

from .gr00t_synthesis import GR00TSynthesis

__all__ = ["GR00TSynthesis"]
