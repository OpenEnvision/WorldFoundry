"""
This module serves as the main entry point for the worldfm_synthesis package,
re-exporting key components for direct access.

It makes the default WorldFM repository path and the main WorldFMSynthesis class
available at the top level of the package.
"""

from .worldfm_synthesis import DEFAULT_WORLDFM_REPO, WorldFMSynthesis

__all__ = ["DEFAULT_WORLDFM_REPO", "WorldFMSynthesis"]
