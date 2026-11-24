"""
This module serves as the public interface for the fantasy world synthesis package.

It re-exports constants defining default repository URLs for various fantasy world
configurations (MOGE2, WAN21, WAN22) and provides access to classes
responsible for the synthesis of WAN21 and WAN22 fantasy world components.
"""

from .fantasy_world_wan21_synthesis import FantasyWorldWan21Synthesis
from .fantasy_world_wan22_synthesis import FantasyWorldWan22Synthesis
from .runtime_env import (
    DEFAULT_FANTASY_WORLD_MOGE2_REPO,
    DEFAULT_FANTASY_WORLD_WAN21_REPO,
    DEFAULT_FANTASY_WORLD_WAN22_REPO,
)

__all__ = [
    "DEFAULT_FANTASY_WORLD_MOGE2_REPO",
    "DEFAULT_FANTASY_WORLD_WAN21_REPO",
    "DEFAULT_FANTASY_WORLD_WAN22_REPO",
    "FantasyWorldWan21Synthesis",
    "FantasyWorldWan22Synthesis",
]
