"""mira.data: loader for the 4-player Rocket League dataset.

Each sample bundles the 4 perspectives of one ~4 s match chunk; a clip is taken from within one chunk.

Public API:
    RocketScienceDataset, MatchClip — load time-aligned 4-perspective clips (random access / streaming)
    Index, MatchEntry, Perspective, Anchor — typed schema for the dataset index (`index.json`)
    Vec3, Quat, GameInfo, BallState, CarState, FrameState — typed per-frame game state
    KeyVocab, DEFAULT_RL_KEYS, tensorize_actions — multi-hot keyboard action parsing
    Event, replay_spans — discrete game events with frame-index mapping

The `physics` and `viz` submodules are optional helpers over a clip's per-frame state: `physics` is
numpy-only, while `viz` needs the `viz` extra plus a system `ffmpeg` on PATH.
"""

from .actions import DEFAULT_RL_KEYS, KeyVocab, tensorize_actions
from .dataset import MatchClip, RocketScienceDataset
from .events import Event, replay_spans
from .schema import Anchor, Index, MatchEntry, Perspective
from .state import BallState, CarState, FrameState, GameInfo, Quat, Vec3

__all__ = [
    "RocketScienceDataset",
    "MatchClip",
    "Index",
    "MatchEntry",
    "Perspective",
    "Anchor",
    "Vec3",
    "Quat",
    "GameInfo",
    "BallState",
    "CarState",
    "FrameState",
    "KeyVocab",
    "DEFAULT_RL_KEYS",
    "tensorize_actions",
    "Event",
    "replay_spans",
]
