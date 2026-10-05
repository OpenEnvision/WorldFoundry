"""WorldFoundry memory infrastructure.

One canonical stack:

- :mod:`.store` / :mod:`.retrieval` — bounded records and deterministic top-k
  scoring (type, metadata, text, recency).
- :mod:`.base` — ABC for ingest / select / compress / process / manage.
- :mod:`.conditioning` — how retrieved memory participates in denoising
  (sequence segments, CFG, freeze-after-step).
- :mod:`.mosaic` — MosaicMem-style 3D patch store and viewpoint retrieval.
- :mod:`.media` — path / tensor / PIL / video normalization for artifacts.
"""

# ──────────────────────────────────────────────────────────────────────────
# Eager exports — this package is torch-free at import except mosaic geometry
# ──────────────────────────────────────────────────────────────────────────

from .base import BaseMemory
from .conditioning import (
    DenoisingLayout,
    DenoisingMemoryAdapter,
    GuidanceMode,
    MemoryCondition,
    SequenceRole,
    SequenceSegment,
    SequenceUpdate,
)
from .mosaic import (
    CameraIntrinsics,
    CameraPose,
    LatentCanvas,
    MemoryRetriever,
    MosaicFrame,
    MosaicMemoryConfig,
    MosaicMemoryStore,
    Patch3D,
    RetrievedPatch,
)
from .store import MemoryQuery, MemoryRecord, MemorySelection, MemoryStore

__all__ = [
    "BaseMemory",
    "CameraIntrinsics",
    "CameraPose",
    "DenoisingLayout",
    "DenoisingMemoryAdapter",
    "GuidanceMode",
    "LatentCanvas",
    "MemoryCondition",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRetriever",
    "MemorySelection",
    "MemoryStore",
    "MosaicFrame",
    "MosaicMemoryConfig",
    "MosaicMemoryStore",
    "Patch3D",
    "RetrievedPatch",
    "SequenceRole",
    "SequenceSegment",
    "SequenceUpdate",
]
