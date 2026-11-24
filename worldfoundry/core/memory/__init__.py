"""WorldFoundry memory infrastructure.

One canonical stack:

- :mod:`.store` / :mod:`.retrieval` — bounded records and deterministic top-k
  scoring (type, metadata, text, recency).
- :mod:`.base` — ABC for ingest / select / compress / process / manage.
- :mod:`.conditioning` — how retrieved memory participates in denoising
  (sequence segments, CFG, freeze-after-step).
- :mod:`.media` — path / tensor / PIL / video normalization for artifacts.
"""

# ──────────────────────────────────────────────────────────────────────────
# Eager exports — this package is torch-free at import
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
from .store import MemoryQuery, MemoryRecord, MemorySelection, MemoryStore

__all__ = [
    "BaseMemory",
    "DenoisingLayout",
    "DenoisingMemoryAdapter",
    "GuidanceMode",
    "MemoryCondition",
    "MemoryQuery",
    "MemoryRecord",
    "MemorySelection",
    "MemoryStore",
    "SequenceRole",
    "SequenceSegment",
    "SequenceUpdate",
]
