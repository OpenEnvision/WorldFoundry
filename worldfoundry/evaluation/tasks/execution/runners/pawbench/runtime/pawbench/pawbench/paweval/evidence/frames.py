"""Sampled video frames used by PAWEval."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class EvidenceFrame:
    frame_id: str
    phase: str
    uri: str
    timestamp_s: float | None = None
