"""The local media passed to one PAWEval judgment."""

from __future__ import annotations

from dataclasses import dataclass

from .frames import EvidenceFrame
from .source_image import SourceImageIdentity


@dataclass(frozen=True)
class EvidencePackage:
    sample_id: str
    scene_id: str
    frames: tuple[EvidenceFrame, ...]
    source_image: SourceImageIdentity
