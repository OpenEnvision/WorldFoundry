"""Source image used to create one PAWBench video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceImageIdentity:
    attached: bool
    uri: str | None = None
    absent_reason: str | None = None
