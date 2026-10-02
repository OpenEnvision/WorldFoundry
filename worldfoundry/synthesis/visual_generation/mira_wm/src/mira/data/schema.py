"""Typed schema for the Rocket League WebDataset index (`index.json`).

One WebDataset **sample = one (match, chunk)**: a ~4 s window of a match bundling all perspectives —
members `p{i}.mp4` / `p{i}.jsonl` per perspective (ordered by `player_id`) + `meta.json`; sample key
`{match_id}_c{chunk:05d}`. A clip is taken from within a single chunk.

`index.json` holds one `MatchEntry` per match (the random-access map). Unknown fields are allowed so
the schema tolerates extra metadata (e.g. `content_id`).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class Anchor(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_type: int
    event_name: str
    master_sec: float


class Perspective(BaseModel):
    """One player's view of a match. Anchors are on the shared match clock (identical across
    perspectives); `recording_offset_sec` places this view's frame 0 on that clock. `frames` /
    `duration` are the per-perspective totals across all chunks."""

    model_config = ConfigDict(extra="allow")
    player_id: int
    team: int
    frames: int
    duration: float
    recording_offset_sec: float = 0.0
    anchors: list[Anchor] = []


class MatchEntry(BaseModel):
    """One match. `chunk_frames` are the frame counts of the ordered chunks (shared across
    perspectives, since the POVs are frame-aligned); they sum to each perspective's `frames` and
    define the chunk boundaries (each clip stays within one chunk; boundaries also map events).
    `perspectives` are ordered by `player_id` (the p0..p3 axis). All of a match's chunk samples
    live in `shard`.

    `chunk_indices[c]` is the ORIGINAL source chunk index of the c-th present chunk. A match's
    present chunks can be non-contiguous (e.g. `[0,1,3,4,...]`); the sample key for the c-th chunk
    is `{match_id}_c{chunk_indices[c]:05d}`, not
    `_c{c:05d}`. When absent (contiguous datasets) it defaults to `range(len(chunk_frames))`."""

    model_config = ConfigDict(extra="allow")
    match_id: str
    shard: str
    n_players: int
    chunk_frames: list[int]
    chunk_indices: list[int] | None = None
    arena: str | None = None
    perspectives: list[Perspective]

    @model_validator(mode="after")
    def _check_chunk_indices(self) -> "MatchEntry":
        """`chunk_indices` must be a per-present-chunk map: one entry per `chunk_frames`, all
        distinct. A length mismatch or a duplicate would make `chunk_id` map a chunk to the wrong
        original index -- i.e. to the wrong `.mp4`/`.jsonl` member -- producing silently corrupt
        samples, so it is rejected at parse time rather than surfacing as bad data downstream."""
        if self.chunk_indices is not None:
            if len(self.chunk_indices) != len(self.chunk_frames):
                raise ValueError(
                    f"{self.match_id}: chunk_indices has {len(self.chunk_indices)} entries but "
                    f"chunk_frames has {len(self.chunk_frames)}"
                )
            if len(set(self.chunk_indices)) != len(self.chunk_indices):
                raise ValueError(f"{self.match_id}: chunk_indices has duplicate entries")
        return self

    def chunk_id(self, position: int) -> int:
        """Original source chunk index of the `position`-th present chunk (for the sample key)."""
        return self.chunk_indices[position] if self.chunk_indices is not None else position


class Index(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_samples: int  # number of matches (each match expands to len(chunk_frames) chunk samples)
    entries: list[MatchEntry]

    @classmethod
    def load(cls, index_path: str | Path) -> "Index":
        with open(index_path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))
