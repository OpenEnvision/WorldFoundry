"""Game events ("anchors") and their alignment to video frames.

Discrete events live in the per-perspective `anchors`, each `{event_type, event_name, master_sec}`,
where `master_sec` is on a single match-wide master clock shared by all 4 perspectives.

Each perspective's recording starts at a slightly different point on that master clock, captured
by `recording_offset_sec` (per perspective). To place an event on a *specific perspective's*
frame timeline:

    frame_index = round((master_sec - recording_offset_sec) * fps)

Note `recording_offset_sec` differs across the 4 perspectives of a match (typically by a few
frames), so the same event maps to slightly different frame indices per perspective. Frame-index
alignment ignores this offset; offset-corrected alignment uses the formula above.
"""

from __future__ import annotations

from dataclasses import dataclass

REPLAY_START = "GoalReplayStarted"
REPLAY_END = "GoalReplayEnded"


@dataclass(frozen=True)
class Event:
    event_type: int
    event_name: str
    master_sec: float

    def frame_index(self, fps: float, recording_offset_sec: float) -> int:
        """Frame index of this event on a given perspective's timeline (may be out of range)."""
        return round((self.master_sec - recording_offset_sec) * fps)


def parse_anchors(anchors: list) -> list[Event]:
    """Build Events from anchors given as dicts or as objects with the matching attributes."""

    def get(a, k):
        return a[k] if isinstance(a, dict) else getattr(a, k)

    return [Event(get(a, "event_type"), get(a, "event_name"), get(a, "master_sec")) for a in anchors]


def events_in_frame_window(
    events: list[Event], f_start: int, f_end: int, fps: float, recording_offset_sec: float
) -> list[Event]:
    """Events whose frame index falls in [f_start, f_end) on this perspective's timeline."""
    return [e for e in events if f_start <= e.frame_index(fps, recording_offset_sec) < f_end]


def replay_spans(
    events: list[Event], fps: float, recording_offset_sec: float, n_frames: int
) -> list[tuple[int, int]]:
    """Frame-index [start, end) spans covering goal-replay segments (non-live-gameplay).

    Pairs each GoalReplayStarted with the next GoalReplayEnded. Spans are clamped to
    [0, n_frames); a dangling start (replay running at recording end) extends to n_frames.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for e in sorted(events, key=lambda x: x.master_sec):
        f = e.frame_index(fps, recording_offset_sec)
        if e.event_name == REPLAY_START:
            start = f
        elif e.event_name == REPLAY_END and start is not None:
            lo, hi = max(0, min(start, f)), min(n_frames, max(start, f))
            if hi > lo:
                spans.append((lo, hi))
            start = None
    if start is not None:
        lo = max(0, min(start, n_frames))
        if n_frames > lo:
            spans.append((lo, n_frames))
    return spans


def overlaps_any(f_start: int, f_end: int, spans: list[tuple[int, int]]) -> bool:
    """Whether [f_start, f_end) intersects any span."""
    return any(f_start < hi and lo < f_end for lo, hi in spans)
