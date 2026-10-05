"""Keyboard actions: parsing the per-frame `.jsonl` into a multi-hot tensor.

The 4-player Rocket League data is keyboard-only (no mouse/analog): every `.jsonl` line is
`{"keys": [...]}` with one line per video frame. We turn that into a multi-hot tensor over a
fixed key vocabulary, downsampling to `target_fps` by OR-ing key presses over each window.

Downsampling contract: integer-only downsampling, keys OR-ed over the window, int32 multi-hot. The
fixed vocabulary order below must stay stable so the multi-hot ordering is consistent across
consumers.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .clips import compute_stride

if TYPE_CHECKING:
    import torch

# Stable, documented ordering of the 9 keys present in the 4-player Rocket League data.
# (drive: W/S, steer/air: A/D, jump: Space, boost: LShiftKey, drift/air-roll: LControlKey,
#  plus Q/E powerslide/air-roll binds). These are the keys present in the data.
DEFAULT_RL_KEYS: tuple[str, ...] = (
    "W",
    "A",
    "S",
    "D",
    "Q",
    "E",
    "Space",
    "LShiftKey",
    "LControlKey",
)


@dataclass(frozen=True)
class KeyVocab:
    """Fixed key vocabulary defining the multi-hot ordering.

    `on_unknown` controls what happens when a `.jsonl` line contains a key outside the vocab:
    "warn" (default, drop + warn once per distinct key), "ignore" (drop silently), or "error" (raise).
    """

    keys: tuple[str, ...]
    on_unknown: Literal["warn", "ignore", "error"] = "warn"
    _index: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.on_unknown not in ("warn", "ignore", "error"):
            raise ValueError(f"on_unknown must be 'warn', 'ignore', or 'error', got {self.on_unknown!r}")
        object.__setattr__(self, "_index", {k: i for i, k in enumerate(self.keys)})

    @classmethod
    def default_rl(cls, on_unknown: Literal["warn", "ignore", "error"] = "warn") -> "KeyVocab":
        return cls(DEFAULT_RL_KEYS, on_unknown=on_unknown)

    def __len__(self) -> int:
        return len(self.keys)


def tensorize_actions(
    lines: Iterable[str | bytes],
    vocab: KeyVocab,
    source_fps: float,
    target_fps: int,
    keep_last_partial: bool = False,
) -> torch.Tensor:
    """Parse `.jsonl` action lines into a multi-hot tensor of shape (n_steps, len(vocab)) int32.

    Downsampling uses the same integer stride as frame decoding (`clips.compute_stride`), so frames
    and actions stay aligned. n_steps = floor(n_lines / stride); the trailing partial window is
    dropped by default, since a step is normally only emitted once a full window has accumulated.
    Set `keep_last_partial=True` to instead emit that trailing window OR-ed over whatever frames
    remain — used when a clip runs to the end of a chunk and its final window is short, so the real
    keys held over those frames are kept rather than lost.
    """
    import torch

    factor = compute_stride(source_fps, target_fps)
    n_keys = len(vocab)

    steps: list[torch.Tensor] = []
    window = torch.zeros(n_keys, dtype=torch.int32)
    count = 0
    unknown_seen: set[str] = set()

    for line in lines:
        # A blank line counts as a frame with no keys held (preserve per-frame alignment rather than
        # dropping the frame). `keys` may be absent or an explicit null ("no keys this frame"); both
        # mean an empty set.
        line = line.strip()
        keys = (json.loads(line).get("keys") or []) if line else []
        for k in keys:
            idx = vocab._index.get(k)
            if idx is None:
                if k not in unknown_seen and vocab.on_unknown != "ignore":
                    if vocab.on_unknown == "error":
                        raise ValueError(f"Unknown key {k!r} not in vocab {vocab.keys}")
                    warnings.warn(f"Dropping unknown key {k!r} (not in vocab)", stacklevel=2)
                    unknown_seen.add(k)
                continue
            window[idx] = 1  # OR over the downsampling window
        count += 1
        if count == factor:
            steps.append(window)
            window = torch.zeros(n_keys, dtype=torch.int32)
            count = 0

    if keep_last_partial and count > 0:
        steps.append(window)

    if not steps:
        return torch.zeros((0, n_keys), dtype=torch.int32)
    return torch.stack(steps, dim=0)
