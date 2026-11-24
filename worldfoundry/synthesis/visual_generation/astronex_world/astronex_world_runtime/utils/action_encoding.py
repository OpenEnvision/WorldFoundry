"""Build the 64-wide action vectors the CrossFPS-trained embedder reads.

The lane layout comes from build_crossfps_lmdb.py:

    [0:2]   left stick   (LX, LY)  -- movement, continuous in [-1, 1]
    [2:4]   right stick  (RX, RY)  -- camera,   continuous in [-1, 1]
    [4:21]  17 buttons, binary
    [21:]   zero, reserved

The dataset ships no sign convention, so each axis was read off the footage by
correlating the recorded stick value against optical flow. Every number below
is that measurement, not an assumption:

    LX  positive = strafe right.  LX +0.81 -> content drifts left  (-157 px)
                                  LX -0.65 -> content drifts right (+85 px)
    LY  positive = move forward.  LY +0.91 -> +150% radial zoom
                                  LY -0.91 ->  +13% (players rarely back up,
                                  so the negative end is weak evidence)
    RX  positive = turn right.    n=20, r = -0.707 against horizontal flow
    RY  positive = look down.     n=16, r = -0.640 against vertical flow,
                                  measured on clips where both left-stick axes
                                  and RX are near zero so pitch is isolated

Frame 0 is always all-zero. The encoder does the same thing, for the reason its
docstring gives: nothing has happened at t=0, and seeding it with the first
command shifts the whole control stream one frame early.
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 64
LX, LY, RX, RY = 0, 1, 2, 3

# WBench's navigation vocabulary onto the sticks. Its W/A/S/D are translations
# and its arrows are rotations, which is exactly the left/right stick split.
WBENCH_ACTION = {
    "W": {LY: +1.0},
    "S": {LY: -1.0},
    "D": {LX: +1.0},
    "A": {LX: -1.0},
    "right": {RX: +1.0},
    "left": {RX: -1.0},
    "down": {RY: +1.0},
    "up": {RY: -1.0},
}

# A keyboard/mouse binding over the same lanes, so a live session can drive the
# model with the keys players actually use.
KEYBOARD = {
    "w": {LY: +1.0},
    "s": {LY: -1.0},
    "d": {LX: +1.0},
    "a": {LX: -1.0},
    "arrowright": {RX: +1.0},
    "arrowleft": {RX: -1.0},
    "arrowdown": {RY: +1.0},
    "arrowup": {RY: -1.0},
}


def from_tokens(tokens, table=WBENCH_ACTION, magnitude: float = 1.0) -> np.ndarray:
    """(len(tokens)+1, 64) from one action name per latent frame.

    ``tokens`` is a per-frame sequence, so a four-turn case is expanded to its
    frames before being passed here rather than being summarised into one
    vector. Unknown names raise: a silently zeroed action is a held stick, which
    is a legitimate command, and would hide a typo as "the model ignored it".
    """
    tokens = list(tokens)
    out = np.zeros((len(tokens) + 1, ACTION_DIM), dtype=np.float32)
    for i, tok in enumerate(tokens):
        for part in str(tok).split("+"):
            part = part.strip()
            if not part or part in ("h", "hold", "none"):
                continue
            if part not in table:
                raise ValueError(f"unknown action {part!r}; known: {sorted(table)}")
            for lane, v in table[part].items():
                # Compound actions address distinct lanes; a repeat would mean
                # two commands on one axis, which the sticks cannot express.
                if out[i + 1, lane] != 0.0:
                    raise ValueError(f"{tok!r} drives lane {lane} twice")
                out[i + 1, lane] = v * magnitude
    return out
