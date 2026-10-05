"""BiWM's 81 camera labels and latent-frame action grammar."""
from __future__ import annotations

import random
import re

import torch

TOKEN_LABELS = {"w": 9, "s": 18, "d": 27, "a": 36,
                "up": 1, "down": 2, "right": 3, "left": 4}
_TRANSLATIONS = ["does not move", "moves forward", "moves backward", "moves right",
                 "moves left", "moves forward-right", "moves forward-left",
                 "moves backward-right", "moves backward-left"]
_ROTATIONS = ["does not rotate", "pitches up", "pitches down", "yaws right", "yaws left",
              "pitches up and yaws right", "pitches up and yaws left",
              "pitches down and yaws right", "pitches down and yaws left"]
ACTION_TEXTS = [f"Camera {translation}. Camera {rotation}."
                for translation in _TRANSLATIONS for rotation in _ROTATIONS]


def action_labels(num_frames, *, actions=None, action_label=None, seed=0):
    """Frame zero is static; segment lengths count latent frames, not pixels."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if actions is not None and action_label is not None:
        raise ValueError("Provide actions or action_label, not both")
    labels = [0]
    if action_label is not None:
        if type(action_label) is not int or not 0 <= action_label <= 80:
            raise ValueError("action_label must be an integer in [0,80]")
        labels.extend([action_label] * (num_frames - 1))
    elif actions is not None:
        for segment in actions.split(","):
            match = re.fullmatch(r"\s*([a-zA-Z]+|\d+)\s*-\s*(\d+)\s*", segment)
            if match is None:
                raise ValueError(f"Invalid action segment: {segment!r}; expected 'w-8,right-12'")
            token, count = match.groups()
            label = int(token) if token.isdigit() else TOKEN_LABELS.get(token.lower(), -1)
            if not 0 <= label <= 80 or int(count) < 1:
                raise ValueError(f"Invalid action label or duration: {segment!r}")
            labels.extend([label] * min(int(count), max(0, num_frames - len(labels))))
        labels.extend([labels[-1]] * (num_frames - len(labels)))
    else:
        rng = random.Random(seed)
        choices = [0, 9, 18, 36, 27, 1, 2, 4, 3]
        while len(labels) < num_frames:
            label, count = rng.choice(choices), rng.randint(4, 16)
            labels.extend([label] * count)
    return torch.tensor(labels[:num_frames], dtype=torch.long).unsqueeze(0)
