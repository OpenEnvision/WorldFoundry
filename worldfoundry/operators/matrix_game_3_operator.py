"""Module for the MatrixGame3 operator implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from PIL import Image

from .base_operator import BaseOperator


def load_pil_image(image_input) -> Image.Image:
    """Return a PIL image or an image file path as an RGB PIL image.

    Raises:
        TypeError: The input is neither a PIL image nor a string path.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


class MatrixGame3Operator(BaseOperator):
    """Build Matrix-Game-3 keyboard and mouse control sequences."""

    KEYBOARD_DIM = 6
    MOUSE_STEP = 0.1

    KEYBOARD_IDX: ClassVar[dict[str, int]] = {
        "forward": 0,
        "back": 1,
        "left": 2,
        "right": 3,
        "jump": 4,
        "attack": 5,
    }

    CAMERA_VALUE_MAP: ClassVar[dict[str, list[Any]]] = {
        "camera_up": [MOUSE_STEP, 0.0],
        "camera_down": [-MOUSE_STEP, 0.0],
        "camera_l": [0.0, -MOUSE_STEP],
        "camera_r": [0.0, MOUSE_STEP],
        "camera_ul": [MOUSE_STEP, -MOUSE_STEP],
        "camera_ur": [MOUSE_STEP, MOUSE_STEP],
        "camera_dl": [-MOUSE_STEP, -MOUSE_STEP],
        "camera_dr": [-MOUSE_STEP, MOUSE_STEP],
    }

    COMBINATION_MAP: ClassVar[dict[str, Any]] = {
        "forward_left": ["forward", "left"],
        "forward_right": ["forward", "right"],
        "back_left": ["back", "left"],
        "back_right": ["back", "right"],
        "nomove": [],
        "idle": [],
    }

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
        frames_per_action: int = 12,
        first_clip_frames: int = 57,
        clip_overlap_frames: int = 16,
    ):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=operation_types or ["action_instruction", "visual_instruction"])
        self.frames_per_action = int(frames_per_action)
        self.first_clip_frames = int(first_clip_frames)
        self.clip_frame = 56
        self.clip_overlap_frames = int(clip_overlap_frames)
        self.followup_frames = self.clip_frame - self.clip_overlap_frames
        default_template = (
            list(self.KEYBOARD_IDX.keys()) + list(self.CAMERA_VALUE_MAP.keys()) + list(self.COMBINATION_MAP.keys())
        )
        self.interaction_template = interaction_template or sorted(set(default_template))
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        normalized = self._normalize_interaction(interaction)
        action = normalized["action"]
        if action not in self.interaction_template:
            raise ValueError(f"{action} not in template. Available: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, (list, tuple)):
            normalized = [self._normalize_interaction(item) for item in interaction]
        else:
            normalized = [self._normalize_interaction(interaction)]
        for item in normalized:
            self.check_interaction(item)
        self.current_interaction.append(normalized)

    def process_interaction(
        self,
        num_frames: int | None = None,
        num_iterations: int | None = None,
        frames_per_action: int | None = None,
    ) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise RuntimeError("No interaction registered")

        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        actions = [item["action"] for item in latest_interaction]
        if len(actions) == 0:
            actions = ["nomove"]

        frames_per_action = int(frames_per_action or self.frames_per_action)
        if num_iterations is None:
            target_frames = (
                num_frames
                if num_frames is not None
                else max(
                    self.first_clip_frames,
                    len(actions) * frames_per_action,
                )
            )
            if target_frames <= self.first_clip_frames:
                num_iterations = 1
            else:
                num_iterations = 1 + math.ceil((int(target_frames) - self.first_clip_frames) / self.followup_frames)
        num_iterations = max(int(num_iterations), 1)
        total_frames = self.first_clip_frames + (num_iterations - 1) * self.followup_frames

        keyboard_condition, mouse_condition = self._build_sequence(
            actions,
            total_frames=total_frames,
            frames_per_action=frames_per_action,
        )
        return {
            "actions": actions,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition,
            "num_frames": total_frames,
            "num_iterations": num_iterations,
            "frames_per_action": frames_per_action,
        }

    def process_perception(self, input_image):
        """Process perception inputs like images, videos, and reference frames."""
        return load_pil_image(input_image)

    def _normalize_interaction(self, interaction: str | dict[str, Any]) -> dict[str, str]:
        if isinstance(interaction, str):
            return {"action": interaction}
        if isinstance(interaction, dict):
            action = interaction.get("action") or interaction.get("signal") or interaction.get("interaction")
            if action is None:
                raise ValueError("MatrixGame3 interaction dict requires 'action', 'signal', or 'interaction'.")
            return {"action": action}
        raise TypeError(f"Unsupported interaction type: {type(interaction)}")

    def _encode_action(self, action: str):
        action_list = self.COMBINATION_MAP.get(action, [action])
        keyboard = torch.zeros(self.KEYBOARD_DIM, dtype=torch.float32)
        mouse = torch.zeros(2, dtype=torch.float32)

        for item in action_list:
            if item in self.KEYBOARD_IDX:
                keyboard[self.KEYBOARD_IDX[item]] = 1.0
            elif item in self.CAMERA_VALUE_MAP:
                mouse = torch.tensor(self.CAMERA_VALUE_MAP[item], dtype=torch.float32)
            elif item in {"nomove", "idle"}:
                continue
            else:
                raise ValueError(f"Unsupported MatrixGame3 action: {action}")
        return keyboard, mouse

    def _build_sequence(
        self,
        actions: Sequence[str],
        *,
        total_frames: int,
        frames_per_action: int,
    ):
        keyboard_rows: list[torch.Tensor] = []
        mouse_rows: list[torch.Tensor] = []

        for action in actions:
            keyboard, mouse = self._encode_action(action)
            keyboard_rows.extend([keyboard.clone()] * frames_per_action)
            mouse_rows.extend([mouse.clone()] * frames_per_action)

        if not keyboard_rows:
            keyboard_rows.append(torch.zeros(self.KEYBOARD_DIM, dtype=torch.float32))
            mouse_rows.append(torch.zeros(2, dtype=torch.float32))

        while len(keyboard_rows) < total_frames:
            keyboard_rows.append(keyboard_rows[-1].clone())
            mouse_rows.append(mouse_rows[-1].clone())

        keyboard_condition = torch.stack(keyboard_rows[:total_frames])
        mouse_condition = torch.stack(mouse_rows[:total_frames])
        return keyboard_condition, mouse_condition
