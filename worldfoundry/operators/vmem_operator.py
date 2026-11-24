"""Module for the VMem operator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base_operator import BaseOperator

_ACTION_ALIASES = {
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "camera_l": "camera_l",
    "camera_left": "camera_l",
    "camera_r": "camera_r",
    "camera_right": "camera_r",
    "forward_left": "forward_left",
    "forward_right": "forward_right",
    "backward_left": "backward_left",
    "backward_right": "backward_right",
}

_ACTION_TO_VMEM_COMMANDS = {
    "forward": ["w"],
    "backward": ["s"],
    "left": ["a"],
    "right": ["d"],
    "camera_l": ["a"],
    "camera_r": ["d"],
    "forward_left": ["a", "w"],
    "forward_right": ["d", "w"],
    "backward_left": ["a", "s"],
    "backward_right": ["d", "s"],
}


def _normalize_action(action: str) -> str:
    key = action.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ACTION_ALIASES:
        raise ValueError(f"Unsupported VMem interaction: {action}")
    return _ACTION_ALIASES[key]


def _to_pil_image(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, str):
        return Image.open(data).convert("RGB")
    if isinstance(data, np.ndarray):
        array = np.asarray(data)
        if array.ndim == 4:
            array = array[0]
        if array.dtype in (np.float16, np.float32, np.float64):
            if array.min() >= -1.0 and array.max() <= 1.0:
                array = (array + 1.0) * 127.5 if array.min() < 0.0 else array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array).convert("RGB")
    if isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        return _to_pil_image(tensor.numpy())
    raise TypeError(f"Unsupported VMem perception input: {type(data)!r}")


class VMemOperator(BaseOperator):
    """Translate WorldFoundry navigation actions into VMem commands."""

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types or ["textual_instruction", "action_instruction", "visual_instruction"]
        )
        self.interaction_template = interaction_template or list(_ACTION_TO_VMEM_COMMANDS)
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        _normalize_action(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, str):
            interaction = [interaction]
        if not isinstance(interaction, Sequence):
            raise TypeError("interaction must be a string or a sequence of strings.")
        normalized = [_normalize_action(item) for item in interaction]
        self.current_interaction.append(normalized)

    def process_perception(self, images):
        """Process perception inputs like images, videos, and reference frames."""
        return {"image": _to_pil_image(images)}

    def process_interaction(self) -> dict[str, list[str]]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction registered. Use get_interaction() first.")

        actions = list(self.current_interaction[-1])
        self.interaction_history.extend(actions)
        commands: list[str] = []
        for action in actions:
            commands.extend(_ACTION_TO_VMEM_COMMANDS[action])

        return {"actions": actions, "commands": commands}


__all__ = ["VMemOperator"]
