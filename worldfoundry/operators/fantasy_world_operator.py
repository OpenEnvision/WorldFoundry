"""Module for the FantasyWorld operator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base_operator import BaseOperator


def _to_rgb_pil_image(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")

    if isinstance(data, (str, Path)):
        return Image.open(data).convert("RGB")

    array = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported FantasyWorld image input shape: {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.min() >= -1.0 and array.max() <= 1.0:
            array = (array + 1.0) * 127.5 if array.min() < 0.0 else array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


def _is_pose_sequence(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return False
    return array.ndim == 3 and array.shape[-2:] == (4, 4)


class FantasyWorldOperator(BaseOperator):
    """Normalize FantasyWorld image inputs and camera-trajectory inputs."""

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=operation_types or ["visual_instruction", "action_instruction"])
        self.interaction_template = interaction_template or ["camera_trajectory"]
        self.interaction_template_init()

    def resolve_interaction_source(
        self,
        *,
        interactions=None,
        camera_json_path: str | Path | None = None,
        camera_data: dict[str, Any] | None = None,
        camera_poses: Sequence | None = None,
    ):
        """Resolve the source or sequence of interaction inputs."""
        explicit = [value for value in (camera_json_path, camera_data, camera_poses) if value is not None]
        if len(explicit) > 1:
            raise ValueError("Pass only one of `camera_json_path`, `camera_data`, or `camera_poses`.")
        if explicit:
            return explicit[0]
        return interactions

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction is None:
            raise ValueError("FantasyWorld requires a camera trajectory input.")
        if isinstance(interaction, (str, Path)):
            path = Path(interaction).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"FantasyWorld camera json not found: {path}")
            return True
        if isinstance(interaction, dict):
            if "cameras_interp" not in interaction:
                raise ValueError("FantasyWorld camera dict must contain `cameras_interp`.")
            return True
        if _is_pose_sequence(interaction):
            return True
        if isinstance(interaction, Sequence) and interaction and hasattr(interaction[0], "w2c_mat"):
            return True
        if isinstance(interaction, Sequence) and interaction and np.asarray(interaction[0]).shape == (4, 4):
            return True
        raise TypeError("FantasyWorld interactions must be camera json data, a json path, or a sequence of 4x4 poses.")

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self):
        """Process the recorded interactions and return the generated actions."""
        if not self.current_interaction:
            raise ValueError("No FantasyWorld interaction registered. Use get_interaction() first.")
        interaction = self.current_interaction[-1]
        self.interaction_history.append(interaction)
        return interaction

    def process_perception(
        self,
        images=None,
        end_image=None,
        scene_name: str | None = None,
    ):
        """Process perception inputs like images, videos, and reference frames."""
        if images is None:
            raise ValueError("FantasyWorld requires `images` for the reference image.")

        input_image = _to_rgb_pil_image(images)
        end_frame = _to_rgb_pil_image(end_image) if end_image is not None else None
        derived_scene_name = scene_name
        if derived_scene_name is None and isinstance(images, (str, Path)):
            derived_scene_name = Path(images).stem

        return {
            "image": input_image,
            "end_image": end_frame,
            "scene_name": derived_scene_name or "fantasyworld_scene",
            "image_size": (input_image.height, input_image.width),
        }
