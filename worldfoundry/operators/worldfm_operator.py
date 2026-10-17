"""Module for the WorldFM operator implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from .base_operator import BaseOperator

_NAVIGATION_DELTAS: dict[str, tuple[float, float, float, float, float]] = {
    "forward": (0.0, 0.0, -0.5, 0.0, 0.0),
    "backward": (0.0, 0.0, 0.5, 0.0, 0.0),
    "left": (-0.5, 0.0, 0.0, 0.0, 0.0),
    "right": (0.5, 0.0, 0.0, 0.0, 0.0),
    "forward_left": (-0.35, 0.0, -0.35, 0.0, 0.0),
    "forward_right": (0.35, 0.0, -0.35, 0.0, 0.0),
    "backward_left": (-0.35, 0.0, 0.35, 0.0, 0.0),
    "backward_right": (0.35, 0.0, 0.35, 0.0, 0.0),
    "camera_up": (0.0, 0.0, 0.0, -0.15, 0.0),
    "camera_down": (0.0, 0.0, 0.0, 0.15, 0.0),
    "camera_l": (0.0, 0.0, 0.0, 0.0, -0.15),
    "camera_r": (0.0, 0.0, 0.0, 0.0, 0.15),
    "camera_ul": (0.0, 0.0, 0.0, -0.1, -0.1),
    "camera_ur": (0.0, 0.0, 0.0, -0.1, 0.1),
    "camera_dl": (0.0, 0.0, 0.0, 0.1, -0.1),
    "camera_dr": (0.0, 0.0, 0.0, 0.1, 0.1),
    "camera_zoom_in": (0.0, 0.0, -0.3, 0.0, 0.0),
    "camera_zoom_out": (0.0, 0.0, 0.3, 0.0, 0.0),
}


def _navigation_poses(interactions: Sequence[str]) -> List[np.ndarray]:
    """Include the initial pose, then apply world translation and local Rx/Rz rotation."""
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    for token in interactions:
        delta = _NAVIGATION_DELTAS.get(str(token))
        if delta is None:
            raise ValueError(f"WorldFM can convert only navigation tokens into poses. Unsupported token: {token}")
        dx, dy, dz, theta_x, theta_z = delta
        rx, rz = np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)
        cx, sx = math.cos(theta_x), math.sin(theta_x)
        rx[1, 1], rx[1, 2], rx[2, 1], rx[2, 2] = cx, -sx, sx, cx
        cz, sz = math.cos(theta_z), math.sin(theta_z)
        rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = cz, -sz, sz, cz
        current = current @ rx @ rz
        current[:3, 3] += (dx, dy, dz)
        poses.append(current.copy())
    return poses


def _image_intrinsics(image: str | Path | Image.Image) -> np.ndarray:
    """Use the source image center and max(width, height) focal length."""
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        with Image.open(image) as source:
            width, height = source.size
    focal = float(max(width, height))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _as_pose_matrix(value: Any) -> np.ndarray:
    """As pose matrix implementation."""
    if isinstance(value, dict):
        if "c2w" not in value:
            raise ValueError("WorldFM pose dict must contain a `c2w` key.")
        value = value["c2w"]

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float64)
        output[:3, :4] = matrix
        return output
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"WorldFM expects poses shaped (4, 4) or (3, 4), got {matrix.shape}.")


def _looks_like_single_pose(value: Any) -> bool:
    """Looks like single pose implementation."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except Exception:
        return False
    return matrix.shape in {(3, 4), (4, 4)}


class WorldFMOperator(BaseOperator):
    """Normalize WorldFM scene inputs and target camera poses."""

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=operation_types or ["visual_instruction", "action_instruction"])
        self.interaction_template = interaction_template or ["camera_pose"]
        self.interaction_template_init()

    def _load_meta(self, meta_path: str | Path) -> Dict[str, Any]:
        """Load meta implementation."""
        meta_path = Path(meta_path).expanduser().resolve()
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)

        required_keys = {"name", "image", "K", "c2w"}
        missing = required_keys.difference(meta)
        if missing:
            raise KeyError(f"WorldFM meta file is missing keys: {sorted(missing)}")

        image_path = (meta_path.parent / meta["image"]).resolve()
        c2w_array = np.asarray(meta["c2w"], dtype=np.float64)
        if c2w_array.ndim == 2:
            c2w_list = [_as_pose_matrix(c2w_array)]
        elif c2w_array.ndim == 3:
            c2w_list = [_as_pose_matrix(item) for item in c2w_array]
        else:
            raise ValueError(f"meta['c2w'] must be shaped (4,4) or (N,4,4), got {c2w_array.shape}")

        return {
            "meta_path": str(meta_path),
            "scene_name": str(meta["name"]),
            "image_path": str(image_path),
            "K": np.asarray(meta["K"], dtype=np.float64),
            "c2w_list": c2w_list,
        }

    def _normalize_interactions(self, interaction: Any) -> List[np.ndarray]:
        """Normalize interactions implementation."""
        if isinstance(interaction, np.ndarray) and interaction.ndim == 3:
            return [_as_pose_matrix(item) for item in interaction]

        if isinstance(interaction, Sequence) and not isinstance(interaction, (str, bytes)):
            if interaction and isinstance(interaction[0], str):
                return _navigation_poses(interaction)

        if _looks_like_single_pose(interaction):
            return [_as_pose_matrix(interaction)]

        if isinstance(interaction, Sequence) and not isinstance(interaction, (str, bytes)):
            poses = [_as_pose_matrix(item) for item in interaction]
            if poses:
                return poses

        raise TypeError("WorldFM interactions must be a pose matrix or a sequence of pose matrices.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._normalize_interactions(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        normalized = self._normalize_interactions(interaction)
        self.current_interaction.append(normalized)

    def process_interaction(self):
        """Process the recorded interactions and return the generated actions."""
        if not self.current_interaction:
            raise ValueError("No WorldFM interactions registered. Use get_interaction() first.")

        current = [pose.copy() for pose in self.current_interaction[-1]]
        self.interaction_history.extend(current)
        return current

    def process_perception(
        self,
        images=None,
        K=None,
        meta_path: str | Path | None = None,
        scene_name: str | None = None,
        panorama_image=None,
        panorama_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        image_path = None
        default_interactions = None
        resolved_meta_path = None

        if meta_path is not None:
            meta = self._load_meta(meta_path)
            resolved_meta_path = meta["meta_path"]
            if images is None and panorama_image is None and panorama_path is None:
                image_path = meta["image_path"]
            if K is None:
                K = meta["K"]
            if scene_name is None:
                scene_name = meta["scene_name"]
            default_interactions = meta["c2w_list"]

        if K is None and isinstance(images, (str, Path, Image.Image)):
            K = _image_intrinsics(images)
        if K is None:
            raise ValueError("WorldFM requires `K` or a meta file containing intrinsics.")

        K_array = np.asarray(K, dtype=np.float64)
        if K_array.shape != (3, 3):
            raise ValueError(f"WorldFM intrinsics K must be shaped (3, 3), got {K_array.shape}")

        if images is None and image_path is None and panorama_image is None and panorama_path is None:
            raise ValueError("WorldFM requires either `images`, `meta_path`, or `panorama_image`/`panorama_path`.")

        return {
            "images": images,
            "image_path": image_path,
            "panorama_image": panorama_image,
            "panorama_path": str(Path(panorama_path).expanduser().resolve()) if panorama_path else None,
            "K": K_array,
            "scene_name": scene_name or "worldfm_scene",
            "default_interactions": default_interactions,
            "meta_path": resolved_meta_path,
        }
