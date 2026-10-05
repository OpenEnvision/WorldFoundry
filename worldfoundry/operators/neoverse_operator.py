"""Module for the NeoVerse operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from .base_operator import BaseOperator


_RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

_ACTION_ALIASES = {
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "camera_l": "camera_l",
    "camera_left": "camera_l",
    "camera_r": "camera_r",
    "camera_right": "camera_r",
    "camera_up": "camera_up",
    "camera_down": "camera_down",
    "forward_left": "forward_left",
    "forward_right": "forward_right",
    "backward_left": "backward_left",
    "backward_right": "backward_right",
    "no-op": "no-op",
    "noop": "no-op",
    "none": "no-op",
}


def _normalize_action(action: str) -> str:
    """Normalize action implementation."""
    key = action.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ACTION_ALIASES:
        raise ValueError(f"Unsupported NeoVerse interaction: {action}")
    return _ACTION_ALIASES[key]


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    """To uint8 image implementation."""
    data = np.asarray(array)
    if data.dtype in (np.float16, np.float32, np.float64):
        if data.min() >= -1.0 and data.max() <= 1.0:
            if data.min() < 0.0:
                data = (data + 1.0) * 127.5
            else:
                data = data * 255.0
        data = np.clip(data, 0.0, 255.0).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return data


def _center_crop(image: Image.Image, resolution: tuple[int, int]) -> Image.Image:
    """Center crop implementation."""
    width, height = image.size
    target_width, target_height = resolution

    scale = max(target_width / width, target_height / height)
    resized = image.resize(
        (int(round(width * scale)), int(round(height * scale))),
        resample=_RESAMPLE_LANCZOS,
    )
    left = max((resized.size[0] - target_width) // 2, 0)
    top = max((resized.size[1] - target_height) // 2, 0)
    return resized.crop((left, top, left + target_width, top + target_height))


def _video_first_frame(path: Path) -> Image.Image:
    """Video first frame implementation."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError("NeoVerse video inputs require imageio to read the first frame.") from exc

    reader = imageio.get_reader(str(path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    return Image.fromarray(_to_uint8_image(np.asarray(frame))).convert("RGB")


def _to_pil_image(data: Any) -> Image.Image:
    """To pil image implementation."""
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, (str, Path)):
        path = Path(data).expanduser()
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            return _video_first_frame(path)
        return Image.open(path).convert("RGB")

    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported NeoVerse perception input: {type(data)!r}")

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported NeoVerse perception shape: {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    array = _to_uint8_image(array)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


def _is_path_like_sequence(data: Any) -> bool:
    """Is path like sequence implementation."""
    return isinstance(data, (list, tuple)) and all(isinstance(item, (str, Path)) for item in data)


def _is_official_video_loader_input(data: Any) -> bool:
    """Is official video loader input implementation."""
    if _is_path_like_sequence(data):
        return True
    if not isinstance(data, (str, Path)):
        return False
    path = Path(data).expanduser()
    return path.is_dir() or path.suffix.lower() in _VIDEO_SUFFIXES


def _load_official_input_frames(
    data: Any,
    *,
    num_frames: int,
    resolution: tuple[int, int],
    resize_mode: str,
    static_scene: bool,
) -> List[Image.Image]:
    """Load official input frames implementation."""
    from worldfoundry.synthesis.visual_generation.neoverse.native.auxiliary import (
        load_video,
    )

    if isinstance(data, (str, Path)):
        loader_input: Any = str(Path(data).expanduser())
    else:
        resolved_items = [str(Path(item).expanduser()) for item in data]
        if len(resolved_items) == 1 and Path(resolved_items[0]).suffix.lower() in _VIDEO_SUFFIXES:
            loader_input = resolved_items[0]
        else:
            loader_input = resolved_items
    frames = load_video(
        loader_input,
        num_frames,
        resolution=resolution,
        resize_mode=resize_mode,
        static_scene=static_scene,
    )
    return [frame.convert("RGB") for frame in frames]


class NeoVerseOperator(BaseOperator):
    """Convert WorldBench navigation actions into NeoVerse camera keyframes."""

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
        *,
        height: int = 336,
        width: int = 560,
        resize_mode: str = "center_crop",
        frames_per_action: int = 20,
        translation_distance: float = 0.08,
        rotation_angle_deg: float = 10.0,
        zoom_ratio: float = 1.0,
        trajectory_mode: str = "relative",
    ):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types
            or ["textual_instruction", "action_instruction", "visual_instruction"]
        )
        self.interaction_template = interaction_template or sorted(set(_ACTION_ALIASES.values()))
        self.interaction_template_init()
        self.height = int(height)
        self.width = int(width)
        self.resize_mode = resize_mode
        self.frames_per_action = int(frames_per_action)
        self.translation_distance = float(translation_distance)
        self.rotation_angle_deg = float(rotation_angle_deg)
        self.zoom_ratio = float(zoom_ratio)
        self.trajectory_mode = trajectory_mode

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if isinstance(interaction, dict):
            return True
        if isinstance(interaction, str):
            _normalize_action(interaction)
            return True
        if isinstance(interaction, Sequence):
            for item in interaction:
                _normalize_action(item)
            return True
        raise TypeError(f"Unsupported NeoVerse interaction type: {type(interaction)!r}")

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        if isinstance(interaction, dict):
            spec = dict(interaction)
            actions = spec.get("actions")
            if actions is not None:
                if isinstance(actions, str):
                    actions = [actions]
                spec["actions"] = [_normalize_action(item) for item in actions]
        else:
            if isinstance(interaction, str):
                interaction = [interaction]
            spec = {"actions": [_normalize_action(item) for item in interaction]}
        self.current_interaction.append(spec)

    def process_perception(
        self,
        images,
        *,
        height: int | None = None,
        width: int | None = None,
        resize_mode: str | None = None,
        static_scene: bool | None = None,
        num_frames: int | None = None,
    ):
        """Process perception inputs like images, videos, and reference frames."""
        resize_h = int(height or self.height)
        resize_w = int(width or self.width)
        mode = resize_mode or self.resize_mode
        resolved_num_frames = int(num_frames or 81)

        if _is_official_video_loader_input(images):
            resolved_static_scene = bool(static_scene) if static_scene is not None else False
            processed_frames = _load_official_input_frames(
                images,
                num_frames=resolved_num_frames,
                resolution=(resize_w, resize_h),
                resize_mode=mode,
                static_scene=resolved_static_scene,
            )
            return {
                "input_frames": processed_frames,
                "height": resize_h,
                "width": resize_w,
                "static_scene": resolved_static_scene,
            }

        if isinstance(images, (list, tuple)) and len(images) > 0:
            frame_candidates = list(images)
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            frame_candidates = [frame for frame in images]
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            frame_candidates = [frame for frame in images]
        else:
            frame_candidates = [images]

        processed_frames: List[Image.Image] = []
        for frame in frame_candidates:
            image = _to_pil_image(frame)
            if mode == "resize":
                image = image.resize((resize_w, resize_h), resample=_RESAMPLE_LANCZOS)
            else:
                image = _center_crop(image, (resize_w, resize_h))
            processed_frames.append(image)

        inferred_static_scene = len(processed_frames) == 1 if static_scene is None else bool(static_scene)
        return {
            "input_frames": processed_frames,
            "height": resize_h,
            "width": resize_w,
            "static_scene": inferred_static_scene,
        }

    def _action_to_operations(self, action: str) -> list[dict[str, dict[str, float]]]:
        """Action to operations implementation."""
        distance = self.translation_distance
        angle = self.rotation_angle_deg

        if action == "forward":
            return [{"push_in": {"distance": distance}}]
        if action == "backward":
            return [{"pull_out": {"distance": distance}}]
        if action == "left":
            return [{"move_left": {"distance": distance}}]
        if action == "right":
            return [{"move_right": {"distance": distance}}]
        if action == "up":
            return [{"boom_up": {"distance": distance}}]
        if action == "down":
            return [{"boom_down": {"distance": distance}}]
        if action == "camera_l":
            return [{"pan_left": {"angle": angle}}]
        if action == "camera_r":
            return [{"pan_right": {"angle": angle}}]
        if action == "camera_up":
            return [{"tilt_up": {"angle": angle}}]
        if action == "camera_down":
            return [{"tilt_down": {"angle": angle}}]
        if action == "forward_left":
            return [
                {"push_in": {"distance": distance}},
                {"move_left": {"distance": distance}},
            ]
        if action == "forward_right":
            return [
                {"push_in": {"distance": distance}},
                {"move_right": {"distance": distance}},
            ]
        if action == "backward_left":
            return [
                {"pull_out": {"distance": distance}},
                {"move_left": {"distance": distance}},
            ]
        if action == "backward_right":
            return [
                {"pull_out": {"distance": distance}},
                {"move_right": {"distance": distance}},
            ]
        if action == "no-op":
            return [{"static": {}}]
        raise ValueError(f"Unsupported NeoVerse action: {action}")

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction registered. Use get_interaction() first.")

        spec = dict(self.current_interaction[-1])
        actions = list(spec.get("actions") or [])
        self.interaction_history.extend(actions)

        if spec.get("trajectory_file") is not None:
            return {
                "actions": actions,
                "predefined_trajectory": None,
                "trajectory_file": spec["trajectory_file"],
                "trajectory_data": None,
                "keyframes": None,
                "num_frames": spec.get("num_frames"),
                "trajectory_mode": spec.get("trajectory_mode", self.trajectory_mode),
                "trajectory_name": spec.get("trajectory_name", "neoverse_trajectory"),
                "zoom_ratio": float(spec.get("zoom_ratio", self.zoom_ratio)),
                "use_first_frame": bool(spec.get("use_first_frame", True)),
                "angle": spec.get("angle"),
                "distance": spec.get("distance"),
                "orbit_radius": spec.get("orbit_radius"),
            }

        trajectory_data = spec.get("trajectory_data")
        if trajectory_data is not None:
            resolved_num_frames = spec.get("num_frames")
            if resolved_num_frames is None and isinstance(trajectory_data, dict):
                resolved_num_frames = trajectory_data.get("num_frames")
            return {
                "actions": actions,
                "predefined_trajectory": None,
                "trajectory_file": None,
                "trajectory_data": trajectory_data,
                "keyframes": None,
                "num_frames": resolved_num_frames,
                "trajectory_mode": spec.get("trajectory_mode", self.trajectory_mode),
                "trajectory_name": spec.get("trajectory_name", "neoverse_trajectory"),
                "zoom_ratio": float(spec.get("zoom_ratio", self.zoom_ratio)),
                "use_first_frame": bool(spec.get("use_first_frame", True)),
                "angle": spec.get("angle"),
                "distance": spec.get("distance"),
                "orbit_radius": spec.get("orbit_radius"),
            }

        predefined_trajectory = spec.get("predefined_trajectory")
        if predefined_trajectory is not None:
            return {
                "actions": actions,
                "predefined_trajectory": predefined_trajectory,
                "trajectory_file": None,
                "trajectory_data": None,
                "keyframes": None,
                "num_frames": spec.get("num_frames"),
                "trajectory_mode": spec.get("trajectory_mode", self.trajectory_mode),
                "trajectory_name": predefined_trajectory,
                "zoom_ratio": float(spec.get("zoom_ratio", self.zoom_ratio)),
                "use_first_frame": bool(spec.get("use_first_frame", True)),
                "angle": spec.get("angle"),
                "distance": spec.get("distance"),
                "orbit_radius": spec.get("orbit_radius"),
            }

        if len(actions) == 0:
            raise ValueError("NeoVerse requires either actions or a trajectory specification.")

        keyframes = [{0: [{"static": {}}]}]
        for action_idx, action in enumerate(actions, start=1):
            keyframes.append({action_idx * self.frames_per_action: self._action_to_operations(action)})

        return {
            "actions": actions,
            "predefined_trajectory": None,
            "trajectory_file": None,
            "trajectory_data": None,
            "keyframes": keyframes,
            "num_frames": 1 + self.frames_per_action * len(actions),
            "trajectory_mode": spec.get("trajectory_mode", self.trajectory_mode),
            "trajectory_name": spec.get("trajectory_name", "neoverse_actions"),
            "zoom_ratio": float(spec.get("zoom_ratio", self.zoom_ratio)),
            "use_first_frame": bool(spec.get("use_first_frame", True)),
            "angle": spec.get("angle"),
            "distance": spec.get("distance"),
            "orbit_radius": spec.get("orbit_radius"),
        }
