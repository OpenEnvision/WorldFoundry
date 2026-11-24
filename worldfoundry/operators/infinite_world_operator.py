"""Module for the InfiniteWorld operator implementation."""

import os
import re
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base_operator import BaseOperator

MOVE_ACTION_MAP = {
    "no-op": 0,
    "go forward": 1,
    "go back": 2,
    "go left": 3,
    "go right": 4,
    "go forward and go left": 5,
    "go forward and go right": 6,
    "go back and go left": 7,
    "go back and go right": 8,
    "uncertain": 9,
}

VIEW_ACTION_MAP = {
    "no-op": 0,
    "turn up": 1,
    "turn down": 2,
    "turn left": 3,
    "turn right": 4,
    "turn up and turn left": 5,
    "turn up and turn right": 6,
    "turn down and turn left": 7,
    "turn down and turn right": 8,
    "uncertain": 9,
}


_STRING_ACTION_ALIASES = {
    "noop": {"move": "no-op", "view": "no-op"},
    "no_op": {"move": "no-op", "view": "no-op"},
    "no-op": {"move": "no-op", "view": "no-op"},
    "none": {"move": "no-op", "view": "no-op"},
    "forward": {"move": "go forward", "view": "no-op"},
    "go_forward": {"move": "go forward", "view": "no-op"},
    "back": {"move": "go back", "view": "no-op"},
    "backward": {"move": "go back", "view": "no-op"},
    "go_back": {"move": "go back", "view": "no-op"},
    "left": {"move": "go left", "view": "no-op"},
    "go_left": {"move": "go left", "view": "no-op"},
    "right": {"move": "go right", "view": "no-op"},
    "go_right": {"move": "go right", "view": "no-op"},
    "forward_left": {"move": "go forward and go left", "view": "no-op"},
    "forward_right": {"move": "go forward and go right", "view": "no-op"},
    "back_left": {"move": "go back and go left", "view": "no-op"},
    "back_right": {"move": "go back and go right", "view": "no-op"},
    "camera_up": {"move": "no-op", "view": "turn up"},
    "look_up": {"move": "no-op", "view": "turn up"},
    "up": {"move": "no-op", "view": "turn up"},
    "camera_down": {"move": "no-op", "view": "turn down"},
    "look_down": {"move": "no-op", "view": "turn down"},
    "down": {"move": "no-op", "view": "turn down"},
    "camera_left": {"move": "no-op", "view": "turn left"},
    "camera_l": {"move": "no-op", "view": "turn left"},
    "turn_left": {"move": "no-op", "view": "turn left"},
    "camera_right": {"move": "no-op", "view": "turn right"},
    "camera_r": {"move": "no-op", "view": "turn right"},
    "turn_right": {"move": "no-op", "view": "turn right"},
    "camera_up_left": {"move": "no-op", "view": "turn up and turn left"},
    "camera_up_right": {"move": "no-op", "view": "turn up and turn right"},
    "camera_down_left": {"move": "no-op", "view": "turn down and turn left"},
    "camera_down_right": {"move": "no-op", "view": "turn down and turn right"},
    "uncertain": {"move": "uncertain", "view": "uncertain"},
}

_MOVE_PART_ALIASES = {
    "forward": "go forward",
    "go_forward": "go forward",
    "back": "go back",
    "backward": "go back",
    "go_back": "go back",
    "left": "go left",
    "go_left": "go left",
    "right": "go right",
    "go_right": "go right",
}

_VIEW_PART_ALIASES = {
    "camera_up": "turn up",
    "up": "turn up",
    "look_up": "turn up",
    "camera_down": "turn down",
    "down": "turn down",
    "look_down": "turn down",
    "camera_left": "turn left",
    "camera_l": "turn left",
    "turn_left": "turn left",
    "camera_right": "turn right",
    "camera_r": "turn right",
    "turn_right": "turn right",
}

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _sanitize_token(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _resize_and_center_crop(frame: np.ndarray, target_size) -> np.ndarray:
    target_h, target_w = target_size
    orig_h, orig_w = frame.shape[:2]
    scale = max(target_h / orig_h, target_w / orig_w)
    resized_h = int(np.ceil(scale * orig_h))
    resized_w = int(np.ceil(scale * orig_w))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    cropped = TF.center_crop(tensor, [target_h, target_w])
    return cropped.permute(1, 2, 0).contiguous().numpy()


def _normalize_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
    video_tensor = video_tensor.float()
    if video_tensor.max() <= 1.0 and video_tensor.min() >= -1.0:
        if video_tensor.min() >= 0.0:
            return video_tensor * 2.0 - 1.0
        return video_tensor
    if video_tensor.min() >= 0.0:
        return video_tensor / 127.5 - 1.0
    return video_tensor.clamp(-1.0, 1.0)


def _frames_to_video_tensor(frames: Sequence[np.ndarray], target_size) -> torch.Tensor:
    processed_frames = []
    for frame in frames:
        frame = _resize_and_center_crop(frame, target_size)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        tensor = tensor / 127.5 - 1.0
        processed_frames.append(tensor)
    stacked = torch.stack(processed_frames, dim=1)
    return stacked.unsqueeze(0)


def _tensor_to_video_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError("Infinite-World currently supports batch size 1 only.")
        if tensor.shape[1] in (1, 3):
            return _normalize_video_tensor(tensor)
        if tensor.shape[2] in (1, 3):
            return _normalize_video_tensor(tensor.permute(0, 2, 1, 3, 4))
        raise ValueError("Unsupported 5D tensor layout for condition video.")
    if tensor.ndim == 4:
        if tensor.shape[0] in (1, 3):
            return _normalize_video_tensor(tensor.unsqueeze(0))
        if tensor.shape[1] in (1, 3):
            return _normalize_video_tensor(tensor.permute(1, 0, 2, 3).unsqueeze(0))
        if tensor.shape[-1] in (1, 3):
            return _normalize_video_tensor(tensor.permute(3, 0, 1, 2).unsqueeze(0))
        raise ValueError("Unsupported 4D tensor layout for condition video.")
    if tensor.ndim == 3:
        if tensor.shape[0] in (1, 3):
            return _normalize_video_tensor(tensor.unsqueeze(0).unsqueeze(2))
        if tensor.shape[-1] in (1, 3):
            return _normalize_video_tensor(tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
        raise ValueError("Unsupported 3D tensor layout for condition image.")
    raise ValueError("Unsupported tensor input for Infinite-World operator.")


def _select_target_size(bucket_config: dict[str, Any], frame: np.ndarray):
    aspect_ratio = frame.shape[0] / frame.shape[1]
    closest_bucket = min(bucket_config, key=lambda key: abs(float(key) - aspect_ratio))
    target_h, target_w = bucket_config[closest_bucket][0]
    return int(target_h), int(target_w)


def _load_frames_from_path(path: str) -> list[np.ndarray]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Condition path not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension in _VIDEO_EXTENSIONS:
        frames = []
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"Failed to read frames from video: {path}")
        return frames

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]


class InfiniteWorldOperator(BaseOperator):
    """Operator for Infinite-World action encoding and condition preprocessing."""

    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
    ):
        """Initialize the operator with specific configurations."""
        super().__init__(
            operation_types=operation_types or ["textual_instruction", "visual_instruction", "action_instruction"]
        )
        self.interaction_template = interaction_template or [
            "no-op",
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "back_left",
            "back_right",
            "camera_up",
            "camera_down",
            "camera_left",
            "camera_right",
            "camera_up_left",
            "camera_up_right",
            "camera_down_left",
            "camera_down_right",
            "uncertain",
        ]
        self.interaction_template_init()

    def _normalize_interaction(self, interaction: str | dict[str, str]) -> dict[str, str]:
        if isinstance(interaction, dict):
            move = interaction.get("move", "no-op")
            view = interaction.get("view", "no-op")
            if move not in MOVE_ACTION_MAP:
                raise ValueError(f"Unsupported move action: {move}")
            if view not in VIEW_ACTION_MAP:
                raise ValueError(f"Unsupported view action: {view}")
            return {"move": move, "view": view}

        if not isinstance(interaction, str):
            raise TypeError(f"Unsupported interaction type: {type(interaction)}")

        normalized = _sanitize_token(interaction)
        if normalized in _STRING_ACTION_ALIASES:
            return dict(_STRING_ACTION_ALIASES[normalized])

        if any(separator in normalized for separator in ("+", "|", ",")):
            move = "no-op"
            view = "no-op"
            for part in re.split(r"[+,|]", normalized):
                token = part.strip()
                if not token:
                    continue
                if token in _MOVE_PART_ALIASES:
                    if move != "no-op":
                        raise ValueError(f"Multiple move actions found in interaction: {interaction}")
                    move = _MOVE_PART_ALIASES[token]
                    continue
                if token in _VIEW_PART_ALIASES:
                    if view != "no-op":
                        raise ValueError(f"Multiple view actions found in interaction: {interaction}")
                    view = _VIEW_PART_ALIASES[token]
                    continue
                raise ValueError(f"Unsupported combined action token: {part}")
            return {"move": move, "view": view}

        raise ValueError(f"Unsupported interaction: {interaction}")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._normalize_interaction(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if isinstance(interaction, (str, dict)):
            interaction = [interaction]
        if not isinstance(interaction, Sequence):
            raise TypeError("interaction should be a string, dict, or a sequence of them.")
        normalized = [self._normalize_interaction(item) for item in interaction]
        self.current_interaction.append(normalized)

    def process_interaction(self, prefix_length: int = 0):
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise RuntimeError("No interaction registered. Use get_interaction() first.")

        current_actions = self.current_interaction[-1]
        if current_actions:
            self.interaction_history.extend(current_actions)

        move_ids = [MOVE_ACTION_MAP["no-op"]] * max(prefix_length, 0)
        view_ids = [VIEW_ACTION_MAP["no-op"]] * max(prefix_length, 0)
        for action in current_actions:
            move_ids.append(MOVE_ACTION_MAP[action["move"]])
            view_ids.append(VIEW_ACTION_MAP[action["view"]])

        return {
            "move_ids": torch.tensor(move_ids, dtype=torch.long),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
            "actions": current_actions,
            "prefix_length": prefix_length,
        }

    def process_perception(self, input_data, bucket_config=None):
        """Process perception inputs like images, videos, and reference frames."""
        if bucket_config is None:
            raise ValueError("bucket_config must be provided for Infinite-World preprocessing.")

        if isinstance(input_data, torch.Tensor):
            video_tensor = _tensor_to_video_tensor(input_data)
            return {
                "condition_video": video_tensor,
                "num_condition_frames": int(video_tensor.shape[2]),
                "target_size": tuple(video_tensor.shape[-2:]),
            }

        if isinstance(input_data, np.ndarray):
            video_tensor = _tensor_to_video_tensor(torch.from_numpy(input_data))
            return {
                "condition_video": video_tensor,
                "num_condition_frames": int(video_tensor.shape[2]),
                "target_size": tuple(video_tensor.shape[-2:]),
            }

        frames: list[np.ndarray] = []
        if isinstance(input_data, str):
            frames.extend(_load_frames_from_path(input_data))
        elif isinstance(input_data, Image.Image):
            frames.append(np.asarray(input_data.convert("RGB")))
        elif isinstance(input_data, Sequence):
            for item in input_data:
                if isinstance(item, Image.Image):
                    frames.append(np.asarray(item.convert("RGB")))
                elif isinstance(item, np.ndarray):
                    frames.append(item)
                elif isinstance(item, str):
                    frames.extend(_load_frames_from_path(item))
                else:
                    raise TypeError(f"Unsupported item type in input sequence: {type(item)}")
        else:
            raise TypeError(f"Unsupported input type for perception: {type(input_data)}")

        if len(frames) == 0:
            raise ValueError("No valid condition frames found for Infinite-World.")

        target_size = _select_target_size(bucket_config, frames[0])
        video_tensor = _frames_to_video_tensor(frames, target_size)
        return {
            "condition_video": video_tensor,
            "num_condition_frames": int(video_tensor.shape[2]),
            "target_size": target_size,
        }
