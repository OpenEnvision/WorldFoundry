"""Module for the FlashWorld operator implementation."""

import os
import numpy as np
import torch
from typing import List, Mapping, Optional, Union, Dict, Any
from PIL import Image
import base64
import io

from .base_operator import BaseOperator

from ..base_models.three_dimensions.point_clouds.flash_world.utils import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def _text_from_flash_world_interaction(interaction: Any) -> Optional[str]:
    """Return prompt text from a mapping interaction; None if it is not text."""
    if isinstance(interaction, Mapping):
        raw = interaction.get("text_prompt")
        if raw is None:
            raw = interaction.get("prompt")
        if raw is None:
            return None
        return str(raw)
    return None


def _split_flash_world_interactions(
    interactions: List[Any],
    text_prompt: str = "",
    prompt: str = "",
) -> tuple:
    """Split recorded interactions into prompt text and camera-action tokens.

    The literal template token ``"text_prompt"`` carries no body and is
    dropped from the camera list. Mapping entries with ``text_prompt`` or
    ``prompt`` contribute their text. Explicit kwargs win over collected text.
    """
    collected: List[str] = []
    camera_actions: List[Any] = []
    for item in interactions:
        mapped = _text_from_flash_world_interaction(item)
        if mapped is not None:
            if mapped:
                collected.append(mapped)
            continue
        if item == "text_prompt":
            continue
        camera_actions.append(item)
    resolved = text_prompt or prompt or " ".join(collected)
    return resolved, camera_actions


# Shared focal point for look-at and dolly forward/backward (must stay consistent).
_LOOK_AT_TARGET = np.array([0.0, 0.5, 0.0], dtype=np.float64)


def _look_at_quaternion_wxyz(
    pos: np.ndarray, target: Optional[np.ndarray] = None
) -> List[float]:
    """World-from-camera rotation as wxyz quaternion; camera looks from pos toward target."""
    if target is None:
        target = _LOOK_AT_TARGET
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up = np.cross(right, forward)
    backward = -forward
    r_mat = np.stack([right, up, backward], axis=1).astype(np.float32)
    q = matrix_to_quaternion(torch.from_numpy(r_mat).unsqueeze(0)).squeeze(0)
    return [float(q[i]) for i in range(4)]


# In-place rotation magnitudes per segment (position unchanged for camera_* turns).
_YAW_PER_SEGMENT = np.pi / 4
_PITCH_PER_SEGMENT = np.pi / 6


def _quat_wxyz_to_R(q: List[float]) -> np.ndarray:
    """Quat wxyz to r implementation."""
    t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    return quaternion_to_matrix(t).squeeze(0).detach().cpu().numpy()


def _view_dir_world_from_quat(quat_wxyz: List[float]) -> np.ndarray:
    """Unit vector in world space along the camera viewing axis (into the scene, FlashWorld c2w)."""
    r = _quat_wxyz_to_R(quat_wxyz)
    # Camera looks along -Z in camera coords -> world direction is -third column of R.
    v = -np.asarray(r[:, 2], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return v / n


def _camera_right_world_from_quat(quat_wxyz: List[float]) -> np.ndarray:
    """Unit vector along camera +X in world space (strafe right); left is -this."""
    r = _quat_wxyz_to_R(quat_wxyz)
    v = np.asarray(r[:, 0], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _R_to_quat_wxyz(r: np.ndarray) -> List[float]:
    """R to quat wxyz implementation."""
    t = torch.from_numpy(r.astype(np.float32)).unsqueeze(0)
    q = matrix_to_quaternion(t).squeeze(0)
    return [float(q[i]) for i in range(4)]


def _rot_y(angle: float) -> np.ndarray:
    """Rot y implementation."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis angle matrix implementation."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def _rotate_yaw_world(quat_wxyz: List[float], yaw: float) -> List[float]:
    """Apply world +Y yaw after current camera orientation (pan left/right in place)."""
    r0 = _quat_wxyz_to_R(quat_wxyz)
    r_new = _rot_y(yaw) @ r0
    return _R_to_quat_wxyz(r_new)


def _rotate_pitch_local(quat_wxyz: List[float], pitch: float) -> List[float]:
    """Pitch around camera right axis (tilt up/down in place)."""
    r0 = _quat_wxyz_to_R(quat_wxyz)
    right = r0[:, 0]
    rp = _axis_angle_matrix(right, pitch)
    r_new = rp @ r0
    return _R_to_quat_wxyz(r_new)


def _orientation_in_place_at_t(
    quat_start: List[float], action: str, t: float
) -> List[float]:
    """Camera fixed; orientation interpolated over segment (t in [0, 1])."""
    t = float(np.clip(t, 0.0, 1.0))
    if action == "camera_l":
        return _rotate_yaw_world(quat_start, t * _YAW_PER_SEGMENT)
    if action == "camera_r":
        return _rotate_yaw_world(quat_start, -t * _YAW_PER_SEGMENT)
    if action == "camera_up":
        return _rotate_pitch_local(quat_start, t * _PITCH_PER_SEGMENT)
    if action == "camera_down":
        return _rotate_pitch_local(quat_start, -t * _PITCH_PER_SEGMENT)
    raise ValueError(f"Not an in-place camera orientation action: {action}")


class FlashWorldOperator(BaseOperator):
    """Operator for FlashWorld pipeline utilities."""
    
    def __init__(
        self,
        operation_types=None,
        interaction_template=None,
    ):
        """
        Initialize FlashWorld operator.
        
        Args:
            operation_types: List of operation types
            interaction_template: List of valid interaction types
                - "text_prompt": Text description for scene generation
                - "forward/backward/left/right": Translation along view / strafe (camera-relative)
                - "camera_l/r/up/down": Pan/tilt in place (position fixed; yaw/pitch)
                - "camera_zoom_in/out": Zoom (focal length; position fixed)
        """
        if operation_types is None:
            operation_types = ["textual_instruction", "action_instruction", "visual_instruction"]
        if interaction_template is None:
            interaction_template = [
                "text_prompt",
                "forward", "backward", "left", "right",
                "camera_up", "camera_down", "camera_l", "camera_r",
                "camera_zoom_in", "camera_zoom_out",
            ]
        super(FlashWorldOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """
        Check if interaction is in the interaction template.
        
        Args:
            interaction: Template token, or a mapping with ``text_prompt``/``prompt``.
            
        Returns:
            True if interaction is valid
            
        Raises:
            ValueError: If interaction is not in template and is not text
        """
        if _text_from_flash_world_interaction(interaction) is not None:
            return True
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template. Available: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        Add interaction to current_interaction list after validation.
        
        Args:
            interaction: Template token or ``{"text_prompt": "..."}`` / ``{"prompt": "..."}``.
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)
    
    def process_interaction(
        self, 
        num_frames: Optional[int] = None,
        image_width: int = 704,
        image_height: int = 480,
        text_prompt: str = "",
        prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Process current interactions and convert to features for representation/synthesis.
        Converts camera actions to actual camera parameters that can be used by representation.
        
        Args:
            num_frames: Number of frames for video generation (optional)
            image_width: Image width for camera intrinsics
            image_height: Image height for camera intrinsics
            text_prompt: Optional scene text; wins over recorded mapping entries
            prompt: Alias for ``text_prompt`` (pipeline ``__call__`` uses this name)
            
        Returns:
            Dictionary containing processed interaction features:
                - text_prompt: str, text from kwargs or recorded mapping interactions
                - cameras: List[Dict], camera parameters for each frame
                - num_frames: int, number of frames
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        
        latest_interaction = self.current_interaction[-1]
        self.interaction_history.append(latest_interaction)
        
        num_frames = num_frames or 16
        
        # Preserve list order; text tokens/mappings are not motion segments.
        text_prompt, camera_actions = _split_flash_world_interactions(
            self.current_interaction,
            text_prompt=text_prompt,
            prompt=prompt,
        )
        
        # Convert camera actions to camera parameters
        cameras = self._camera_actions_to_cameras(
            camera_actions=camera_actions,
            num_frames=num_frames,
            image_width=image_width,
            image_height=image_height
        )
        
        result = {
            "text_prompt": text_prompt,
            "cameras": cameras,
            "num_frames": num_frames,
        }
        
        return result
    
    def _segment_frame_counts(self, num_frames: int, n_segments: int) -> List[int]:
        """Split num_frames into n_segments contiguous parts (>=1 each when num_frames >= n_segments)."""
        if n_segments <= 0:
            return []
        if num_frames < n_segments:
            n_segments = num_frames
        base = num_frames // n_segments
        rem = num_frames % n_segments
        return [base + (1 if i < rem else 0) for i in range(n_segments)]

    def _apply_action_end_state(
        self,
        pos: np.ndarray,
        zoom: float,
        action: str,
        quat: Optional[List[float]] = None,
    ) -> tuple:
        """Apply one full action (t=1) from current pose; returns (new_pos, new_zoom)."""
        p = pos.astype(np.float64).copy()
        z = float(zoom)
        q_use = quat if quat is not None else _look_at_quaternion_wxyz(p)

        if action == "forward":
            # Dolly along current viewing direction (after camera_* pan/tilt), not toward old focal.
            fwd = _view_dir_world_from_quat(q_use)
            dist = float(np.linalg.norm(_LOOK_AT_TARGET - p))
            step = 0.40 * max(dist, 0.15)
            p = p + step * fwd
        elif action == "backward":
            # Dolly back along the same optical axis (paired with forward).
            fwd = _view_dir_world_from_quat(q_use)
            dist = float(np.linalg.norm(p - _LOOK_AT_TARGET))
            step = 0.35 * max(dist, 0.15)
            p = p - step * fwd
        elif action == "left":
            right = _camera_right_world_from_quat(q_use)
            p = p - 0.55 * right
        elif action == "right":
            right = _camera_right_world_from_quat(q_use)
            p = p + 0.55 * right
        elif action == "camera_up":
            pass
        elif action == "camera_down":
            pass
        elif action == "camera_l":
            pass
        elif action == "camera_r":
            pass
        elif action == "camera_zoom_in":
            z *= 1.18
        elif action == "camera_zoom_out":
            z *= 0.86
        else:
            pass
        return p.astype(np.float64), z

    def _interp_pose_for_action(
        self,
        pos_start: np.ndarray,
        zoom_start: float,
        action: str,
        t: float,
        quat: Optional[List[float]] = None,
    ) -> tuple:
        """Linear interpolation within one segment: t in [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        pos_end, zoom_end = self._apply_action_end_state(
            pos_start, zoom_start, action, quat=quat
        )
        pos = (1.0 - t) * pos_start + t * pos_end
        zoom = (1.0 - t) * zoom_start + t * zoom_end
        return pos, zoom

    def _camera_actions_to_cameras(
        self,
        camera_actions: List[str],
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Convert camera action strings to camera parameter dictionaries.
        Each list entry runs in order over its own contiguous frame sub-range
        (first action -> first segment, second -> next segment, ...).
        """
        if not camera_actions:
            return self._create_default_cameras(num_frames, image_width, image_height)

        # If more actions than frames, keep one frame per action from the start of the list
        if len(camera_actions) > num_frames:
            camera_actions = camera_actions[:num_frames]

        segment_lengths = self._segment_frame_counts(num_frames, len(camera_actions))
        cameras: List[Dict[str, Any]] = []

        pos = np.array([0.0, 0.5, 2.0], dtype=np.float64)
        zoom = 1.0
        quat: List[float] = _look_at_quaternion_wxyz(pos)

        for action, seg_len in zip(camera_actions, segment_lengths):
            pos_start = pos.copy()
            zoom_start = zoom
            quat_start = quat.copy()
            _, zoom_seg_end = self._apply_action_end_state(
                pos_start, zoom_start, action, quat=quat_start
            )

            for j in range(seg_len):
                if seg_len <= 1:
                    t = 1.0
                else:
                    t = j / (seg_len - 1)

                if action in ("camera_l", "camera_r", "camera_up", "camera_down"):
                    p = pos_start.copy()
                    zm = zoom_start
                    q = _orientation_in_place_at_t(quat_start, action, t)
                elif action in ("camera_zoom_in", "camera_zoom_out"):
                    p = pos_start.copy()
                    zm = (1.0 - t) * zoom_start + t * zoom_seg_end
                    q = quat_start.copy()
                else:
                    # Translation: fixed heading; forward/backward use quat_start so dolly is along view.
                    p, zm = self._interp_pose_for_action(
                        pos_start, zoom_start, action, t, quat=quat_start
                    )
                    q = quat_start.copy()

                fx = image_width * 0.7 * zm
                fy = image_height * 0.7 * zm
                cameras.append({
                    'position': [float(p[0]), float(p[1]), float(p[2])],
                    'quaternion': q,
                    'fx': fx,
                    'fy': fy,
                    'cx': image_width * 0.5,
                    'cy': image_height * 0.5,
                })

            pos, zoom = self._apply_action_end_state(
                pos_start, zoom_start, action, quat=quat_start
            )
            if action in ("forward", "backward", "left", "right"):
                quat = quat_start.copy()
            elif action in ("camera_l", "camera_r", "camera_up", "camera_down"):
                quat = _orientation_in_place_at_t(quat_start, action, 1.0)
            elif action in ("camera_zoom_in", "camera_zoom_out"):
                quat = quat_start.copy()

        return cameras
    
    def _create_default_cameras(
        self,
        num_frames: int,
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Create default camera trajectory (circular path).
        
        Args:
            num_frames: Number of frames
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of camera dictionaries
        """
        cameras = []
        radius = 2.0
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            
            # Circular camera path
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5
            pos = np.array([x, y, z], dtype=np.float64)
            quat = _look_at_quaternion_wxyz(pos)
            
            camera = {
                'position': [float(x), float(y), float(z)],
                'quaternion': quat,
                'fx': image_width * 0.7,
                'fy': image_height * 0.7,
                'cx': image_width * 0.5,
                'cy': image_height * 0.5,
            }
            cameras.append(camera)
        
        return cameras
    
    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, Image.Image, bytes]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Process visual signal (image) for real-time interactive updates.
        
        Args:
            input_signal: Visual input signal - can be:
                - Image file path (str)
                - Numpy array (H, W, 3) in RGB format
                - Torch tensor (C, H, W) or (1, C, H, W) in CHW format
                - PIL Image
                - Base64 encoded image string
                - Bytes of image data
                
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        if isinstance(input_signal, Image.Image):
            # Already a PIL Image, convert to RGB
            return input_signal.convert('RGB')
        
        elif isinstance(input_signal, str):
            # Check if it's a file path or base64
            if os.path.exists(input_signal):
                # File path
                image = Image.open(input_signal)
                return image.convert('RGB')
            elif input_signal.startswith('data:image'):
                # Base64 encoded image
                if ',' in input_signal:
                    image_data = input_signal.split(',', 1)[1]
                else:
                    image_data = input_signal
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert('RGB')
            else:
                raise ValueError(f"Invalid input: {input_signal}")
        
        elif isinstance(input_signal, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(input_signal))
            return image.convert('RGB')
        
        elif isinstance(input_signal, np.ndarray):
            # Numpy array, expected in RGB channel order. Integer dtypes are
            # already 0-255; only float arrays in [0, 1] need rescaling.
            if np.issubdtype(input_signal.dtype, np.integer):
                input_signal = input_signal.astype(np.uint8)
            elif input_signal.max() <= 1.0:
                input_signal = (input_signal * 255).astype(np.uint8)
            else:
                input_signal = input_signal.astype(np.uint8)
            
            image = Image.fromarray(input_signal)
            return image.convert('RGB')
        
        elif isinstance(input_signal, torch.Tensor):
            # Torch tensor
            if input_signal.dim() == 3:
                image_array = input_signal.permute(1, 2, 0).cpu().numpy()
            else:
                image_array = input_signal[0].permute(1, 2, 0).cpu().numpy()
            
            if np.issubdtype(image_array.dtype, np.integer):
                image_array = image_array.astype(np.uint8)
            elif image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            image = Image.fromarray(image_array)
            return image.convert('RGB')
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_signal)}")
