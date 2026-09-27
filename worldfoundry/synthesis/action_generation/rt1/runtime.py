# Inference-only RT-1 source retained in-tree.
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

RUNTIME_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RT1RuntimeConfig:
    """Configure the RT-1 SavedModel runtime.

    Args:
        checkpoint_dir: Directory containing the official TensorFlow SavedModel.
        device: TensorFlow device selector, such as "cpu", "cuda", or "/CPU:0".
    """

    checkpoint_dir: Path
    device: str = "cpu"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_rgb_array(image: Any, *, size: tuple[int, int]) -> np.ndarray:
    if image is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        array = image
    else:
        from PIL import Image

        image_path = Path(str(image)).expanduser()
        array = np.asarray(Image.open(image_path).convert("RGB").resize(size), dtype=np.uint8)
    if array.ndim == 4:
        array = array[0]
    return np.asarray(array, dtype=np.uint8)


def _language_embedding_array(value: Any) -> np.ndarray:
    """Validate the 512-D embedding consumed by the official RT-1 network."""
    if value is None:
        raise ValueError(
            "RT-1 requires language_embedding: a 512-D embedding from the "
            "encoder used for this checkpoint. A prompt string alone is not "
            "encoded by the released SavedModel."
        )
    if isinstance(value, (str, Path)):
        source = Path(value).expanduser()
        if source.suffix.lower() == ".npy":
            value = np.load(source, allow_pickle=False)
        elif source.suffix.lower() == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("RT-1 language_embedding file must be .json or .npy")
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (512,):
        array = array[None, :]
    if array.shape != (1, 512):
        raise ValueError(f"RT-1 language_embedding must have shape (512,) or (1, 512), got {array.shape}")
    if not np.isfinite(array).all() or not np.any(array):
        raise ValueError("RT-1 language_embedding must contain finite, nonzero values")
    return np.ascontiguousarray(array)


class RT1SavedModelRuntime:
    """Run the in-tree RT-1 TensorFlow SavedModel action signature.

    Args:
        config: Runtime configuration with checkpoint location and device.
    """

    def __init__(self, config: RT1RuntimeConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._tf: Any | None = None

    @property
    def tf_device(self) -> str:
        device = self.config.device.lower()
        if device.startswith("/"):
            return self.config.device
        if device.startswith("cuda") or device.startswith("gpu"):
            return "/GPU:0"
        return "/CPU:0"

    def _load_model(self) -> Any:
        """Load TensorFlow lazily and restore the official SavedModel.

        Args:
            None.
        """

        if self._model is None:
            import tensorflow as tf
            import tensorflow_probability as _tfp  # noqa: F401


            self._tf = tf
            with tf.device(self.tf_device):
                self._model = tf.saved_model.load(str(self.config.checkpoint_dir))
        return self._model

    def predict_action(
        self,
        *,
        instruction: str,
        language_embedding: Any = None,
        image: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one RT-1 action step and write an action_trace artifact.

        Args:
            instruction: Natural-language task instruction for the policy.
            language_embedding: Checkpoint-compatible 512-D language embedding.
            image: RGB image path or array used as the current observation.
            output_path: JSON artifact path for the action trace.
            extra_metadata: Additional serializable context included in output.
        """

        embedding = _language_embedding_array(language_embedding)
        started = time.monotonic()
        model = self._load_model()
        tf = self._tf
        zeros = tf.zeros
        with tf.device(self.tf_device):
            initial_state = model.signatures["get_initial_state"](batch_size=tf.constant(1, dtype=tf.int32))
        image_array = _load_rgb_array(image, size=(320, 256))[None]
        with tf.device(self.tf_device):
            action_inputs = {
                "0/observation/natural_language_embedding": tf.convert_to_tensor(embedding, dtype=tf.float32),
                "0/observation/orientation_start": tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32),
                "0/observation/orientation_box": zeros((1, 2, 3), tf.float32),
                "1/step_num": initial_state["step_num"],
                "0/observation/gripper_closedness_commanded": zeros((1, 1), tf.float32),
                "0/observation/workspace_bounds": zeros((1, 3, 3), tf.float32),
                "0/step_type": zeros((1,), tf.int32),
                "0/observation/robot_orientation_positions_box": zeros((1, 3, 3), tf.float32),
                "0/observation/height_to_bottom": zeros((1, 1), tf.float32),
                "0/observation/image": tf.convert_to_tensor(image_array, dtype=tf.uint8),
                "1/t": initial_state["t"],
                "0/observation/gripper_closed": zeros((1, 1), tf.float32),
                "0/observation/base_pose_tool_reached": zeros((1, 7), tf.float32),
                "0/observation/rotation_delta_to_go": zeros((1, 3), tf.float32),
                "0/reward": zeros((1,), tf.float32),
                "0/observation/vector_to_go": zeros((1, 3), tf.float32),
                "0/discount": tf.ones((1,), tf.float32),
                "0/observation/natural_language_instruction": tf.constant([instruction], dtype=tf.string),
                "1/image": initial_state["image"],
                "0/observation/src_rotation": tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32),
                "1/action_tokens": initial_state["action_tokens"],
            }
            outputs = model.signatures["action"](**action_inputs)
        action = {
            "world_vector": _jsonable(outputs["action/world_vector"][0]),
            "rotation_delta": _jsonable(outputs["action/rotation_delta"][0]),
            "gripper_closedness_action": _jsonable(outputs["action/gripper_closedness_action"][0]),
            "terminate_episode": _jsonable(outputs["action/terminate_episode"][0]),
            "base_displacement_vector": _jsonable(outputs["action/base_displacement_vector"][0]),
            "base_displacement_vertical_rotation": _jsonable(outputs["action/base_displacement_vertical_rotation"][0]),
        }
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-rt1-in-tree-action-trace",
            "status": "success",
            "model_id": "rt-1",
            "backend": "worldfoundry.rt1.in_tree_savedmodel.action_signature",
            "backend_quality": "in_tree_runtime",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "runtime_root": str(RUNTIME_ROOT),
            "instruction": instruction,
            "language_embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
            "device": self.tf_device,
            "action": action,
            "actions": [action],
            "state_shapes": {key: list(value.shape) for key, value in initial_state.items()},
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if extra_metadata:
            payload["metadata"] = _jsonable(dict(extra_metadata))
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "model_id": "rt-1",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
