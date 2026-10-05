"""Provides utilities and a runtime for in-tree GR00T checkpoint inference.

This module includes functions for selecting GR00T checkpoints, preparing observations
from various input formats, and a class for managing the GR00T policy inference lifecycle
within WorldFoundry.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.runtime.env import resolve_ckpt_dir, resolve_hfd_root
from worldfoundry.synthesis.action_generation.gr00t.architecture import (
    load_checkpoint_architecture,
    load_embodiment_ids,
)


def _jsonable(value: Any) -> Any:
    """Recursively converts a value into a JSON-serializable format.

    Handles common non-JSON types like Path objects, NumPy arrays, and recursively
    processes mappings and sequences.

    Args:
        value: The value to convert.

    Returns:
        The JSON-serializable representation of the value.
    """
    if isinstance(value, Mapping):
        # Recursively convert keys and values in mappings
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively convert items in sequences
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to strings
        return str(value)
    if hasattr(value, "tolist"):
        # Handle NumPy arrays by converting to a list
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle NumPy scalars by converting to a Python primitive
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Primitives and None are already JSON-serializable
        return value
    # Default to string conversion for any other unsupported type
    return str(value)


def select_gr00t_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    variant: str | None,
) -> Path:
    """Select a local GR00T checkpoint directory from explicit input or profile metadata.

    This function attempts to find a suitable GR00T checkpoint by checking provided
    directories, profile candidates, and optional variants. It resolves paths,
    expands environment variables, and ensures the selected path is a valid directory.

    Args:
        checkpoint_dir: An explicit checkpoint root directory or a variant subdirectory.
            If provided, this path is prioritized.
        checkpoints: A sequence of checkpoint candidates, typically from a profile,
            each being a mapping that may contain a 'local_dir' key.
        variant: An optional GR00T checkpoint variant, such as "libero_10",
            which can specify a subdirectory within a checkpoint root.

    Returns:
        The resolved Path to the selected GR00T checkpoint directory.

    Raises:
        FileNotFoundError: If no local GR00T checkpoint could be found matching
            the criteria.
    """
    def _expand_path_template(value: str) -> str:
        """Expands environment variables and WorldFoundry specific path templates."""
        defaults = {
            "WORLDFOUNDRY_CKPT_DIR": str(resolve_ckpt_dir()),
            "WORLDFOUNDRY_HFD_ROOT": str(resolve_hfd_root()),
        }
        expanded = str(value)
        # Expand custom WorldFoundry path variables
        for key, replacement in defaults.items():
            expanded = expanded.replace(f"${{{key}}}", replacement)
            expanded = expanded.replace(f"${key}", replacement)
        # Expand standard shell environment variables
        return os.path.expandvars(expanded)

    candidates: list[Mapping[str, Any]] = []
    # Prioritize an explicitly provided checkpoint directory
    if checkpoint_dir:
        candidates.append({"local_dir": str(checkpoint_dir), "role": "explicit_checkpoint"})
    # Add profile-based checkpoint candidates
    candidates.extend(dict(item) for item in checkpoints)

    requested = str(variant or "").lower()
    if checkpoint_dir:
        explicit = Path(_expand_path_template(str(checkpoint_dir))).expanduser().resolve()
        if (explicit / "config.json").is_file():
            return explicit

    # First pass: try to find a checkpoint matching the requested variant or role/path substring
    for item in candidates:
        path_text = _expand_path_template(str(item.get("local_dir") or ""))
        if not path_text:
            continue
        # Resolve the absolute path
        path = Path(path_text).expanduser().resolve()
        # Check for a variant-specific subdirectory
        variant_path = path / str(variant) if variant else path
        if requested and variant_path.is_dir():
            return variant_path

        # Check if the requested variant matches the role or a substring of the path
        role = str(item.get("role") or "").lower()
        if requested and (requested in role or requested in path_text.lower()) and path.is_dir():
            return path
        # If no specific variant is requested, return the first valid directory
        if not requested and path.is_dir():
            return path

    # Second pass: if no variant was explicitly requested or found, return any valid checkpoint directory
    # This acts as a fallback if the first pass didn't yield a specific match but a valid directory exists
    for item in candidates:
        path_text = _expand_path_template(str(item.get("local_dir") or ""))
        if not path_text:
            continue
        path = Path(path_text).expanduser().resolve()
        if path.is_dir():
            return path
    raise FileNotFoundError("No local GR00T checkpoint was found.")


def _require_checkpoint_file(checkpoint_dir: Path, filename: str) -> Path:
    """Ensures a specific file exists within the GR00T checkpoint directory.

    Args:
        checkpoint_dir: The root directory of the GR00T checkpoint.
        filename: The name of the file to check for.

    Returns:
        The Path to the required file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = checkpoint_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"GR00T checkpoint file is missing: {path}")
    return path


def _load_image_array(image: Any) -> tuple[Any, str]:
    """Loads an image from various input types into a standard HxWxC uint8 NumPy array.

    Args:
        image: The input image, which can be:
            - None (raises ValueError)
            - A PIL.Image.Image object
            - A string or Path to an image file
            - A Mapping (takes the first value)
            - A Sequence (takes the first item)
            - A NumPy array (will be processed to HxWxC uint8)

    Returns:
        A tuple containing:
            - The image as a NumPy array (HxWxC, dtype=np.uint8).
            - A string indicating the source of the image (e.g., "in-memory:PIL.Image", "path/to/image.png").

    Raises:
        ValueError: If the input image is None, an empty mapping/sequence, or
            an unsupported array shape.
        FileNotFoundError: If the image path does not exist.
    """
    import numpy as np
    from PIL import Image

    if image is None:
        raise ValueError("GR00T runtime requires at least one RGB observation image.")
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB NumPy array
        return np.asarray(image.convert("RGB"), dtype=np.uint8), "in-memory:PIL.Image"
    if isinstance(image, (str, Path)):
        # Load image from file path
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"GR00T image path does not exist: {image_path}")
        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8), str(image_path)
    if isinstance(image, Mapping):
        # If a dictionary (e.g., multiple camera views), take the first image
        if not image:
            raise ValueError("GR00T received an empty camera view mapping.")
        return _load_image_array(next(iter(image.values())))
    if isinstance(image, Sequence) and not isinstance(image, (bytes, bytearray, str)):
        # If a sequence (e.g., list of images), take the first image
        if not image:
            raise ValueError("GR00T received an empty image sequence.")
        return _load_image_array(image[0])

    # Assume input is a NumPy-like array and process it
    array = np.asarray(image)
    # Handle common array shapes from sensor data (e.g., (1, T, H, W, C) or (T, H, W, C))
    if array.ndim == 5:
        array = array[0, -1]  # Take the last frame of the first batch item
    elif array.ndim == 4:
        array = array[0]  # Take the first batch item
    if array.ndim != 3:
        raise ValueError(f"GR00T image array must be HxWxC or CxHxW, got shape {array.shape}.")
    # Transpose if the channel dimension is at the front (CxHxW)
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    # Convert float arrays (0.0-1.0) to uint8 (0-255)
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8), "in-memory:array"


def _zero_state(modality_keys: Sequence[str], state_statistics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Creates a dictionary of zero-initialized state vectors for GR00T.

    Each state vector is a NumPy array of shape (1, 1, size), where size is inferred
    from state statistics if available, otherwise defaults to 1.

    Args:
        modality_keys: A sequence of string keys for the state modalities.
        state_statistics: Optional mapping containing statistics for state modalities,
            used to determine the size of each state vector.

    Returns:
        A dictionary where keys are modality strings and values are zero-initialized
        NumPy arrays.
    """
    import numpy as np

    state: dict[str, Any] = {}
    for key in modality_keys:
        key_text = str(key)
        size = 1
        # Attempt to infer the size of the state vector from provided statistics
        if isinstance(state_statistics, Mapping):
            values = state_statistics.get(key_text)
            if isinstance(values, Mapping) and isinstance(values.get("mean"), Sequence):
                size = max(1, len(values["mean"]))
        state[key_text] = np.zeros((1, 1, size), dtype=np.float32)
    return state


def _resolve_observation(
    *,
    instruction: str,
    image: Any,
    gr00t_observation: Mapping[str, Any] | None,
    checkpoint_dir: Path,
    embodiment_value: str,
) -> tuple[dict[str, Any], str]:
    """Constructs a GR00T-compatible observation dictionary.

    This involves loading checkpoint configuration, processing the input image,
    initializing state modalities, and formatting everything into the expected
    GR00T observation structure.

    Args:
        instruction: The natural language instruction for the policy.
        image: The raw image input (e.g., PIL Image, path, NumPy array).
        gr00t_observation: Optional existing GR00T observation containing, for example,
            'joint_state' or 'proprio' for state modalities.
        checkpoint_dir: The path to the GR00T checkpoint directory.
        embodiment_value: The string value representing the current embodiment.

    Returns:
        A tuple containing:
            - The fully constructed GR00T observation dictionary.
            - A string indicating the source of the main observation image.
    """
    # Load GR00T checkpoint architecture and processor configuration
    architecture = load_checkpoint_architecture(checkpoint_dir)
    processor_config = json.loads((checkpoint_dir / "processor_config.json").read_text(encoding="utf-8"))
    statistics = json.loads((checkpoint_dir / "statistics.json").read_text(encoding="utf-8"))

    # Extract modality keys specific to the current embodiment from the config
    modality_configs = processor_config["processor_kwargs"]["modality_configs"][embodiment_value]
    state_statistics = statistics.get(embodiment_value, {}).get("state")
    video_keys = [str(key) for key in modality_configs["video"]["modality_keys"]]
    language_key = str(modality_configs["language"]["modality_keys"][0])
    state_keys = [str(key) for key in modality_configs["state"]["modality_keys"]]

    # Preserve named camera views instead of duplicating the first camera.
    cameras = image
    if gr00t_observation:
        supplied_video = gr00t_observation.get("video")
        if supplied_video is None:
            supplied_video = gr00t_observation.get("camera_views")
        if supplied_video is not None:
            cameras = supplied_video
    video = {}
    sources = {}
    for video_key in video_keys:
        if isinstance(cameras, Mapping):
            if video_key not in cameras:
                raise ValueError(f"GR00T observation is missing camera {video_key!r}.")
            camera = cameras[video_key]
        else:
            camera = cameras
        image_array, source = _load_image_array(camera)
        video[video_key] = image_array[None, None, ...]
        sources[video_key] = source
    image_source = json.dumps(sources) if isinstance(cameras, Mapping) else next(iter(sources.values()))

    # Initialize all state modalities to zeros
    state = _zero_state(state_keys, state_statistics)
    # If proprioceptive data is provided, update the state modalities
    if gr00t_observation:
        joint_state = gr00t_observation.get("state")
        if joint_state is None:
            joint_state = gr00t_observation.get("joint_state")
        if joint_state is None:
            joint_state = gr00t_observation.get("proprio") # Fallback for older naming conventions
        if isinstance(joint_state, Mapping):
            state.update(joint_state)

    # Construct the final observation dictionary in the format expected by GR00T
    return (
        {
            "video": video,
            "state": state,
            "language": {language_key: [[instruction or "do something"]]},
            "metadata": {"architecture": architecture},
        },
        image_source,
    )


@dataclass(frozen=True)
class GR00TRuntimeConfig:
    """Runtime settings for a vendored GR00T policy inference call."""

    checkpoint_dir: Path
    embodiment_tag: str
    device: str
    torch_dtype: str
    seed: int


class GR00TRuntime:
    """Lazy in-tree GR00T runtime backed by vendored official checkpoint metadata.

    This class manages the loading and inference of a GR00T policy, handling
    configuration, observation preparation, and action trace generation.
    """

    def __init__(self, config: GR00TRuntimeConfig) -> None:
        """Initializes the GR00TRuntime with the given configuration.

        The GR00T policy is loaded lazily on the first call to `load()` or `predict_action()`.

        Args:
            config: The GR00TRuntimeConfig object containing policy settings.
        """
        self.config = config
        self.policy: Any | None = None
        self.embodiment_value: str | None = None

    @staticmethod
    def describe_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
        """Return checkpoint metadata used by WorldFoundry run plans.

        This method verifies the existence of essential checkpoint files and
        loads architectural and embodiment ID information.

        Args:
            checkpoint_dir: The path to the GR00T checkpoint directory.

        Returns:
            A dictionary containing architecture and embodiment IDs metadata.

        Raises:
            FileNotFoundError: If any required checkpoint file is missing.
        """
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        # Verify essential checkpoint files exist
        _require_checkpoint_file(checkpoint, "config.json")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "gr00t_n1":
            from worldfoundry.synthesis.action_generation.gr00t.legacy_n1 import describe_n1_checkpoint

            return describe_n1_checkpoint(checkpoint)
        _require_checkpoint_file(checkpoint, "processor_config.json")
        _require_checkpoint_file(checkpoint, "embodiment_id.json")
        _require_checkpoint_file(checkpoint, "model.safetensors.index.json")
        # Load architecture and embodiment IDs
        architecture = load_checkpoint_architecture(checkpoint)
        architecture["embodiment_ids"] = load_embodiment_ids(checkpoint)
        return architecture

    def _resolve_embodiment_value(self) -> str:
        """Resolves the internal GR00T embodiment value from the configured embodiment tag."""
        # Ensure gr00t modules are correctly aliased before importing EmbodimentTag
        from worldfoundry.synthesis.action_generation.gr00t.preprocessing import EmbodimentTag

        tag = EmbodimentTag.resolve(self.config.embodiment_tag)
        return str(tag.value)

    def load(self) -> None:
        """Load the configured GR00T policy from its local checkpoint.

        This method is idempotent; the policy will only be loaded once.
        It also resolves the internal embodiment value and verifies checkpoint files.
        """
        if self.policy is not None:
            return
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        # Ensure gr00t modules are correctly aliased before policy import
        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        # Verify checkpoint files and load metadata
        self.describe_checkpoint(checkpoint)
        # Resolve the internal string representation of the embodiment tag
        self.embodiment_value = self._resolve_embodiment_value()
        from worldfoundry.synthesis.action_generation.gr00t.policy import Gr00tPolicy

        device = resolve_inference_device(self.config.device, allow_cpu_fallback=True)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)

        self.policy = Gr00tPolicy(
            embodiment_tag=self.config.embodiment_tag,
            model_path=str(checkpoint),
            device=device,
            torch_dtype=dtype,
        )

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        gr00t_observation: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one GR00T policy inference and write a WorldFoundry action trace.

        This method prepares the observation, runs the policy to get an action,
        and then serializes the results into a JSON action trace file.

        Args:
            instruction: The natural language instruction for the GR00T policy.
            image: The current observation image (various formats supported).
            output_path: The file path where the action trace JSON will be saved.
            gr00t_observation: Optional, additional GR00T-specific observation data,
                such as proprioception or joint states.
            extra_metadata: Optional, additional key-value pairs to include in
                the output action trace metadata.

        Returns:
            A dictionary summarizing the inference result and the path to the artifact.
        """
        # Ensure the policy is loaded before inference
        self.load()
        assert self.policy is not None
        assert self.embodiment_value is not None

        import numpy as np

        # Set NumPy random seed for reproducibility
        np.random.seed(self.config.seed)
        started = time.monotonic()

        # Resolve and prepare the observation dictionary for the GR00T policy
        observation, image_source = _resolve_observation(
            instruction=instruction,
            image=image,
            gr00t_observation=gr00t_observation,
            checkpoint_dir=self.config.checkpoint_dir,
            embodiment_value=self.embodiment_value,
        )
        # Run the GR00T policy to predict the action
        action, info = self.policy.get_action(observation)

        # Prepare the output directory and construct the action trace payload
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-gr00t-action-trace",
            "status": "success",
            "model_id": "gr00t",
            "backend": "worldfoundry.gr00t.in_tree_runtime.predict_action",
            "backend_quality": "official_checkpoint_wrapper",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "device": self.config.device,
            "embodiment_tag": self.config.embodiment_tag,
            "image_source": image_source,
            "instruction": instruction,
            "seed": self.config.seed,
            "action": _jsonable(action), # Ensure action is JSON-serializable
            "info": _jsonable(info), # Ensure info is JSON-serializable
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})), # Include extra metadata
        }
        # Write the action trace to the specified output path
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Calculate SHA256 hash of the generated artifact
        artifact_sha256 = file_sha256(target)
        # Return a summary of the action trace artifact
        return {
            "status": "success",
            "model_id": "gr00t",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
