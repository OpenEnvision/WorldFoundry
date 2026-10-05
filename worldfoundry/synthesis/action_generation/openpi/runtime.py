"""
Utility module for managing and interacting with the OpenPI (Open-Vocabulary Policy Integration) runtime.

This module provides functionalities for selecting OpenPI checkpoints, loading image data
for observations, resolving observations for policy inference, and running an in-tree
OpenPI policy to predict actions. It integrates with `worldfoundry` path utilities and
handles necessary environment setups like installing OpenPI aliases.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path

_PARTIAL_CHECKPOINT_SUFFIXES = (".aria2", ".incomplete", ".gstmp")


def _checkpoint_complete(path: Path) -> bool:
    """Accept only a complete local PyTorch or Orbax OpenPI checkpoint."""

    if not path.is_dir():
        return False
    try:
        for candidate in path.rglob("*"):
            if candidate.name.endswith(_PARTIAL_CHECKPOINT_SUFFIXES):
                return False
            if (
                candidate.name == "._____temp"
                and candidate.is_dir()
                and next(candidate.iterdir(), None) is not None
            ):
                return False

        pytorch_weights = path / "model.safetensors"
        if pytorch_weights.is_file() and pytorch_weights.stat().st_size > 0:
            return True

        params = path / "params"
        metadata = params / "_METADATA"
        manifest = params / "manifest.ocdbt"
        process_manifest = params / "ocdbt.process_0" / "manifest.ocdbt"
        data_dir = params / "ocdbt.process_0" / "d"
        return (
            metadata.is_file()
            and metadata.stat().st_size > 0
            and manifest.is_file()
            and manifest.stat().st_size > 0
            and process_manifest.is_file()
            and process_manifest.stat().st_size > 0
            and data_dir.is_dir()
            and next((item for item in data_dir.iterdir() if item.is_file() and item.stat().st_size > 0), None)
            is not None
        )
    except OSError:
        return False


def _jsonable(value: Any) -> Any:
    """
    Recursively converts a value into a JSON-serializable format.

    Handles common Python types, Path objects, numpy arrays (via `tolist` or `item`),
    and falls back to string conversion for other types.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of the value.
    """
    if isinstance(value, Mapping):
        # Recursively process dictionary keys and values.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively process list/tuple items.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to string representation.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle numpy arrays or similar objects by converting to a Python list.
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle numpy scalars or similar objects by converting to a Python scalar.
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Base cases: already JSON-serializable types.
        return value
    # Fallback for any other type, converting to string.
    return str(value)


def select_openpi_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    config_name: str,
    require_exists: bool = True,
) -> Path:
    """
    Selects the most suitable local OpenPI checkpoint path based on provided candidates and configuration.

    The selection process prioritizes explicitly provided directories, then candidates
    matching the `config_name` in their metadata, and finally any available checkpoint.

    Args:
        checkpoint_dir: An optional explicit directory path for the checkpoint.
        checkpoints: A sequence of checkpoint candidate mappings, each potentially containing
            'local_dir', 'role', or 'path' keys.
        config_name: The name of the policy configuration, used for filtering preferred candidates.
        require_exists: If True, only returns a path to an existing directory. If False,
            may return a non-existent path if no existing ones are found.

    Returns:
        The resolved Path to the selected OpenPI checkpoint directory.

    Raises:
        FileNotFoundError: If `require_exists` is True and no existing checkpoint is found.
    """
    if checkpoint_dir:
        explicit_path = _checkpoint_path({"local_dir": str(checkpoint_dir)})
        if explicit_path is not None and (_checkpoint_complete(explicit_path) or not require_exists):
            return explicit_path

    candidates: list[Mapping[str, Any]] = []
    # Extend with other provided checkpoint candidates.
    candidates.extend(dict(item) for item in checkpoints)

    # First, try to find checkpoints whose metadata contains tokens from the config_name.
    preferred = _preferred_checkpoint_candidates(candidates, config_name)
    for item in preferred:
        path = _checkpoint_path(item)
        if path is not None and _checkpoint_complete(path):
            return path

    # If no preferred existing checkpoint, iterate through all candidates.
    for item in candidates:
        path = _checkpoint_path(item)
        if path is None:
            continue
        if _checkpoint_complete(path):
            return path
    # If no existing checkpoint is found and `require_exists` is False,
    # try to return the path of any candidate (preferred first).
    if not require_exists:
        for item in (*preferred, *candidates):
            path = _checkpoint_path(item)
            if path is not None:
                return path
    raise FileNotFoundError("No local OpenPI checkpoint was found.")


def _checkpoint_path(item: Mapping[str, Any]) -> Path | None:
    """
    Resolves a checkpoint item mapping into an absolute Path object.

    Looks for 'local_dir' within the item to determine the checkpoint path.
    Paths are resolved relative to the worldfoundry project root if not absolute.

    Args:
        item: A dictionary representing a checkpoint candidate, expected to have a 'local_dir' key.

    Returns:
        An absolute Path object if 'local_dir' is present and valid, otherwise None.
    """
    local_dir = str(item.get("local_dir") or "")
    if not local_dir:
        return None
    # Resolve the path using worldfoundry specific path resolution logic.
    path = resolve_worldfoundry_path(local_dir)
    # If the path is not absolute, assume it's relative to the project root.
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _preferred_checkpoint_candidates(
    candidates: Sequence[Mapping[str, Any]],
    config_name: str,
) -> tuple[Mapping[str, Any], ...]:
    """
    Filters a list of checkpoint candidates to identify those preferred based on the configuration name.

    A candidate is "preferred" if keywords from the `config_name` are found in its
    'role', 'local_dir', or 'path' metadata fields (case-insensitive, underscore-normalized).

    Args:
        candidates: A sequence of checkpoint candidate mappings.
        config_name: The name of the policy configuration (e.g., "libero-100-test").

    Returns:
        A tuple of preferred checkpoint candidate mappings. If no candidates fully match,
        it returns candidates that partially match any token.
    """
    # Normalize config_name into a tuple of significant tokens.
    tokens = tuple(part for part in config_name.lower().replace("-", "_").split("_") if part)
    if not tokens:
        return ()

    preferred: list[Mapping[str, Any]] = []
    for item in candidates:
        # Create a haystack string from relevant metadata fields for matching.
        haystack = " ".join(str(item.get(key) or "").lower().replace("-", "_") for key in ("role", "local_dir", "path"))
        # A candidate is preferred if ALL tokens from the config_name are found in the haystack.
        if all(token in haystack for token in tokens):
            preferred.append(item)

    if preferred:
        return tuple(preferred)

    # If no candidates fully match, return candidates where ANY token from the config_name is found.
    return tuple(
        item
        for item in candidates
        if any(token in " ".join(str(item.get(key) or "").lower().replace("-", "_") for key in ("role", "local_dir", "path")) for token in tokens)
    )


def _load_rgb_array(image: Any) -> Any:
    """
    Loads one image into an RGB uint8 NumPy array without changing its aspect ratio.

    Supports PIL images, local file paths, and array-like HWC/CHW images. Spatial
    resizing is intentionally left to the retained OpenPI resize-with-padding transform.

    Args:
        image: One input image, as a PIL image, local path, or array-like value.

    Returns:
        An HWC RGB NumPy array with dtype uint8, or None if the input was None.

    Raises:
        FileNotFoundError: If `image` is a path but the file does not exist.
        ValueError: If `image` is a NumPy array with an unsupported number of dimensions.
    """
    if image is None:
        return None
    # Defer import of heavy libraries until they are needed.
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB and preserve its native spatial shape.
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)
    if isinstance(image, (str, Path)):
        # Load image from a local path and preserve its native spatial shape.
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenPI image path does not exist: {image_path}")
        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    # Assume input is a NumPy array or similar array-like object.
    array = np.asarray(image)
    if array.ndim != 3 or array.size == 0:
        raise ValueError(f"OpenPI image array must be HxWxC or CxHxW, got shape {array.shape}.")
    # Transpose array if it's in CxHxW format (e.g., (3, 224, 224)) to HxWxC.
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"OpenPI image must have 1, 3, or 4 channels, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("OpenPI image contains non-finite values.")
    # Normalize unit-range floating arrays, while preserving already byte-scaled floats.
    if array.dtype.kind == "f":
        if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    # Convert through PIL to consistently handle grayscale and alpha channels.
    return np.asarray(Image.fromarray(array.astype(np.uint8)).convert("RGB"), dtype=np.uint8)


def _mapping_sources(value: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    sources = [value]
    for key in ("observation", "images", "rgb_views", "vision"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def _first_observation_value(
    observation: Mapping[str, Any] | None,
    aliases: Sequence[str],
) -> Any:
    for source in _mapping_sources(observation):
        for alias in aliases:
            if alias not in source:
                continue
            value = source[alias]
            if isinstance(value, Mapping):
                for payload_key in ("color", "rgb", "image"):
                    if payload_key in value:
                        value = value[payload_key]
                        break
            if value is not None:
                return value
    return None


def _positional_images(image: Any) -> list[Any]:
    if image is None or isinstance(image, (str, bytes, bytearray, Path, Mapping)):
        return []
    if isinstance(image, Sequence):
        return list(image)
    return []


def _camera_value(
    observation: Mapping[str, Any] | None,
    image: Any,
    aliases: Sequence[str],
    *,
    position: int,
) -> Any:
    value = _first_observation_value(observation, aliases)
    if value is not None:
        return value
    if isinstance(image, Mapping):
        value = _first_observation_value(image, aliases)
        if value is not None:
            return value
    positional = _positional_images(image)
    if position < len(positional):
        return positional[position]
    if position == 0 and image is not None and not isinstance(image, Mapping) and not positional:
        return image
    return None


def _required_image(
    observation: Mapping[str, Any] | None,
    image: Any,
    aliases: Sequence[str],
    *,
    position: int,
    label: str,
) -> Any:
    value = _camera_value(observation, image, aliases, position=position)
    if value is None:
        raise ValueError(
            f"OpenPI {label} is required; accepted camera keys are {tuple(aliases)}."
        )
    return _load_rgb_array(value)


def _state_vector(
    observation: Mapping[str, Any] | None,
    aliases: Sequence[str],
    *,
    width: int,
    label: str,
) -> Any:
    import numpy as np

    value = _first_observation_value(observation, aliases)
    if value is None:
        raise ValueError(
            f"OpenPI {label} is required; pass an explicit {width}-D state using one of {tuple(aliases)}."
        )
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (width,):
        raise ValueError(f"OpenPI {label} must have shape ({width},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"OpenPI {label} contains non-finite values.")
    return array


def _resolve_droid_observation(
    prompt: str,
    image: Any,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    import numpy as np

    joint_aliases = (
        "observation/joint_position",
        "joint_position",
        "joint_positions",
    )
    gripper_aliases = (
        "observation/gripper_position",
        "gripper_position",
        "gripper",
    )
    joint = _first_observation_value(observation, joint_aliases)
    gripper = _first_observation_value(observation, gripper_aliases)
    if joint is None and gripper is None:
        state = _state_vector(
            observation,
            ("observation/state", "state", "proprio", "robot_state", "joint_state"),
            width=8,
            label="DROID state",
        )
        joint_array, gripper_array = state[:7], state[7:]
    elif joint is None or gripper is None:
        missing = "joint_position" if joint is None else "gripper_position"
        raise ValueError(
            f"OpenPI DROID observation is missing {missing!r}; provide both joint/gripper fields "
            "or one explicit 8-D state vector."
        )
    else:
        joint_array = np.asarray(joint, dtype=np.float32).reshape(-1)
        gripper_array = np.asarray(gripper, dtype=np.float32).reshape(-1)
        if joint_array.shape != (7,) or gripper_array.shape != (1,):
            raise ValueError(
                "OpenPI DROID joint_position/gripper_position must have shapes (7,) and (1,), "
                f"got {joint_array.shape} and {gripper_array.shape}."
            )
        if not np.all(np.isfinite(joint_array)) or not np.all(np.isfinite(gripper_array)):
            raise ValueError("OpenPI DROID state contains non-finite values.")

    base = _required_image(
        observation,
        image,
        (
            "observation/exterior_image_1_left",
            "exterior_image_1_left",
            "base_0_rgb",
            "cam_high",
            "cam_head",
            "head_camera",
            "top_camera",
            "observation/image",
            "image",
        ),
        position=0,
        label="DROID exterior image",
    )
    wrist = _required_image(
        observation,
        image,
        (
            "observation/wrist_image_left",
            "wrist_image_left",
            "left_wrist_0_rgb",
            "wrist_0_rgb",
            "cam_left_wrist",
            "left_wrist",
            "observation/wrist_image",
            "wrist_image",
        ),
        position=1,
        label="DROID wrist image",
    )
    return {
        "observation/exterior_image_1_left": base,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_array,
        "observation/gripper_position": gripper_array,
        "prompt": prompt or "do something",
    }


def _resolve_libero_observation(
    prompt: str,
    image: Any,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state = _state_vector(
        observation,
        ("observation/state", "state", "proprio", "robot_state", "joint_state"),
        width=8,
        label="LIBERO state",
    )
    base = _required_image(
        observation,
        image,
        ("observation/image", "image", "base_0_rgb", "agentview", "agentview_image", "cam_high"),
        position=0,
        label="LIBERO base image",
    )
    wrist = _required_image(
        observation,
        image,
        ("observation/wrist_image", "wrist_image", "left_wrist_0_rgb", "cam_left_wrist"),
        position=1,
        label="LIBERO wrist image",
    )
    return {
        "observation/state": state,
        "observation/image": base,
        "observation/wrist_image": wrist,
        "prompt": prompt or "do something",
    }


def _resolve_aloha_observation(
    prompt: str,
    image: Any,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    import numpy as np

    state = _state_vector(
        observation,
        ("observation/state", "state", "proprio", "robot_state", "joint_state"),
        width=14,
        label="ALOHA state",
    )
    cameras = (
        ("cam_high", ("cam_high", "top", "top_camera", "head_camera", "base_0_rgb", "observation/image"), True),
        ("cam_left_wrist", ("cam_left_wrist", "left_wrist", "left_camera", "left_wrist_0_rgb"), False),
        ("cam_right_wrist", ("cam_right_wrist", "right_wrist", "right_camera", "right_wrist_0_rgb"), False),
        ("cam_low", ("cam_low", "low", "low_camera", "base_1_rgb"), False),
    )
    images: dict[str, Any] = {}
    for position, (name, aliases, required) in enumerate(cameras):
        value = _camera_value(observation, image, aliases, position=position)
        if value is None:
            if required:
                raise ValueError(
                    f"OpenPI ALOHA base image is required; accepted camera keys are {aliases}."
                )
            continue
        # The retained AlohaInputs transform consumes CHW uint8 frames.
        images[name] = np.transpose(_load_rgb_array(value), (2, 0, 1))
    return {"state": state, "images": images, "prompt": prompt or "do something"}


def _resolve_observation(
    prompt: str,
    image: Any,
    openpi_observation: Mapping[str, Any] | None,
    *,
    data_family: str,
) -> dict[str, Any]:
    """
    Construct an exact embodiment-specific raw observation for OpenPI transforms.

    Args:
        prompt: The textual instruction for the policy.
        image: The raw image input (can be path, PIL Image, array, etc.).
        openpi_observation: An optional dictionary containing additional OpenPI
            specific observation data, e.g., 'proprio'.

    Returns:
        A dictionary representing the full observation ready for an OpenPI policy.
    """
    family = str(data_family).strip().lower()
    if family == "droid":
        return _resolve_droid_observation(prompt, image, openpi_observation)
    if family == "libero":
        return _resolve_libero_observation(prompt, image, openpi_observation)
    if family == "aloha":
        return _resolve_aloha_observation(prompt, image, openpi_observation)
    raise ValueError(f"Unsupported OpenPI data family: {data_family!r}")


@dataclass(frozen=True)
class OpenPIRuntimeConfig:
    """
    Configuration settings for initializing and running an OpenPI policy.

    Attributes:
        checkpoint_dir: The path to the directory containing the policy checkpoint files.
        config_name: The name of the OpenPI policy configuration (e.g., "libero-100-test").
        pytorch_device: The PyTorch device to use for inference (e.g., "cuda:0", "cpu").
        seed: The random seed to use for reproducible policy inference.
    """

    checkpoint_dir: Path
    config_name: str
    data_family: str
    pytorch_device: str | None
    torch_dtype: str
    seed: int
    paligemma_tokenizer_path: str
    fast_tokenizer_path: str


class OpenPIRuntime:
    """
    A lazy-loading runtime for performing inference with an in-tree OpenPI policy.

    This class encapsulates the logic for loading a specific OpenPI policy from a
    checkpoint and then using it to predict actions based on observations.
    Policy loading is deferred until the first call to `load()` or `predict_action()`.
    """

    def __init__(self, config: OpenPIRuntimeConfig) -> None:
        """
        Initializes the OpenPIRuntime with the given configuration.

        The policy itself is not loaded during initialization but is loaded on demand.

        Args:
            config: An instance of OpenPIRuntimeConfig specifying policy details.
        """
        self.config = config
        self.policy: Any | None = None

    def load(self) -> None:
        """
        Loads the configured OpenPI policy from its local checkpoint directory.

        This method is idempotent; if the policy is already loaded, it does nothing.
        It imports necessary OpenPI modules dynamically and creates the policy instance.

        Raises:
            FileNotFoundError: If the specified checkpoint directory does not exist.
        """
        if self.policy is not None:
            return
        # Resolve and validate the checkpoint directory path.
        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint directory does not exist: {checkpoint}")

        # Resolve the inference configuration for the specified policy.
        from . import config as openpi_config
        from . import policy_loader

        openpi_config.configure_local_tokenizers(
            paligemma=self.config.paligemma_tokenizer_path,
            fast=self.config.fast_tokenizer_path,
        )
        runtime_config = openpi_config.get_config(self.config.config_name)
        if runtime_config.data_family != self.config.data_family:
            raise ValueError(
                f"OpenPI config {self.config.config_name!r} uses data family "
                f"{runtime_config.data_family!r}, not {self.config.data_family!r}."
            )
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        device = resolve_inference_device(
            self.config.pytorch_device or "cuda",
            allow_cpu_fallback=True,
        )
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        # Create the policy instance using the configuration and checkpoint.
        self.policy = policy_loader.create_trained_policy(
            runtime_config,
            checkpoint,
            pytorch_device=device,
            inference_dtype=str(dtype).removeprefix("torch."),
        )

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        openpi_observation: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Runs one OpenPI policy inference step and saves the action trace to a file.

        This method first ensures the policy is loaded, then constructs the observation,
        performs inference, extracts the predicted actions, and saves them along with
        metadata into a JSON file following a WorldFoundry action trace schema.

        Args:
            instruction: The textual instruction for the policy.
            image: The image input for the policy (e.g., path, PIL Image, NumPy array).
            output_path: The file path where the action trace JSON will be saved.
            openpi_observation: Optional additional OpenPI observation data (e.g., proprioception).
            extra_metadata: Optional dictionary of extra metadata to include in the trace file.

        Returns:
            A dictionary containing metadata about the generated action trace artifact.
        """
        # Load the policy if it hasn't been loaded yet.
        self.load()
        # Assert that the policy is now available for inference.
        assert self.policy is not None

        # Defer import of numpy.
        import numpy as np

        started = time.monotonic()
        # Resolve the full observation dictionary for the policy.
        observation = _resolve_observation(
            instruction,
            image,
            openpi_observation,
            data_family=self.config.data_family,
        )
        # Perform inference using the loaded policy.
        result = self.policy.infer(observation)
        # Extract actions from the inference result.
        actions = np.asarray(result["actions"])

        # Resolve the output path and ensure its parent directory exists.
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Construct the payload for the action trace JSON file.
        payload = {
            "schema_version": "worldfoundry-openpi-action-trace",
            "status": "success",
            "model_id": "openpi",
            "backend": "worldfoundry.openpi.in_tree_runtime.create_trained_policy_infer",
            "backend_quality": "official_demo",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "config_name": self.config.config_name,
            "data_family": self.config.data_family,
            "instruction": instruction,
            "seed": self.config.seed,
            "action_shape": list(actions.shape),
            "action": _jsonable(actions[0]),  # Store the first action as a single item for convenience.
            "actions": _jsonable(actions),  # Store all actions.
            "result_keys": sorted(str(key) for key in result.keys()),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        # Write the JSON payload to the target file.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Calculate SHA256 hash of the created artifact.
        artifact_sha256 = file_sha256(target)
        # Return a summary dictionary of the created artifact.
        return {
            "status": "success",
            "model_id": "openpi",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
