"""Provides an in-tree runtime for the Being-H0.5 policy, allowing it to be integrated and run within the WorldFoundry framework."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def _jsonable(value: Any) -> Any:
    """Recursively converts a value to a JSON-serializable type.

    Handles mappings, sequences, Path objects, numpy arrays, and basic types.
    Non-standard types are converted to their string representation.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of the value.
    """
    if isinstance(value, Mapping):
        # Recursively convert all items in a mapping, ensuring keys are strings.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively convert all items in a sequence.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to their string representation.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle numpy arrays or similar objects that have a tolist method.
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle numpy scalars or similar objects that have an item method.
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Base cases: already JSON-serializable.
        return value
    # Default: convert any other type to its string representation.
    return str(value)


def select_being_h05_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> Path:
    """Select the local Being-H0.5 checkpoint directory for a dataset.

    This function prioritizes an explicit `checkpoint_dir` if provided.
    Otherwise, it attempts to find a suitable checkpoint from a list of candidates
    based on the `dataset_name`, preferring existing paths over non-existing ones.

    Args:
        checkpoint_dir: Explicit checkpoint override from the caller. If provided, this path is used directly.
        checkpoints: Profile checkpoint entries, each a mapping possibly containing 'role' and 'local_dir'.
        dataset_name: Dataset name used to prefer LIBERO, RoboCasa, or merged checkpoints.

    Returns:
        The resolved Path to the selected Being-H0.5 checkpoint directory.

    Raises:
        FileNotFoundError: If no suitable checkpoint directory can be found or resolved.
    """

    def expand(value: str | Path) -> Path:
        """Resolves a given path, making it absolute and canonical."""
        path = resolve_worldfoundry_path(value)
        if not path.is_absolute():
            path = project_root() / path
        return path.resolve()

    candidates: list[Mapping[str, Any]] = []
    if checkpoint_dir:
        # If an explicit checkpoint directory is provided, use it directly.
        return expand(checkpoint_dir)
    candidates.extend(dict(item) for item in checkpoints)

    dataset_lower = dataset_name.lower()
    # Determine preferred tokens based on the dataset name to prioritize specific checkpoints.
    # This establishes an order of preference for merged, RoboCasa, or LIBERO specific checkpoints.
    if "robocasa" in dataset_lower and "libero" in dataset_lower:
        preferred_tokens = ("libero_robocasa", "cross")
    elif "robocasa" in dataset_lower:
        preferred_tokens = ("robocasa",)
    elif "libero" in dataset_lower:
        preferred_tokens = ("libero",)
    else:
        preferred_tokens = ()

    # First pass: Look for preferred checkpoints that actually exist on disk.
    for token in preferred_tokens:
        for item in candidates:
            role = str(item.get("role") or "").lower()
            local_dir = str(item.get("local_dir") or "")
            path = expand(local_dir)
            if (token in role or token in local_dir.lower()) and path.exists():
                return path

    # Second pass: If no preferred existing checkpoints, look for preferred checkpoints even if they don't exist yet.
    # This assumes they might be downloaded or created later.
    for token in preferred_tokens:
        for item in candidates:
            role = str(item.get("role") or "").lower()
            local_dir = str(item.get("local_dir") or "")
            if token in role or token in local_dir.lower():
                return expand(local_dir)

    # Third pass: Look for any available checkpoint that exists, regardless of preferred tokens.
    for item in candidates:
        local_dir = str(item.get("local_dir") or "")
        if not local_dir:
            continue
        path = expand(local_dir)
        if path.exists():
            return path
    # Final pass: Look for any available checkpoint, even if it doesn't exist yet.
    for item in candidates:
        local_dir = str(item.get("local_dir") or "")
        if local_dir:
            return expand(local_dir)
    raise FileNotFoundError("No local Being-H0.5 checkpoint directory was found.")


@dataclass(frozen=True)
class BeingH05RuntimeConfig:
    """Runtime settings for the vendored Being-H0.5 policy.

    These settings configure how the Being-H0.5 policy is initialized and run,
    including paths, dataset specifics, hardware device, and various model parameters.

    Attributes:
        checkpoint_dir: Path to the Being-H0.5 model checkpoint directory.
        data_config_name: Name of the data configuration to use (e.g., 'libero_long_horizon').
        dataset_name: Name of the dataset being used (e.g., 'libero_spatial_v1').
        embodiment_tag: Identifier for the robot embodiment (e.g., 'libero_16_dof').
        instruction_template: Template string for formatting instructions.
        device: The compute device to use for inference (e.g., 'cuda:0', 'cpu').
        enable_rtc: Whether to enable real-time compilation (RTC) for performance.
        metadata_variant: Optional variant string for metadata.
        stats_selection_mode: Mode for selecting statistics (e.g., 'dataset_specific').
        attention_mask_kind: Kind of attention mask to use (e.g., 'causal').
    """

    checkpoint_dir: Path
    data_config_name: str
    dataset_name: str
    embodiment_tag: str
    instruction_template: str
    device: str
    torch_dtype: str
    enable_rtc: bool
    metadata_variant: str | None
    stats_selection_mode: str
    attention_mask_kind: str


def _load_rgb_array(image: Any) -> Any:
    """Loads and preprocesses an image into a NumPy array suitable for Being-H0.5 policy input.

    Supports various input types: None, sequences of images (takes the first),
    PIL Image objects, file paths, and existing NumPy arrays.
    Normalizes images to (1, H, W, 3) with dtype uint8.

    Args:
        image: The input image, which can be None, a sequence, PIL Image,
               a string/Path to an image file, or a NumPy array.

    Returns:
        A NumPy array of shape (1, H, W, 3) and dtype `uint8`, or None if the input was None/empty.

    Raises:
        FileNotFoundError: If a provided image path does not exist.
        ValueError: If a NumPy array has an unsupported shape.
    """
    if image is None:
        return None
    if isinstance(image, Sequence) and not isinstance(image, (bytes, bytearray, str)):
        # If a sequence of images is provided, take the first one.
        if not image:
            return None
        return _load_rgb_array(image[0])

    # Defer import of heavy libraries until needed
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB NumPy array.
        return np.asarray(image.convert("RGB"), dtype=np.uint8)[None]
    if isinstance(image, (str, Path)):
        # Load image from file path.
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Being-H0.5 image path does not exist: {image_path}")
        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)[None]

    # Assume input is already an array-like object.
    array = np.asarray(image)
    if array.ndim == 4:
        # If shape is TxHxWxC, assume it's already in the correct format and data type.
        return array.astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Being-H0.5 image array must be HxWxC, TxHxWxC, or CxHxW, got shape {array.shape}.")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        # Transpose from CxHxW to HxWxC if channels are first and not 1 or 3 for the last dimension.
        array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        # Convert float arrays (0.0-1.0) to uint8 (0-255).
        array = np.clip(array, 0.0, 1.0) * 255.0
    # Ensure final array is uint8 and has a batch dimension.
    return array.astype(np.uint8)[None]


def _load_array(value: Any, *, key: str) -> Any:
    """Loads a NumPy array from a file path or converts an arbitrary value to a NumPy array.

    Args:
        value: The input value, which can be None, a string/Path to an array file, or an array-like object.
        key: The key associated with the array, used for error messages.

    Returns:
        A NumPy array, or None if the input was None.

    Raises:
        FileNotFoundError: If a provided path to an array file does not exist.
    """
    if value is None:
        return None

    # Defer import of numpy until needed
    import numpy as np

    if isinstance(value, (str, Path)):
        # Load NumPy array from file.
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Being-H0.5 {key} path does not exist: {path}")
        return np.load(path)
    # Convert any other array-like value to a NumPy array.
    return np.asarray(value)


def _default_state_array(key: str) -> Any:
    """Generates a default zero-filled NumPy array for specific state keys.

    Used when an observation's state modality is missing. The shape and dtype
    are determined by conventions for common state keys like EEF position/rotation.

    Args:
        key: The state key (e.g., 'robot_state.eef_position').

    Returns:
        A NumPy array of zeros with a shape and dtype appropriate for the key.
    """
    # Defer import of numpy until needed
    import numpy as np

    if key.endswith(("eef_position", "eef_rotation")):
        return np.zeros((1, 3), dtype=np.float32)
    if key.endswith("libero_gripper_position"):
        return np.zeros((1, 2), dtype=np.float32)
    return np.zeros((1, 1), dtype=np.float32)


def _policy_modalities(config: BeingH05RuntimeConfig) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Retrieves the video, state, and language keys expected by the Being-H0.5 policy.

    This function relies on the internal data configuration of the Being-H0.5
    library to determine the required input modalities.

    Args:
        config: The runtime configuration for the Being-H0.5 policy.

    Returns:
        A tuple containing:
            - A tuple of strings for video keys.
            - A tuple of strings for state keys.
            - A string for the language key.
    """
    from .preprocessing.data_config import DATA_CONFIG_MAP
    from .preprocessing.schema import EmbodimentTag

    # Load the specific data configuration class based on the config name.
    data_config_cls = DATA_CONFIG_MAP[config.data_config_name]
    # Instantiate the data configuration, setting fixed parameters for inference.
    data_config = data_config_cls(
        embodiment_tag=EmbodimentTag(config.embodiment_tag),
        use_fixed_view=False,
        max_view_num=-1,
        obs_indices=[0],
        action_indices=list(range(16)),
    )
    return (
        tuple(data_config.VIDEO_KEYS),
        tuple(data_config.STATE_KEYS),
        data_config.LANGUAGE_KEYS[0],
    )


def build_being_h05_observation(
    *,
    config: BeingH05RuntimeConfig,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a BeingHPolicy observation from WorldFoundry inputs.

    This function consolidates various inputs (image, instruction, additional observation
    data) into a single dictionary formatted as required by the Being-H0.5 policy.
    It handles loading and preprocessing of different modality types.

    Args:
        config: Resolved runtime configuration.
        instruction: Natural-language task instruction.
        image: Direct RGB image input or path, used as the primary visual observation.
        observation: Operator-provided Being-H0.5 observation mapping, which can
                     supplement or override modalities.

    Returns:
        A dictionary representing the full observation for the Being-H0.5 policy.

    Raises:
        ValueError: If a required image or state modality cannot be loaded.
    """

    source = dict(observation or {})
    # Retrieve the expected input keys for video, state, and language from the policy's data configuration.
    video_keys, state_keys, language_key = _policy_modalities(config)
    result: dict[str, Any] = {}

    # Load the primary image input once, to be used for all video keys by default.
    primary_image = _load_rgb_array(image)
    for key in video_keys:
        value = source.get(key)
        if value is None:
            # If no specific value for this video key is in source, use the primary image.
            value = primary_image
        loaded = _load_rgb_array(value)
        if loaded is None:
            raise ValueError(f"Being-H0.5 observation requires image modality: {key}")
        result[key] = loaded

    for key in state_keys:
        value = source.get(key)
        if value is None:
            # If the full key (e.g., 'robot_state.eef_position') is not found,
            # try looking for a 'short key' (e.g., 'eef_position').
            namespace, short_key = key.split(".", 1)
            value = source.get(short_key)
            if value is None:
                nested_state = source.get(namespace)
                if isinstance(nested_state, Mapping):
                    value = nested_state.get(short_key)
        if value is None:
            # If still not found, provide a default zero-filled array.
            value = _default_state_array(key)
        loaded = _load_array(value, key=key)
        if loaded is None:
            raise ValueError(f"Being-H0.5 observation requires state modality: {key}")
        result[key] = loaded

    result[language_key] = instruction
    # Include optional control parameters if provided in the source observation.
    for key in ("prev_chunk", "inference_delay"):
        if source.get(key) is not None:
            result[key] = source[key]
    return result


class BeingH05Runtime:
    """A lazy-loading runtime for the Being-H0.5 policy.

    This class encapsulates the Being-H0.5 policy and provides an interface
    to run inference, building observations from WorldFoundry inputs and
    saving action traces. The policy is loaded only when `predict_action` is called.
    """

    def __init__(self, config: BeingH05RuntimeConfig) -> None:
        """Create a lazy Being-H0.5 policy runtime.

        Args:
            config: Checkpoint, dataset, embodiment, and device settings.
        """

        self.config = config
        self._policy = None

    def _load_policy(self) -> Any:
        """Loads the BeingHPolicy instance if it hasn't been loaded yet.

        This method ensures the policy is instantiated only once upon its first use.

        Returns:
            The loaded BeingHPolicy instance.
        """
        if self._policy is not None:
            return self._policy

        from .policy import BeingHPolicy
        # Instantiate the policy with parameters from the runtime configuration.
        self._policy = BeingHPolicy(
            model_path=str(self.config.checkpoint_dir),
            data_config_name=self.config.data_config_name,
            dataset_name=self.config.dataset_name,
            embodiment_tag=self.config.embodiment_tag,
            instruction_template=self.config.instruction_template,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
            enable_rtc=self.config.enable_rtc,
            metadata_variant=self.config.metadata_variant,
            stats_selection_mode=self.config.stats_selection_mode,
            attention_mask_kind=self.config.attention_mask_kind,
        )
        return self._policy

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any] | None,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the in-tree BeingHPolicy and write a WorldFoundry action trace.

        This method performs the full inference loop:
        1. Lazily loads the policy.
        2. Builds the policy-specific observation from WorldFoundry inputs.
        3. Calls the policy's `get_action` method.
        4. Constructs an action trace and writes it to a JSON file.
        5. Returns a dictionary summarizing the prediction and artifact details.

        Args:
            instruction: Natural-language task instruction.
            image: Direct RGB image input or path.
            observation: Operator-provided Being-H0.5 observation mapping.
            output_path: JSON artifact path for the action trace.
            extra_metadata: Additional WorldFoundry metadata to include in the trace.

        Returns:
            A dictionary containing prediction status, artifact details, and the predicted action.
        """

        start = time.monotonic()
        policy = self._load_policy()
        # Convert WorldFoundry inputs into the format expected by the Being-H0.5 policy.
        policy_observation = build_being_h05_observation(
            config=self.config,
            instruction=instruction,
            image=image,
            observation=observation,
        )
        action = policy.get_action(policy_observation)
        # Construct the action trace dictionary, including model details and prediction.
        trace = {
            "model_id": "being-h05",
            "artifact_kind": "action_trace",
            "runtime": "worldfoundry.being_h05.in_tree_runtime.BeingHPolicy.get_action",
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "data_config_name": self.config.data_config_name,
            "dataset_name": self.config.dataset_name,
            "embodiment_tag": self.config.embodiment_tag,
            "prediction": _jsonable(action),  # Ensure the action is JSON-serializable.
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write the action trace to the specified JSON file.
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "model_id": "being-h05",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "duration_seconds": round(time.monotonic() - start, 3),
            "runtime": trace["runtime"],
            "prediction": trace["prediction"],
        }
