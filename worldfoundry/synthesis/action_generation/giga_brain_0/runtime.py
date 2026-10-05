"""Utilities and runtime for GigaBrain-0 inference within WorldFoundry.

This module provides functions for path resolution, data normalization, and a lazy-loading
runtime class to interact with the GigaBrain-0 model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def _jsonable(value: Any) -> Any:
    """Recursively converts a value into a JSON-serializable format.

    Handles common types like mappings, sequences, Path objects, and
    NumPy/PyTorch-specific types by converting them to basic Python types.

    Args:
        value: The input value to convert.

    Returns:
        A JSON-serializable representation of the input value.
    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # Handle NumPy arrays or similar objects that can be converted to a list
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    # Handle PyTorch scalar tensors or similar objects that can be converted to a Python scalar
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Fallback for other types, attempt to convert to string
    return str(value)


def _worldfoundry_repository_root() -> Path:
    """Get the root directory of the WorldFoundry repository.

    Returns:
        The Path object pointing to the WorldFoundry project root.
    """
    return project_root()


def _expand_path_value(value: Any) -> Path | None:
    """Expands a given path value, resolving it relative to the WorldFoundry repository root if not absolute.

    Args:
        value: The path value, which can be a string, Path object, or None/empty string.

    Returns:
        An absolute and resolved Path object, or None if the input value is None or empty.
    """
    if value in (None, ""):
        return None
    repo_root = _worldfoundry_repository_root()
    path = resolve_worldfoundry_path(value)
    # If the path is not absolute, treat it as relative to the repository root
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _first_existing_path(*values: Any, require_existing: bool = True) -> Path:
    """Finds the first existing path among a list of candidates, expanding them if necessary.

    Args:
        *values: Variable arguments representing potential path candidates.
        require_existing: If True, raises FileNotFoundError if no candidate path exists.
                          If False, returns the first expanded path, even if it doesn't exist.

    Returns:
        The first existing (and expanded) Path object from the candidates.

    Raises:
        FileNotFoundError: If `require_existing` is True and no path exists, or if no valid
                           path candidates are provided.
    """
    candidates: list[Path] = []
    for value in values:
        path = _expand_path_value(value)
        if path is None:
            continue
        candidates.append(path)
        # Return the path immediately if it exists or if we don't require it to exist
        if not require_existing or path.exists():
            return path
    # If no existing path was found and candidates were provided, raise a specific error
    if candidates:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No existing GigaBrain-0 path was provided. Checked: {joined}")
    # If no path candidates were provided at all, raise a generic error
    raise FileNotFoundError("No GigaBrain-0 path was provided.")


def _role_path(roles: Mapping[str, str], *needles: str) -> str | None:
    """Finds a path in a dictionary of roles where the role string contains all specified `needles`.

    Args:
        roles: A mapping from role strings to path strings.
        *needles: Variable arguments of substrings to search for in the role names.

    Returns:
        The path string associated with the first matching role, or None if no match is found.
    """
    for role, path in roles.items():
        # Check if all 'needle' substrings are present in the current role
        if all(needle in role for needle in needles):
            return path
    return None


def _existing_role_path(roles: Mapping[str, str], *needles: str) -> str | None:
    """Finds a path in a dictionary of roles where the role string contains all specified `needles`
    and the path actually exists on the filesystem.

    Args:
        roles: A mapping from role strings to path strings.
        *needles: Variable arguments of substrings to search for in the role names.

    Returns:
        The path string associated with the first matching and existing role, or None if no match is found.
    """
    for role, path in roles.items():
        expanded = _expand_path_value(path)
        # Check if all 'needle' substrings are present and the expanded path exists
        if all(needle in role for needle in needles) and expanded is not None and expanded.exists():
            return path
    return None


def _giga_brain_checkpoint_role_path(roles: Mapping[str, str], variant_id: Any = None) -> str | None:
    """Determines the GigaBrain-0 checkpoint path based on roles and an optional variant ID.

    Prioritizes specific variant matches (e.g., "0.1") before falling back to generic "giga" or "model" roles.

    Args:
        roles: A mapping from role strings to path strings.
        variant_id: An optional variant identifier, used to select specific checkpoint roles.

    Returns:
        The path string for the GigaBrain-0 checkpoint, or None if no suitable path is found.
    """
    # Normalize the variant ID for consistent matching
    variant = str(variant_id or "").lower().replace("_", "-")
    # Prioritize specific GigaBrain-0.1 variant roles
    if "0.1" in variant or "0p1" in variant:
        return _role_path(roles, "0p1", "checkpoint") or _role_path(roles, "0.1", "checkpoint")
    # Fallback to general GigaBrain checkpoint or model roles
    return _role_path(roles, "giga", "checkpoint") or _role_path(roles, "model")


def _local_path_or_repo_id(value: Any) -> str:
    """Converts a value to a string, expanding it to an absolute path if it looks like a local file path.
    Otherwise, returns it as-is (e.g., for a Hugging Face repository ID).

    Local paths are identified if they start with '/', '~', '.' or contain '$' (for environment variables).

    Args:
        value: The input value, typically a string representing a path or a repository ID.

    Returns:
        An absolute path string if it's a local path, otherwise the original string value.
    """
    text = "" if value is None else str(value)
    # Check if the text resembles a local file path
    if "$" in text or text.startswith(("/", "~", ".")):
        path = _expand_path_value(text)
        if path is not None:
            return str(path)
    return text


def _as_tensor(value: Any) -> Any:
    """Converts a value to a PyTorch tensor, if it's not already one.

    Args:
        value: The input value to convert. Can be a PyTorch tensor, NumPy array, list, etc.

    Returns:
        A PyTorch tensor.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


GIGA_BRAIN_0_IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _as_chw_float_tensor(value: Any) -> Any:
    """Converts various input types (tensor, path, PIL Image, numpy array) into a
    PyTorch tensor with shape `(C, H, W)`, `float32` dtype, and values normalized to `[0, 1]`.

    Args:
        value: The input image data. Can be a PyTorch tensor, file path (str or Path),
               PIL Image, or NumPy array.

    Returns:
        A PyTorch tensor of shape (C, H, W), dtype float32, with values in [0, 1].

    Raises:
        ValueError: If the input tensor shape or channel count is unsupported.
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(value, torch.Tensor):
        tensor = value
    # Load image from file path
    elif isinstance(value, (str, Path)):
        with Image.open(Path(value).expanduser()) as image:
            # Convert to RGB and then to NumPy array
            tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    # Convert PIL Image to tensor
    elif isinstance(value, Image.Image):
        tensor = torch.from_numpy(np.asarray(value.convert("RGB")).copy())
    # Convert NumPy array or other array-like to tensor
    else:
        tensor = torch.as_tensor(np.asarray(value))

    # Handle batch dimension if present (e.g., (1, H, W, C))
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    # Handle 2D grayscale image (H, W), expand to (3, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    # Handle 3D image (H, W, C) or (C, H, W)
    elif tensor.ndim == 3:
        # If channels are last (H, W, C), permute to (C, H, W)
        if tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"GigaBrain-0 images must be 2D/3D tensors or image paths, got shape {tuple(tensor.shape)}")

    # Ensure 3 channels
    if tensor.shape[0] == 4:  # Remove alpha channel if present (RGBA to RGB)
        tensor = tensor[:3]
    elif tensor.shape[0] == 1:  # Repeat grayscale channel to 3 (L to RGB)
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"GigaBrain-0 images must have 1, 3, or 4 channels, got {tensor.shape[0]}")

    # Convert to float32 and normalize to [0, 1] if needed
    original_dtype = tensor.dtype
    tensor = tensor.contiguous().to(dtype=torch.float32)
    # Scale from [0, 255] to [0, 1] if original dtype was uint8 or max value is > 2.0
    if original_dtype == torch.uint8 or (tensor.numel() and float(tensor.detach().max().cpu()) > 2.0):
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _normalize_image_mapping(images: Any) -> dict[str, Any]:
    """Normalizes a given image input into a dictionary of `(C, H, W)` float tensors,
    keyed by official GigaBrain-0 camera names.

    Args:
        images: Input images, either as a mapping from camera names to image data,
                or as a sequence of image data corresponding to `GIGA_BRAIN_0_IMAGE_KEYS`.

    Returns:
        A dictionary with official GigaBrain-0 image keys mapping to normalized `(C, H, W)`
        float tensors (values in [0, 1]).

    Raises:
        ValueError: If the input format is unsupported or required image keys are missing.
    """
    if isinstance(images, Mapping):
        items = images.items()
    # If a sequence is provided, assume it corresponds to GIGA_BRAIN_0_IMAGE_KEYS
    elif isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray, Path)):
        items = zip(GIGA_BRAIN_0_IMAGE_KEYS, images)
    else:
        raise ValueError(
            "GigaBrain-0 requires a multi-view image mapping keyed by official camera names: "
            + ", ".join(GIGA_BRAIN_0_IMAGE_KEYS)
        )
    # Normalize each image in the mapping/sequence
    normalized = {str(key): _as_chw_float_tensor(value) for key, value in items}
    # Check for any missing required image keys
    missing = [key for key in GIGA_BRAIN_0_IMAGE_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"GigaBrain-0 missing required image keys: {', '.join(missing)}")
    return normalized


def _normalize_state(value: Any) -> Any:
    """Normalizes the robot state vector into a flattened `float32` PyTorch tensor.

    Args:
        value: The robot state input, which can be any type convertible to a PyTorch tensor.

    Returns:
        A 1D PyTorch tensor of type `float32` representing the robot state.

    Raises:
        ValueError: If the input state is None.
    """
    import torch

    if value is None:
        raise ValueError("GigaBrain-0 requires a robot state vector.")
    state = _as_tensor(value)
    # Convert to float32 and flatten the tensor
    return state.to(dtype=torch.float32).reshape(-1)


def _normalize_stats_for_official_pipeline(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Normalizes a statistics dictionary to include 'q01' and 'q99' keys if 'min' and 'max'
    are present, respectively. This is done to match the expected format of the official
    GigaBrain-0 pipeline.

    Args:
        stats: The input statistics dictionary.

    Returns:
        A new dictionary with 'q01' and 'q99' keys added if their 'min'/'max' equivalents exist.
    """
    normalized = dict(stats)
    if "q01" not in normalized and "min" in normalized:
        normalized["q01"] = normalized["min"]
    if "q99" not in normalized and "max" in normalized:
        normalized["q99"] = normalized["max"]
    return normalized


def select_giga_brain_0_paths(
    *,
    model_path: Any = None,
    norm_stats_path: Any = None,
    tokenizer_model_path: Any = None,
    fast_tokenizer_path: Any = None,
    variant_id: Any = None,
    checkpoints: Sequence[Mapping[str, Any]] = (),
    require_existing: bool = True,
) -> dict[str, Path | str]:
    """Resolves GigaBrain-0 checkpoint, normalization statistics, and tokenizer paths.

    Prioritizes explicitly provided paths, then looks for paths within `checkpoints` metadata,
    and finally expands local paths or retains Hugging Face repository IDs for tokenizers.

    Args:
        model_path: Explicitly provided local GigaBrain checkpoint directory path.
        norm_stats_path: Explicitly provided normalization statistics JSON path.
        tokenizer_model_path: Explicitly provided PaliGemma tokenizer path or Hugging Face repo ID.
        fast_tokenizer_path: Explicitly provided FAST tokenizer path or Hugging Face repo ID.
        variant_id: Optional variant selector string (e.g., "giga-brain-0.1-3.5b-base")
                    used to find specific checkpoint roles.
        checkpoints: A sequence of runtime profile checkpoint records, typically from
                     a YAML configuration, containing 'role' and 'local_dir' keys.
        require_existing: If True, `model_path` and `norm_stats_path` must point to existing files/directories.

    Returns:
        A dictionary containing the resolved paths for "model_path" (Path),
        "norm_stats_path" (Path), "tokenizer_model_path" (str), and "fast_tokenizer_path" (str).

    Raises:
        ValueError: If required tokenizer paths cannot be resolved.
        FileNotFoundError: If `require_existing` is True and a required local path does not exist.
    """
    roles: dict[str, str] = {}
    # Build a dictionary of roles from the provided checkpoint metadata
    for item in checkpoints:
        role = str(item.get("role") or item.get("name") or "").lower()
        local_dir = str(item.get("local_dir") or item.get("path") or "")
        if role and local_dir:
            roles[role] = local_dir

    # Resolve tokenizer paths, prioritizing explicit arguments, then roles
    tokenizer = tokenizer_model_path or _role_path(roles, "tokenizer", "model")
    fast_tokenizer = fast_tokenizer_path or _role_path(roles, "fast", "tokenizer")

    if not tokenizer:
        raise ValueError("GigaBrain-0 requires tokenizer_model_path in runtime YAML or checkpoint metadata.")
    if not fast_tokenizer:
        raise ValueError("GigaBrain-0 requires fast_tokenizer_path in runtime YAML or checkpoint metadata.")

    return {
        "model_path": _first_existing_path(
            model_path,
            _giga_brain_checkpoint_role_path(roles, variant_id),
            require_existing=require_existing,
        ),
        "norm_stats_path": _first_existing_path(
            norm_stats_path,
            _role_path(roles, "norm", "stats"),
            _role_path(roles, "stats"),
            require_existing=require_existing,
        ),
        "tokenizer_model_path": _local_path_or_repo_id(tokenizer),
        "fast_tokenizer_path": _local_path_or_repo_id(fast_tokenizer),
    }


@dataclass(frozen=True)
class GigaBrain0RuntimeConfig:
    """Runtime settings for in-tree GigaBrain-0 inference.

    Attributes:
        model_path: The local path to the GigaBrain-0 model checkpoint.
        norm_stats_path: The local path to the JSON file containing normalization statistics.
        tokenizer_model_path: The path or Hugging Face repository ID for the PaliGemma tokenizer model.
        fast_tokenizer_path: The path or Hugging Face repository ID for the FAST tokenizer model.
        embodiment_id: An integer ID for the robot embodiment.
        delta_mask: A sequence of booleans indicating which action dimensions are delta values.
        original_action_dim: The original dimensionality of the action space.
        action_chunk: The number of actions in an action chunk for prediction.
        device: The PyTorch device to run the model on (e.g., "cuda", "cpu").
        compile_policy: If True, compile the policy using `torch.compile` for performance.
        torch_dtype: The desired PyTorch data type for model weights (e.g., "bfloat16", "float32").
        autoregressive_mode_only: If True, run the policy in autoregressive inference mode only.
        enable_2d_traj_output: If True, enable 2D trajectory output from the policy.
        depth_img_prefix_name: Optional prefix name for depth images, if used.
    """
    model_path: Path
    norm_stats_path: Path
    tokenizer_model_path: str
    fast_tokenizer_path: str
    embodiment_id: int
    delta_mask: Sequence[bool]
    original_action_dim: int
    action_chunk: int
    device: str
    compile_policy: bool
    torch_dtype: str | None
    autoregressive_mode_only: bool
    enable_2d_traj_output: bool
    depth_img_prefix_name: str | None


class GigaBrain0Runtime:
    """Lazy in-tree GigaBrain-0 inference runtime.

    This class handles the loading and inference of the GigaBrain-0 model,
    deferring the actual model loading until the first `predict_action` call.
    """

    def __init__(self, config: GigaBrain0RuntimeConfig) -> None:
        """Initialize the GigaBrain-0 runtime with a given configuration.

        The policy model is not loaded until `load()` or `predict_action()` is called.

        Args:
            config: An instance of `GigaBrain0RuntimeConfig` containing all necessary
                    runtime settings.
        """
        self.config = config
        self.policy: Any | None = None

    def load(self) -> None:
        """Load the configured GigaBrain-0 policy from its local checkpoint.

        This method imports necessary GigaBrain-0 pipeline components, loads normalization
        statistics, initializes the GigaBrain0Pipeline, and transfers it to the specified device.
        If `compile_policy` is true, it also compiles the model.
        """
        if self.policy is not None:
            return
        import torch

        from worldfoundry.synthesis.action_generation.giga_brain_0.pipeline import GigaBrain0Pipeline

        # Load and parse normalization statistics from the JSON file
        payload = json.loads(self.config.norm_stats_path.read_text(encoding="utf-8"))
        norm_stats = payload.get("norm_stats", payload) # Handle cases where 'norm_stats' might be top-level

        # Initialize the GigaBrain-0 pipeline with configuration and normalized stats
        pipe = GigaBrain0Pipeline(
            model_path=str(self.config.model_path),
            tokenizer_model_path=self.config.tokenizer_model_path,
            fast_tokenizer_path=self.config.fast_tokenizer_path,
            embodiment_id=self.config.embodiment_id,
            state_norm_stats=_normalize_stats_for_official_pipeline(norm_stats["observation.state"]),
            action_norm_stats=_normalize_stats_for_official_pipeline(norm_stats["action"]),
            delta_mask=list(self.config.delta_mask),
            original_action_dim=self.config.original_action_dim,
            autoregressive_inference_mode=self.config.autoregressive_mode_only,
            depth_img_prefix_name=self.config.depth_img_prefix_name,
            torch_dtype=self.config.torch_dtype,
        )
        pipe.to(self.config.device)
        # Compile the policy if configured, but not in autoregressive mode
        if self.config.compile_policy and not self.config.autoregressive_mode_only:
            pipe.compile()
        self.policy = {"pipe": pipe, "torch": torch}

    def predict_action(
        self,
        *,
        prompt: str,
        images: Any,
        state: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one GigaBrain-0 action prediction and write an action trace to a JSON file.

        Args:
            prompt: The task instruction string.
            images: Multi-view RGB image input, can be a mapping of camera names to image data,
                    or a sequence of image data in `GIGA_BRAIN_0_IMAGE_KEYS` order.
            state: The robot state vector.
            output_path: The file path where the JSON action trace will be written.
            extra_metadata: Optional dictionary of additional WorldFoundry context or metadata
                            to include in the trace.

        Returns:
            A dictionary summarizing the prediction result, including artifact path and SHA256.
        """
        self.load() # Ensure the model is loaded before prediction
        assert self.policy is not None

        started = time.monotonic()
        # Normalize inputs to the expected tensor formats
        image_tensors = _normalize_image_mapping(images)
        state_tensor = _normalize_state(state)

        # Perform the actual prediction using the loaded pipeline
        outputs = self.policy["pipe"](
            image_tensors,
            prompt,
            state_tensor,
            enable_2d_traj_output=self.config.enable_2d_traj_output,
            autoregressive_mode_only=self.config.autoregressive_mode_only,
        )

        # Unpack outputs based on whether 2D trajectory output is enabled
        if self.config.enable_2d_traj_output:
            action, traj = outputs
        else:
            action, traj = outputs, None

        # Prepare the output path and ensure its directory exists
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Construct the payload for the action trace JSON file
        payload = {
            "schema_version": "worldfoundry-giga-brain-0-action-trace",
            "status": "success",
            "model_id": "giga-brain-0",
            "backend": "worldfoundry.giga_brain_0.in_tree_runtime",
            "backend_quality": "official_in_tree",
            "artifact_kind": "action_trace",
            "instruction": prompt,
            "action_shape": list(action.shape),
            "actions": _jsonable(action.detach().cpu()), # Detach from GPU and convert to JSON-serializable format
            "trajectory_2d": _jsonable(traj.detach().cpu()) if traj is not None else None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        # Write the JSON payload to the specified output file
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Calculate SHA256 hash of the generated artifact
        artifact_sha256 = file_sha256(target)

        # Return a summary of the prediction
        return {
            "status": "success",
            "model_id": "giga-brain-0",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
