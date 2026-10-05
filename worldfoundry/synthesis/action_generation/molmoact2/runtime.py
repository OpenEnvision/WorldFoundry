"""
This module provides utilities and a runtime for interacting with MolmoAct2 models,
including checkpoint resolution, model loading, and action prediction.

It defines default configurations for various MolmoAct2 embodiments (e.g., DROID, YAM)
and helper functions for processing inputs and outputs for inference.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import (
    project_root,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config


@lru_cache(maxsize=1)
def _embodiment_defaults() -> dict[str, dict[str, Any]]:
    """Load MolmoAct2 variant metadata from the packaged data configuration."""
    config = load_vla_va_wam_runtime_config("molmoact2")
    variants = config.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("MolmoAct2 runtime config has no variants mapping")
    defaults: dict[str, dict[str, Any]] = {}
    for name, value in variants.items():
        if not isinstance(value, Mapping):
            continue
        payload = dict(value)
        canonical = str(payload.get("embodiment") or name).strip().lower().replace("-", "_")
        payload["variant_name"] = str(name)
        payload["camera_keys"] = tuple(payload.get("camera_keys") or ())
        defaults[canonical] = payload
    if not defaults:
        raise ValueError("MolmoAct2 runtime config defines no usable variants")
    return defaults


def _default_embodiment() -> str:
    config = load_vla_va_wam_runtime_config("molmoact2")
    requested = str(config.get("default_variant") or "droid").lower().replace("-", "_")
    defaults = _embodiment_defaults()
    return requested if requested in defaults else next(iter(defaults))


def _jsonable(value: Any) -> Any:
    """
    Recursively converts an object into a JSON-serializable format.

    Handles mappings, sequences, Path objects, numpy arrays (via .tolist()/.item()),
    and basic types (str, int, float, bool, None). Other types are converted to string.

    Args:
        value: The object to convert.

    Returns:
        A JSON-serializable representation of the object.
    """
    if isinstance(value, Mapping):
        # Recursively process items in a dictionary
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively process items in a list or tuple
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to string
        return str(value)
    if hasattr(value, "tolist"):
        # Handle numpy arrays or similar objects with a tolist method
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle numpy scalars or similar objects with an item method
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Basic JSON-serializable types
        return value
    # Default to string conversion for other types
    return str(value)


def _worldfoundry_repository_root() -> Path:
    """
    Returns the root path of the `worldfoundry` repository.

    This is typically used to resolve relative paths within the project.

    Returns:
        The Path object representing the project root.
    """
    return project_root()


def _expand_path_value(value: Any) -> Path | None:
    """
    Expands a given path value to an absolute, resolved Path object.

    Handles `None` or empty strings, relative paths (by resolving against the
    `worldfoundry` repository root), and user-home directory expansion.

    Args:
        value: The path value to expand. Can be a string, Path, or None.

    Returns:
        A resolved Path object, or None if the input value was None or empty.
    """
    if value in (None, ""):
        return None
    project_dir = _worldfoundry_repository_root()
    # Resolve against worldfoundry project paths first
    path = resolve_worldfoundry_path(value)
    # If still not absolute, assume it's relative to the project root
    if not path.is_absolute():
        path = project_dir / path
    # Resolve any symlinks and `.` `..` components to get the canonical path
    return path.resolve()


def _normalize_embodiment(value: Any = None, *, repo_id: str = "", norm_tag: str = "") -> str:
    """
    Normalizes an embodiment name to its canonical form using aliases and defaults.

    Prioritizes explicit matches, then aliases, then hints from `repo_id` or `norm_tag`.
    Defaults to "droid" if no clear match is found.

    Args:
        value: The embodiment name to normalize (e.g., "bimanual-yam", "franka").
        repo_id: Optional string hint from a Hugging Face repository ID.
        norm_tag: Optional string hint from a normalization tag.

    Returns:
        The canonical embodiment name (e.g., "yam", "droid", "so100").
    """
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    defaults = _embodiment_defaults()
    for canonical, payload in defaults.items():
        candidates = {
            canonical.replace("_", "-"),
            str(payload.get("variant_name") or "").lower().replace("_", "-"),
            str(payload.get("embodiment") or "").lower().replace("_", "-"),
            *(str(item).lower().replace("_", "-") for item in payload.get("aliases") or ()),
        }
        if text and text in candidates:
            return canonical
    # Use repository and normalization hints if direct matches fail.
    repo_value = str(repo_id or "").strip().lower()
    norm_value = str(norm_tag or "").strip().lower()
    for canonical, payload in defaults.items():
        repo_hint = str(payload.get("repo_id") or "").lower()
        norm_hint = str(payload.get("norm_tag") or "").lower()
        if (repo_value and repo_hint == repo_value) or (
            not repo_value and norm_value and norm_hint == norm_value
        ):
            return canonical
    return _default_embodiment()


def _role_matches_checkpoint(item: Mapping[str, Any], needles: Sequence[str]) -> bool:
    """
    Checks if a given checkpoint item's metadata matches a sequence of keywords.

    It concatenates various metadata fields (role, variant, repo_id, etc.) into
    a single string and checks if all `needles` (case-insensitive) are present.

    Args:
        item: A dictionary representing a checkpoint's metadata.
        needles: A sequence of strings that must all be found in the checkpoint's metadata.

    Returns:
        True if all needles are found in the checkpoint's metadata, False otherwise.
    """
    # Concatenate relevant fields into a single search string
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("role", "variant", "variant_id", "repo_id", "id", "name", "norm_tag")
    ).lower()
    # Check if all specified needles are present in the haystack
    return all(needle.lower() in haystack for needle in needles)


def _checkpoint_local_dir(item: Mapping[str, Any]) -> Path | None:
    """
    Extracts and expands the local directory path from a checkpoint metadata item.

    Looks for keys like "local_dir", "path", or "checkpoint_dir" in that order.

    Args:
        item: A dictionary representing a checkpoint's metadata.

    Returns:
        A resolved Path object to the local directory, or None if no valid path is found.
    """
    # Prioritize keys for local directory path
    return _expand_path_value(item.get("local_dir") or item.get("path") or item.get("checkpoint_dir"))


def select_molmoact2_checkpoint(
    *,
    repo_id: Any = None,
    checkpoint_dir: Any = None,
    embodiment: Any = None,
    variant_id: Any = None,
    checkpoints: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """
    Resolves a MolmoAct2 checkpoint configuration based on explicit arguments
    or a list of available checkpoint metadata.

    This function attempts to find the best match for a MolmoAct2 model,
    prioritizing explicitly provided paths or repository IDs, then matching
    against known variants or embodiments in a provided list of checkpoints,
    and finally falling back to default embodiment settings.

    Args:
        repo_id: The Hugging Face repository ID for the model (e.g., "allenai/MolmoAct2-DROID").
        checkpoint_dir: A local path to the model checkpoint directory.
        embodiment: The desired embodiment name (e.g., "yam", "droid").
        variant_id: An alias for embodiment or a specific variant string.
        checkpoints: An optional sequence of dictionaries, each representing a
                     pre-configured checkpoint with metadata (e.g., 'role', 'repo_id').

    Returns:
        A dictionary containing the resolved checkpoint configuration, including:
        - "embodiment": The canonical embodiment name.
        - "repo_id": The Hugging Face repository ID.
        - "local_dir": The resolved local path to the checkpoint, or None.
    """
    explicit_dir = _expand_path_value(checkpoint_dir)
    explicit_repo = str(repo_id or "")
    # Normalize the embodiment based on provided hints
    resolved_embodiment = _normalize_embodiment(embodiment or variant_id, repo_id=explicit_repo)
    defaults = _embodiment_defaults()

    # If explicit directory or repository ID is provided, use it directly
    if explicit_dir is not None or explicit_repo:
        return {
            "embodiment": resolved_embodiment,
            "repo_id": explicit_repo or defaults[resolved_embodiment]["repo_id"],
            "local_dir": explicit_dir,
        }

    preferred: Mapping[str, Any] | None = None
    variant = str(variant_id or embodiment or "").lower()
    # Try to find a checkpoint matching the explicit variant_id/embodiment first
    if variant:
        # Split variant into parts to create multiple "needles" for matching
        needles = [part for part in variant.replace("_", "-").split("-") if part and part != "molmoact2"]
        for item in checkpoints:
            if needles and _role_matches_checkpoint(item, needles):
                preferred = item
                break
    # If no variant-specific match, try to find a checkpoint matching the resolved embodiment
    if preferred is None:
        for item in checkpoints:
            if _role_matches_checkpoint(item, (resolved_embodiment,)):
                preferred = item
                break
    # If still no match, and checkpoints are available, pick the first one as a fallback
    if preferred is None and checkpoints:
        preferred = checkpoints[0]

    # If a preferred checkpoint was found, construct the return dict using its metadata
    if preferred is not None:
        repo = str(preferred.get("repo_id") or defaults[resolved_embodiment]["repo_id"])
        # Re-normalize embodiment using hints from the preferred checkpoint's metadata
        resolved_embodiment = _normalize_embodiment(resolved_embodiment, repo_id=repo, norm_tag=str(preferred.get("norm_tag") or ""))
        return {
            "embodiment": resolved_embodiment,
            "repo_id": repo,
            "local_dir": _checkpoint_local_dir(preferred),
        }

    # If no explicit or preferred checkpoint was found, return defaults for the resolved embodiment
    return {
        "embodiment": resolved_embodiment,
        "repo_id": defaults[resolved_embodiment]["repo_id"],
        "local_dir": None,
    }


def _to_pil(arr: Any) -> Any:
    """
    Converts various image representations to a PIL Image in RGB mode.

    Handles existing PIL Image objects, file paths (strings or Path objects),
    and numpy arrays. Performs necessary type and dimension checks.

    Args:
        arr: The image input, can be a PIL Image, string path, Path object, or numpy array.

    Returns:
        A PIL Image object in "RGB" mode.

    Raises:
        FileNotFoundError: If a provided image path does not exist.
        ValueError: If a numpy array is not HxWx3 or if image conversion fails.
    """
    import numpy as np
    from PIL import Image

    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    if isinstance(arr, (str, Path)):
        image_path = Path(arr).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"MolmoAct2 image path does not exist: {image_path}")
        return Image.open(image_path).convert("RGB")
    # Handle numpy arrays or array-like objects
    value = np.asarray(arr)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {value.shape}")
    if value.dtype != np.uint8:
        # Convert to uint8 and clip values if necessary
        value = np.clip(value, 0, 255).astype(np.uint8)
    return Image.fromarray(value, mode="RGB")


def _ordered_images(images: Any, camera_keys: Sequence[str]) -> list[Any]:
    """
    Orders and converts input images to a list of PIL Images, matching `camera_keys`.

    Supports inputs as:
    - A mapping (dict) where keys match `camera_keys`.
    - A sequence (list/tuple) where order must match `camera_keys`.
    - A single image if `camera_keys` has only one element.

    Args:
        images: The input images, can be a dict, list, tuple, or single image object/path.
        camera_keys: A sequence of strings representing the expected camera order.

    Returns:
        A list of PIL Image objects, ordered according to `camera_keys`.

    Raises:
        ValueError: If `images` format is invalid or missing/mismatching camera keys/count.
    """
    if isinstance(images, Mapping):
        # If images are provided as a dictionary, ensure all required camera_keys are present
        missing = [key for key in camera_keys if key not in images]
        if missing:
            raise ValueError(f"MolmoAct2 images mapping is missing camera keys: {missing}")
        return [_to_pil(images[key]) for key in camera_keys]
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        # If images are provided as a sequence, ensure the count matches camera_keys
        values = list(images)
        if len(values) != len(camera_keys):
            raise ValueError(f"MolmoAct2 expected {len(camera_keys)} images, got {len(values)}")
        return [_to_pil(item) for item in values]
    if len(camera_keys) == 1:
        # If only one camera is expected, a single image input is valid
        return [_to_pil(images)]
    raise ValueError(f"MolmoAct2 requires images for camera keys {list(camera_keys)}")


def _load_in_tree_processor(local_dir: Path) -> Any:
    """Build the official processor classes without executing checkpoint Python."""
    from transformers import AutoTokenizer

    from .image_processing_molmoact2 import MolmoAct2ImageProcessor
    from .processing_molmoact2 import MolmoAct2Processor
    from .video_processing_molmoact2 import MolmoAct2VideoProcessor

    config_path = local_dir / "processor_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MolmoAct2 processor config is missing: {config_path}")
    processor_config = json.loads(config_path.read_text(encoding="utf-8"))

    def component_options(key: str, type_key: str) -> dict[str, Any]:
        options = dict(processor_config.get(key) or {})
        options.pop("auto_map", None)
        options.pop(type_key, None)
        options.pop("processor_class", None)
        return options

    image_processor = MolmoAct2ImageProcessor(
        **component_options("image_processor", "image_processor_type")
    )
    video_processor = MolmoAct2VideoProcessor(
        **component_options("video_processor", "video_processor_type")
    )
    tokenizer_options: dict[str, Any] = {}
    tokenizer_config = json.loads((local_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    extra_tokens = tokenizer_config.get("extra_special_tokens")
    if isinstance(extra_tokens, list):
        # Transformers 4 expects named extra tokens; the checkpoint uses the
        # Transformers 5 list form. Preserve token IDs and special-token status.
        tokenizer_options["extra_special_tokens"] = {}
        tokenizer_options["additional_special_tokens"] = list(dict.fromkeys(
            [*tokenizer_config.get("additional_special_tokens", []), *extra_tokens]
        ))
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir),
        trust_remote_code=False,
        local_files_only=True,
        **tokenizer_options,
    )
    chat_template_path = local_dir / "chat_template.jinja"
    chat_template = (
        chat_template_path.read_text(encoding="utf-8")
        if chat_template_path.is_file()
        else processor_config.get("chat_template")
    )
    optional = {
        key: processor_config[key]
        for key in MolmoAct2Processor.optional_attributes
        if key in processor_config and key != "chat_template"
    }
    return MolmoAct2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        **optional,
    )


@dataclass(frozen=True)
class MolmoAct2RuntimeConfig:
    """
    Runtime settings for in-tree MolmoAct2 inference.

    This dataclass encapsulates all configuration parameters required to
    initialize and run a MolmoAct2 model.
    """

    repo_id: str
    local_dir: Path | None
    embodiment: str
    norm_tag: str
    camera_keys: tuple[str, ...]
    state_dim: int
    action_mode_key: str
    device: str = "cuda:0"
    torch_dtype: str = "auto"
    num_steps: int = 10
    enable_cuda_graph: bool = False
    enable_depth_reasoning: bool = False
    enable_adaptive_depth: bool = False
    normalize_language: bool = True


class MolmoAct2Runtime:
    """
    Lazy-loading MolmoAct2 model runtime using Hugging Face's `predict_action`
    and in-tree helper code.

    This class manages the loading of the MolmoAct2 model and processor,
    and provides a method to predict actions based on observations.
    It ensures thread-safe model loading and inference.
    """

    def __init__(self, config: MolmoAct2RuntimeConfig) -> None:
        """
        Initializes the MolmoAct2Runtime with the given configuration.

        The model and processor are not loaded until the first call to `load()`
        or `predict_action()`.

        Args:
            config: An instance of `MolmoAct2RuntimeConfig` specifying
                    model details and runtime parameters.
        """
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None
        self.local_dir: Path | None = None
        self._lock = threading.Lock()  # Ensures thread-safe model loading and inference

    def load(self) -> None:
        """
        Loads the MolmoAct2 model and processor if they haven't been loaded already.

        This method resolves an already staged local checkpoint and initializes
        WorldFoundry's pinned in-tree processor and model classes.
        It also customizes the model's input movement to handle specific dtypes.
        """
        if self.model is not None and self.processor is not None:
            return  # Model already loaded

        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration_molmoact2 import MolmoAct2Config
        from .modeling_molmoact2 import MolmoAct2ForConditionalGeneration

        local_dir = self.config.local_dir
        # Inference is deliberately local-only. Checkpoint acquisition is handled
        # by WorldFoundry's hfd/model preparation workflow, never from model code.
        if local_dir is not None:
            if not local_dir.is_dir():
                raise FileNotFoundError(f"MolmoAct2 checkpoint directory does not exist: {local_dir}")
            resolved = local_dir.resolve()
        else:
            resolved = resolve_local_hf_model_path(
                self.config.repo_id,
                required_files=("config.json", "processor_config.json"),
            )
        if not (
            (resolved / "model.safetensors").is_file()
            or (resolved / "model.safetensors.index.json").is_file()
        ):
            raise FileNotFoundError(
                f"MolmoAct2 model weights are missing from local checkpoint: {resolved}"
            )

        resolved_device = resolve_inference_device(self.config.device, allow_cpu_fallback=True)
        dtype = resolve_inference_dtype(resolved_device, self.config.torch_dtype)

        self.processor = _load_in_tree_processor(resolved)
        model_config = MolmoAct2Config.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = MolmoAct2ForConditionalGeneration.from_pretrained(
            str(resolved),
            config=model_config,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(resolved_device).eval()
        target_dtype = next(model.parameters()).dtype

        # Override the model's internal method for moving inputs to device
        # This ensures that inputs are moved to the correct device and cast to the target dtype (e.g., bf16)
        # for floating point tensors, which is crucial for mixed-precision inference.
        def _move_and_cast(inputs: Mapping[str, Any], dev: Any, _target: Any = target_dtype) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        model._move_inputs_to_device = _move_and_cast
        self.model = model
        self.local_dir = resolved

    def predict_action(
        self,
        *,
        prompt: str,
        images: Any,
        state: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Predicts actions for a given prompt, images, and robot state.

        This method loads the model if not already loaded, processes inputs,
        calls the model's `predict_action` method, and saves the results
        to a JSON file.

        Args:
            prompt: The natural language instruction for the task.
            images: Image observations. Can be a dictionary mapping camera keys
                    to images, a sequence of images in the order of `camera_keys`,
                    or a single image if only one camera is configured. Images can
                    be PIL objects, file paths, or numpy arrays.
            state: The current robot state vector (e.g., joint positions).
                   Must be a 1D array-like object of shape `(self.config.state_dim,)`.
            output_path: The file path where the action trace JSON should be saved.
            extra_metadata: Optional additional metadata to include in the output JSON.

        Returns:
            A dictionary containing metadata about the prediction, including
            status, model ID, artifact path, and SHA256 hash.

        Raises:
            ValueError: If `state` is None or has an incorrect shape, or if
                        `images` are in an invalid format.
            FileNotFoundError: If an image path provided in `images` does not exist.
        """
        import numpy as np
        import torch

        self.load()  # Ensure the model and processor are loaded
        assert self.model is not None
        assert self.processor is not None

        # Validate and format the robot state input
        if state is None:
            raise ValueError("MolmoAct2 requires a robot state vector.")
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (self.config.state_dim,):
            raise ValueError(f"state must be shape ({self.config.state_dim},), got {state_f32.shape}")

        # Process and order images according to camera_keys
        pil_images = _ordered_images(images, self.config.camera_keys)

        action_mode_key = self.config.action_mode_key
        parameters = None
        # Dynamically determine the correct `action_mode` parameter name for `predict_action`
        # as it can vary (e.g., 'inference_action_mode', 'action_mode').
        try:
            import inspect
            parameters = inspect.signature(self.model.predict_action).parameters
            if action_mode_key not in parameters:
                if "inference_action_mode" in parameters:
                    action_mode_key = "inference_action_mode"
                elif "action_mode" in parameters:
                    action_mode_key = "action_mode"
        except (TypeError, ValueError):
            pass

        # Construct keyword arguments for the model's predict_action method
        kwargs = {
            "processor": self.processor,
            "images": pil_images,
            "task": prompt,
            "state": state_f32,
            "norm_tag": self.config.norm_tag,
            "enable_depth_reasoning": self.config.enable_depth_reasoning,
            "num_steps": self.config.num_steps,
            "normalize_language": self.config.normalize_language,
            "enable_cuda_graph": self.config.enable_cuda_graph,
            action_mode_key: "continuous",  # Default to continuous action mode
        }
        # Conditionally add `enable_adaptive_depth` and `depth_cache` if supported by the model's signature
        if self.config.enable_adaptive_depth and (parameters is None or "enable_adaptive_depth" in parameters):
            kwargs["enable_adaptive_depth"] = self.config.enable_adaptive_depth
        elif parameters is not None and "enable_adaptive_depth" in parameters:
            # If the model supports adaptive depth but it's not enabled in config, set to False
            kwargs["enable_adaptive_depth"] = False
        if self.config.enable_adaptive_depth and (parameters is None or "depth_cache" in parameters):
            kwargs["depth_cache"] = None

        started = time.monotonic()
        # Ensure thread-safe inference and disable gradient computation for performance
        with self._lock, torch.inference_mode():
            output = self.model.predict_action(**kwargs)
        
        # Post-process the raw action output
        raw = output.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        # Remove batch dimension if it's a single item (e.g., [1, N, M] -> [N, M])
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        # Prepare and save the action trace to a JSON file
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-molmoact2-action-trace",
            "status": "success",
            "model_id": "molmoact2",
            "backend": "worldfoundry.molmoact2.in_tree_hf_predict_action",
            "backend_quality": "official_hf_in_tree_runtime",
            "artifact_kind": "action_trace",
            "instruction": prompt,
            "embodiment": self.config.embodiment,
            "repo_id": self.config.repo_id,
            "local_dir": "" if self.local_dir is None else str(self.local_dir),
            "norm_tag": self.config.norm_tag,
            "camera_keys": list(self.config.camera_keys),
            "state_shape": list(state_f32.shape),
            "action_shape": list(actions.shape),
            "actions": _jsonable(actions),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        
        # Return a summary dictionary for external consumption
        return {
            "status": "success",
            "model_id": "molmoact2",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
