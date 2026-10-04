from __future__ import annotations

import importlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def _call_policy_method(method: Any, payload: Mapping[str, Any], *, unpack: bool = False) -> Any:
    """Select the supported payload signature before executing a stateful policy."""
    args, kwargs = ((), dict(payload)) if unpack else ((payload,), {})
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        # Opaque callables use the backend's normal calling convention.
        return method(*args, **kwargs)
    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        args, kwargs = ((payload,), {}) if unpack else ((), dict(payload))
        signature.bind(*args, **kwargs)
    return method(*args, **kwargs)


def _jsonable(value: Any) -> Any:
    """
    Converts various Python types into a JSON-serializable format.

    This function recursively processes dictionaries, lists, and tuples.
    It converts Path objects to strings, handles NumPy array-like objects
    by calling their `tolist()` or `item()` methods, and passes through
    basic JSON types directly. Any other type is converted to its string representation.

    Args:
        value: The Python object to convert.

    Returns:
        A JSON-serializable representation of the input `value`.
    """
    if isinstance(value, Mapping):
        # Recursively convert dictionary keys and values. Keys are ensured to be strings.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively convert items in lists or tuples.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to their string representation.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle NumPy arrays or similar objects by converting them to a list.
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle NumPy scalars or similar objects by extracting their scalar value.
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Basic JSON types and None are returned as-is.
        return value
    # For any other type, return its string representation.
    return str(value)


def _expand_path(value: str | Path | None) -> Path | None:
    """
    Expands a given path string or Path object into an absolute, resolved Path.

    Handles `None` or empty strings by returning `None`. If the path is not
    absolute, it's treated as relative to the project root.

    Args:
        value: The path to expand, which can be a string, Path object, or None.

    Returns:
        An absolute, resolved Path object, or None if the input `value` was None or empty.
    """
    if value in (None, ""):
        return None
    # Resolve the path using worldfoundry' path resolution logic.
    path = resolve_worldfoundry_path(str(value))
    if not path.is_absolute():
        # If the path is not absolute, assume it's relative to the project root.
        path = project_root() / path
    # Return the fully resolved path, normalizing any '..' or symlinks.
    return path.resolve()


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse config booleans without treating the string ``"false"`` as true."""

    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _first_checkpoint(
    *,
    explicit: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
) -> Path | None:
    """
    Determines the primary checkpoint path based on explicit input or a sequence of checkpoint definitions.

    It prioritizes an explicitly provided path. If none is given, it iterates through
    a sequence of checkpoint mappings, searching for common checkpoint keys like
    'local_dir', 'path', or 'checkpoint_path'.

    Args:
        explicit: An explicit path to a checkpoint, or None.
        checkpoints: A sequence of dictionaries, each potentially containing checkpoint path information.

    Returns:
        The resolved Path to the first found checkpoint, or None if no valid checkpoint path is found.
    """
    explicit_path = _expand_path(explicit)
    if explicit_path is not None:
        # If an explicit path is provided and valid, it takes precedence.
        return explicit_path
    for item in checkpoints:
        # Iterate through the provided checkpoint definitions.
        for key in ("local_dir", "path", "checkpoint_path"):
            # Check common keys for checkpoint paths within each definition.
            candidate = _expand_path(item.get(key))
            if candidate is not None:
                # Return the first valid candidate path found.
                return candidate
    return None


def _first_checkpoint_ref(
    *,
    explicit: str | None,
    checkpoints: Sequence[Mapping[str, Any]],
) -> str | None:
    if explicit not in (None, ""):
        return str(explicit)
    for item in checkpoints:
        for key in ("checkpoint_ref", "checkpoint_repo_id", "repo_id", "hf_repo_id"):
            candidate = item.get(key)
            if candidate not in (None, ""):
                return str(candidate)
    return None


def _import_attr(target: str) -> Any:
    """
    Dynamically imports an attribute from a module specified by a string.

    The `target` string must be in the format "module:attribute" or "module:class.attribute".
    For example, "my_module.sub_module:MyClass.method".

    Args:
        target: The string specifying the module and attribute to import.

    Returns:
        The imported attribute (e.g., a function, class, or variable).

    Raises:
        ValueError: If the `target` string format is incorrect.
        ImportError: If the module cannot be found.
        AttributeError: If the attribute cannot be found within the module.
    """
    # Split the target string into module name and attribute path.
    module_name, separator, attr_name = target.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"import target must use module:attribute syntax, got {target!r}")
    # Import the specified module.
    module = importlib.import_module(module_name)
    obj = module
    # Traverse the attribute path to get the final object.
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def _load_image(value: Any) -> Any:
    """
    Loads an image from a path or extracts it from a sequence, converting it to RGB.

    If `value` is a sequence (and not a string/bytes), it attempts to load the first element.
    If `value` is a string or Path, it opens the image file using Pillow, checks its existence,
    and converts it to 'RGB' format. Other types are returned as-is.

    Args:
        value: The image source, which can be a path (str or Path), a sequence containing a path, or None.

    Returns:
        A PIL Image object in 'RGB' format, or None, or the original value if it's not a recognized image source.

    Raises:
        FileNotFoundError: If a specified image path does not exist.
        ImportError: If PIL (Pillow) is not installed.
    """
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # If it's a sequence, try to load the first element.
        return _load_image(value[0]) if value else None
    if isinstance(value, (str, Path)):
        # Defer PIL import to avoid unnecessary dependency unless image loading is needed.
        from PIL import Image

        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image path does not exist: {path}")
        # Open the image and convert to RGB format for consistency.
        return Image.open(path).convert("RGB")
    return value


@dataclass(frozen=True)
class OfficialPolicyRuntimeConfig:
    """
    Configuration for an official policy runtime.

    This dataclass holds all necessary parameters to load and execute
    a pre-trained policy model, including paths, backend type, device,
    data types, and policy/processor targets.

    Attributes:
        model_id: A unique identifier for the model.
        backend: The backend type for loading the policy (e.g., "hf_auto_action_model", "lerobot_policy").
        checkpoint_path: The resolved Path to the model's checkpoint.
        device: The computational device to use (e.g., "cuda", "cpu").
        torch_dtype: The PyTorch data type to use for model weights (e.g., "bfloat16", "float32").
        trust_remote_code: Whether to trust remote code when loading models from Hugging Face.
        require_checkpoint: Whether a checkpoint path is strictly required for this policy.
        policy_target: The import path string for the policy class/function (e.g., "my_module:PolicyClass").
        processor_target: The import path string for the processor class/function, if separate.
        predict_method: The name of the method to call on the policy object for prediction (e.g., "predict_action").
        required_assets: A tuple of strings, paths to any additional assets required by the policy.
        optional_assets: A mapping of optional assets.
    """

    model_id: str
    backend: str
    checkpoint_path: Path | None
    checkpoint_ref: str | None
    device: str
    torch_dtype: str
    trust_remote_code: bool
    require_checkpoint: bool
    policy_target: str | None
    processor_target: str | None
    predict_method: str
    required_assets: tuple[str, ...]
    optional_assets: Mapping[str, Any]
    runtime_options: Mapping[str, Any]

    @property
    def checkpoint_location(self) -> str | None:
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            return str(self.checkpoint_path)
        if self.checkpoint_ref:
            from worldfoundry.core.io.paths import resolve_local_hf_model_path

            try:
                return str(resolve_local_hf_model_path(self.checkpoint_ref))
            except FileNotFoundError:
                return None
        return None


def build_runtime_config(
    *,
    model_id: str,
    profile_checkpoints: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, Any],
    options: Mapping[str, Any],
    device: str,
) -> OfficialPolicyRuntimeConfig:
    """
    Constructs an `OfficialPolicyRuntimeConfig` by merging default settings,
    user options, and available checkpoint information.

    This function prioritizes options over defaults and attempts to find a
    suitable checkpoint path.

    Args:
        model_id: The identifier for the model.
        profile_checkpoints: A sequence of checkpoint profiles to search for a path.
        defaults: Default configuration parameters.
        options: User-provided options that override defaults.
        device: The default device to use if not specified in options/defaults.

    Returns:
        An `OfficialPolicyRuntimeConfig` instance with the resolved configuration.
    """
    # Merge defaults and options, with options taking precedence.
    merged = {**dict(defaults), **dict(options)}
    option_checkpoint_path = (
        options.get("checkpoint_path")
        or options.get("checkpoint_dir")
        or options.get("ckpt_path")
    )
    option_checkpoint_ref = (
        options.get("checkpoint_ref")
        or options.get("checkpoint_repo_id")
        or options.get("repo_id")
        or options.get("hf_repo_id")
    )
    configured_checkpoint_path = (
        merged.get("checkpoint_path")
        or merged.get("checkpoint_dir")
        or merged.get("ckpt_path")
    )
    configured_checkpoint_ref = (
        merged.get("checkpoint_ref")
        or merged.get("checkpoint_repo_id")
        or merged.get("repo_id")
        or merged.get("hf_repo_id")
    )
    if option_checkpoint_ref not in (None, "") and option_checkpoint_path in (None, ""):
        configured_checkpoint_path = None
    if option_checkpoint_path not in (None, "") and option_checkpoint_ref in (None, ""):
        configured_checkpoint_ref = None
    # Determine the primary checkpoint path.
    checkpoint_path = _first_checkpoint(
        explicit=configured_checkpoint_path,
        # An explicit repository reference must not be shadowed by an unrelated
        # profile-local path.  Fall back to the profile only when neither side
        # of the checkpoint location was explicitly selected.
        checkpoints=(
            profile_checkpoints
            if option_checkpoint_ref in (None, "") or option_checkpoint_path not in (None, "")
            else ()
        ),
    )
    checkpoint_ref = _first_checkpoint_ref(
        explicit=configured_checkpoint_ref,
        # Likewise, a custom local checkpoint must not inherit stale remote
        # provenance from the profile's default checkpoint.
        checkpoints=(
            profile_checkpoints
            if option_checkpoint_path in (None, "") or option_checkpoint_ref not in (None, "")
            else ()
        ),
    )
    requested_remote_code = _as_bool(merged.get("trust_remote_code"), False)
    if requested_remote_code:
        raise ValueError(
            f"{model_id} requests remote checkpoint code, which is forbidden for in-tree inference"
        )
    external_source = (
        merged.get("source_repo")
        or merged.get("source_subdir")
        or merged.get("source_workdir")
        or merged.get("pythonpath_dirs")
    )
    if external_source and (
        not isinstance(external_source, str) or external_source.strip()
    ):
        raise ValueError(
            f"{model_id} requests an external source checkout, which is forbidden for in-tree inference"
        )
    return OfficialPolicyRuntimeConfig(
        model_id=model_id,
        # Default backend to "hf_auto_action_model" if not specified.
        backend=str(merged.get("backend") or "hf_auto_action_model"),
        checkpoint_path=checkpoint_path,
        checkpoint_ref=checkpoint_ref,
        # Use provided device if not overridden in config.
        device=str(merged.get("device") or device),
        torch_dtype=str(merged.get("torch_dtype") or "auto"),
        # In-tree integrations do not execute Python shipped by checkpoints.
        trust_remote_code=False,
        # Default require_checkpoint to True.
        require_checkpoint=_as_bool(merged.get("require_checkpoint"), True),
        policy_target=str(merged.get("policy_target") or "") or None,
        processor_target=str(merged.get("processor_target") or "") or None,
        # Default prediction method to "predict_action".
        predict_method=str(merged.get("predict_method") or "predict_action"),
        required_assets=tuple(str(item) for item in merged.get("required_assets") or ()),
        optional_assets=dict(merged.get("optional_assets") or {}),
        runtime_options=dict(merged),
    )


class OfficialPolicyRuntime:
    """
    Manages the lifecycle and execution of an official policy model.

    This class handles loading the model based on the provided configuration,
    checking for missing assets, performing predictions, and logging outputs.
    It supports various backends for loading models (e.g., Hugging Face, custom).
    """

    def __init__(self, config: OfficialPolicyRuntimeConfig) -> None:
        """
        Initializes the OfficialPolicyRuntime with a given configuration.

        Args:
            config: An instance of `OfficialPolicyRuntimeConfig` defining the policy.
        """
        self.config = config
        self._policy: Any = None  # Lazily loaded policy object.

    def missing_assets(self) -> list[dict[str, str]]:
        """
        Checks for any missing required assets or paths defined in the configuration.

        Returns:
            A list of dictionaries, where each dictionary describes a missing asset
            with its kind, path (if applicable), and reason for being missing.
        """
        missing: list[dict[str, str]] = []

        # Check for checkpoint requirements
        if self.config.require_checkpoint and self.config.checkpoint_location is None:
            if self.config.checkpoint_path is not None:
                reason = "configured checkpoint path does not exist"
                if self.config.checkpoint_ref:
                    reason += " and checkpoint repository is not cached locally"
                missing.append(
                    {"kind": "checkpoint", "path": str(self.config.checkpoint_path), "reason": reason}
                )
            elif self.config.checkpoint_ref:
                missing.append(
                    {
                        "kind": "checkpoint",
                        "path": self.config.checkpoint_ref,
                        "reason": "configured checkpoint repository is not cached locally",
                    }
                )
            else:
                missing.append(
                    {"kind": "checkpoint", "path": "", "reason": "no checkpoint_path/checkpoint_dir/checkpoint_ref configured"}
                )

        # Check for other required assets
        for item in self.config.required_assets:
            path = _expand_path(item)
            if path is None or not path.exists():
                missing.append(
                    {
                        "kind": "required_asset",
                        "path": "" if path is None else str(path),
                        "reason": "required runtime asset does not exist",
                    }
                )
        return missing

    def plan_payload(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any] | None,
        action_context: Sequence[Any],
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generates a planning payload dictionary that describes the policy execution plan.

        This payload includes all configuration details, input arguments, and
        information about any missing assets. It does not perform the actual prediction,
        but rather prepares the data structure for it.

        Args:
            instruction: The natural language instruction for the task.
            image: The current visual observation.
            observation: The current observation data.
            action_context: A sequence of previous actions or context.
            output_path: The path where the action trace will be saved.
            extra_metadata: Additional metadata to include in the payload.

        Returns:
            A dictionary representing the planning payload.
        """
        return {
            "schema_version": "worldfoundry-official-policy-runtime-plan-v1",
            "model_id": self.config.model_id,
            "backend": self.config.backend,
            "checkpoint_path": "" if self.config.checkpoint_path is None else str(self.config.checkpoint_path),
            "checkpoint_ref": "" if self.config.checkpoint_ref is None else self.config.checkpoint_ref,
            "checkpoint_location": "" if self.config.checkpoint_location is None else self.config.checkpoint_location,
            "device": self.config.device,
            "torch_dtype": self.config.torch_dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "require_checkpoint": self.config.require_checkpoint,
            "policy_target": self.config.policy_target,
            "processor_target": self.config.processor_target,
            "predict_method": self.config.predict_method,
            "runtime_options": _jsonable(self.config.runtime_options),
            "missing_assets": self.missing_assets(),
            "inputs": {
                "instruction": instruction,
                "image": _jsonable(image),
                "observation": _jsonable(observation or {}),
                "action_context": _jsonable(action_context),
            },
            "output_path": str(output_path),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }

    def _load_policy(self) -> Any:
        """
        Loads the policy model based on the configured backend and parameters.

        This method is idempotent: if the policy is already loaded, it returns the
        existing instance. It handles various backends like 'lerobot_policy',
        'custom_from_pretrained', 'callable_entrypoint', 'hf_auto_action_model',
        and 'hf_image_text_to_text'.

        Returns:
            The loaded policy object (or a tuple of processor and policy for some backends).

        Raises:
            FileNotFoundError: If a required checkpoint path is missing.
            RuntimeError: If a policy target is missing for a backend that requires it,
                          or if an unsupported backend is specified.
            ImportError: If required libraries (e.g., torch, transformers) are not installed.
        """
        if self._policy is not None:
            return self._policy

        if self.config.policy_target:
            target_module = self.config.policy_target.partition(":")[0]
            if not target_module.startswith("worldfoundry."):
                raise RuntimeError(
                    f"{self.config.model_id} policy_target must be in-tree under worldfoundry; "
                    f"got {self.config.policy_target!r}"
                )
        if self.config.processor_target:
            target_module = self.config.processor_target.partition(":")[0]
            if not target_module.startswith("worldfoundry."):
                raise RuntimeError(
                    f"{self.config.model_id} processor_target must be in-tree under worldfoundry; "
                    f"got {self.config.processor_target!r}"
                )

        # Ensure checkpoint exists if required
        checkpoint_location = self.config.checkpoint_location
        if self.config.require_checkpoint and checkpoint_location is None:
            raise FileNotFoundError(f"{self.config.model_id} checkpoint path is not configured.")

        # Backend-specific policy loading logic
        if self.config.backend == "lerobot_policy":
            if not self.config.policy_target:
                raise RuntimeError(f"{self.config.model_id} requires policy_target for lerobot_policy backend.")
            policy_cls = _import_attr(self.config.policy_target)
            self._policy = policy_cls.from_pretrained(str(checkpoint_location)).to(self.config.device).eval()
            return self._policy
        if self.config.backend == "custom_from_pretrained":
            if not self.config.policy_target:
                raise RuntimeError(f"{self.config.model_id} requires policy_target for custom_from_pretrained backend.")
            policy_cls = _import_attr(self.config.policy_target)
            policy = policy_cls.from_pretrained(
                str(checkpoint_location),
                local_files_only=True,
                trust_remote_code=False,
            )
            # Move policy to device and set to eval mode if methods are available
            if hasattr(policy, "to"):
                policy = policy.to(self.config.device)
            if hasattr(policy, "eval"):
                policy = policy.eval()
            processor = None
            if self.config.processor_target:
                processor_cls = _import_attr(self.config.processor_target)
                processor = processor_cls.from_pretrained(
                    str(checkpoint_location),
                    local_files_only=True,
                    trust_remote_code=False,
                )
            self._policy = (processor, policy)
            return self._policy
        if self.config.backend == "callable_entrypoint":
            if not self.config.policy_target:
                raise RuntimeError(f"{self.config.model_id} requires policy_target for callable_entrypoint backend.")
            self._policy = _import_attr(self.config.policy_target)
            return self._policy
        if self.config.backend in {"hf_auto_action_model", "processor_select_action"}:
            import torch
            from transformers import AutoModel, AutoProcessor

            dtype = None
            if self.config.torch_dtype in {"bfloat16", "bf16"}:
                dtype = torch.bfloat16
            elif self.config.torch_dtype in {"float16", "fp16"}:
                dtype = torch.float16
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            # Load processor and model from Hugging Face
            processor = AutoProcessor.from_pretrained(
                str(checkpoint_location),
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(str(checkpoint_location), **model_kwargs)
            model = model.to(self.config.device).eval()
            self._policy = (processor, model)
            return self._policy
        if self.config.backend == "hf_image_text_to_text":
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype = None
            if self.config.torch_dtype in {"bfloat16", "bf16"}:
                dtype = torch.bfloat16
            elif self.config.torch_dtype in {"float16", "fp16"}:
                dtype = torch.float16
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            # Load processor and model from Hugging Face specifically for image-text-to-text
            processor = AutoProcessor.from_pretrained(
                str(checkpoint_location),
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForImageTextToText.from_pretrained(str(checkpoint_location), **model_kwargs)
            model = model.to(self.config.device).eval()
            self._policy = (processor, model)
            return self._policy
        raise RuntimeError(f"{self.config.model_id} has unsupported backend {self.config.backend!r}.")

    def _predict_with_policy(
        self,
        policy: Any,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any] | None,
        action_context: Sequence[Any],
    ) -> Any:
        """
        Executes a prediction using the loaded policy and provided inputs.

        The prediction method varies based on the configured `backend`.

        Args:
            policy: The loaded policy object (or a tuple of processor and policy).
            instruction: The natural language instruction.
            image: The current visual observation.
            observation: The current observation data.
            action_context: A sequence of previous actions or context.

        Returns:
            The raw output from the policy's prediction method.

        Raises:
            RuntimeError: If the policy or processor does not expose an expected prediction method.
        """
        if self.config.backend == "callable_entrypoint":
            # For callable entrypoints, directly call the policy with explicit arguments.
            call_kwargs = {
                "instruction": instruction,
                "image": image,
                "observation": observation or {},
                "action_context": list(action_context),
                "checkpoint_path": str(self.config.checkpoint_location or ""),
                "device": self.config.device,
            }
            signature = inspect.signature(policy)
            if "runtime_options" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
            ):
                call_kwargs["runtime_options"] = dict(self.config.runtime_options)
            return policy(**call_kwargs)
        if self.config.backend in {"custom_from_pretrained", "processor_select_action"}:
            processor, model = policy
            loaded_image = _load_image(image)
            batch = dict(observation or {})
            if loaded_image is not None:
                batch.setdefault("observation.images.image", [loaded_image])
                batch.setdefault("image", loaded_image)
            if instruction:
                batch.setdefault("task", [instruction])
                batch.setdefault("prompt", instruction)
            if processor is not None and hasattr(processor, "select_action"):
                # Prioritize processor.select_action if available.
                output = processor.select_action(model, batch)
                return getattr(output, "action", output)
            method = getattr(model, self.config.predict_method, None)
            if callable(method):
                return _call_policy_method(method, batch)
            raise RuntimeError(
                f"{self.config.model_id} custom policy loaded, but no processor.select_action "
                f"or {self.config.predict_method} method is available."
            )
        if self.config.backend == "hf_auto_action_model":
            processor, model = policy
            loaded_image = _load_image(image)
            processor_kwargs: dict[str, Any] = {"text": instruction, "return_tensors": "pt"}
            if loaded_image is not None:
                processor_kwargs["images"] = loaded_image
            inputs = processor(**processor_kwargs)
            # Move inputs to the model's device.
            inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            # Try configured predict method or 'get_action'.
            method = getattr(model, self.config.predict_method, None) or getattr(model, "get_action", None)
            if callable(method):
                return _call_policy_method(method, inputs, unpack=True)
            raise RuntimeError(
                f"{self.config.model_id} HF model loaded, but exposes neither "
                f"{self.config.predict_method} nor get_action."
            )
        if self.config.backend == "hf_image_text_to_text":
            import torch

            processor, model = policy
            loaded_image = _load_image(image)
            # Construct chat-like messages for image-text-to-text models.
            messages = [
                {
                    "role": "user",
                    "content": [
                        *([{"type": "image", "image": loaded_image}] if loaded_image is not None else []),
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                # Generate output tokens.
                generated = model.generate(**inputs, max_new_tokens=256, use_cache=True)
            output_ids = generated[:, input_length:]
            return processor.batch_decode(output_ids, skip_special_tokens=True)
        
        # Generic fallback for other backends: try configured predict method or 'select_action'.
        method = getattr(policy, self.config.predict_method, None) or getattr(policy, "select_action", None)
        if not callable(method):
            raise RuntimeError(
                f"{self.config.model_id} policy loaded, but exposes neither "
                f"{self.config.predict_method} nor select_action."
            )
        payload = {
            "observation": dict(observation or {}),
            "task": instruction,
            "action_context": list(action_context),
        }
        return _call_policy_method(method, payload)

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any] | None,
        action_context: Sequence[Any],
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Performs an action prediction, saves the result, and returns structured output.

        This is the main public method for running the policy. It first checks for
        missing assets, then loads the policy (if not already loaded), executes
        the prediction, and writes an action trace JSON file to `output_path`.

        Args:
            instruction: The natural language instruction for the task.
            image: The current visual observation.
            observation: The current observation data.
            action_context: A sequence of previous actions or context.
            output_path: The file path where the action trace JSON will be saved.
            extra_metadata: Additional metadata to include in the saved trace.

        Returns:
            A dictionary containing the prediction status, model ID, artifact path,
            backend quality, and the predicted actions.

        Raises:
            FileNotFoundError: If any required checkpoint or assets are missing.
            Exception: Any exception raised during policy loading or prediction.
        """
        missing = self.missing_assets()
        if missing:
            raise FileNotFoundError(
                f"{self.config.model_id} cannot run because required checkpoint/assets are missing: "
                + json.dumps(missing, ensure_ascii=False)
            )

        started = time.time()
        # Load the policy and then use it to predict actions.
        raw_actions = self._predict_with_policy(
            self._load_policy(),
            instruction=instruction,
            image=image,
            observation=observation,
            action_context=action_context,
        )
        action_values = raw_actions.get("actions") if isinstance(raw_actions, Mapping) and "actions" in raw_actions else raw_actions
        
        # Prepare the output path and ensure its parent directory exists.
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        # Construct the payload for the action trace file.
        payload = {
            "schema_version": "worldfoundry-action-trace-v1",
            "model_id": self.config.model_id,
            "runtime": "worldfoundry.official_policy.in_tree_runtime",
            "backend": self.config.backend,
            "checkpoint_path": "" if self.config.checkpoint_path is None else str(self.config.checkpoint_path),
            "checkpoint_ref": "" if self.config.checkpoint_ref is None else self.config.checkpoint_ref,
            "checkpoint_location": "" if self.config.checkpoint_location is None else self.config.checkpoint_location,
            "instruction": instruction,
            "actions": _jsonable(action_values),
            "raw_output": _jsonable(raw_actions),
            "observation": _jsonable(observation or {}),
            "metadata": _jsonable(dict(extra_metadata or {})),
            "elapsed_seconds": time.time() - started,
        }
        
        # Write the action trace payload to the specified output file.
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Return a summary of the prediction outcome.
        return {
            "status": "completed",
            "model_id": self.config.model_id,
            "artifact_kind": "action_trace",
            "artifact_path": str(output),
            "runtime": "worldfoundry.official_policy.in_tree_runtime",
            "backend_quality": "checkpoint_backed" if self.config.require_checkpoint else "algorithm_runtime",
            "actions": payload["actions"],
            "metadata": payload["metadata"],
        }
