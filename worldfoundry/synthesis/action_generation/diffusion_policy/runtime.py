"""This module provides an in-tree runtime for executing Diffusion Policy models.

It includes utilities for configuring the runtime, selecting checkpoints,
preprocessing observations, and performing action prediction, with the capability
to output WorldFoundry-compatible action traces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


@dataclass(frozen=True)
class DiffusionPolicyRuntimeConfig:
    """Configuration class for the Diffusion Policy in-tree runtime.

    Attributes:
        checkpoint_path: Path to the Diffusion Policy model checkpoint file.
        device: The PyTorch device (e.g., "cuda" or "cpu") on which to run the policy.
    """
    checkpoint_path: Path
    device: str = "cuda"


_TARGET_REWRITES = {
    "diffusion_policy.policy.diffusion_unet_lowdim_policy": (
        "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.policy"
    ),
    "diffusion_policy.model.diffusion.conditional_unet1d": (
        "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.unet"
    ),
    "diffusion_policy.model.diffusion.conv1d_components": (
        "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.convolution"
    ),
    "diffusion_policy.model.diffusion.positional_embedding": (
        "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.embeddings"
    ),
    "diffusion_policy.model.common.normalizer": (
        "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.normalizer"
    ),
}

_ALLOWED_HYDRA_TARGET_PREFIXES = (
    "worldfoundry.synthesis.action_generation.diffusion_policy.modeling.",
    "diffusers.schedulers.",
)


def _rewrite_checkpoint_targets(value: Any) -> Any:
    """Map serialized upstream Hydra targets to their shallow in-tree modules."""
    if isinstance(value, str):
        for old, new in _TARGET_REWRITES.items():
            if value == old or value.startswith(old + "."):
                return new + value[len(old):]
        return value
    if isinstance(value, Mapping):
        return {key: _rewrite_checkpoint_targets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rewrite_checkpoint_targets(item) for item in value]
    return value


def _validate_safe_config(value: Any, *, path: str = "cfg") -> None:
    """Reject executable or unresolved values in a converted policy config."""

    if isinstance(value, str):
        if "${" in value:
            raise ValueError(
                f"Diffusion Policy safe config must be fully resolved before conversion: {path}"
            )
        return
    if isinstance(value, Mapping):
        target = value.get("_target_")
        if target is not None:
            target_text = str(target)
            if not target_text.startswith(_ALLOWED_HYDRA_TARGET_PREFIXES):
                raise ValueError(
                    f"Diffusion Policy config target is not allowlisted at {path}: {target_text}"
                )
        for key, item in value.items():
            _validate_safe_config(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_config(item, path=f"{path}[{index}]")
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(
            f"Diffusion Policy safe config contains unsupported value at {path}: {type(value).__name__}"
        )


def _tensor_state_dict(value: Any, *, name: str) -> dict[str, Any]:
    import torch

    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"Diffusion Policy {name} must be a non-empty tensor mapping")
    if not all(isinstance(key, str) and torch.is_tensor(item) for key, item in value.items()):
        raise TypeError(f"Diffusion Policy {name} may contain only named tensors")
    return dict(value)


def _expand_runtime_path(value: str | Path) -> Path:
    """Expands a given path, resolving WorldFoundry specific prefixes and ensuring it's an absolute path.

    If the path is not absolute after resolving WorldFoundry prefixes, it's treated as
    relative to the project root directory.

    Args:
        value: The path to expand, which can be a string or a Path object.

    Returns:
        The fully resolved, absolute path.
    """
    # Resolve WorldFoundry-specific path prefixes (e.g., "worldfoundry://").
    path = resolve_worldfoundry_path(value)
    # If the path is still not absolute, assume it's relative to the project's root directory.
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def select_diffusion_policy_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
    require_exists: bool = True,
) -> Path:
    """Select the checkpoint path for the in-tree Diffusion Policy runtime.

    Prioritizes an explicit `checkpoint_path` provided by the caller. If not provided,
    it falls back to the first checkpoint's `local_dir` found in the `checkpoints`
    list (typically from a runtime profile).

    Args:
        checkpoint_path: Explicit checkpoint override supplied by the caller.
        checkpoints: Runtime-profile checkpoint records used when no override exists.

    Raises:
        ValueError: If no checkpoint path can be determined from the inputs.
        FileNotFoundError: If the determined checkpoint path does not exist as a file.

    Returns:
        The resolved and validated path to the Diffusion Policy checkpoint file.
    """
    candidate = checkpoint_path
    # If no explicit checkpoint path is provided, try to use the first checkpoint from the profile.
    if candidate is None and checkpoints:
        candidate = checkpoints[0].get("local_dir")
    if candidate is None:
        raise ValueError("Diffusion Policy requires a checkpoint_path or profile checkpoint.")
    
    # Expand the path to be absolute and resolve any project-relative references.
    path = _expand_runtime_path(candidate)
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"Diffusion Policy checkpoint not found: {path}")
    return path




def _coerce_observation(observation: Any, *, device: str) -> Mapping[str, Any]:
    """Normalize caller observations into the upstream lowdim policy input format.

    This function handles converting various observation formats (e.g., NumPy arrays,
    PyTorch tensors, or dictionaries containing an 'obs' key) into a standardized
    `{"obs": torch.Tensor}` dictionary suitable for the Diffusion Policy.
    It ensures the tensor is on the correct device and has the expected dimensions.

    Args:
        observation: Tensor, ndarray, or mapping containing an ``obs`` value.
                     Expected shape is `[B, T, D]` or `[T, D]`.
        device: Torch device used for inference tensors (e.g., "cuda" or "cpu").

    Raises:
        ValueError: If `observation` does not contain a valid 'obs' input or
                    if the final tensor shape is incorrect.

    Returns:
        A dictionary `{"obs": torch.Tensor}` with the observation coerced to
        the required format, device, and dtype.
    """
    import numpy as np
    import torch

    # Extract the observation value, supporting both direct input and dictionary with "obs" key.
    obs_value = observation.get("obs") if isinstance(observation, Mapping) else observation
    if obs_value is None:
        raise ValueError("Diffusion Policy predict_action requires observation/obs input.")
    
    # Convert NumPy array to PyTorch tensor.
    if isinstance(obs_value, np.ndarray):
        obs_tensor = torch.from_numpy(obs_value)
    else:
        obs_tensor = obs_value
    
    # Ensure the input is a PyTorch tensor, coercing if necessary.
    if not torch.is_tensor(obs_tensor):
        obs_tensor = torch.as_tensor(obs_tensor)
    
    # Move the tensor to the specified device and ensure float32 dtype.
    obs_tensor = obs_tensor.to(device=device, dtype=torch.float32)
    
    # Add a batch dimension if the observation is a single sequence ([T, D] -> [1, T, D]).
    if obs_tensor.ndim == 2:
        obs_tensor = obs_tensor.unsqueeze(0)
    
    # Validate the final shape: expected [B, T, D].
    if obs_tensor.ndim != 3:
        raise ValueError("Diffusion Policy obs must have shape [B, T, D] or [T, D].")
    return {"obs": obs_tensor}


class DiffusionPolicyRuntime:
    """A class to manage and execute an in-tree Diffusion Policy model for action prediction.

    This class provides a lazy loading mechanism for the policy model, ensuring that
    the model is only loaded into memory when inference is first requested. It supports
    configuring the checkpoint, device, and output directory, and generates
    WorldFoundry-compatible action traces.
    """
    def __init__(self, config: DiffusionPolicyRuntimeConfig) -> None:
        """Create a lazy in-tree Diffusion Policy checkpoint runtime.

        Initializes the runtime with the provided configuration but defers
        the actual loading of the policy model until `_load_policy` is called.

        Args:
            config: Checkpoint, device, and output-directory settings for inference.
        """
        self.config = config
        self._policy = None

    def _load_policy(self) -> Any:
        """Load a Diffusion Policy checkpoint and prepare the policy for inference.

        This method is called lazily and ensures the policy model is loaded and ready
        for prediction. It handles dynamic import of the Diffusion Policy package,
        deserialization, Hydra policy construction, and model-state loading.

        Returns:
            The loaded and configured Diffusion Policy model instance.

        Raises:
            RuntimeError: If the in-tree runtime cannot be installed or if the
                          model loading process encounters issues.
        """
        # If the policy is already loaded, return the cached instance.
        if self._policy is not None:
            return self._policy

        import copy
        import hydra
        import torch
        from omegaconf import OmegaConf
        from worldfoundry.core.model_loading.checkpoints.file import load_torch_checkpoint

        # Official workspace checkpoints use dill and can execute arbitrary code while
        # deserializing.  Runtime inference accepts only an offline-converted payload:
        # a fully resolved primitive cfg plus tensor-only model/EMA state dictionaries.
        try:
            payload = load_torch_checkpoint(
                self.config.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Diffusion Policy rejected the executable official dill checkpoint. "
                "Convert it once in a trusted offline environment to a weights-only payload "
                "containing a fully resolved primitive 'cfg' mapping and tensor-only "
                "'state_dicts' before WorldFoundry inference."
            ) from error
        if not isinstance(payload, Mapping):
            raise TypeError("Diffusion Policy converted checkpoint root must be a mapping")
        cfg_container = payload.get("cfg")
        if not isinstance(cfg_container, Mapping):
            raise TypeError(
                "Diffusion Policy converted checkpoint requires a primitive 'cfg' mapping"
            )
        rewritten_cfg = _rewrite_checkpoint_targets(dict(cfg_container))
        _validate_safe_config(rewritten_cfg)
        cfg = OmegaConf.create(rewritten_cfg)
        state_dicts = payload.get("state_dicts")
        if not isinstance(state_dicts, Mapping):
            raise TypeError("Diffusion Policy converted checkpoint requires 'state_dicts'")
        policy = hydra.utils.instantiate(cfg.policy)
        model_state = state_dicts.get("model")
        if model_state is None:
            model_state = state_dicts.get("ema_model")
        if model_state is None:
            raise KeyError("Diffusion Policy checkpoint has neither model nor ema_model state")
        policy.load_state_dict(_tensor_state_dict(model_state, name="model state"), strict=True)

        training_cfg = getattr(cfg, "training", None)
        use_ema = bool(getattr(training_cfg, "use_ema", False))
        if use_ema and "ema_model" in state_dicts:
            policy = copy.deepcopy(policy)
            policy.load_state_dict(
                _tensor_state_dict(state_dicts["ema_model"], name="EMA state"),
                strict=True,
            )
        
        # Move the selected policy model to the configured device and set it to evaluation mode.
        policy.to(torch.device(self.config.device))
        policy.eval()
        self._policy = policy
        return policy

    def predict_action(
        self,
        *,
        observation: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run upstream ``policy.predict_action`` and write a WorldFoundry trace.

        This method performs inference using the loaded Diffusion Policy model and
        serializes the prediction results into a JSON-formatted WorldFoundry action trace file.

        Args:
            observation: Lowdim observation with shape ``[B, T, D]`` or ``[T, D]``.
                         Can be a NumPy array, PyTorch tensor, or a dictionary
                         containing an "obs" key.
            output_path: JSON artifact path for the action trace file.
            extra_metadata: Optional additional WorldFoundry metadata to include in the trace.

        Returns:
            A dictionary containing the prediction status, model ID, artifact details,
            runtime path, and the generated prediction.
        """
        import torch

        # Lazily load the policy model if it hasn't been loaded yet.
        policy = self._load_policy()
        
        # Coerce the input observation into the expected PyTorch tensor format for the policy.
        obs_dict = _coerce_observation(observation, device=self.config.device)
        normalizer = policy.normalizer["obs"]
        expected_features = int(normalizer.params_dict["scale"].shape[0])
        actual_features = int(obs_dict["obs"].shape[-1])
        if actual_features != expected_features:
            raise ValueError(
                "Diffusion Policy observation feature dimension does not match the "
                f"checkpoint normalizer: expected {expected_features}, got {actual_features}"
            )
        
        # Perform inference without gradient computation.
        with torch.no_grad():
            raw_result = policy.predict_action(obs_dict)
        
        # Construct the WorldFoundry action trace dictionary.
        trace = {
            "model_id": "diffusion-policy",
            "artifact_kind": "action_trace",
            "runtime": "worldfoundry.diffusion_policy.in_tree_runtime.predict_action",
            "checkpoint_path": str(self.config.checkpoint_path),
            "device": self.config.device,
            # Convert prediction tensors to CPU NumPy arrays/lists and make them JSON-safe.
            "prediction": jsonable({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_result.items()}),
            "metadata": jsonable(dict(extra_metadata or {})),
        }
        
        # Resolve the output path, create parent directories, and write the trace to a JSON file.
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        
        # Return a summary of the prediction, including the path to the saved artifact.
        return {
            "status": "success",
            "model_id": "diffusion-policy",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "runtime": trace["runtime"],
            "prediction": trace["prediction"],
        }
