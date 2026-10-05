"""Checkpoint-gated in-tree inference for GigaWorld-Policy-0.5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import (
    resolve_data_path,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    runtime_options_cache_key,
)


@dataclass(frozen=True)
class GigaWorldPolicyRuntimeConfig:
    checkpoint_location: str
    base_model_location: str
    norm_stats_location: str
    fixed_t5_location: str | None
    device: str
    torch_dtype: str
    compile_transformer: bool
    compile_mode: str
    compile_fullgraph: bool
    compile_scope: str
    enable_model_cpu_offload: bool
    seed: int | None
    camera_keys: tuple[str, ...]
    state_dim: int
    action_dim: int


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration must contain a mapping: {path}")
    return dict(payload)


def _runtime_defaults() -> dict[str, Any]:
    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "giga-world-policy-0.5.yaml")
    return _load_yaml(path)


def _local_model_directory(location: str, *, markers: Sequence[str], label: str) -> Path:
    direct = resolve_worldfoundry_path(location)
    root = direct.resolve() if direct.is_dir() else resolve_local_hf_model_path(location)
    missing = [marker for marker in markers if not (root / marker).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete under {root}; missing {missing}")
    return root


def _checkpoint_directory(location: str) -> Path:
    root = _local_model_directory(location, markers=("config.json",), label="GigaWorld-Policy checkpoint")
    if not any(root.glob("*.safetensors")):
        raise FileNotFoundError(
            f"GigaWorld-Policy requires safetensors checkpoint shards under {root}; "
            "pickle-based .bin weights are not accepted"
        )
    return root


def _local_file(location: str | None, label: str, *, required: bool = True) -> Path | None:
    if location in (None, "", "null", "None"):
        if required:
            raise ValueError(f"{label} must be configured")
        return None
    path = resolve_worldfoundry_path(str(location)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _state_vector(value: Any, dimension: int) -> Any:
    import numpy as np

    while isinstance(value, Mapping):
        nested = first_present(value, "vector", "value", "state", "proprio", "joint_positions", "qpos")
        if nested is None:
            break
        value = nested
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != dimension:
        raise ValueError(f"GigaWorld-Policy requires a {dimension}-D robot state, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError("robot state contains non-finite values")
    return array


def _rgb_chw(value: Any) -> Any:
    import numpy as np
    import torch

    if hasattr(value, "convert"):
        array = np.asarray(value.convert("RGB"))
    else:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").numpy()
        array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"camera image must be unbatched, got {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        chw = array.astype(np.float32, copy=False)
    elif array.ndim == 3 and array.shape[-1] == 3:
        chw = np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"camera image must be RGB HWC or CHW, got {array.shape}")
    if not np.isfinite(chw).all():
        raise ValueError("camera image contains non-finite values")
    low = float(chw.min(initial=0.0))
    high = float(chw.max(initial=0.0))
    if low < 0.0:
        if low < -1.0 or high > 1.0:
            raise ValueError(f"unsupported floating camera range [{low}, {high}]")
        chw = (chw + 1.0) * 0.5
    elif high > 1.0:
        chw = chw / 255.0
    return torch.from_numpy(chw.copy()).to(dtype=torch.float32)


class GigaWorldPolicyRuntime:
    """Persistent official MoT action-only diffusion runtime."""

    def __init__(self, config: GigaWorldPolicyRuntimeConfig) -> None:
        self.config = config
        self._policy: Any = None
        self._checkpoint = ""
        self._device = ""
        self._dtype: Any = None

    def _load(self) -> Any:
        if self._policy is not None:
            return self._policy

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .runtime_upstream import get_policy

        checkpoint = _checkpoint_directory(self.config.checkpoint_location)
        base_model = _local_model_directory(
            self.config.base_model_location,
            markers=("model_index.json", "vae/config.json", "tokenizer/tokenizer_config.json"),
            label="local Wan2.2 base model",
        )
        norm_stats = _local_file(self.config.norm_stats_location, "normalization statistics")
        fixed_t5 = _local_file(self.config.fixed_t5_location, "fixed T5 embedding", required=False)
        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        dtype_name = {
            "torch.float16": "fp16",
            "torch.bfloat16": "bf16",
            "torch.float32": "fp32",
        }[str(dtype)]
        policy = get_policy(
            checkpoint=str(checkpoint),
            base_model=str(base_model),
            norm_stats=str(norm_stats),
            data_paths=(),
            device=device,
            fixed_t5_path=None if fixed_t5 is None else str(fixed_t5),
            model_dtype=dtype_name,
            compile_transformer=self.config.compile_transformer,
            compile_mode=self.config.compile_mode,
            compile_fullgraph=self.config.compile_fullgraph,
            compile_scope=self.config.compile_scope,
            seed=self.config.seed,
            enable_model_cpu_offload=self.config.enable_model_cpu_offload,
        )
        self._policy = policy
        self._checkpoint = str(checkpoint)
        self._device = device
        self._dtype = dtype
        return policy

    def predict_action(self, *, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        policy = self._load()
        views = collect_images(observation, image, self.config.camera_keys)
        if len(views) < 3:
            raise ValueError(
                "GigaWorld-Policy requires three RGB views ordered as head, left wrist, and right wrist"
            )
        state_value = first_present(
            observation,
            "state",
            "proprio",
            "robot_state",
            "joint_positions",
            "joint_action",
            "qpos",
        )
        if state_value is None:
            raise ValueError(f"GigaWorld-Policy requires a {self.config.state_dim}-D robot state")
        state = torch.from_numpy(_state_vector(state_value, self.config.state_dim))
        request = {
            "observation.images.cam_high": _rgb_chw(views[0]),
            "observation.images.cam_left_wrist": _rgb_chw(views[1]),
            "observation.images.cam_right_wrist": _rgb_chw(views[2]),
            "observation.state": state,
            "prompt": instruction,
            "_quiet": True,
            "_skip_ref_image_save": True,
        }
        with torch.inference_mode():
            actions = np.asarray(policy.inference(request), dtype=np.float32)
        expected = (48, self.config.action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"GigaWorld-Policy returned action shape {actions.shape}, expected {expected}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("GigaWorld-Policy produced non-finite actions")
        return completed_action_result(
            model_id="giga-world-policy-0.5",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._checkpoint,
            device=self._device,
            runtime="worldfoundry.giga_world_policy.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "dtype": str(self._dtype),
                "architecture": "action-centered-mixture-of-transformers",
                "model_cpu_offload": self.config.enable_model_cpu_offload,
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], GigaWorldPolicyRuntime] = {}


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """WorldFoundry callable entrypoint for local GigaWorld-Policy inference."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    if not checkpoint:
        raise ValueError("GigaWorld-Policy requires a staged local checkpoint directory")
    fixed_t5 = options.get("fixed_t5_embedding_path")
    seed = options.get("seed")
    config = GigaWorldPolicyRuntimeConfig(
        checkpoint_location=checkpoint,
        base_model_location=str(options.get("base_model_path") or ""),
        norm_stats_location=str(options.get("norm_stats_path") or ""),
        fixed_t5_location=None if fixed_t5 in (None, "", "null", "None") else str(fixed_t5),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        compile_transformer=option_bool(options.get("compile_transformer"), True),
        compile_mode=str(options.get("compile_mode") or "reduce-overhead"),
        compile_fullgraph=option_bool(options.get("compile_fullgraph"), False),
        compile_scope=str(options.get("compile_scope") or "action-blocks"),
        enable_model_cpu_offload=option_bool(options.get("enable_model_cpu_offload"), False),
        seed=None if seed in (None, "", "null", "None") else int(seed),
        camera_keys=tuple(str(item) for item in options["camera_keys"]),
        state_dim=int(options.get("state_dim") or 14),
        action_dim=int(options.get("action_dim") or 14),
    )
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = GigaWorldPolicyRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["GigaWorldPolicyRuntime", "GigaWorldPolicyRuntimeConfig", "predict_action"]
