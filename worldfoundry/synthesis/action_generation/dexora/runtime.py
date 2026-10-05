"""Checkpoint-backed in-tree Dexora inference runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    option_float,
    runtime_options_cache_key,
    to_numpy_image,
)


@dataclass(frozen=True)
class DexoraRuntimeConfig:
    checkpoint_location: str
    config_path: str
    vision_encoder_path: str
    text_encoder_path: str
    statistics_path: str | None
    device: str
    torch_dtype: str
    local_files_only: bool
    control_frequency: float


def _runtime_defaults() -> dict[str, Any]:
    import yaml

    path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", "dexora-1b.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Dexora runtime config must contain a mapping: {path}")
    return payload


def _resolve_local(value: str, *, required_files: tuple[str, ...] = ()) -> str:
    expanded = resolve_worldfoundry_path(value)
    if expanded.exists():
        return str(expanded.resolve())
    return str(resolve_local_hf_model_path(value, required_files=required_files))


class DexoraRuntime:
    """Persistent 36-DoF dual-arm/dual-hand policy runtime."""

    def __init__(self, config: DexoraRuntimeConfig) -> None:
        self.config = config
        self._policy: Any = None
        self._statistics: dict[str, Any] | None = None
        self._resolved_checkpoint = ""

    def _load(self) -> Any:
        if self._policy is not None:
            return self._policy

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .policy import DexoraPolicy, DexoraPolicyConfig

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        checkpoint = _resolve_local(self.config.checkpoint_location)
        config_path = _resolve_local(self.config.config_path)
        vision_encoder = _resolve_local(
            self.config.vision_encoder_path,
            required_files=("config.json", "preprocessor_config.json"),
        )
        text_encoder = _resolve_local(
            self.config.text_encoder_path,
            required_files=("config.json", "tokenizer_config.json"),
        )
        policy_config = DexoraPolicyConfig(
            model_config_path=config_path,
            text_encoder_path=text_encoder,
            vision_encoder_path=vision_encoder,
            dtype=dtype,
            device=device,
            local_files_only=self.config.local_files_only,
        )
        policy = DexoraPolicy(checkpoint, policy_config)
        policy.policy.requires_grad_(False)
        policy.vision_encoder.requires_grad_(False)
        policy.text_encoder.requires_grad_(False)
        self._policy = policy
        self._resolved_checkpoint = checkpoint
        self._statistics = self._load_statistics()
        return policy

    def _load_statistics(self) -> dict[str, Any] | None:
        if not self.config.statistics_path:
            return None
        path = Path(_resolve_local(self.config.statistics_path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "state" not in payload or "action" not in payload:
            raise ValueError("Dexora statistics must define state and action entries")
        return payload

    @staticmethod
    def _normalization_bounds(statistics: Mapping[str, Any]) -> tuple[Any, Any]:
        import numpy as np

        low = first_present(statistics, "percentile_1", "q01", "min")
        high = first_present(statistics, "percentile_99", "q99", "max")
        if low is None or high is None:
            raise KeyError("Dexora statistics require percentile_1/percentile_99, q01/q99, or min/max")
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def _normalize_state(self, state: Any) -> Any:
        import numpy as np

        value = np.asarray(state, dtype=np.float32).reshape(-1)
        if self._statistics is None:
            return value
        low, high = self._normalization_bounds(self._statistics["state"])
        return np.clip(2.0 * (value - low) / np.maximum(high - low, 1e-8) - 1.0, -1.0, 1.0)

    def _unnormalize_action(self, action: Any) -> Any:
        import numpy as np

        value = np.asarray(action, dtype=np.float32)
        if self._statistics is None:
            return value
        low, high = self._normalization_bounds(self._statistics["action"])
        return 0.5 * (value + 1.0) * (high - low) + low

    def predict_action(self, *, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        policy = self._load()
        state = first_present(observation, "state", "proprio", "robot_state")
        if state is None:
            raise ValueError(f"Dexora requires a {policy.cfg.state_dim}D robot state")
        state = self._normalize_state(state)
        if state.shape != (policy.cfg.state_dim,):
            raise ValueError(f"Dexora requires exactly {policy.cfg.state_dim} state values, got {state.shape}")

        camera_names = tuple(policy.cfg.cameras)
        views = collect_images(observation, image, camera_names)
        if len(views) != len(camera_names):
            raise ValueError(
                f"Dexora requires {len(camera_names)} RGB camera observations "
                f"in the configured order {list(camera_names)}, got {len(views)}"
            )
        images = {name: to_numpy_image(view) for name, view in zip(camera_names, views, strict=False)}
        actions = self._unnormalize_action(
            policy.get_action(
                {
                    "state": state,
                    "images": images,
                    "instruction": instruction,
                    "ctrl_freq": self.config.control_frequency,
                }
            )
        )
        if actions.shape != (policy.cfg.chunk_size, policy.cfg.state_dim):
            raise RuntimeError(f"Dexora returned unexpected action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise FloatingPointError("Dexora produced non-finite actions")
        return completed_action_result(
            model_id="dexora-1b",
            instruction=instruction,
            actions=actions.tolist(),
            checkpoint_path=self._resolved_checkpoint,
            device=str(policy.device),
            runtime="worldfoundry.dexora.in_tree_runtime",
            metadata={
                "action_shape": list(actions.shape),
                "camera_views": len(images),
                "control_frequency": self.config.control_frequency,
                "normalized_with_statistics": self._statistics is not None,
                "dtype": str(policy.cfg.dtype),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], DexoraRuntime] = {}


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
    """WorldFoundry callable policy entrypoint."""

    del action_context
    options = {**_runtime_defaults(), **dict(runtime_options or {})}
    checkpoint = checkpoint_path or str(options.get("checkpoint_path") or "")
    if not checkpoint:
        raise ValueError("Dexora requires a local checkpoint_path")
    config = DexoraRuntimeConfig(
        checkpoint_location=checkpoint,
        config_path=str(options["config_path"]),
        vision_encoder_path=str(options["vision_encoder_path"]),
        text_encoder_path=str(options["text_encoder_path"]),
        statistics_path=str(options["statistics_path"]) if options.get("statistics_path") else None,
        device=device,
        torch_dtype=str(options["torch_dtype"]),
        local_files_only=option_bool(options["local_files_only"], True),
        control_frequency=option_float(options["control_frequency"], 0.0),
    )
    if not config.local_files_only:
        raise ValueError("Dexora runtime is local-only")
    key = (checkpoint, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = DexoraRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime.predict_action(instruction=instruction, image=image, observation=observation)


__all__ = ["DexoraRuntime", "DexoraRuntimeConfig", "predict_action"]
