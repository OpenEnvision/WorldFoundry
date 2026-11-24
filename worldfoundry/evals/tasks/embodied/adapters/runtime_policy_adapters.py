"""In-tree embodied policy adapters backed by callable, official-policy, or synthesis runtimes."""

from __future__ import annotations

import importlib
import inspect
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.embodied.adapters.runtime_bridge import (
    extract_action_values,
    first_image,
    load_synthesis_class,
    normalize_model_id,
)
from worldfoundry.evals.tasks.embodied.policy_adapter import (
    default_delta_pose_action_spec,
    default_rgb_language_observation_spec,
    normalize_action_payload,
)
from worldfoundry.evals.tasks.embodied.simulators.specs import DimSpec
from worldfoundry.synthesis.action_generation.official_policy.runtime import OfficialPolicyRuntime, build_runtime_config
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile


def _load_target(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"policy_target must be 'module:attribute', got {target!r}")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attr.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"policy_target is not callable: {target}")
    return value


def load_vla_runtime_config(model_id: str) -> dict[str, Any]:
    """Load ``data/models/runtime/configs/vla_va_wam/<model_id>.yaml`` if present."""
    filename = model_id.strip().lower().replace("_", "-")
    try:
        return load_vla_va_wam_runtime_config(filename)
    except (FileNotFoundError, TypeError, OSError):
        return {}


class CallableRuntimePolicyAdapter:
    """Wrap a configured callable runtime behind the embodied policy protocol."""

    def __init__(self, model_id: str, parameters: Mapping[str, Any] | None = None) -> None:
        self.model_id = str(model_id)
        runtime_config = load_vla_runtime_config(self.model_id)
        self.parameters = {**runtime_config, **dict(parameters or {})}
        target = self.parameters.get("policy_target")
        if not target:
            raise ValueError(f"model_id={model_id!r} has no policy_target runtime config")
        self._target = str(target)
        self._fn: Any | None = None

    def _get_fn(self) -> Any:
        if self._fn is None:
            self._fn = _load_target(self._target)
        return self._fn

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        fn = self._get_fn()
        kwargs = {
            **self.parameters,
            "instruction": instruction,
            "image": first_image(obs),
            "observation": obs,
            "action_context": obs.get("action_context", ()),
            "checkpoint_path": self.parameters.get("checkpoint_path", ""),
            "device": self.parameters.get("device", "cuda"),
        }
        signature = inspect.signature(fn)
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        result = fn(**kwargs)
        if isinstance(result, Mapping) and result.get("status") == "blocked":
            raise RuntimeError(str(result.get("blocked_reason") or "policy runtime blocked"))
        return normalize_action_payload(result)

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_delta_pose_action_spec()

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_rgb_language_observation_spec()

    def cleanup(self) -> None:
        return None


class OfficialPolicyPolicyAdapter:
    """Wrap ``OfficialPolicyRuntime`` behind the embodied policy protocol."""

    def __init__(self, model_id: str, parameters: Mapping[str, Any] | None = None) -> None:
        self.model_id = normalize_model_id(model_id)
        self.parameters = dict(parameters or {})
        self._runtime: OfficialPolicyRuntime | None = None
        self._artifact_dir = Path(
            self.parameters.get("artifact_dir")
            or self.parameters.get("output_dir")
            or tempfile.mkdtemp(prefix="wf-official-policy-")
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _get_runtime(self) -> OfficialPolicyRuntime:
        if self._runtime is None:
            profile = load_runtime_profile(
                self.model_id,
                manifest_path=self.parameters.get("manifest_path"),
                profile_path=self.parameters.get("profile_path"),
                acquisition_root=self.parameters.get("acquisition_root"),
                hf_models_root=self.parameters.get("hf_models_root"),
            )
            defaults = load_vla_va_wam_runtime_config(
                self.model_id,
                self.parameters.get("runtime_config_path"),
            )
            config = build_runtime_config(
                model_id=self.model_id,
                profile_checkpoints=profile.checkpoints,
                defaults=defaults,
                options=self.parameters,
                device=str(self.parameters.get("device") or "cuda"),
            )
            self._runtime = OfficialPolicyRuntime(config)
        return self._runtime

    def _next_artifact_path(self) -> Path:
        self._counter += 1
        return self._artifact_dir / f"action_{self._counter:06d}.json"

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        action_context = obs.get("action_context", ())
        if not isinstance(action_context, (list, tuple)):
            action_context = ()
        result = self._get_runtime().predict_action(
            instruction=instruction,
            image=first_image(obs),
            observation=dict(obs),
            action_context=action_context,
            output_path=self._next_artifact_path(),
            extra_metadata={"closed_loop": True},
        )
        normalized = normalize_action_payload(extract_action_values(result))
        normalized["_metadata"] = {**dict(normalized.get("_metadata") or {}), "official_policy_result": dict(result)}
        return normalized

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_delta_pose_action_spec()

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_rgb_language_observation_spec()

    def cleanup(self) -> None:
        if self._runtime is None:
            return
        cleanup = getattr(self._runtime, "cleanup", None)
        if callable(cleanup):
            cleanup()


class PipelineSynthesisPolicyAdapter:
    """Wrap a profile-backed synthesis runtime behind the embodied policy protocol."""

    def __init__(self, model_id: str, parameters: Mapping[str, Any] | None = None) -> None:
        self.model_id = normalize_model_id(model_id)
        self.parameters = dict(parameters or {})
        self._synthesis: Any | None = None

    def _get_synthesis(self) -> Any:
        if self._synthesis is None:
            synthesis_cls = load_synthesis_class(self.model_id)
            init_kwargs = {
                **self.parameters,
                "model_id": self.model_id,
                "device": str(self.parameters.get("device") or "cuda"),
            }
            from_pretrained = getattr(synthesis_cls, "from_pretrained", None)
            if not callable(from_pretrained):
                raise TypeError(f"{synthesis_cls.__name__} does not expose from_pretrained")
            self._synthesis = from_pretrained(init_kwargs, device=init_kwargs["device"])
        return self._synthesis

    def _predict_kwargs(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        action_context = obs.get("action_context", ())
        if not isinstance(action_context, (list, tuple)):
            action_context = ()
        return {
            **self.parameters,
            "instruction": instruction,
            "observation": dict(obs),
            "official_policy_observation": dict(obs),
            "gr00t_observation": dict(obs),
            "openpi_observation": dict(obs),
            "action_context": action_context,
            "task_description": instruction,
        }

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        kwargs = self._predict_kwargs(obs, instruction)
        predict = getattr(self._get_synthesis(), "predict", None)
        if not callable(predict):
            raise TypeError(f"{type(self._synthesis).__name__} does not expose predict")

        signature = inspect.signature(predict)
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

        result = predict(
            prompt=instruction,
            images=first_image(obs),
            interactions=list(obs.get("action_context") or ()),
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{self.model_id} synthesis predict returned non-mapping: {type(result).__name__}")
        status = str(result.get("status") or "")
        if status in {"blocked", "error"}:
            raise RuntimeError(
                str(result.get("blocked_reason") or result.get("error") or f"{self.model_id} synthesis predict blocked")
            )
        normalized = normalize_action_payload(extract_action_values(result))
        normalized["_metadata"] = {
            **dict(normalized.get("_metadata") or {}),
            "pipeline_synthesis_result": dict(result),
        }
        return normalized

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_delta_pose_action_spec()

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return default_rgb_language_observation_spec()

    def cleanup(self) -> None:
        runtime = getattr(self._synthesis, "_runtime", None) if self._synthesis is not None else None
        cleanup = getattr(runtime, "cleanup", None)
        if callable(cleanup):
            cleanup()


__all__ = [
    "CallableRuntimePolicyAdapter",
    "OfficialPolicyPolicyAdapter",
    "PipelineSynthesisPolicyAdapter",
    "load_vla_runtime_config",
]
