"""Policy adapter interfaces for embodied closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from worldfoundry.evals.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    DimSpec,
)

ActionPayload = Mapping[str, Any] | Sequence[Any]


@runtime_checkable
class EmbodiedPolicyAdapter(Protocol):
    """Small policy surface consumed by simulator rollout runners."""

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        """Return an action payload for one simulator observation."""

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        """Declare action components produced by this policy."""

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        """Declare observation components expected by this policy."""

    def cleanup(self) -> None:
        """Release model or connection resources."""


def _flatten_numeric(value: Any) -> list[float]:
    """Coerce a nested numeric action value into a flat Python float list."""
    if value is None:
        return []
    try:
        import numpy as np

        return [float(item) for item in np.asarray(value, dtype=float).reshape(-1).tolist()]
    except Exception:
        pass
    if isinstance(value, (str, bytes)):
        raise TypeError("action values must be numeric, not text")
    if isinstance(value, Sequence):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return [float(value)]


def normalize_action_payload(payload: ActionPayload) -> dict[str, Any]:
    """Normalize common policy action shapes to ``{"actions": [...]}``.

    Supported inputs include:
    - ``{"actions": [...]}`` or ``{"action": [...]}``
    - structured ``{"position": [...], "rotation": [...], "gripper": x}``
    - a raw numeric sequence.
    """
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in ("artifact_path", "artifact_uri", "metadata"):
            if key in payload:
                metadata[key] = payload[key]
        if "token" in payload:
            normalized = {"token": str(payload["token"])}
            if metadata:
                normalized["_metadata"] = metadata
            return normalized
        if "discrete_action" in payload:
            normalized = {"discrete_action": str(payload["discrete_action"])}
            if metadata:
                normalized["_metadata"] = metadata
            return normalized
        if "actions" in payload:
            action = payload["actions"]
        elif "action" in payload:
            action = payload["action"]
        elif {"position", "rotation", "gripper"} & set(payload.keys()):
            action = [
                *_flatten_numeric(payload.get("position", [])),
                *_flatten_numeric(payload.get("rotation", [])),
                *_flatten_numeric(payload.get("gripper", [])),
            ]
        else:
            raise ValueError("policy action payload must contain actions/action or structured components")
    else:
        action = payload

    normalized = {"actions": _flatten_numeric(action)}
    if metadata:
        normalized["_metadata"] = metadata
    return normalized


def default_delta_pose_action_spec() -> dict[str, DimSpec]:
    """Default 7D delta-pose action convention used by LIBERO/OpenVLA."""
    return {
        "position": POSITION_DELTA,
        "rotation": ROTATION_AA,
        "gripper": GRIPPER_CLOSE_POS,
    }


def default_rgb_language_observation_spec() -> dict[str, DimSpec]:
    """Default VLA observation convention: agent-view RGB plus language."""
    return {
        "agentview": IMAGE_RGB,
        "language": LANGUAGE,
    }


class CallablePolicyAdapter:
    """Adapter for lightweight callables used in tests and local experiments."""

    def __init__(
        self,
        fn: Callable[[Mapping[str, Any], str], ActionPayload],
        *,
        action_spec: Mapping[str, DimSpec | Mapping[str, Any]] | None = None,
        observation_spec: Mapping[str, DimSpec | Mapping[str, Any]] | None = None,
    ) -> None:
        self._fn = fn
        self._action_spec = dict(action_spec or default_delta_pose_action_spec())
        self._observation_spec = dict(observation_spec or default_rgb_language_observation_spec())

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        return normalize_action_payload(self._fn(obs, instruction))

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return self._action_spec

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        return self._observation_spec

    def cleanup(self) -> None:
        return None


OFFICIAL_POLICY_BACKENDS = frozenset(
    {
        "lerobot_policy",
        "hf_auto_action_model",
        "custom_from_pretrained",
        "processor_select_action",
        "hf_image_text_to_text",
    }
)


def _normalize_model_id(model_id: str) -> str:
    return str(model_id or "").strip().lower().replace("_", "-")


def _load_runtime_config(model_id: str) -> dict[str, Any]:
    from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

    normalized_id = _normalize_model_id(model_id)
    try:
        return load_vla_va_wam_runtime_config(normalized_id)
    except (FileNotFoundError, TypeError, OSError):
        return {}


def _policy_target_is_predict_action(policy_target: str) -> bool:
    _, _, attr = str(policy_target).partition(":")
    return attr == "predict_action"


def _vla_va_wam_model_ids() -> frozenset[str]:
    from worldfoundry.evals.utils import worldfoundry_data_path

    root = worldfoundry_data_path("models", "catalog", "vla_va_wam")
    if not root.is_dir():
        return frozenset()
    return frozenset(path.stem.replace("_", "-") for path in root.glob("*.yaml"))


def _has_action_pipeline_binding(model_id: str) -> bool:
    from worldfoundry.evals.tasks.embodied.adapters.runtime_bridge import load_pipeline_target

    normalized_id = _normalize_model_id(model_id)
    if normalized_id not in _vla_va_wam_model_ids():
        return False
    return load_pipeline_target(normalized_id) is not None


MODEL_ID_ALIASES: dict[str, tuple[str, dict[str, Any]]] = {
    "openvla-libero": ("openvla", {"unnorm_key": "libero_spatial"}),
    "openvla-libero-spatial": ("openvla", {"unnorm_key": "libero_spatial"}),
}


def _resolve_model_parameters(model_id: str, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_id = _normalize_model_id(model_id)
    alias = MODEL_ID_ALIASES.get(normalized_id)
    if alias is None:
        return normalized_id, dict(params)
    canonical_id, defaults = alias
    return canonical_id, {**defaults, **dict(params)}


def build_policy_adapter(
    model_id: str,
    model_parameters: Mapping[str, Any] | None = None,
    *,
    server_url: str | None = None,
) -> EmbodiedPolicyAdapter:
    """Build a policy adapter from WF model id, parameters, or WebSocket URL."""
    params = dict(model_parameters or {})
    normalized_id, params = _resolve_model_parameters(model_id, params)
    if server_url:
        from worldfoundry.evals.tasks.embodied.adapters.websocket_adapter import WebSocketPolicyAdapter

        return WebSocketPolicyAdapter(
            server_url,
            timeout=float(params.get("server_timeout", params.get("timeout", 30.0))),
            benchmark=str(params.get("benchmark_id") or params.get("benchmark") or ""),
        )

    explicit = params.get("policy_runner")
    if isinstance(explicit, EmbodiedPolicyAdapter):
        return explicit
    if callable(explicit):
        return CallablePolicyAdapter(explicit)
    if callable(params.get("policy")):
        return CallablePolicyAdapter(params["policy"])

    normalized_id = _normalize_model_id(normalized_id)
    runtime_config = _load_runtime_config(normalized_id)
    backend = str(runtime_config.get("backend") or params.get("backend") or "").strip()
    policy_target = str(runtime_config.get("policy_target") or params.get("policy_target") or "").strip()

    from worldfoundry.evals.tasks.embodied.adapters.runtime_policy_adapters import (
        CallableRuntimePolicyAdapter,
        OfficialPolicyPolicyAdapter,
        PipelineSynthesisPolicyAdapter,
    )

    if policy_target and (backend in {"", "callable_entrypoint"} or _policy_target_is_predict_action(policy_target)):
        return CallableRuntimePolicyAdapter(normalized_id, params)

    if backend in OFFICIAL_POLICY_BACKENDS or (policy_target and backend == "lerobot_policy"):
        return OfficialPolicyPolicyAdapter(model_id=normalized_id, parameters=params)

    if _has_action_pipeline_binding(normalized_id):
        return PipelineSynthesisPolicyAdapter(model_id=normalized_id, parameters=params)

    if policy_target:
        return CallableRuntimePolicyAdapter(normalized_id, params)

    raise ValueError(f"unsupported embodied policy adapter for model_id={model_id!r}")


__all__ = [
    "ActionPayload",
    "CallablePolicyAdapter",
    "EmbodiedPolicyAdapter",
    "build_policy_adapter",
    "default_delta_pose_action_spec",
    "default_rgb_language_observation_spec",
    "normalize_action_payload",
]
