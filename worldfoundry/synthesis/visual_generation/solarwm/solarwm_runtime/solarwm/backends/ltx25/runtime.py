"""Heavy-runtime provider protocol and shared-engine adapters for LTX-2.5.

The core package never imports Torch or LTX-Core at module import time.
The runtime supplies one provider implementing this protocol. Receipts
make successful imports insufficient: the provider must prove exact base,
codec, adapter, and shared-checkpoint identities before a step is allowed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, Protocol

from ...checkpoint import CheckpointContract, VerifiedCheckpoint, assert_resume_compatible
from ...errors import BackendContractError
from ...inference import InferenceAdapter, InferenceCase
from .checkpoint import (
    LORA_CHECKPOINT_CONTRACT,
    BaseCheckpointInspection,
    InferenceAdapterCheckpoint,
    StrictCodecLoadReceipt,
    StrictModelLoadReceipt,
    runtime_fingerprint,
)
from .inference import InferencePlan

RUNTIME_PROVIDER_API = "solarwm.ltx25.runtime-provider.v1"
RUNTIME_ENTRYPOINT_GROUP = "solarwm.ltx25_runtime"
DEFAULT_RUNTIME_PROVIDER = f"{__package__}.official:create_provider"
REQUIRED_CHECKPOINT_COMPONENTS = (
    "adapter",
    "ema",
    "optimizer",
    "scheduler",
    "runtime",
)


@dataclass(frozen=True)
class ProviderCheck:
    """One provider-owned readiness fact."""

    name: str
    status: str
    detail: str
    required: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or self.status not in {"pass", "fail", "warning"}:
            raise BackendContractError("provider readiness check is malformed")
        if not self.detail.strip():
            raise BackendContractError("provider readiness check lacks detail")


@dataclass(frozen=True)
class InferenceRuntimeBundle:
    adapter: InferenceAdapter
    cases: Sequence[InferenceCase]
    model_receipt: StrictModelLoadReceipt
    codec_receipt: StrictCodecLoadReceipt
    adapter_checkpoint_manifest_digest: str
    non_writer_run: Callable[[Sequence[InferenceCase], str], None] | None = None
    close: Callable[[], None] | None = None


class LTX25RuntimeProvider(Protocol):
    api_version: str
    identity: str

    def readiness(self, config: Mapping[str, Any], action: str) -> Sequence[ProviderCheck]: ...

    def create_training_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        validation_plan: InferencePlan,
    ) -> NoReturn: ...

    def create_inference_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        plan: InferencePlan,
        adapter_checkpoint: InferenceAdapterCheckpoint,
    ) -> InferenceRuntimeBundle: ...

    def create_preencode_runtime(self, config: Mapping[str, Any]) -> NoReturn: ...


@dataclass(frozen=True)
class ProviderResolution:
    provider: LTX25RuntimeProvider | None
    entrypoint: str
    error: str = ""


def _runtime_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("runtime", {})
    return value if isinstance(value, Mapping) else {}


def _entrypoint_candidates(config: Mapping[str, Any]) -> tuple[str, ...]:
    runtime = _runtime_mapping(config)
    explicit = str(runtime.get("provider_entrypoint") or "").strip()
    if explicit:
        return (explicit,)
    environment = os.environ.get("SOLARWM_LTX25_RUNTIME_PROVIDER", "").strip()
    if environment:
        return (environment,)
    # Installed entry points are not consulted: the vendored provider is the default.
    return (DEFAULT_RUNTIME_PROVIDER,)


def _validate_provider(provider: Any, entrypoint: str) -> LTX25RuntimeProvider:
    if getattr(provider, "api_version", None) != RUNTIME_PROVIDER_API:
        raise BackendContractError(f"LTX provider {entrypoint!r} has unsupported api_version")
    if not str(getattr(provider, "identity", "")).strip():
        raise BackendContractError(f"LTX provider {entrypoint!r} has no identity")
    for name in (
        "readiness",
        "create_training_runtime",
        "create_inference_runtime",
        "create_preencode_runtime",
    ):
        if not callable(getattr(provider, name, None)):
            raise BackendContractError(f"LTX provider {entrypoint!r} lacks callable {name}")
    return provider


def _load_entrypoint(value: str) -> LTX25RuntimeProvider:
    if value.startswith("entrypoint://"):
        requested = value.removeprefix("entrypoint://")
        points = importlib.metadata.entry_points()
        selected = (
            points.select(group=RUNTIME_ENTRYPOINT_GROUP, name=requested)
            if hasattr(points, "select")
            else [point for point in points.get(RUNTIME_ENTRYPOINT_GROUP, ()) if point.name == requested]
        )
        if len(selected) != 1:
            raise BackendContractError(f"LTX runtime entry point {requested!r} is not unique")
        factory = selected[0].load()
    else:
        module_name, separator, attribute = value.partition(":")
        if not separator or not module_name or not attribute:
            raise BackendContractError("runtime.provider_entrypoint must use 'module.path:factory'")
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    if not callable(factory):
        raise BackendContractError(f"LTX runtime provider factory {value!r} is not callable")
    return _validate_provider(factory(), value)


def resolve_runtime_provider(
    config: Mapping[str, Any],
    *,
    injected: LTX25RuntimeProvider | None = None,
) -> ProviderResolution:
    """Discover exactly one provider without hiding import/protocol failures."""

    if injected is not None:
        try:
            provider = _validate_provider(injected, "<injected>")
        except Exception as exc:  # provider boundary normalization
            return ProviderResolution(None, "<injected>", f"{type(exc).__name__}: {exc}")
        return ProviderResolution(provider, "<injected>")
    candidates = _entrypoint_candidates(config)
    if len(candidates) != 1:
        return ProviderResolution(
            None,
            ",".join(candidates),
            "multiple LTX runtime providers were discovered; configure one explicitly",
        )
    entrypoint = candidates[0]
    try:
        provider = _load_entrypoint(entrypoint)
    except Exception as exc:  # provider boundary normalization
        return ProviderResolution(
            None,
            entrypoint,
            f"{type(exc).__name__}: {exc}",
        )
    return ProviderResolution(provider, entrypoint)


def checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    model = config["model"]
    data = config["data"]
    distributed = config["distributed"]
    if not all(isinstance(value, Mapping) for value in (model, data, distributed)):
        raise BackendContractError("LTX checkpoint contract requires model/data/distributed")
    transform = str(model["camera_translation_transform"])
    generation = str(data["generation"])
    fingerprint = runtime_fingerprint(
        camera_translation_transform=transform,
        data_generation=generation,
    )
    extras = {"runtime": fingerprint}
    return CheckpointContract(
        family="ltx25_video",
        stage="stage0p5",
        causal_mode="bidirectional",
        objective="native_rectified_flow",
        objective_variant="shifted_logit_normal",
        camera_translation_transform=transform,
        parameterization="lora-r384-alpha384",
        sp_size=int(distributed["sequence_parallel_size"]),
        data_generation=generation,
        extras=extras,
    )


def inference_checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    """Return the portable, optimizer-free LTX inference-weight contract."""

    model = config["model"]
    distributed = config["distributed"]
    if not all(isinstance(value, Mapping) for value in (model, distributed)):
        raise BackendContractError("LTX inference checkpoint contract requires model/distributed")
    transform = str(model["camera_translation_transform"])
    extras = {
        "lora": {
            "rank": LORA_CHECKPOINT_CONTRACT.rank,
            "alpha": LORA_CHECKPOINT_CONTRACT.alpha,
            "dropout": LORA_CHECKPOINT_CONTRACT.dropout,
            "target_count": LORA_CHECKPOINT_CONTRACT.target_count,
            "trainable_parameters": LORA_CHECKPOINT_CONTRACT.trainable_parameters,
            "base_scale_tables_trainable": (LORA_CHECKPOINT_CONTRACT.base_scale_tables_trainable),
        }
    }
    return CheckpointContract(
        family="ltx25_video",
        stage="stage0p5",
        causal_mode="bidirectional",
        objective="native_rectified_flow",
        objective_variant="shifted_logit_normal",
        camera_translation_transform=transform,
        parameterization="lora-r384-alpha384",
        sp_size=1,
        data_generation="inference-portable:v1",
        extras=extras,
    )


def validate_ltx_inference_checkpoint(
    checkpoint: VerifiedCheckpoint,
    config: Mapping[str, Any],
    *,
    weights: str,
) -> None:
    """Validate one optimizer-free LTX release transaction for inference only."""

    selected = str(weights).strip().lower()
    if selected not in {"live", "ema"}:
        raise BackendContractError("LTX inference checkpoint weights must be live or ema")
    expected = inference_checkpoint_contract(config)
    assert_resume_compatible(expected, checkpoint.contract)
    component = "adapter" if selected == "live" else "ema"
    paths = {item.path for item in checkpoint.files}
    required = {f"{component}/model.safetensors"}
    if paths != required:
        raise BackendContractError(f"LTX inference checkpoint inventory differs: {sorted(paths)}")
    metadata = checkpoint.metadata
    expected_metadata = {
        "schema": "solarwm.ltx25.inference-checkpoint.v1",
        "selected_weights": selected,
        "lora": expected.extras["lora"],
        "optimizer_state_present": False,
        "scheduler_state_present": False,
    }
    drift = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if drift:
        raise BackendContractError(f"LTX inference checkpoint metadata differs: {drift}")


def validate_ltx_checkpoint(
    checkpoint: VerifiedCheckpoint,
    config: Mapping[str, Any],
    *,
    model_load_receipt: Mapping[str, Any] | None = None,
) -> None:
    """Apply the LTX semantic and component inventory on the shared store."""

    expected = checkpoint_contract(config)
    assert_resume_compatible(expected, checkpoint.contract)
    roots = {Path(item.path).parts[0] for item in checkpoint.files}
    missing = sorted(set(REQUIRED_CHECKPOINT_COMPONENTS) - roots)
    if missing:
        raise BackendContractError(f"LTX shared checkpoint lacks required components: {missing}")
    metadata = checkpoint.metadata
    if model_load_receipt is not None and metadata.get("model_load_receipt") != dict(model_load_receipt):
        raise BackendContractError("LTX checkpoint does not bind its strict model load")


__all__ = [
    "DEFAULT_RUNTIME_PROVIDER",
    "REQUIRED_CHECKPOINT_COMPONENTS",
    "RUNTIME_ENTRYPOINT_GROUP",
    "RUNTIME_PROVIDER_API",
    "InferenceRuntimeBundle",
    "LTX25RuntimeProvider",
    "ProviderCheck",
    "ProviderResolution",
    "checkpoint_contract",
    "inference_checkpoint_contract",
    "resolve_runtime_provider",
    "validate_ltx_checkpoint",
    "validate_ltx_inference_checkpoint",
]
