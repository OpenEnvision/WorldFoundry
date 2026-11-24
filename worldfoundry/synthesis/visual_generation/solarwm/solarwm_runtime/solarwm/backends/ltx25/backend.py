"""Executable LTX-2.5 routes backed by a heavy-runtime provider."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...checkpoint import verify_checkpoint
from ...errors import BackendContractError
from ...inference import InferenceEngine
from .checkpoint import InferenceAdapterCheckpoint, inspect_base_checkpoint
from .config import LTX25RunContract, validate_ltx25_config
from .inference import GuidanceSpec, InferenceSpec, build_inference_plan
from .readiness import probe_ltx25_runtime, write_readiness_report
from .runtime import (
    InferenceRuntimeBundle,
    LTX25RuntimeProvider,
    resolve_runtime_provider,
    validate_ltx_checkpoint,
    validate_ltx_inference_checkpoint,
)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise BackendContractError(f"{key} must be a mapping")
    return value


def _rank() -> int:
    try:
        value = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise BackendContractError("RANK must be an integer") from exc
    if value < 0:
        raise BackendContractError("RANK must be nonnegative")
    return value


def _inference_adapter_checkpoint(
    config: Mapping[str, Any],
) -> InferenceAdapterCheckpoint:
    model = _mapping(config, "model")
    path = Path(str(model["adapter_checkpoint_path"])).expanduser()
    weights = str(model["adapter_weights"]).lower()
    checkpoint_format = str(model.get("adapter_checkpoint_format", "transaction_v1")).lower()
    verified = verify_checkpoint(path)
    if checkpoint_format == "inference_transaction_v1":
        validate_ltx_inference_checkpoint(verified, config, weights=weights)
    else:
        validate_ltx_checkpoint(verified, config)
    component = "adapter" if weights == "live" else "ema"
    tensor_path = verified.path / component / "model.safetensors"
    if not tensor_path.is_file() or tensor_path.is_symlink():
        raise BackendContractError(f"LTX checkpoint lacks {component} safetensors")
    return InferenceAdapterCheckpoint(
        path=verified.path,
        tensor_path=tensor_path,
        manifest_digest=verified.manifest_digest,
        weights=weights,
        checkpoint_format=checkpoint_format,
    )


def _inference_plan(config: Mapping[str, Any], *, validation: bool) -> Any:
    section = _mapping(config, "validation" if validation else "inference")
    if validation:
        value = section.get("inference")
        if not isinstance(value, Mapping):
            raise BackendContractError("validation.inference must be a mapping")
        section = value
    guidance = section["guidance"]
    if not isinstance(guidance, Mapping):
        raise BackendContractError("inference.guidance must be a mapping")
    return build_inference_plan(
        InferenceSpec(
            num_inference_steps=int(section["num_inference_steps"]),
            seed=int(section["seed"]),
            fps=int(section["fps"]),
            decoder_mode=str(section["decoder_mode"]),
        ),
        GuidanceSpec(
            cfg_scale=float(guidance["cfg_scale"]),
            stg_scale=float(guidance["stg_scale"]),
            rescale_scale=float(guidance["rescale_scale"]),
            stg_blocks=tuple(int(item) for item in guidance["stg_blocks"]),
        ),
    )


@dataclass(frozen=True)
class LTX25Backend:
    """Dependency-light public boundary for the independent LTX backend."""

    family: str = "ltx25_video"
    provider: LTX25RuntimeProvider | None = None
    enforce_environment: bool = True

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_ltx25_config(config)

    def _prepare(
        self, config: Mapping[str, Any], expected_action: str
    ) -> tuple[LTX25RunContract, LTX25RuntimeProvider, Path]:
        contract = validate_ltx25_config(config)
        if contract.action != expected_action:
            raise BackendContractError(f"backend method {expected_action} cannot execute action {contract.action!r}")
        resolution = resolve_runtime_provider(config, injected=self.provider)
        report = probe_ltx25_runtime(
            config,
            resolution=resolution,
            enforce_environment=self.enforce_environment,
        )
        output_dir = Path(str(_mapping(config, "runtime")["output_dir"])).expanduser()
        report_path = output_dir / f"ltx25-readiness.rank-{_rank():05d}.json"
        write_readiness_report(report_path, report)
        report.require_ready(report_path=report_path)
        if resolution.provider is None:
            raise BackendContractError("LTX readiness passed without a runtime provider")
        return contract, resolution.provider, output_dir

    def train(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM training is not part of the WorldFoundry inference runtime")

    def infer(self, config: Mapping[str, Any]) -> int:
        _contract, provider, output_dir = self._prepare(config, "infer")
        model = _mapping(config, "model")
        inspection = inspect_base_checkpoint(str(model["checkpoint_path"]))
        adapter_checkpoint = _inference_adapter_checkpoint(config)
        plan = _inference_plan(config, validation=False)
        bundle = provider.create_inference_runtime(
            config,
            inspection,
            plan,
            adapter_checkpoint,
        )
        if not isinstance(bundle, InferenceRuntimeBundle):
            raise BackendContractError("provider returned an invalid InferenceRuntimeBundle")
        if (
            bundle.model_receipt.provider_identity != provider.identity
            or bundle.codec_receipt.provider_identity != provider.identity
        ):
            raise BackendContractError("inference receipt provider identity differs")
        if bundle.adapter_checkpoint_manifest_digest != adapter_checkpoint.manifest_digest:
            raise BackendContractError("inference bundle is not bound to the adapter checkpoint")
        bundle.model_receipt.validate(
            inspection,
            adapter_checkpoint_manifest_digest=adapter_checkpoint.manifest_digest,
        )
        bundle.codec_receipt.validate(
            require_gemma=False,
        )
        weights_id = f"{adapter_checkpoint.manifest_digest}:{str(model['adapter_weights']).lower()}"
        cases = tuple(bundle.cases)
        if not cases:
            raise BackendContractError("inference provider returned no cases")
        try:
            if _rank() == 0:
                InferenceEngine(bundle.adapter).run(
                    cases,
                    weights_id=weights_id,
                    output_dir=output_dir / "inference",
                )
            elif bundle.non_writer_run is not None:
                bundle.non_writer_run(cases, weights_id)
            else:
                raise BackendContractError("distributed inference requires a provider non_writer_run hook")
        finally:
            if bundle.close is not None:
                bundle.close()
        return 0

    def preencode(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM preencoding is not part of the WorldFoundry inference runtime")


def create_backend(*, family: str = "ltx25_video") -> LTX25Backend:
    if family != "ltx25_video":
        raise BackendContractError(f"LTX-2.5 backend cannot serve family {family!r}")
    return LTX25Backend(family=family)


__all__ = ["LTX25Backend", "create_backend"]
