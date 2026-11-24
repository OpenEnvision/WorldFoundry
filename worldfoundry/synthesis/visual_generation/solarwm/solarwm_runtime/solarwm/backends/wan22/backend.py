"""Public Wan2.2 backend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import BackendContractError
from .codec import validate_preencode_config
from .contracts import WAN_FAMILIES, validate_wan_config
from .generation import resolve_generation_plan


@dataclass(frozen=True)
class Wan22Backend:
    """Validated Wan2.2 family plugin."""

    family: str

    def validate_config(self, config: Mapping[str, Any]) -> None:
        action = str(config.get("action", "")).strip().lower()
        if action == "preencode":
            validate_preencode_config(config, expected_family=self.family)
            return
        validate_wan_config(config, expected_family=self.family)
        if action == "infer":
            # Inference and inline validation resolve the same immutable plan.
            resolve_generation_plan(config)

    def train(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM training is not part of the WorldFoundry inference runtime")

    def readiness(self, config: Mapping[str, Any], *, require_cuda: bool = False) -> Any:
        """Inspect dependencies, assets, indexes, and optionally CUDA."""

        self.validate_config(config)
        from .runtime.readiness import probe_runtime

        train = config.get("train", {})
        stage2 = isinstance(train, Mapping) and str(train.get("stage")) == "stage2"
        return probe_runtime(
            config,
            family=self.family,
            require_cuda=require_cuda,
            require_transformer_weights=not stage2,
        )

    def infer(self, config: Mapping[str, Any]) -> int:
        self.validate_config(config)
        from .runtime.readiness import probe_runtime

        train = config.get("train", {})
        stage2 = isinstance(train, Mapping) and str(train.get("stage")) == "stage2"
        probe_runtime(
            config,
            family=self.family,
            require_cuda=True,
            require_transformer_weights=not stage2,
        ).require_ready()
        if self.family == "wan22_ti2v_5b" and stage2:
            from .runtime.stage2 import run_stage2_inference

            run_stage2_inference(config)
            return 0
        from .runtime.inference import run_wan_inference

        run_wan_inference(config)
        return 0

    def preencode(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM preencoding is not part of the WorldFoundry inference runtime")


def create_backend(*, family: str) -> Wan22Backend:
    """Create one strictly named Wan backend without importing torch."""

    normalized = str(family).strip().lower()
    if normalized not in WAN_FAMILIES:
        raise BackendContractError(
            f"solarwm.backends.wan22 does not implement family {family!r}; expected one of {sorted(WAN_FAMILIES)}"
        )
    return Wan22Backend(family=normalized)
