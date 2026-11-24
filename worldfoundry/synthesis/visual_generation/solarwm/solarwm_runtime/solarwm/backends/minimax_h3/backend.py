"""Lazy MiniMax-H3 backend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import BackendContractError
from .config import validate_h3_config


@dataclass(frozen=True)
class MiniMaxH3Backend:
    """Public plugin; heavy H3 dependencies load only after an action starts."""

    family: str = "minimax_h3"

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_h3_config(config)

    def train(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM training is not part of the WorldFoundry inference runtime")

    def infer(self, config: Mapping[str, Any]) -> int:
        self.validate_config(config)
        from .runtime import run_inference

        return int(run_inference(config))

    def preencode(self, config: Mapping[str, Any]) -> int:
        raise BackendContractError("SolarWM preencoding is not part of the WorldFoundry inference runtime")


def create_backend(*, family: str = "minimax_h3") -> MiniMaxH3Backend:
    """Factory used by the global lazy backend registry."""

    if family != "minimax_h3":
        raise BackendContractError(f"MiniMax-H3 backend cannot serve family {family!r}")
    return MiniMaxH3Backend(family=family)


__all__ = ["MiniMaxH3Backend", "create_backend"]
