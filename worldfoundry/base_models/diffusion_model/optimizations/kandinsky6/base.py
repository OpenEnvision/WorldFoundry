"""Shared config model and the apply/clear protocol for runtime policies."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Runtime(Protocol):
    """A policy the factory can attach to a live pipeline and later remove."""

    def apply(self, pipe: object) -> None: ...

    def clear(self, pipe: object) -> None: ...


# Factory applies these stages in order. Offload strategy is read before
# weights land on a device; the wrapper itself is attached after cache.
ORDER: tuple[str, ...] = (
    "weights",
    "kernels",
    "compile",
    "cache",
    "offload",
    "profile",
)
