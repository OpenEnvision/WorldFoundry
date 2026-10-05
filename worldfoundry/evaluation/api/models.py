from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Mapping, Protocol, Sequence, runtime_checkable

from worldfoundry.evaluation.api.json_contract import JsonContract, copy_mapping
from worldfoundry.evaluation.api.world_model_manifest import (
    WorldModelManifest,
)

from .generation import GenerationRequest, GenerationResult

WORLD_MODEL_CONFIG_SCHEMA_VERSION = "worldfoundry-world-model-config"


@dataclass(frozen=True)
class WorldModelConfig(JsonContract):
    """Configuration used to construct a world-model runner."""

    model_id: str
    runner: str
    variant: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    manifest: WorldModelManifest | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORLD_MODEL_CONFIG_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __post_init__(self) -> None:
        if self.schema_version != WORLD_MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported WorldModelConfig schema_version: {self.schema_version}")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "runner", str(self.runner))
        object.__setattr__(self, "parameters", copy_mapping(self.parameters))
        object.__setattr__(self, "runtime", copy_mapping(self.runtime))
        if self.manifest is not None and not isinstance(self.manifest, WorldModelManifest):
            to_dict = getattr(self.manifest, "to_dict", None)
            payload = to_dict() if callable(to_dict) else self.manifest
            object.__setattr__(self, "manifest", WorldModelManifest.from_dict(payload))
        object.__setattr__(self, "metadata", copy_mapping(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelConfig":
        manifest = data.get("manifest")
        if manifest is not None and not isinstance(manifest, WorldModelManifest):
            to_dict = getattr(manifest, "to_dict", None)
            manifest = to_dict() if callable(to_dict) else manifest
        return cls(
            model_id=str(data["model_id"]),
            runner=str(data["runner"]),
            variant=data.get("variant", ""),
            parameters=data.get("parameters"),
            runtime=data.get("runtime"),
            seed=data.get("seed"),
            manifest=WorldModelManifest.from_dict(manifest) if isinstance(manifest, Mapping) else manifest,
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", WORLD_MODEL_CONFIG_SCHEMA_VERSION),
        )


@runtime_checkable
class WorldModelRunner(Protocol):
    """Minimum runner surface shared by local and remote world models."""

    model_id: str
    capabilities: Collection[str]

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> "WorldModelRunner":
        ...

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]:
        ...

    def cleanup(self) -> None:
        ...
