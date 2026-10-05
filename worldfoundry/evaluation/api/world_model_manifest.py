"""Public world-model manifest DTO (model-zoo and evaluation share this contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .json_contract import JsonContract, copy_mapping, tuple_of_str

WORLD_MODEL_MANIFEST_SCHEMA_VERSION = "worldfoundry-world-model-manifest"


@dataclass(frozen=True)
class WorldModelManifest(JsonContract):
    """Public metadata describing a world-model runner."""

    model_id: str
    name: str = ""
    aliases: tuple[str, ...] = ()
    version: str = ""
    provider: str = ""
    capabilities: tuple[str, ...] = ()
    supported_tasks: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    output_artifacts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORLD_MODEL_MANIFEST_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __post_init__(self) -> None:
        if self.schema_version != WORLD_MODEL_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported WorldModelManifest schema_version: {self.schema_version}")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "aliases", tuple_of_str(self.aliases))
        object.__setattr__(self, "capabilities", tuple_of_str(self.capabilities))
        object.__setattr__(self, "supported_tasks", tuple_of_str(self.supported_tasks))
        object.__setattr__(self, "required_artifacts", tuple_of_str(self.required_artifacts))
        object.__setattr__(self, "output_artifacts", tuple_of_str(self.output_artifacts))
        object.__setattr__(self, "tags", tuple_of_str(self.tags))
        object.__setattr__(self, "metadata", copy_mapping(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldModelManifest":
        return cls(
            model_id=str(data["model_id"]),
            name=data.get("name", ""),
            aliases=tuple_of_str(data.get("aliases")),
            version=data.get("version", ""),
            provider=data.get("provider", ""),
            capabilities=tuple_of_str(data.get("capabilities")),
            supported_tasks=tuple_of_str(data.get("supported_tasks")),
            required_artifacts=tuple_of_str(data.get("required_artifacts")),
            output_artifacts=tuple_of_str(data.get("output_artifacts")),
            tags=tuple_of_str(data.get("tags")),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", WORLD_MODEL_MANIFEST_SCHEMA_VERSION),
        )
