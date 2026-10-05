"""Published Echo-Memory model IDs, aliases, and upstream checkpoint status.

Recipes look up a :class:`EchoMemoryModelSpec` by ``model_id`` (or alias)
via :func:`get_echo_memory_model_spec`.  The spec pins the immutable
:class:`~..networks.echo_memory.schema.EchoMemoryRecipe` and the relative
safetensors path inside the official Hugging Face snapshot.

WorldFoundry integrates Context K=1 with its published checkpoint.
Unreleased and retracted research variants are outside the integration scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..networks.echo_memory.schema import EchoMemoryRecipe, get_echo_memory_recipe

ECHO_SOURCE_REPOSITORY = "https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory"
ECHO_SOURCE_REVISION = "30eaffb55b264e1d8dfef70a8934d34c86e29947"
ECHO_CHECKPOINT_REPOSITORY = "Echo-Team/Echo-Memory"
ECHO_CHECKPOINT_REVISION = "7645f33efbfd02c58dc471570d0c7bd6a50eec83"
ECHO_BACKBONE_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B"


class EchoCheckpointAvailability(str, Enum):
    """Current upstream checkpoint state, kept separate from model identity."""

    PUBLIC_CURRENT = "public_current"
    UNRELEASED_BY_UPSTREAM = "unreleased_by_upstream"
    RETRACTED_MISALIGNED_BY_UPSTREAM = "retracted_misaligned_by_upstream"


@dataclass(frozen=True)
class EchoMemoryModelSpec:
    """One independently selectable WorldFoundry model."""

    model_id: str
    display_name: str
    recipe_id: str
    checkpoint_file: str
    aliases: tuple[str, ...] = ()
    release_tier: str = "paper_primary"
    checkpoint_availability: EchoCheckpointAvailability = EchoCheckpointAvailability.PUBLIC_CURRENT

    @property
    def recipe(self) -> EchoMemoryRecipe:
        """Return the immutable recipe pinned by this model."""

        return get_echo_memory_recipe(self.recipe_id)

    @property
    def has_public_checkpoint(self) -> bool:
        """Whether the official Hugging Face ``main`` branch serves this file."""

        return self.checkpoint_availability is EchoCheckpointAvailability.PUBLIC_CURRENT

    @property
    def integration_status(self) -> str:
        """Catalog status implied by code readiness and upstream weight access."""

        return "integrated" if self.has_public_checkpoint else "blocked"


_SPECS = (
    EchoMemoryModelSpec(
        model_id="echo-memory-context-k1",
        display_name="Echo-Memory Context K=1",
        recipe_id="context-k1",
        checkpoint_file="context_k1/epoch-0.safetensors",
        aliases=("echo-context-k1",),
    ),
)

ECHO_MEMORY_MODELS: Mapping[str, EchoMemoryModelSpec] = MappingProxyType({spec.model_id: spec for spec in _SPECS})


def get_echo_memory_model_spec(model_id: str) -> EchoMemoryModelSpec:
    """Resolve an independent Echo model ID or a declared alias."""

    normalized = str(model_id).strip().lower().replace("_", "-")
    for spec in ECHO_MEMORY_MODELS.values():
        candidates = (spec.model_id, *spec.aliases)
        if normalized in {candidate.lower().replace("_", "-") for candidate in candidates}:
            return spec
    choices = ", ".join(ECHO_MEMORY_MODELS)
    raise KeyError(f"unknown Echo-Memory model {model_id!r}; choices: {choices}")


__all__ = [
    "ECHO_BACKBONE_REPOSITORY",
    "ECHO_CHECKPOINT_REPOSITORY",
    "ECHO_CHECKPOINT_REVISION",
    "ECHO_MEMORY_MODELS",
    "ECHO_SOURCE_REPOSITORY",
    "ECHO_SOURCE_REVISION",
    "EchoCheckpointAvailability",
    "EchoMemoryModelSpec",
    "get_echo_memory_model_spec",
]
