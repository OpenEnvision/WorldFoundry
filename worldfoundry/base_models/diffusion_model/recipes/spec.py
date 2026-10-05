"""Immutable, declarative model recipes: the only input contract for the pipeline.

This module only describes which pure components and which framework strategy to use.
It sits at the top of:

    NativeDiffusionRecipe (this module)
        → NativeDiffusionRegistry.resolve / assembler.build
        → ExecutionStrategyRegistry
        → runner.run(DiffusionRequest)

Inputs: model_id, a ComponentSpec tuple, ExecutionSpec, and a CheckpointSpec mapping.
Output: a recipe frozen by ``__post_init__`` for read-only assembler consumption.
Must not provide a runner or pipeline factory, or perform loading / sampling here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..components import ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec


@dataclass(frozen=True, slots=True)
class NativeDiffusionRecipe:
    """One public model assembled exclusively from canonical infra roles.

    A recipe cannot provide a runner. It selects pure component factories and a
    framework-owned execution strategy instead.

    Fields:
        model_id: Canonical ID; empty is illegal. Case/underscore normalization
            happens at registry registration time.
        components: Non-empty ComponentSpec tuple; ComponentKey values must be unique.
        execution: ExecutionSpec (strategy ID + bindings). Bindings must point at
            declared components.
        checkpoints: Logical name → CheckpointSpec. Every ComponentSpec.checkpoints
            value must exist here.
        aliases: Extra resolve names reserved by the registry.
        capabilities / options / metadata: Capability tags and read-only extras.
            The assembler does not interpret strategy from these.

    ``__post_init__`` raises :exc:`ValueError` for an empty model_id, no components,
    duplicate ComponentKey, unknown checkpoint reference, or a binding to a missing component.
    """

    model_id: str
    components: tuple[ComponentSpec, ...]
    execution: ExecutionSpec = ExecutionSpec()
    checkpoints: Mapping[str, CheckpointSpec] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)
    options: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _key(self.model_id):
            raise ValueError("model_id cannot be empty")
        components = tuple(self.components)
        if not components:
            raise ValueError("native diffusion recipe requires components")
        component_keys = [component.key for component in components]
        if len(component_keys) != len(set(component_keys)):
            raise ValueError(f"recipe {self.model_id!r} contains duplicate component keys")

        checkpoints = dict(self.checkpoints)
        missing_checkpoints = sorted(
            {
                checkpoint_key
                for component in components
                for checkpoint_key in component.checkpoints.values()
                if checkpoint_key not in checkpoints
            }
        )
        if missing_checkpoints:
            raise ValueError(f"recipe {self.model_id!r} references unknown checkpoints: {missing_checkpoints}")
        missing_bindings = sorted(str(key) for key in self.execution.bindings.values() if key not in component_keys)
        if missing_bindings:
            raise ValueError(f"recipe {self.model_id!r} binds missing components: {missing_bindings}")

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _key(value: str) -> str:
    """Same identity normalization as NativeDiffusionRegistry: strip, lower, ``_`` → ``-``."""
    return str(value).strip().lower().replace("_", "-")


__all__ = ["NativeDiffusionRecipe"]
