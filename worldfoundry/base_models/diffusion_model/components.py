"""Declarative component identity and execution specs.

This is the upstream spec layer shared by NativeDiffusionRecipe and the assembler:

    recipe fills ComponentSpec / ExecutionSpec
        → assembler calls each factory with ComponentBuildContext
        → strategy looks up ComponentKey via ExecutionSpec.bindings
        → runner consumes protocol objects only and never reads these dataclasses

This module does not run a model; it only defines immutable specs and BuildPurpose.
Model packages must not provide a runner, pipeline, loader, or optimization manager
here (or on a recipe). They may only supply checkpoint-compatible component implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from .loaders import CheckpointSpec
from .optimizations import AttentionBackend, OffloadMode, QuantizationMode, RuntimePolicy


class BuildPurpose(str, Enum):
    """Why a component graph is being built. The assembler writes this onto ComponentBuildContext.purpose.

    Factories default to inference for backwards compatibility. A training assembler
    must pass TRAINING explicitly so factories can reject TeaCache, cuda_graph, and
    other inference-only wrappers before parameters or optimizer state exist.
    ROLLOUT / REWARD are reserved for RL-side paths.
    """

    INFERENCE = "inference"
    TRAINING = "training"
    ROLLOUT = "rollout"
    REWARD = "reward"


class ComponentKind(str, Enum):
    """Component role. Maps 1:1 onto contracts protocols via assembler._EXPECTED_PROTOCOLS."""

    DENOISER = "denoiser"
    CONDITIONER = "conditioner"
    LATENT_ENCODER = "latent_encoder"
    LATENT_INITIALIZER = "latent_initializer"
    LATENT_PROCESSOR = "latent_processor"
    SCHEDULER = "scheduler"
    DECODER = "decoder"


@dataclass(frozen=True, slots=True, order=True)
class ComponentKey:
    """Stable kind/name identity inside one recipe. Strategy bindings and checkpoints address by this.

    Fields:
        kind: ComponentKind; chooses which protocol the assembler isinstance-checks.
        name: Instance name under that kind. Defaults to ``main``. Normalized to
            lowercase with ``_`` → ``-``.

    ``str(key)`` looks like ``"denoiser:main"`` and is a valid component_options string key.
    An empty name raises :exc:`ValueError`.
    """

    kind: ComponentKind
    name: str = "main"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ComponentKind(self.kind))
        normalized_name = str(self.name).strip().lower().replace("_", "-")
        if not normalized_name:
            raise ValueError("component name cannot be empty")
        object.__setattr__(self, "name", normalized_name)

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}"


@dataclass(frozen=True, slots=True)
class ComponentBuildContext:
    """Framework-owned immutable input to a pure component factory. Model packages must not hold a runner.

    Fields:
        model_id: Public ID of the owning recipe.
        key: ComponentKey being constructed.
        policy: RuntimePolicy. TRAINING rejects inference accelerators via
            :func:`validate_runtime_policy_for_purpose`.
        checkpoints: Local ``name → CheckpointSpec`` bindings (recipe keys already resolved).
        recipe_options / component_options: Read-only mappings; the latter overrides same-name options.
        purpose: BuildPurpose, default INFERENCE.

    Failures: empty checkpoint binding name → :exc:`ValueError`; non-CheckpointSpec value →
    :exc:`TypeError`; illegal training policy → :exc:`ValueError`.
    """

    model_id: str
    key: ComponentKey
    policy: RuntimePolicy
    checkpoints: Mapping[str, CheckpointSpec] = field(default_factory=dict)
    recipe_options: Mapping[str, object] = field(default_factory=dict)
    component_options: Mapping[str, object] = field(default_factory=dict)
    purpose: BuildPurpose = BuildPurpose.INFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", BuildPurpose(self.purpose))
        validate_runtime_policy_for_purpose(self.policy, self.purpose)
        checkpoints = {str(name): value for name, value in self.checkpoints.items()}
        if any(not name.strip() for name in checkpoints):
            raise ValueError(f"checkpoint binding names for {self.key} cannot be empty")
        if not all(isinstance(value, CheckpointSpec) for value in checkpoints.values()):
            raise TypeError(f"checkpoint bindings for {self.key} must contain CheckpointSpec values")
        object.__setattr__(self, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(self, "recipe_options", MappingProxyType(dict(self.recipe_options)))
        object.__setattr__(
            self,
            "component_options",
            MappingProxyType(dict(self.component_options)),
        )

    def checkpoint(self, name: str = "weights") -> CheckpointSpec | None:
        """Return one optional named checkpoint bound by the recipe, or ``None``."""

        return self.checkpoints.get(name)

    def require_checkpoint(self, name: str = "weights") -> CheckpointSpec:
        """Return a named checkpoint or raise :exc:`KeyError` before model construction."""

        try:
            return self.checkpoints[name]
        except KeyError as error:
            raise KeyError(f"component {self.key} requires checkpoint binding {name!r}") from error


# RuntimePolicy.options keys forbidden on a training build (inference accelerators / wrappers).
_TRAINING_FORBIDDEN_OPTIONS = frozenset(
    {
        "accelerations",
        "adacache",
        "block_swap",
        "cache_skip",
        "cuda_graph",
        "enable_cuda_graph",
        "feature_cache",
        "approximate_attention",
        "fuse_qkv",
        "inplace_residual",
        "rms_norm_precision",
        "inference_mode",
        "magcache",
        "static_cross_kv",
        "step_cache",
        "taylorseer",
        "teacache",
        "teacache_thresh",
    }
)


def validate_runtime_policy_for_purpose(policy: RuntimePolicy, purpose: BuildPurpose | str) -> None:
    """Fail closed when a training build requests inference-only behavior.

    All purposes reject the unimplemented combined residual/AdaLN opt-in.
    Otherwise non-TRAINING returns immediately. TRAINING rejects offload,
    quantization, SAGE attention, and ``_TRAINING_FORBIDDEN_OPTIONS`` (TeaCache,
    cuda_graph, cache_skip, …).
    P0 deliberately accepts a narrow policy; frozen-component offload and
    training-safe quantization must go through training-owned contracts instead of
    silently reusing inference wrappers. Raises :exc:`ValueError` on violation.
    """

    resolved = BuildPurpose(purpose)
    fused_residual_adaln = policy.options.get("fused_residual_adaln")
    if fused_residual_adaln is not None and not isinstance(fused_residual_adaln, bool):
        raise TypeError("fused_residual_adaln must be a bool")
    if fused_residual_adaln:
        raise ValueError("fused_residual_adaln is unavailable: the combined kernel is not implemented")
    if resolved is not BuildPurpose.TRAINING:
        return

    errors: list[str] = []
    if policy.offload.mode is not OffloadMode.NONE:
        errors.append(f"offload={policy.offload.mode.value}")
    if policy.quantization.mode is not QuantizationMode.NONE:
        errors.append(f"quantization={policy.quantization.mode.value}")
    if policy.attention in {AttentionBackend.SAGE, AttentionBackend.SOL_ATTN}:
        errors.append(f"attention={policy.attention.value}")
    enabled_options = sorted(key for key in _TRAINING_FORBIDDEN_OPTIONS if bool(policy.options.get(key)))
    if enabled_options:
        errors.append(f"inference-only options={enabled_options}")
    if errors:
        raise ValueError("training component build rejects " + ", ".join(errors))


# Pure component factory: consumes only ComponentBuildContext; must not return a runner/pipeline.
ComponentFactory = Callable[[ComponentBuildContext], object]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One architecture component selected by a recipe: identity + pure factory + checkpoint names.

    Fields:
        key: ComponentKey.
        factory: ``ComponentBuildContext → object``. Must not close over a runner.
        checkpoints: Local name → key in ``recipe.checkpoints`` (not a filesystem path).
        options: Component-level defaults, overridable by assembler component_options.

    A non-callable factory raises :exc:`TypeError`; empty binding names raise :exc:`ValueError`.
    """

    key: ComponentKey
    factory: ComponentFactory
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError(f"component factory for {self.key} must be callable")
        checkpoints = {str(name): str(value) for name, value in self.checkpoints.items()}
        if any(not name.strip() or not value.strip() for name, value in checkpoints.items()):
            raise ValueError(f"checkpoint bindings for {self.key} cannot be empty")
        object.__setattr__(self, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


# Default role → ComponentKey bindings for standard / most single-stage strategies.
STANDARD_COMPONENT_BINDINGS = MappingProxyType(
    {
        "denoiser": ComponentKey(ComponentKind.DENOISER),
        "conditioner": ComponentKey(ComponentKind.CONDITIONER),
        "latent_initializer": ComponentKey(ComponentKind.LATENT_INITIALIZER),
        "scheduler": ComponentKey(ComponentKind.SCHEDULER),
        "decoder": ComponentKey(ComponentKind.DECODER),
    }
)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Select a framework-owned execution strategy and declare role → ComponentKey bindings.

    A recipe may only fill a strategy ID and bindings; it cannot supply a runner factory.
    The strategy name is normalized to lowercase with ``_`` → ``-`` (same as
    ExecutionStrategyRegistry). An empty strategy raises :exc:`ValueError`. An unknown
    ID raises UnsupportedExecutionStrategyError only when the assembler asks the registry.
    """

    strategy: str = "standard"
    bindings: Mapping[str, ComponentKey] = field(default_factory=lambda: dict(STANDARD_COMPONENT_BINDINGS))
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower().replace("_", "-")
        if not strategy:
            raise ValueError("execution strategy cannot be empty")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


__all__ = [
    "BuildPurpose",
    "ComponentBuildContext",
    "ComponentFactory",
    "ComponentKey",
    "ComponentKind",
    "ComponentSpec",
    "ExecutionSpec",
    "STANDARD_COMPONENT_BINDINGS",
    "validate_runtime_policy_for_purpose",
]
