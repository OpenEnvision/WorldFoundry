"""Framework-owned execution strategy registry.

Recipes select a strategy by ID. This module instantiates the runner.

    assembler.build_components(...) → dict[ComponentKey, object]
        → ExecutionStrategyRegistry.build(recipe.execution.strategy, ExecutionBuildContext)
        → each build_*_strategy validates bindings, then constructs a runner

Inputs: ExecutionBuildContext (recipe + built components + RuntimePolicy + extensions).
Output: a runner that implements DiffusionExecutor.
Must not: let a recipe supply a runner factory; register a strategy as an import
side effect; let a model package construct a runner while skipping binding checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, runtime_checkable

from ..components import ComponentKey
from ..contracts import DiffusionOutput, DiffusionRequest
from ..extensions import DiffusionExtension, FrozenContextSuffixExtension, FrozenLatentMaskExtension
from ..optimizations import RuntimePolicy
from ..recipes.spec import NativeDiffusionRecipe
from .autoregressive import AutoregressiveWindowRunner
from .base import (
    DualConditionGuidanceRunner,
    NativeDiffusionRunner,
    RunnerComponents,
    Wan22DualExpertGuidanceRunner,
)
from .chunked import ChunkedAdditiveCacheRunner, ChunkedKVCacheRunner
from .multistage import JointMultiStageDiffusionRunner, MultiStageComponents
from .prefix_recompute import PrefixRecomputeRunner


@runtime_checkable
class DiffusionExecutor(Protocol):
    """Common execution surface consumed by the native public pipeline: ``model_id`` and ``run``.

    The assembler / registry isinstance-check builder return values against this protocol.
    """

    model_id: str

    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Execute one normalized native DiffusionRequest and return DiffusionOutput."""


@dataclass(frozen=True, slots=True)
class ExecutionBuildContext:
    """Complete framework-owned input for an execution strategy builder.

    The component mapping is frozen read-only in post_init.

    Fields:
        recipe: Still carries ExecutionSpec.bindings / options for validation and guidance params.
        components: Assembler-built ``ComponentKey → protocol object``.
        policy: RuntimePolicy providing device / dtype.
        extensions: Caller-injected DiffusionExtension values; some strategies append a built-in one.
    """

    recipe: NativeDiffusionRecipe
    components: Mapping[ComponentKey, object]
    policy: RuntimePolicy
    extensions: tuple[DiffusionExtension, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


# Strategy builder: consume only ExecutionBuildContext and return a DiffusionExecutor.
ExecutionStrategyBuilder = Callable[[ExecutionBuildContext], DiffusionExecutor]


class UnsupportedExecutionStrategyError(ValueError):
    """Raised when recipe.execution.strategy has no builder in this registry."""


class ExecutionStrategyRegistry:
    """Instance-local strategy → builder table. The default table is frozen by :func:`default_execution_strategy_registry`.

    register after freeze → :exc:`RuntimeError`; duplicate ID → :exc:`ValueError`;
    non-callable builder → :exc:`TypeError`. :meth:`build` on an unknown strategy →
    :exc:`UnsupportedExecutionStrategyError`; a return value that is not a
    DiffusionExecutor → :exc:`TypeError`.
    """

    def __init__(self) -> None:
        self._builders: dict[str, ExecutionStrategyBuilder] = {}
        self._frozen = False

    def register(self, strategy: str, builder: ExecutionStrategyBuilder) -> None:
        """Register a normalized strategy ID. The default table is filled by the framework before freeze."""
        if self._frozen:
            raise RuntimeError("execution strategy registry is frozen")
        key = _strategy_key(strategy)
        if key in self._builders:
            raise ValueError(f"execution strategy is already registered: {key}")
        if not callable(builder):
            raise TypeError("execution strategy builder must be callable")
        self._builders[key] = builder

    def freeze(self) -> None:
        """Reject further register calls so a model import cannot sneak in a private strategy."""
        self._frozen = True

    def build(self, strategy: str, context: ExecutionBuildContext) -> DiffusionExecutor:
        """Invoke the builder for this ID after ``_strategy_key`` normalization."""
        key = _strategy_key(strategy)
        cfg_degree = _cfg_parallel_degree(context)
        if cfg_degree > 1 and key not in {
            "standard",
            "frozen-context",
            "masked-latent",
            "wan22-dual-expert-guidance",
        }:
            raise ValueError(
                f"cfg_parallel is not implemented for execution strategy {key!r}"
            )
        cfg_gate_step = _cfg_gate_step(context)
        if cfg_gate_step < 1.0 and key not in {
            "standard",
            "frozen-context",
            "masked-latent",
        }:
            raise ValueError(
                f"cfg_gate_step is not implemented for execution strategy {key!r}"
            )
        try:
            builder = self._builders[key]
        except KeyError as error:
            raise UnsupportedExecutionStrategyError(key) from error
        executor = builder(context)
        if not isinstance(executor, DiffusionExecutor):
            raise TypeError(
                f"execution strategy {key!r} returned {type(executor).__name__}; expected DiffusionExecutor"
            )
        return executor

    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def build_standard_strategy(
    context: ExecutionBuildContext,
    *,
    runner_type: type[NativeDiffusionRunner] = NativeDiffusionRunner,
) -> NativeDiffusionRunner:
    """Build the shared condition → initialize → denoise → decode loop.

    Bindings must cover denoiser / conditioner / latent_initializer / scheduler / decoder,
    optionally latent_encoder. Extra roles or missing required ones raise :exc:`ValueError`.
    ``execution.options.guidance_mode`` is passed to NativeDiffusionRunner (``standard`` / ``positive``).
    """

    required_bindings = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
    }
    optional_bindings = {"latent_encoder"}
    actual_bindings = set(context.recipe.execution.bindings)
    if not required_bindings.issubset(actual_bindings) or actual_bindings - required_bindings - optional_bindings:
        raise ValueError(
            "standard execution requires the canonical bindings and optionally latent_encoder: "
            f"required={sorted(required_bindings)}; got={sorted(actual_bindings)}"
        )
    bound = {role: context.components[key] for role, key in context.recipe.execution.bindings.items()}
    return runner_type(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound.get("latent_encoder"),  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        guidance_mode=str(context.recipe.execution.options.get("guidance_mode", "standard")),
        cfg_parallel_degree=_cfg_parallel_degree(context),
        cfg_gate_step=_cfg_gate_step(context),
    )


def build_vchitect_strategy(context: ExecutionBuildContext) -> NativeDiffusionRunner:
    """Use the standard component graph with Vchitect's official CFG curve."""
    from .vchitect import VchitectGuidanceRunner

    return build_standard_strategy(context, runner_type=VchitectGuidanceRunner)


def build_dual_condition_guidance_strategy(
    context: ExecutionBuildContext,
) -> DualConditionGuidanceRunner:
    """Standard lifecycle plus framework-owned three-branch CFG (text + secondary condition).

    Bindings must be exactly the five canonical roles plus latent_encoder, else :exc:`ValueError`.
    secondary_guidance_scale / secondary_guidance_input are read from execution.options.
    """

    required_bindings = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
        "latent_encoder",
    }
    actual_bindings = set(context.recipe.execution.bindings)
    if actual_bindings != required_bindings:
        raise ValueError(
            "dual-condition-guidance execution requires canonical codec bindings: "
            f"required={sorted(required_bindings)}; got={sorted(actual_bindings)}"
        )
    bound = {role: context.components[key] for role, key in context.recipe.execution.bindings.items()}
    return DualConditionGuidanceRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound["latent_encoder"],  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        secondary_guidance_scale=float(
            context.recipe.execution.options.get("secondary_guidance_scale", 1.0)
        ),
        secondary_guidance_input=str(
            context.recipe.execution.options.get(
                "secondary_guidance_input",
                "secondary_guidance_scale",
            )
        ),
    )


def build_wan22_dual_expert_guidance_strategy(
    context: ExecutionBuildContext,
) -> Wan22DualExpertGuidanceRunner:
    """Wan2.2 A14B: timestep-routed low/high expert CFG lifecycle.

    Required bindings match standard; latent_encoder is optional. options must provide
    boundary_ratio / low_noise_guidance_scale / high_noise_guidance_scale
    (missing keys fail at float()).
    """

    required = {"denoiser", "conditioner", "latent_initializer", "scheduler", "decoder"}
    optional = {"latent_encoder"}
    bindings = context.recipe.execution.bindings
    if not required.issubset(bindings) or set(bindings) - required - optional:
        raise ValueError(
            "wan22-dual-expert-guidance requires canonical bindings and optionally latent_encoder"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    return Wan22DualExpertGuidanceRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound.get("latent_encoder"),  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        boundary_ratio=float(options["boundary_ratio"]),
        low_noise_guidance_scale=float(options["low_noise_guidance_scale"]),
        high_noise_guidance_scale=float(options["high_noise_guidance_scale"]),
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        cfg_parallel_degree=_cfg_parallel_degree(context),
    )


def build_frozen_context_strategy(
    context: ExecutionBuildContext,
) -> NativeDiffusionRunner:
    """Reuse the standard loop and auto-install FrozenContextSuffixExtension for suffix relocking.

    If the caller already supplied the same extension_id, raise :exc:`ValueError` to avoid double-mount.
    """

    extension = FrozenContextSuffixExtension()
    if any(item.extension_id == extension.extension_id for item in context.extensions):
        raise ValueError("frozen-context strategy installs its suffix extension automatically")
    enriched = ExecutionBuildContext(
        recipe=context.recipe,
        components=context.components,
        policy=context.policy,
        extensions=(*context.extensions, extension),
    )
    return build_standard_strategy(enriched)


def build_masked_latent_strategy(
    context: ExecutionBuildContext,
) -> NativeDiffusionRunner:
    """Reuse the standard loop; FrozenLatentMaskExtension projects frozen latent regions after each step.

    Pre-installing the same extension_id also raises :exc:`ValueError`.
    """

    extension = FrozenLatentMaskExtension()
    if any(item.extension_id == extension.extension_id for item in context.extensions):
        raise ValueError("masked-latent strategy installs its projection extension automatically")
    enriched = ExecutionBuildContext(
        recipe=context.recipe,
        components=context.components,
        policy=context.policy,
        extensions=(*context.extensions, extension),
    )
    return build_standard_strategy(enriched)


def build_joint_multistage_strategy(
    context: ExecutionBuildContext,
) -> JointMultiStageDiffusionRunner:
    """Joint-modality multi-stage runner. Bindings declare one scheduler per stage via ``scheduler-*``.

    Required: denoiser / conditioner / latent_initializer / decoder. Optional: processor.
    Missing ``scheduler-*``, a stage_steps length mismatch, or multi-stage without a
    processor all raise :exc:`ValueError`.
    """

    bindings = context.recipe.execution.bindings
    required = {"denoiser", "conditioner", "latent_initializer", "decoder"}
    if not required.issubset(bindings):
        missing = sorted(required - set(bindings))
        raise ValueError(f"joint-multistage execution is missing bindings: {missing}")
    scheduler_roles = tuple(sorted(role for role in bindings if role.startswith("scheduler-")))
    if not scheduler_roles:
        raise ValueError("joint-multistage execution requires scheduler-* bindings")
    optional = {"processor"}
    unexpected = sorted(set(bindings) - required - optional - set(scheduler_roles))
    if unexpected:
        raise ValueError(f"joint-multistage execution has unsupported bindings: {unexpected}")
    stage_steps = tuple(int(value) for value in context.recipe.execution.options.get("stage_steps", ()))
    if len(stage_steps) != len(scheduler_roles):
        raise ValueError("joint-multistage stage_steps must match scheduler binding count")
    if len(stage_steps) > 1 and "processor" not in bindings:
        raise ValueError("multi-stage execution requires a processor binding")
    bound = {role: context.components[key] for role, key in bindings.items()}
    return JointMultiStageDiffusionRunner(
        model_id=context.recipe.model_id,
        components=MultiStageComponents(
            denoiser=bound["denoiser"],
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            schedulers=tuple(bound[role] for role in scheduler_roles),  # type: ignore[arg-type]
            processor=bound.get("processor"),  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
        ),
        stage_steps=stage_steps,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )


def build_autoregressive_window_strategy(
    context: ExecutionBuildContext,
) -> AutoregressiveWindowRunner:
    """Multi-view / block autoregressive diffusion with an optional latent encoder."""

    required = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
    }
    optional = {"latent_encoder"}
    bindings = context.recipe.execution.bindings
    if not required.issubset(bindings) or set(bindings) - required - optional:
        raise ValueError(
            "autoregressive-window execution requires canonical bindings and optionally "
            f"latent_encoder: required={sorted(required)}; got={sorted(bindings)}"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    return AutoregressiveWindowRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound.get("latent_encoder"),  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        prediction_mode=str(options.get("prediction_mode", "flow")),
        fixed_timesteps=tuple(int(value) for value in options.get("fixed_timesteps", ())),
        context_timestep=int(options.get("context_timestep", 0)),
        guidance_mode=str(options.get("guidance_mode", "standard")),
    )


def build_chunked_kv_cache_strategy(
    context: ExecutionBuildContext,
) -> ChunkedKVCacheRunner:
    """Temporal chunk traversal with token or additive attention caches."""

    required = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
    }
    additive = context.recipe.execution.strategy == "chunked-additive-cache"
    if not additive:
        required.add("latent_encoder")
    bindings = context.recipe.execution.bindings
    if set(bindings) != required:
        raise ValueError(
            "chunked-kv-cache execution requires canonical encoded-latent bindings: "
            f"required={sorted(required)}; got={sorted(bindings)}"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    runner_class = ChunkedAdditiveCacheRunner if additive else ChunkedKVCacheRunner
    return runner_class(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound.get("latent_encoder"),  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        base_chunk_frames=int(options.get("base_chunk_frames", 3)),
        num_cached_chunks=int(options.get("num_cached_chunks", 2)),
        sink_token=bool(options.get("sink_token", True)),
    )


def build_prefix_recompute_strategy(
    context: ExecutionBuildContext,
) -> PrefixRecomputeRunner:
    """Bounded clean-prefix recomputation with an optional paired refiner."""

    required = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "latent_encoder",
        "scheduler",
        "decoder",
    }
    optional = {"refiner", "refiner_conditioner"}
    bindings = context.recipe.execution.bindings
    if not required.issubset(bindings) or set(bindings) - required - optional:
        raise ValueError(
            "prefix-recompute execution requires canonical encoded-latent bindings "
            f"and an optional refiner pair: required={sorted(required)}; "
            f"got={sorted(bindings)}"
        )
    if ("refiner" in bindings) != ("refiner_conditioner" in bindings):
        raise ValueError(
            "prefix-recompute refiner and refiner_conditioner bindings must be provided together"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    return PrefixRecomputeRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound["latent_encoder"],  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        guidance_mode=str(options.get("guidance_mode", "standard")),
        chunk_size=int(options.get("chunk_size", 3)),
        history_frames=int(options.get("history_frames", 6)),
        sink_frames=int(options.get("sink_frames", 1)),
        refiner=bound.get("refiner"),  # type: ignore[arg-type]
        refiner_conditioner=bound.get("refiner_conditioner"),  # type: ignore[arg-type]
        refiner_max_frames=int(options.get("refiner_max_frames", 11)),
    )


def default_execution_strategy_registry() -> ExecutionStrategyRegistry:
    """Create and freeze the built-in strategy set. Each call is a new instance; no global mutation."""

    registry = ExecutionStrategyRegistry()
    registry.register("standard", build_standard_strategy)
    registry.register("vchitect-guidance", build_vchitect_strategy)
    registry.register("dual-condition-guidance", build_dual_condition_guidance_strategy)
    registry.register("wan22-dual-expert-guidance", build_wan22_dual_expert_guidance_strategy)
    registry.register("frozen-context", build_frozen_context_strategy)
    registry.register("masked-latent", build_masked_latent_strategy)
    registry.register("joint-multistage", build_joint_multistage_strategy)
    registry.register("autoregressive-window", build_autoregressive_window_strategy)
    registry.register("chunked-kv-cache", build_chunked_kv_cache_strategy)
    registry.register("chunked-additive-cache", build_chunked_kv_cache_strategy)
    registry.register("prefix-recompute", build_prefix_recompute_strategy)
    registry.freeze()
    return registry


def _strategy_key(value: str) -> str:
    """Same strategy-ID normalization as ExecutionSpec. Empty string raises :exc:`ValueError`."""
    key = str(value).strip().lower().replace("_", "-")
    if not key:
        raise ValueError("execution strategy cannot be empty")
    return key


def _cfg_parallel_degree(context: ExecutionBuildContext) -> int:
    value = context.policy.options.get(
        "cfg_parallel",
        context.policy.options.get("cfg_parallel_degree", 1),
    )
    if value is True:
        return 2
    if value in (False, None):
        return 1
    degree = int(value)
    if degree not in {1, 2}:
        raise ValueError("cfg_parallel must be 1 or 2")
    return degree


def _cfg_gate_step(context: ExecutionBuildContext) -> float:
    """Resolve FastVideo-compatible stale-uncond gating from runtime policy."""

    options = context.policy.options
    primary = options.get("cfg_gate_step")
    alias = options.get("cfg_gate_fraction")
    if primary is not None and alias is not None and float(primary) != float(alias):
        raise ValueError("cfg_gate_step and cfg_gate_fraction must match")
    value = 1.0 if primary is None and alias is None else float(
        primary if primary is not None else alias
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError("cfg_gate_step must be in [0.0, 1.0]")
    return value


__all__ = [
    "DiffusionExecutor",
    "ExecutionBuildContext",
    "ExecutionStrategyBuilder",
    "ExecutionStrategyRegistry",
    "UnsupportedExecutionStrategyError",
    "build_dual_condition_guidance_strategy",
    "build_wan22_dual_expert_guidance_strategy",
    "build_autoregressive_window_strategy",
    "build_chunked_kv_cache_strategy",
    "build_prefix_recompute_strategy",
    "build_standard_strategy",
    "build_vchitect_strategy",
    "build_frozen_context_strategy",
    "build_joint_multistage_strategy",
    "build_masked_latent_strategy",
    "default_execution_strategy_registry",
]
