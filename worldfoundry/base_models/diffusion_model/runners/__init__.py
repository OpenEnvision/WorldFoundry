"""Framework-owned canonical runner and strategy export surface.

This package is the downstream execution layer: after the assembler selects a
strategy, runners are instantiated here and ``run(DiffusionRequest)``. It only
re-exports the DiffusionExecutor protocol and the runner types strategy builders
need. It does not register recipes at import time.
Must not: put a runner factory on a recipe; implement a sampling loop in this file.
"""

from .autoregressive import AutoregressiveWindowRunner, WindowedDenoiser
from .base import DualConditionGuidanceRunner, NativeDiffusionRunner, RunnerComponents
from .chunked import ChunkedCacheDenoiser, ChunkedKVCacheRunner
from .multistage import (
    JointMultiStageDiffusionRunner,
    MultiModalLatentDecoder,
    MultiStageComponents,
    MultiStageLatentInitializer,
)
from .prefix_recompute import (
    PrefixRecomputeRunner,
    PrefixRecomputeWindow,
    PrefixRefiner,
    prefix_chunk_boundaries,
    prefix_recompute_window,
)
from .staged import InferenceStage, InferenceStageGraph, InferenceStageRunner, StagedDiffusionPipeline
from .strategies import (
    DiffusionExecutor,
    ExecutionBuildContext,
    ExecutionStrategyRegistry,
    UnsupportedExecutionStrategyError,
    default_execution_strategy_registry,
)


def __getattr__(name: str):
    # Wan's tokenizer dependencies are optional for other runner families.
    if name in {"TeaCache", "WanStagedPipeline", "model_fn_wan_video"}:
        from . import wan_staged

        value = getattr(wan_staged, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffusionExecutor",
    "AutoregressiveWindowRunner",
    "DualConditionGuidanceRunner",
    "ChunkedCacheDenoiser",
    "ChunkedKVCacheRunner",
    "ExecutionBuildContext",
    "ExecutionStrategyRegistry",
    "JointMultiStageDiffusionRunner",
    "InferenceStage",
    "InferenceStageGraph",
    "InferenceStageRunner",
    "MultiModalLatentDecoder",
    "MultiStageComponents",
    "MultiStageLatentInitializer",
    "NativeDiffusionRunner",
    "PrefixRecomputeRunner",
    "PrefixRecomputeWindow",
    "PrefixRefiner",
    "RunnerComponents",
    "StagedDiffusionPipeline",
    "TeaCache",
    "WanStagedPipeline",
    "WindowedDenoiser",
    "UnsupportedExecutionStrategyError",
    "default_execution_strategy_registry",
    "model_fn_wan_video",
    "prefix_chunk_boundaries",
    "prefix_recompute_window",
]
