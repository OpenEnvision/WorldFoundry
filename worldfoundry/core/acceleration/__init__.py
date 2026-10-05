"""Opt-in inference approximations and acceleration reporting.

- cache: cross-step residual/prediction reuse.
- token_pruning: token selection and reconstruction.
- quantization: low-precision Linear execution and FP8/NVFP4 packing.
- technology: inventory of active acceleration families.

CUDA graph dispatch, frame prefetch, overlap and prewarm are owned by
core.execution. One-shot encoder placement is owned by core.vram.
Existing package-level APIs continue to expose those helpers.
"""

# ──────────────────────────────────────────────────────────────────────────
# Re-exports — keep ``from worldfoundry.core.acceleration import X`` stable
# ──────────────────────────────────────────────────────────────────────────

from worldfoundry.core.acceleration.cache import (
    AdaCacheResidualCache,
    AdaptiveResidualCache,
    BlockTaylorSeerCache,
    CustomTaylorResidualCache,
    FixedStepCache,
    MagCacheResidualCache,
    TaylorSeerResidualCache,
    TeaCacheResidualCache,
)
from worldfoundry.core.execution.graphs.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from worldfoundry.core.vram.encoder_lifecycle import (
    collect_and_release_cuda_memory,
    ensure_one_shot_encoder,
    move_tensors_to_cpu,
    offload_module_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
    setup_one_shot_encoder,
)
from worldfoundry.core.execution.realtime.frame_prefetch import (
    CudaHostPrefetch,
    LazyCudaFrame,
    prefetch_to_numpy,
)
from worldfoundry.core.acceleration.quantization.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4,
    quantize_nvfp4,
    replace_linear_with_nvfp4,
)
from worldfoundry.core.execution.realtime.overlap import (
    CudaStreamOverlap,
    HostThreadOverlap,
    SynchronousOverlap,
)
from worldfoundry.core.execution.realtime.prewarm import (
    PrewarmDeadline,
    PrewarmSequenceTiming,
    PrewarmTimeoutError,
    PrewarmTiming,
    cuda_graph_prewarm_steps,
    run_async_prewarm_sequence,
    run_prewarm_sequence,
    run_timed_prewarm,
)
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    WeightOnlyLinear,
    quantization_runtime_report,
    replace_linear_with_float8,
    replace_linear_with_weight_only,
    reset_quantization_runtime_window,
    set_low_precision_enabled,
)
from worldfoundry.core.acceleration.technology import (
    AccelerationTechnology,
    acceleration_technology_report,
)
from worldfoundry.core.acceleration.token_pruning import (
    TokenPruner,
    TokenPruneState,
    prune_tokens,
    restore_tokens,
    select_token_indices,
)

__all__ = [
    "AdaCacheResidualCache",
    "AdaptiveResidualCache",
    "AccelerationTechnology",
    "BlockTaylorSeerCache",
    "CUDAGraphDispatch",
    "CudaHostPrefetch",
    "CudaStreamOverlap",
    "CustomTaylorResidualCache",
    "FixedStepCache",
    "Float8Linear",
    "HostThreadOverlap",
    "LazyCudaFrame",
    "MagCacheResidualCache",
    "NVFP4Linear",
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "SynchronousOverlap",
    "TaylorSeerResidualCache",
    "TeaCacheResidualCache",
    "TokenPruneState",
    "TokenPruner",
    "WeightOnlyLinear",
    "collect_and_release_cuda_memory",
    "cuda_graph_capture_ar_index",
    "cuda_graph_prewarm_steps",
    "ensure_one_shot_encoder",
    "move_tensors_to_cpu",
    "offload_module_to_cpu",
    "prefetch_to_numpy",
    "prune_tokens",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "quantization_runtime_report",
    "reset_quantization_runtime_window",
    "replace_linear_with_float8",
    "replace_linear_with_nvfp4",
    "replace_linear_with_weight_only",
    "release_one_shot_encoder_references",
    "restore_tokens",
    "run_async_prewarm_sequence",
    "run_one_shot_encoder_stage",
    "run_prewarm_sequence",
    "run_timed_prewarm",
    "select_token_indices",
    "set_low_precision_enabled",
    "setup_one_shot_encoder",
    "acceleration_technology_report",
]
