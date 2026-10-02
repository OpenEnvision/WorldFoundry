"""Attention operators and state shared by model families.

- backends: capability probes, exact SDPA, layout dispatch and model adapters.
- rotary: 2D/3D/n-D, complex and projective position embeddings.
- cache: KV storage, causal geometry, eviction and quantization.
- sequence: packed, variable-length and multimodal layouts.
- sparse: block masks and sparse kernels.
- parallel: context/sequence-parallel attention and rotary adapters.

Public symbols resolve lazily; package import does not initialize model dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy export table — names resolve on first access so import stays cheap
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "KVSegmentArena": "worldfoundry.core.attention.cache.kv_arena",
    "KVSegmentLayout": "worldfoundry.core.attention.cache.kv_arena",
    "AttentionKernelCapability": "worldfoundry.core.attention.backends.probe",
    "ModelSpecificAttentionBackendError": "worldfoundry.core.attention.backends.probe",
    "apply_complex_rotary_embedding": "worldfoundry.core.attention.rotary.complex_rope",
    "AttentionCallable": "worldfoundry.core.attention.backends.model_backends",
    "AttentionFunction": "worldfoundry.core.attention.backends.model_backends",
    "AttentionBackendInfo": "worldfoundry.core.attention.backends.native",
    "BlockKVCache": "worldfoundry.core.attention.cache.kvcache",
    "InferenceParams": "worldfoundry.core.attention.cache.inference_state",
    "CausalVideoCacheGeometry": "worldfoundry.core.attention.cache.causal_cache",
    "allocate_causal_video_cache": "worldfoundry.core.attention.cache.causal_cache",
    "begin_causal_video_cache_block": "worldfoundry.core.attention.cache.causal_cache",
    "causal_video_cache_geometry": "worldfoundry.core.attention.cache.causal_cache",
    "causal_video_cache_state": "worldfoundry.core.attention.cache.causal_cache",
    "commit_causal_video_cache_block": "worldfoundry.core.attention.cache.causal_cache",
    "finish_causal_video_cache_call": "worldfoundry.core.attention.cache.causal_cache",
    "ContextParallelAttention": "worldfoundry.core.attention.parallel.cp",
    "KVCacheRelativeRotaryPositionEmbedding3D": "worldfoundry.core.attention.rotary.rope",
    "ModelMetaArgs": "worldfoundry.core.attention.sequence.packed_sequence",
    "MaskedAttentionCallable": "worldfoundry.core.attention.backends.model_backends",
    "MaskedAttentionFunction": "worldfoundry.core.attention.backends.model_backends",
    "NativeAttention": "worldfoundry.core.attention.backends.native",
    "PackedCoreAttnParams": "worldfoundry.core.attention.sequence.packed_sequence",
    "PackedCrossAttnParams": "worldfoundry.core.attention.sequence.packed_sequence",
    "PositionGetter": "worldfoundry.core.attention.rotary.rope_2d",
    "RotaryPositionEmbedding3D": "worldfoundry.core.attention.rotary.rope",
    "RotaryPositionEmbedding2D": "worldfoundry.core.attention.rotary.rope_2d",
    "attention_backend_capability": "worldfoundry.core.attention.backends.probe",
    "attention_backend_from_env": "worldfoundry.core.attention.backends.probe",
    "attention_dispatch_report": "worldfoundry.core.attention.backends.dispatch",
    "attention_compile_receipt_scope": "worldfoundry.core.attention.backends.dispatch",
    "attention_forward": "worldfoundry.core.attention.backends.dispatch",
    "attention_provider_runtime_report": "worldfoundry.core.attention.backends.dispatch",
    "clear_attention_dispatch_cache": "worldfoundry.core.attention.backends.dispatch",
    "complex_rotary_frequencies": "worldfoundry.core.attention.rotary.complex_rope",
    "complex_rotary_frequencies_3d": "worldfoundry.core.attention.rotary.complex_rope",
    "flash_attention": "worldfoundry.core.attention.sequence.varlen",
    "flattened_attention": "worldfoundry.core.attention.backends.hybrid",
    "masked_attention": "worldfoundry.core.attention.sequence.varlen",
    "hybrid_provider_attention": "worldfoundry.core.attention.backends.hybrid",
    "flattened_multihead_attention": "worldfoundry.core.attention.backends.native",
    "apply_nd_rotary_embedding": "worldfoundry.core.attention.rotary.rope_nd",
    "apply_rope_freqs": "worldfoundry.core.attention.rotary.rope",
    "apply_rotary_embedding": "worldfoundry.core.attention.rotary.rope",
    "apply_sequence_parallel_rope": "worldfoundry.core.attention.parallel.sequence_parallel_rope",
    "attention_backend_report": "worldfoundry.core.attention.backends.probe",
    "attention_backend_info": "worldfoundry.core.attention.backends.native",
    "attention_backend_context": "worldfoundry.core.attention.backends.native",
    "get_1d_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "get_cu_seqlens": "worldfoundry.core.attention.sequence.sequence_metadata",
    "get_meshgrid_nd": "worldfoundry.core.attention.rotary.rope_nd",
    "get_nd_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "gpu_supports_flash_attention": "worldfoundry.core.attention.backends.probe",
    "normalize_attention_backend": "worldfoundry.core.attention.backends.probe",
    "native_sdpa_priority": "worldfoundry.core.attention.backends.native",
    "normalize_fully_masked_rows": "worldfoundry.core.attention.backends.native",
    "piecewise_attention": "worldfoundry.core.attention.sparse.piecewise",
    "piecewise_attention_available": "worldfoundry.core.attention.sparse.piecewise",
    "prope_dot_product_attention": "worldfoundry.core.attention.rotary.projective_rope",
    "pad_freqs": "worldfoundry.core.attention.parallel.sequence_parallel_rope",
    "packed_sequence_attention": "worldfoundry.core.attention.backends.dispatch",
    "attention": "worldfoundry.core.attention.sequence.varlen",
    "probe_attention_backends": "worldfoundry.core.attention.backends.probe",
    "require_generic_attention_backend": "worldfoundry.core.attention.backends.probe",
    "reset_attention_provider_runtime": "worldfoundry.core.attention.backends.dispatch",
    "reshape_rotary_for_broadcast": "worldfoundry.core.attention.rotary.rope_nd",
    "invert_camera_intrinsics": "worldfoundry.core.attention.rotary.projective_rope",
    "invert_se3": "worldfoundry.core.attention.rotary.projective_rope",
    "lift_camera_intrinsics": "worldfoundry.core.attention.rotary.projective_rope",
    "rotary_frequencies": "worldfoundry.core.attention.rotary.rope",
    "rotate_half": "worldfoundry.core.attention.rotary.rope",
    "sequence_parallel_attention_forward": "worldfoundry.core.attention.parallel.sequence_parallel_rope",
    "QKVSelfAttention": "worldfoundry.core.nn.transformer.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.nn.transformer.vit_qkv",
    "CSOHelper": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "UlyssesScheduler": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cp_post_process": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cp_pre_process": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cso_communication": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "scaled_dot_product_attention": "worldfoundry.core.attention.backends.native",
    "resolve_attention_backend": "worldfoundry.core.attention.backends.probe",
    "resolve_transformers_attention_implementation": "worldfoundry.core.attention.backends.probe",
    "varlen_scaled_dot_product_attention": "worldfoundry.core.attention.sequence.varlen",
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from ``_EXPORT_MODULES`` on first access.

    The result is written back into ``globals()`` so later lookups skip
    ``import_module``. Unknown names raise :class:`AttributeError`.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Merge materialized globals with ``__all__`` so completion sees every export."""
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
