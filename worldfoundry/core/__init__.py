"""WorldFoundry first-party primitives with no model identity.

This package is the public facade for :mod:`worldfoundry.core`.
``_EXPORT_MODULES`` maps each exported name to its real submodule; the object
itself is resolved only when ``from worldfoundry.core import Xxx`` or attribute
access runs :func:`__getattr__`, which ``import_module``s the target and writes
the result back into ``globals()``. That keeps:

- ``import worldfoundry.core`` from eagerly pulling torch, Hydra, DCP, or the
  video stack;
- control-plane and CPU-only processes able to import this package safely;
- repeated lookups of the same symbol from calling ``import_module`` again
  (the first successful resolve is cached on the module).

:func:`__dir__` merges already-materialized globals with the lazy ``__all__``
list so completion tools see every public name. ``log_filters`` is the only
import-time side effect: it demotes Inductor autotuner noise and is not
listed in the lazy export table.

Implementation groups:

- ``attention``: attention backends, rotary embeddings, sequence layouts and KV caches.
- ``nn``: reusable neural blocks, transformers, diffusion and latent operations.
- ``distributed``: process setup, collectives, parallel groups and parameter sharding.
- ``model_loading``: model materialization and checkpoint loading / state conversion.
- ``io``: assets, filesystems, structured formats and input datasets.
- ``media``: image / audio / video codecs, processing and artifact previews.
- ``execution``: inference bootstrap, devices, processes, CUDA graphs and realtime execution.
- ``configuration``: structured / Python configuration and validation.
- ``acceleration``: reusable cache, token-pruning and low-precision operators.
- ``vram``: parameter placement, offload and encoder lifetime.
- ``memory``: context / semantic memory, retrieval and persistence.
- ``geometry``: camera transforms, conditioning, trajectories and RGBD warping.
- ``observability``: logging, terminal formatting, timing and profiling.
- ``utils``: general Python object and tensor helpers.

The root retains the public facade and cross-cutting contracts, registries,
environment-name compatibility, and action / prompt input normalization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from worldfoundry.core.observability import log_filters as _log_filters  # noqa: F401

_EXPORT_MODULES = {
    "PromptProcessor": "worldfoundry.core.prompting",
    "assign_state_dict_strict": "worldfoundry.core.model_loading.checkpoints",
    "adaptive_batched_inference": "worldfoundry.core.execution.inference_runtime",
    "batched_image_features": "worldfoundry.core.utils.tensors.features",
    "euler_angles_to_rotation_matrix_zyx": "worldfoundry.core.geometry.transforms",
    "compose_horizontal_views": "worldfoundry.core.utils",
    "split_horizontal_views": "worldfoundry.core.utils",
    "AttentionBackendInfo": "worldfoundry.core.attention.backends.native",
    "AttentionKernelCapability": "worldfoundry.core.attention",
    "AutoTorchModule": "worldfoundry.core.vram",
    "AutoWrappedLinear": "worldfoundry.core.vram",
    "AutoWrappedModule": "worldfoundry.core.vram",
    "AutoWrappedNonRecurseModule": "worldfoundry.core.vram",
    "CudaSyncTimer": "worldfoundry.core.observability.time",
    "DiffusionModelConfig": "worldfoundry.core.configuration",
    "DiskMap": "worldfoundry.core.vram",
    "DTensorFastEmaModelUpdater": "worldfoundry.core.distributed",
    "FlowMatchScheduler": "worldfoundry.core.nn.diffusion.schedulers",
    "LayerwiseOffloadHandle": "worldfoundry.core.vram",
    "LoaderModelConfig": "worldfoundry.core.model_loading",
    "DuplicateRegistryKeyError": "worldfoundry.core.registry",
    "DropPath": "worldfoundry.core.nn.blocks.layers",
    "LayerScale": "worldfoundry.core.nn.blocks.layers",
    "Mlp": "worldfoundry.core.nn.blocks.layers",
    "ModelConfig": "worldfoundry.core.model_loading",
    "ModelMetaArgs": "worldfoundry.core.attention",
    "ModelLoaderRegistry": "worldfoundry.core.model_loading",
    "PackedCoreAttnParams": "worldfoundry.core.attention",
    "PackedCrossAttnParams": "worldfoundry.core.attention",
    "PatchGridSpec": "worldfoundry.core.nn.blocks.patching",
    "PatchEmbed": "worldfoundry.core.nn.blocks.layers",
    "PatchEmbed_Mlp": "worldfoundry.core.nn.blocks.layers",
    "Permute": "worldfoundry.core.nn.blocks.layers",
    "PixelUnshuffle": "worldfoundry.core.nn.blocks.layers",
    "PositionGetter": "worldfoundry.core.attention.rotary.rope_2d",
    "RegistryError": "worldfoundry.core.registry",
    "RegistryItem": "worldfoundry.core.registry",
    "SchedulerInterface": "worldfoundry.core.nn.diffusion.schedulers",
    "TransformerShapeSpec": "worldfoundry.core.nn.transformer.shape",
    "TypedRegistry": "worldfoundry.core.registry",
    "UnknownRegistryKeyError": "worldfoundry.core.registry",
    "InferenceArtifactSpec": "worldfoundry.core.execution.inference",
    "InferenceCheckpointRef": "worldfoundry.core.execution.inference",
    "InferenceFieldSpec": "worldfoundry.core.execution.inference",
    "InferenceParams": "worldfoundry.core.attention.cache.inference_state",
    "InferenceTaskProfile": "worldfoundry.core.execution.inference",
    "InferenceVariantSpec": "worldfoundry.core.execution.inference",
    "ModelInferenceSpec": "worldfoundry.core.execution.inference",
    "WorldFoundryInferenceInfraState": "worldfoundry.core.execution.inference",
    "SwiGLU": "worldfoundry.core.nn.blocks.layers",
    "SwiGLUFFN": "worldfoundry.core.nn.blocks.layers",
    "SwiGLUFFNFused": "worldfoundry.core.nn.blocks.layers",
    "TileProcessor": "worldfoundry.core.io",
    "UlyssesScheduler": "worldfoundry.core.attention",
    "RotaryPositionEmbedding2D": "worldfoundry.core.attention.rotary.rope_2d",
    "apply_nd_rotary_embedding": "worldfoundry.core.attention.rotary.rope_nd",
    "apply_rotary_embedding": "worldfoundry.core.attention.rotary.rope",
    "attention_backend_capability": "worldfoundry.core.attention",
    "attention_backend_from_env": "worldfoundry.core.attention",
    "attention_forward": "worldfoundry.core.attention.backends.dispatch",
    "attention_backend_report": "worldfoundry.core.attention",
    "attention_backend_info": "worldfoundry.core.attention.backends.native",
    "attention_backend_context": "worldfoundry.core.attention.backends.native",
    "attention_head_dim": "worldfoundry.core.nn.transformer.shape",
    "autocast_context": "worldfoundry.core.execution.inference",
    "causal_attention_mask": "worldfoundry.core.nn.transformer.shape",
    "broadcast_dtensor_model_states": "worldfoundry.core.distributed",
    "clip_grad_norm_": "worldfoundry.core.nn.checkpointing.gradient",
    "clip_grads_with_norm_": "worldfoundry.core.nn.checkpointing.gradient",
    "compile_module_if_enabled": "worldfoundry.core.execution.inference",
    "bind_log_context": "worldfoundry.core.observability.logging_setup",
    "clear_log_context": "worldfoundry.core.observability.logging_setup",
    "configure_logging": "worldfoundry.core.observability.logging_setup",
    "clear_inference_runtime_cache": "worldfoundry.core.execution.runtime_cache",
    "coerce_video_frames": "worldfoundry.core.io",
    "Color": "worldfoundry.core.io",
    "crop_and_resize": "worldfoundry.core.io",
    "CSOHelper": "worldfoundry.core.attention",
    "cp_post_process": "worldfoundry.core.attention",
    "cp_pre_process": "worldfoundry.core.attention",
    "copy_uri": "worldfoundry.core.io",
    "cso_communication": "worldfoundry.core.attention",
    "cuda_visible_devices_from_device": "worldfoundry.core.execution.device",
    "destroy_model_parallel": "worldfoundry.core.distributed",
    "dist_init": "worldfoundry.core.distributed",
    "dump_serialized": "worldfoundry.core.io",
    "divide": "worldfoundry.core.utils",
    "drop_path": "worldfoundry.core.nn.blocks.layers",
    "enable_layerwise_cpu_offload": "worldfoundry.core.vram",
    "env_is_true": "worldfoundry.core.utils",
    "exists_uri": "worldfoundry.core.io",
    "extract_frames_from_video_url": "worldfoundry.core.io",
    "extract_video_frames_to_directory": "worldfoundry.core.io",
    "file_sha256": "worldfoundry.core.io",
    "get_1d_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "get_meshgrid_nd": "worldfoundry.core.attention.rotary.rope_nd",
    "get_nd_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "gpu_supports_flash_attention": "worldfoundry.core.attention",
    "flattened_multihead_attention": "worldfoundry.core.attention",
    "hash_model_file": "worldfoundry.core.model_loading",
    "hash_state_dict_keys": "worldfoundry.core.model_loading",
    "get_log_context": "worldfoundry.core.observability.logging_setup",
    "get_logger": "worldfoundry.core.observability.logging_setup",
    "log_context_environment": "worldfoundry.core.observability.logging_setup",
    "get_video_details": "worldfoundry.core.io",
    "get_local_tensor_if_dtensor": "worldfoundry.core.distributed",
    "get_total_norm": "worldfoundry.core.nn.checkpointing.gradient",
    "build_rename_dict": "worldfoundry.core.model_loading",
    "infer_serialization_format": "worldfoundry.core.io",
    "init_pp_scheduler": "worldfoundry.core.distributed",
    "init_weights_on_device": "worldfoundry.core.vram",
    "initialize_model_parallel": "worldfoundry.core.distributed",
    "inference_infra_state": "worldfoundry.core.execution.inference",
    "install_worldfoundry_inference_infra": "worldfoundry.core.execution.inference",
    "is_last_rank": "worldfoundry.core.distributed",
    "is_last_tp_cp_rank": "worldfoundry.core.distributed",
    "is_accelerator_out_of_memory": "worldfoundry.core.execution.inference_runtime",
    "is_remote_uri": "worldfoundry.core.io",
    "join_uri": "worldfoundry.core.io",
    "layer_scale": "worldfoundry.core.nn.blocks.normalization",
    "layerwise_offload_mutation_scope": "worldfoundry.core.vram",
    "load_model": "worldfoundry.core.model_loading",
    "load_model_loader_registry": "worldfoundry.core.model_loading",
    "load_model_with_disk_offload": "worldfoundry.core.model_loading",
    "load_serialized": "worldfoundry.core.io",
    "load_frames_from_video": "worldfoundry.core.io",
    "load_safetensors_into_model_streaming": "worldfoundry.core.model_loading.checkpoints",
    "load_sharded_safetensors_parallel_with_progress": "worldfoundry.core.model_loading.checkpoints",
    "load_state_dict": "worldfoundry.core.model_loading",
    "load_state_dict_from_safetensors_index": "worldfoundry.core.model_loading",
    "load_torch_checkpoint": "worldfoundry.core.model_loading",
    "load_torch_state_dict": "worldfoundry.core.model_loading",
    "load_video_frames": "worldfoundry.core.io",
    "log_context": "worldfoundry.core.observability.logging_setup",
    "write_jsonl_event": "worldfoundry.core.observability.logging_setup",
    "LowMemoryImageFolder": "worldfoundry.core.io",
    "LowMemoryVideo": "worldfoundry.core.io",
    "load_pil_image": "worldfoundry.core.utils",
    "materialize_video_input": "worldfoundry.core.io",
    "materialize_image_input": "worldfoundry.core.utils",
    "materialize_hf_snapshot": "worldfoundry.core.io.assets.hf",
    "maybe_download_hf_repo_on_rank0": "worldfoundry.core.io",
    "merge_video_audio": "worldfoundry.core.io",
    "merge_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "mlp_hidden_size": "worldfoundry.core.nn.transformer.shape",
    "named_apply": "worldfoundry.core.nn.blocks.module_utils",
    "model_parallel_is_initialized": "worldfoundry.core.distributed",
    "normalize_attention_backend": "worldfoundry.core.attention",
    "normalize_action_values": "worldfoundry.core.action_normalization",
    "named_camera_trajectory_tensors": "worldfoundry.core.geometry.trajectory",
    "piecewise_attention": "worldfoundry.core.attention",
    "piecewise_attention_available": "worldfoundry.core.attention",
    "normalize_registry_key": "worldfoundry.core.registry",
    "open_uri": "worldfoundry.core.io",
    "parse_uri_scheme": "worldfoundry.core.io",
    "package_module_root": "worldfoundry.core.io",
    "patchify_image": "worldfoundry.core.nn.blocks.patching",
    "probe_attention_backends": "worldfoundry.core.attention",
    "read_binary_uri": "worldfoundry.core.io",
    "read_image_as_video_tensor": "worldfoundry.core.io",
    "read_text_uri": "worldfoundry.core.io",
    "read_text_tail": "worldfoundry.core.execution.process",
    "read_video": "worldfoundry.core.io",
    "read_video_window_rgb": "worldfoundry.core.io",
    "resize_video_tensor_to_resolution": "worldfoundry.core.io",
    "HF_URI_SCHEME": "worldfoundry.core.io",
    "hf_download_or_fpath": "worldfoundry.core.io",
    "resolve_hf_path": "worldfoundry.core.io",
    "resolve_hf_snapshot_path": "worldfoundry.core.io",
    "resolve_local_checkpoint_file": "worldfoundry.core.io",
    "resolve_local_hf_model_path": "worldfoundry.core.io",
    "get_current_torch_device": "worldfoundry.core.execution.device",
    "resolve_inference_device": "worldfoundry.core.execution.device",
    "resolve_inference_batch_size": "worldfoundry.core.execution.inference_runtime",
    "resize_and_letterbox": "worldfoundry.core.utils",
    "resolve_inference_dtype": "worldfoundry.core.execution.device",
    "run_torchrun_module": "worldfoundry.core.execution.process",
    "run_logged_subprocess": "worldfoundry.core.execution.process",
    "stack_or_pad_tensors": "worldfoundry.core.utils",
    "terminate_process_group": "worldfoundry.core.execution.process",
    "terminate_process_tree": "worldfoundry.core.execution.process",
    "standardize_quaternion_xyzw": "worldfoundry.core.geometry.transforms",
    "torchrun_module_command": "worldfoundry.core.execution.process",
    "TorchProfileResult": "worldfoundry.core.observability.torchprofile",
    "TORCHPROFILE_SCHEMA_VERSION": "worldfoundry.core.observability.torchprofile",
    "profile_torch_module": "worldfoundry.core.observability.torchprofile",
    "resolve_attention_backend": "worldfoundry.core.attention",
    "resolve_transformers_attention_implementation": "worldfoundry.core.attention",
    "routed_swiglu_moe_pytorch": "worldfoundry.core.kernels",
    "routed_swiglu_moe": "worldfoundry.core.kernels",
    "routed_swiglu_moe_triton": "worldfoundry.core.kernels",
    "rms_norm": "worldfoundry.core.nn.blocks.normalization",
    "ray_condition": "worldfoundry.core.geometry.transforms",
    "quaternion_xyzw_to_rotation_matrix": "worldfoundry.core.geometry.transforms",
    "rotation_matrix_to_euler_angles_opencv": "worldfoundry.core.geometry.transforms",
    "rotation_matrix_to_euler_angles_zyx": "worldfoundry.core.geometry.transforms",
    "rotation_matrix_to_quaternion_wxyz": "worldfoundry.core.geometry.transforms",
    "rotation_matrix_to_quaternion_xyzw": "worldfoundry.core.geometry.transforms",
    "rotary_frequencies": "worldfoundry.core.attention.rotary.rope",
    "rotate_half": "worldfoundry.core.attention.rotary.rope",
    "scaled_dot_product_attention": "worldfoundry.core.attention.backends.native",
    "save_frames": "worldfoundry.core.io",
    "save_image_or_video_tensor": "worldfoundry.core.io",
    "save_video": "worldfoundry.core.io",
    "save_video_frames": "worldfoundry.core.io",
    "save_video_with_audio": "worldfoundry.core.io",
    "search_for_images": "worldfoundry.core.io",
    "search_for_embeddings": "worldfoundry.core.model_loading",
    "search_for_files": "worldfoundry.core.model_loading",
    "search_parameter": "worldfoundry.core.model_loading",
    "select_modality_statistics": "worldfoundry.core.action_normalization",
    "as_list": "worldfoundry.core.utils",
    "jsonable": "worldfoundry.core.io",
    "mean_flat": "worldfoundry.core.utils",
    "mean_pairwise_cosine_distance": "worldfoundry.core.utils.tensors.features",
    "ParallelHelper": "worldfoundry.core.io",
    "PPScheduler": "worldfoundry.core.distributed",
    "pp_scheduler": "worldfoundry.core.distributed",
    "print_per_rank": "worldfoundry.core.distributed",
    "print_rank_0": "worldfoundry.core.distributed",
    "set_random_seed": "worldfoundry.core.utils",
    "skip_model_initialization": "worldfoundry.core.vram",
    "split_state_dict_with_prefix": "worldfoundry.core.model_loading",
    "varlen_scaled_dot_product_attention": "worldfoundry.core.attention.sequence.varlen",
    "validate_state_dict_compatibility": "worldfoundry.core.model_loading.checkpoints",
    "split_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "transformer_shape_spec": "worldfoundry.core.nn.transformer.shape",
    "unpatchify_image": "worldfoundry.core.nn.blocks.patching",
    "unnormalize_action_values": "worldfoundry.core.action_normalization",
    "utc_now_iso": "worldfoundry.core.observability.time",
    "worldfoundry_inference_context": "worldfoundry.core.execution.inference",
    "wrap_runner_for_worldfoundry_core": "worldfoundry.core.execution.inference",
    "video_tensor_to_uint8_frames": "worldfoundry.core.io",
    "VideoData": "worldfoundry.core.io",
    "IdentityVideoPostProcessor": "worldfoundry.core.media.processing.postprocess",
    "VideoChunk": "worldfoundry.core.media.processing.postprocess",
    "VideoPostProcessor": "worldfoundry.core.media.processing.postprocess",
    "VideoPostProcessorSession": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessChain": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessStepStats": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessStream": "worldfoundry.core.media.processing.postprocess",
    "VideoSpec": "worldfoundry.core.media.processing.postprocess",
    "RTX_POSTPROCESS_PRESET_NAMES": "worldfoundry.core.media.processing.rtx",
    "RTXVFXCapability": "worldfoundry.core.media.processing.rtx",
    "RTXVFXProbeStatus": "worldfoundry.core.media.processing.rtx",
    "RTXVideoFloatRange": "worldfoundry.core.media.processing.rtx",
    "RTXVideoSuperResolutionConfig": "worldfoundry.core.media.processing.rtx",
    "RTXVideoSuperResolutionPostProcessor": "worldfoundry.core.media.processing.rtx",
    "RTXVideoSuperResolutionQuality": "worldfoundry.core.media.processing.rtx",
    "inspect_rtx_vfx_capability": "worldfoundry.core.media.processing.rtx",
    "probe_rtx_vfx_runtime": "worldfoundry.core.media.processing.rtx",
    "require_rtx_vfx_runtime": "worldfoundry.core.media.processing.rtx",
    "rtx_postprocessor_from_preset": "worldfoundry.core.media.processing.rtx",
    "write_binary_uri": "worldfoundry.core.io",
    "write_text_uri": "worldfoundry.core.io",
    "write_video": "worldfoundry.core.io",
    "write_video_torchvision": "worldfoundry.core.io",
    "zero_module": "worldfoundry.core.nn.blocks.layers",
}


def __getattr__(name: str) -> Any:
    """Resolve ``name`` from ``_EXPORT_MODULES`` and cache it on this module."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    # Cache on globals so the next attribute access skips import_module.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Merge materialized globals with lazy ``__all__`` for completion and ``dir()``."""
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
