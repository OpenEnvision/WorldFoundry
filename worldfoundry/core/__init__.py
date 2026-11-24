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
- ``execution``: inference bootstrap, processes, CUDA graphs and realtime execution.
- ``platform``: accelerator detection, torch backend selection and the persistent compile cache.
- ``configuration``: structured / Python configuration and validation.
- ``acceleration``: reusable cache, token-pruning and low-precision operators.
- ``vram``: parameter placement, offload and encoder lifetime.
- ``memory``: context / semantic memory, retrieval and persistence.
- ``geometry``: camera transforms, conditioning, trajectories and RGBD warping.
- ``observability``: logging, rank-aware output, terminal formatting, timing and profiling.
- ``utils``: general Python object and tensor helpers.

The root retains the public facade and cross-cutting contracts, registries,
environment-name compatibility, and action / prompt input normalization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from worldfoundry.core.observability import (
    log_filters as _log_filters,  # noqa: F401 - Imported to install the log filters.
)

_EXPORT_MODULES = {
    "PromptProcessor": "worldfoundry.core.prompting",
    "assign_state_dict_strict": "worldfoundry.core.model_loading.checkpoints.assignment",
    "adaptive_batched_inference": "worldfoundry.core.execution.inference_runtime",
    "euler_angles_to_rotation_matrix_zyx": "worldfoundry.core.geometry.transforms",
    "compose_horizontal_views": "worldfoundry.core.media.processing.image_utils",
    "split_horizontal_views": "worldfoundry.core.media.processing.image_utils",
    "AttentionBackendInfo": "worldfoundry.core.attention.backends.native",
    "AttentionKernelCapability": "worldfoundry.core.attention.backends.probe",
    "AutoTorchModule": "worldfoundry.core.vram.layers",
    "AutoWrappedLinear": "worldfoundry.core.vram.layers",
    "AutoWrappedModule": "worldfoundry.core.vram.layers",
    "AutoWrappedNonRecurseModule": "worldfoundry.core.vram.layers",
    "CudaSyncTimer": "worldfoundry.core.observability.time",
    "DiffusionModelConfig": "worldfoundry.core.configuration.model_config",
    "DiskMap": "worldfoundry.core.vram.disk_map",
    "FlowMatchScheduler": "worldfoundry.core.nn.diffusion.schedulers",
    "LayerwiseOffloadHandle": "worldfoundry.core.vram.layerwise_offload",
    "LoaderModelConfig": "worldfoundry.core.model_loading.config",
    "DuplicateRegistryKeyError": "worldfoundry.core.registry",
    "DropPath": "worldfoundry.core.nn.blocks.layers",
    "LayerScale": "worldfoundry.core.nn.blocks.layers",
    "Mlp": "worldfoundry.core.nn.blocks.layers",
    "ModelConfig": "worldfoundry.core.model_loading.config",
    "ModelMetaArgs": "worldfoundry.core.attention.sequence.packed_sequence",
    "ModelLoaderRegistry": "worldfoundry.core.model_loading.registry_config",
    "PackedCoreAttnParams": "worldfoundry.core.attention.sequence.packed_sequence",
    "PackedCrossAttnParams": "worldfoundry.core.attention.sequence.packed_sequence",
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
    "TileProcessor": "worldfoundry.core.media.processing.video_tiling",
    "UlyssesScheduler": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "RotaryPositionEmbedding2D": "worldfoundry.core.attention.rotary.rope_2d",
    "apply_nd_rotary_embedding": "worldfoundry.core.attention.rotary.rope_nd",
    "apply_rotary_embedding": "worldfoundry.core.attention.rotary.rope",
    "attention_backend_capability": "worldfoundry.core.attention.backends.probe",
    "attention_backend_from_env": "worldfoundry.core.attention.backends.probe",
    "attention_forward": "worldfoundry.core.attention.backends.dispatch",
    "attention_backend_report": "worldfoundry.core.attention.backends.probe",
    "attention_backend_info": "worldfoundry.core.attention.backends.native",
    "attention_backend_context": "worldfoundry.core.attention.backends.native",
    "attention_head_dim": "worldfoundry.core.nn.transformer.shape",
    "autocast_context": "worldfoundry.core.execution.inference",
    "causal_attention_mask": "worldfoundry.core.nn.transformer.shape",
    "broadcast_dtensor_model_states": "worldfoundry.core.distributed.collectives.device_mesh",
    "clip_grad_norm_": "worldfoundry.core.nn.checkpointing.gradient",
    "clip_grads_with_norm_": "worldfoundry.core.nn.checkpointing.gradient",
    "compile_module_if_enabled": "worldfoundry.core.execution.inference",
    "bind_log_context": "worldfoundry.core.observability.logging_setup",
    "clear_log_context": "worldfoundry.core.observability.logging_setup",
    "configure_logging": "worldfoundry.core.observability.logging_setup",
    "clear_inference_runtime_cache": "worldfoundry.core.execution.runtime_cache",
    "coerce_video_frames": "worldfoundry.core.media.codecs.video",
    "Color": "worldfoundry.core.observability.colors",
    "crop_and_resize": "worldfoundry.core.io.inputs.video_data",
    "CSOHelper": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cp_post_process": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cp_pre_process": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "copy_uri": "worldfoundry.core.io.filesystem.storage",
    "cso_communication": "worldfoundry.core.attention.parallel.context_parallel_runtime",
    "cuda_visible_devices_from_device": "worldfoundry.core.platform.device",
    "destroy_model_parallel": "worldfoundry.core.distributed.model_parallel.groups",
    "dist_init": "worldfoundry.core.distributed.runtime.inference_runtime",
    "dump_serialized": "worldfoundry.core.io.formats.serialization",
    "divide": "worldfoundry.core.utils.python.misc_utils",
    "drop_path": "worldfoundry.core.nn.blocks.layers",
    "enable_layerwise_cpu_offload": "worldfoundry.core.vram.layerwise_offload",
    "env_is_true": "worldfoundry.core.utils.python.misc_utils",
    "exists_uri": "worldfoundry.core.io.filesystem.storage",
    "extract_frames_from_video_url": "worldfoundry.core.media.codecs.video",
    "extract_video_frames_to_directory": "worldfoundry.core.media.codecs.video",
    "file_sha256": "worldfoundry.core.io.filesystem.file_utils",
    "get_1d_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "get_meshgrid_nd": "worldfoundry.core.attention.rotary.rope_nd",
    "get_nd_rotary_pos_embed": "worldfoundry.core.attention.rotary.rope_nd",
    "gpu_supports_flash_attention": "worldfoundry.core.attention.backends.probe",
    "flattened_multihead_attention": "worldfoundry.core.attention.backends.native",
    "hash_model_file": "worldfoundry.core.model_loading.checkpoints.file",
    "hash_state_dict_keys": "worldfoundry.core.model_loading.checkpoints.file",
    "get_log_context": "worldfoundry.core.observability.logging_setup",
    "get_logger": "worldfoundry.core.observability.logging_setup",
    "log_context_environment": "worldfoundry.core.observability.logging_setup",
    "get_video_details": "worldfoundry.core.media.codecs.video",
    "get_local_tensor_if_dtensor": "worldfoundry.core.distributed.collectives.device_mesh",
    "get_total_norm": "worldfoundry.core.nn.checkpointing.gradient",
    "build_rename_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "infer_serialization_format": "worldfoundry.core.io.formats.serialization",
    "init_pp_scheduler": "worldfoundry.core.distributed.model_parallel.pipeline",
    "init_weights_on_device": "worldfoundry.core.vram.initialization",
    "initialize_model_parallel": "worldfoundry.core.distributed.model_parallel.groups",
    "inference_infra_state": "worldfoundry.core.execution.inference",
    "install_worldfoundry_inference_infra": "worldfoundry.core.execution.inference",
    "is_last_rank": "worldfoundry.core.distributed.runtime.inference_runtime",
    "is_last_tp_cp_rank": "worldfoundry.core.distributed.runtime.inference_runtime",
    "is_accelerator_out_of_memory": "worldfoundry.core.execution.inference_runtime",
    "is_remote_uri": "worldfoundry.core.io.filesystem.storage",
    "join_uri": "worldfoundry.core.io.filesystem.storage",
    "layer_scale": "worldfoundry.core.nn.blocks.normalization",
    "layerwise_offload_mutation_scope": "worldfoundry.core.vram.layerwise_offload",
    "load_model": "worldfoundry.core.model_loading.model",
    "load_model_loader_registry": "worldfoundry.core.model_loading.registry_config",
    "load_model_with_disk_offload": "worldfoundry.core.model_loading.model",
    "load_serialized": "worldfoundry.core.io.formats.serialization",
    "load_frames_from_video": "worldfoundry.core.media.codecs.video",
    "load_safetensors_into_model_streaming": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "load_sharded_safetensors_parallel_with_progress": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "load_state_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "load_state_dict_from_safetensors_index": "worldfoundry.core.model_loading.checkpoints.file",
    "load_torch_checkpoint": "worldfoundry.core.model_loading.checkpoints.file",
    "load_torch_state_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "load_video_frames": "worldfoundry.core.media.codecs.video",
    "log_context": "worldfoundry.core.observability.logging_setup",
    "write_jsonl_event": "worldfoundry.core.observability.logging_setup",
    "LowMemoryImageFolder": "worldfoundry.core.io.inputs.video_data",
    "LowMemoryVideo": "worldfoundry.core.io.inputs.video_data",
    "load_pil_image": "worldfoundry.core.media.processing.image_utils",
    "materialize_video_input": "worldfoundry.core.media.codecs.video",
    "materialize_image_input": "worldfoundry.core.media.processing.image_utils",
    "materialize_hf_snapshot": "worldfoundry.core.io.assets.hf",
    "maybe_download_hf_repo_on_rank0": "worldfoundry.core.io.assets.hf",
    "merge_video_audio": "worldfoundry.core.io.inputs.video_data",
    "merge_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "mlp_hidden_size": "worldfoundry.core.nn.transformer.shape",
    "named_apply": "worldfoundry.core.nn.blocks.module_utils",
    "model_parallel_is_initialized": "worldfoundry.core.distributed.model_parallel.groups",
    "normalize_attention_backend": "worldfoundry.core.attention.backends.probe",
    "normalize_action_values": "worldfoundry.core.action_normalization",
    "named_camera_trajectory_tensors": "worldfoundry.core.geometry.trajectory",
    "piecewise_attention": "worldfoundry.core.attention.sparse.piecewise",
    "piecewise_attention_available": "worldfoundry.core.attention.sparse.piecewise",
    "normalize_registry_key": "worldfoundry.core.registry",
    "open_uri": "worldfoundry.core.io.filesystem.storage",
    "parse_uri_scheme": "worldfoundry.core.io.filesystem.storage",
    "package_module_root": "worldfoundry.core.io.paths",
    "patchify_image": "worldfoundry.core.nn.blocks.patching",
    "probe_attention_backends": "worldfoundry.core.attention.backends.probe",
    "read_binary_uri": "worldfoundry.core.io.filesystem.storage",
    "read_image_as_video_tensor": "worldfoundry.core.media.codecs.video",
    "read_text_uri": "worldfoundry.core.io.filesystem.storage",
    "package_module_command": "worldfoundry.core.execution.process",
    "read_text_tail": "worldfoundry.core.execution.process",
    "read_video": "worldfoundry.core.media.codecs.video",
    "read_video_window_rgb": "worldfoundry.core.media.codecs.video",
    "resize_video_tensor_to_resolution": "worldfoundry.core.media.codecs.video",
    "HF_URI_SCHEME": "worldfoundry.core.io.assets.hf",
    "hf_download_or_fpath": "worldfoundry.core.io.assets.hf",
    "resolve_hf_path": "worldfoundry.core.io.assets.hf",
    "resolve_hf_snapshot_path": "worldfoundry.core.io.assets.hf",
    "resolve_local_checkpoint_file": "worldfoundry.core.io.paths",
    "resolve_local_hf_model_path": "worldfoundry.core.io.paths",
    "get_current_torch_device": "worldfoundry.core.platform.device",
    "resolve_inference_device": "worldfoundry.core.platform.device",
    "resolve_inference_batch_size": "worldfoundry.core.execution.inference_runtime",
    "resize_and_letterbox": "worldfoundry.core.media.processing.image_utils",
    "resolve_inference_dtype": "worldfoundry.core.platform.device",
    "run_torchrun_module": "worldfoundry.core.execution.process",
    "run_logged_subprocess": "worldfoundry.core.execution.process",
    "stack_or_pad_tensors": "worldfoundry.core.utils.tensors.batch",
    "terminate_process_group": "worldfoundry.core.execution.process",
    "terminate_process_tree": "worldfoundry.core.execution.process",
    "standardize_quaternion_xyzw": "worldfoundry.core.geometry.transforms",
    "torchrun_module_command": "worldfoundry.core.execution.process",
    "TorchProfileResult": "worldfoundry.core.observability.torchprofile",
    "TORCHPROFILE_SCHEMA_VERSION": "worldfoundry.core.observability.torchprofile",
    "profile_torch_module": "worldfoundry.core.observability.torchprofile",
    "resolve_attention_backend": "worldfoundry.core.attention.backends.probe",
    "resolve_transformers_attention_implementation": "worldfoundry.core.attention.backends.probe",
    "routed_swiglu_moe_pytorch": "worldfoundry.core.kernels.moe",
    "routed_swiglu_moe": "worldfoundry.core.kernels.moe",
    "routed_swiglu_moe_triton": "worldfoundry.core.kernels.triton_moe",
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
    "save_frames": "worldfoundry.core.io.inputs.video_data",
    "save_image_or_video_tensor": "worldfoundry.core.media.codecs.video",
    "save_video": "worldfoundry.core.io.inputs.video_data",
    "save_video_frames": "worldfoundry.core.media.codecs.video",
    "save_video_with_audio": "worldfoundry.core.io.inputs.video_data",
    "search_for_images": "worldfoundry.core.io.inputs.video_data",
    "search_for_embeddings": "worldfoundry.core.model_loading.checkpoints.file",
    "search_for_files": "worldfoundry.core.model_loading.checkpoints.file",
    "search_parameter": "worldfoundry.core.model_loading.checkpoints.file",
    "select_modality_statistics": "worldfoundry.core.action_normalization",
    "as_list": "worldfoundry.core.utils.python.functional_utils",
    "jsonable": "worldfoundry.core.io.formats.serialization",
    "mean_flat": "worldfoundry.core.utils.tensors.torch",
    "ParallelHelper": "worldfoundry.core.media.processing.video_tiling",
    "PPScheduler": "worldfoundry.core.distributed.model_parallel.pipeline",
    "pp_scheduler": "worldfoundry.core.distributed.model_parallel.pipeline",
    "print_per_rank": "worldfoundry.core.observability.rank_logging",
    "print_rank_0": "worldfoundry.core.observability.rank_logging",
    "set_random_seed": "worldfoundry.core.utils.tensors.torch",
    "skip_model_initialization": "worldfoundry.core.vram.initialization",
    "split_state_dict_with_prefix": "worldfoundry.core.model_loading.checkpoints.file",
    "varlen_scaled_dot_product_attention": "worldfoundry.core.attention.sequence.varlen",
    "validate_state_dict_compatibility": "worldfoundry.core.model_loading.checkpoints.assignment",
    "split_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "transformer_shape_spec": "worldfoundry.core.nn.transformer.shape",
    "unpatchify_image": "worldfoundry.core.nn.blocks.patching",
    "unnormalize_action_values": "worldfoundry.core.action_normalization",
    "utc_now_iso": "worldfoundry.core.observability.time",
    "worldfoundry_inference_context": "worldfoundry.core.execution.inference",
    "wrap_runner_for_worldfoundry_core": "worldfoundry.core.execution.inference",
    "video_tensor_to_uint8_frames": "worldfoundry.core.media.codecs.video",
    "VideoData": "worldfoundry.core.io.inputs.video_data",
    "IdentityVideoPostProcessor": "worldfoundry.core.media.processing.postprocess",
    "VideoChunk": "worldfoundry.core.media.processing.postprocess",
    "VideoPostProcessor": "worldfoundry.core.media.processing.postprocess",
    "VideoPostProcessorSession": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessChain": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessStepStats": "worldfoundry.core.media.processing.postprocess",
    "VideoPostprocessStream": "worldfoundry.core.media.processing.postprocess",
    "VideoSpec": "worldfoundry.core.media.processing.postprocess",
    "available_postprocess_presets": "worldfoundry.core.media.processing.presets",
    "build_postprocessor": "worldfoundry.core.media.processing.presets",
    "normalize_postprocess_preset_name": "worldfoundry.core.media.processing.presets",
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
    "write_binary_uri": "worldfoundry.core.io.filesystem.storage",
    "write_text_uri": "worldfoundry.core.io.filesystem.storage",
    "write_video": "worldfoundry.core.media.codecs.video",
    "write_video_torchvision": "worldfoundry.core.media.codecs.video",
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
