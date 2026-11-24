"""Native LTX audio-video denoiser component.

Adapts the official LTX velocity transformer to the multi-modal
``Denoiser`` contract used by the joint AV runner.  Config comes from
safetensors metadata via :func:`~...loaders.metadata.safetensors_json_metadata`.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from threading import RLock
from types import SimpleNamespace
from typing import Any

import torch

from worldfoundry.core.nn.transformer.dit import velocity_to_denoised

from ...components import ComponentBuildContext
from ...contracts import MultiModalDenoiserInput, MultiModalDenoiserOutput
from ...loaders import (
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
    safetensors_json_metadata,
)
from ...optimizations import OffloadMode, QuantizationMode, RuntimePolicy
from ..networks.ltx.fused_ops import LTX_DEFAULT_FP8_EXCLUDES, enable_ltx_fused_fp8_ffn
from ..networks.ltx.modality import Modality
from ..networks.ltx.model import LTXModel, resolve_cached_video_grid_shape
from ..networks.ltx.transformer import BasicAVTransformerBlock
from .graph_wrapped import (
    FeatureCacheDenoiserMixin,
    GraphWrappedDenoiserMixin,
    resolve_cuda_graph_mode,
    resolve_teacache_threshold,
    validate_cuda_graph_options,
)
from .ltx_configurator import LTXModelConfigurator, LTXVideoOnlyModelConfigurator


class LTXAVTransformerModule(torch.nn.Module):
    """Checkpoint-configured LTX transformer with a stable loader surface."""

    CONFIGURATOR = LTXModelConfigurator

    def __init__(self, checkpoint_config: Mapping[str, object], *, split_checkpoint: bool = False) -> None:
        super().__init__()
        self.velocity_model = (
            self.CONFIGURATOR.from_metadata({"config": dict(checkpoint_config)})
            if split_checkpoint
            else self.CONFIGURATOR.from_config(dict(checkpoint_config))
        )

    def forward(self, *args, **kwargs):
        return self.velocity_model(*args, **kwargs)


class LTXVideoTransformerModule(torch.nn.Module):
    """Checkpoint-configured LTX-Video transformer."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.velocity_model = LTXVideoOnlyModelConfigurator.from_config(dict(checkpoint_config))

    def forward(self, *args, **kwargs):
        return self.velocity_model(*args, **kwargs)


def convert_ltx_transformer_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    """Select transformer weights and strip the official container prefix."""

    prefix = "model.diffusion_model."
    excluded = (
        f"{prefix}video_embeddings_connector.",
        f"{prefix}audio_embeddings_connector.",
    )
    keys = {
        f"velocity_model.{key.removeprefix(prefix)}": key
        for key in state_dict
        if key.startswith(prefix) and not key.startswith(excluded)
    }
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        return TransformedStateDict(state_dict, keys)
    return {destination: state_dict[source] for destination, source in keys.items()}


def convert_ltx_video_transformer_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map an official LTX-Video 0.9.x transformer into the native module."""

    prefix = "model.diffusion_model."
    return {
        f"velocity_model.{key.removeprefix(prefix)}": state_dict[key] for key in state_dict if key.startswith(prefix)
    }


def _feature_cache_input_view(model_input: MultiModalDenoiserInput) -> Any:
    """Adapt ``MultiModalDenoiserInput`` to ``FeatureCacheDenoiserMixin``'s explicit ``request_id`` contract."""
    if getattr(model_input, "request_id", None) is not None:
        return model_input
    request_id = model_input.conditioning.get("request_id")
    if isinstance(request_id, str) and request_id.strip():
        return SimpleNamespace(
            request_id=request_id,
            branch=model_input.branch,
            step_index=model_input.step_index,
            total_steps=model_input.total_steps,
            conditioning=model_input.conditioning,
        )
    return model_input


_APPROXIMATE_ATTENTION_BACKENDS = frozenset({"sol_attn", "pisa", "sage_attention", "sage_attention_3", "cudnn_fp8"})


def _normalize_ltx_cuda_graph_mode(enable_cuda_graph: bool | str) -> bool | str:
    """Normalize ``enable_cuda_graph`` so ``'off'``/``'false'`` truly disable CUDA Graphs."""
    if isinstance(enable_cuda_graph, bool):
        return enable_cuda_graph
    normalized = str(enable_cuda_graph).strip().casefold()
    if normalized in {"", "false", "0", "off", "none", "disabled"}:
        return False
    if normalized == "piecewise":
        return "piecewise"
    if normalized in {"whole", "full", "true", "1"}:
        return True
    raise ValueError(
        f"unsupported cuda_graph mode {enable_cuda_graph!r}; expected bool, 'off', 'whole', or 'piecewise'"
    )


def _validate_ltx_graph_attention_compatibility(model: torch.nn.Module, enable_cuda_graph: bool | str) -> None:
    """Reject attention backend options that cannot execute inside CUDA Graph capture."""
    if _normalize_ltx_cuda_graph_mode(enable_cuda_graph) is False:
        return

    from ..networks.ltx.attention import Attention

    for child in model.modules():
        if not isinstance(child, Attention):
            continue
        backend = child.attention_backend
        opts = child._attention_backend_options or {}
        if backend == "cudnn_fp8" and bool(opts.get("dynamic_scale", False)):
            raise ValueError(
                "cudnn_fp8 with dynamic_scale=True requires host synchronization and cannot be used "
                "with CUDA Graph capture; use static calibrated descale_q/descale_k/descale_v "
                "or default unit scaling"
            )
        if backend == "scaled_int8_fp8":
            raise ValueError("scaled_int8_fp8 attention does not support CUDA Graph capture")


def _with_ltx_default_fp8_excludes(policy: RuntimePolicy) -> RuntimePolicy:
    """Merge LTX-calibrated FP8 exclusions into ``policy`` so stem/AdaLN/head stay dense."""
    if policy.quantization.mode is not QuantizationMode.FP8:
        return policy
    merged = tuple(dict.fromkeys((*LTX_DEFAULT_FP8_EXCLUDES, *policy.quantization.exclude)))
    if merged == policy.quantization.exclude:
        return policy
    return replace(policy, quantization=replace(policy.quantization, exclude=merged))


def _resolve_ltx_fused_ops(context: ComponentBuildContext) -> bool:
    options = context.policy.options
    comp = context.component_options
    return bool(
        comp.get(
            "ltx_fused_ops",
            comp.get("fused_rope", options.get("ltx_fused_ops", options.get("fused_rope", False))),
        )
    )


def _parse_dense_blocks(raw: object) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError("dense_blocks must be a sequence of block indices")
    parsed: list[int] = []
    for item in raw:  # type: ignore[union-attr]
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"dense_blocks entries must be integers, got {item!r}")
        parsed.append(int(item))
    return tuple(parsed)


def _parse_tile_3d(raw: object) -> tuple[int, int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError("Sol-Attn tile_3d must be a 3-tuple of positive integers")
    vals = list(raw)  # type: ignore[arg-type]
    if len(vals) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in vals):
        raise ValueError("Sol-Attn tile_3d must contain three positive integers (tf, th, tw)")
    return (int(vals[0]), int(vals[1]), int(vals[2]))


_TensorCacheEntry = tuple[
    weakref.ref[torch.Tensor],
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    bool,
]


def _evict_inference_cache_entries(cache: dict[int, _TensorCacheEntry]) -> None:
    """Drop inference-mode entries (which lack ``_version``) at request/stage boundaries."""
    stale = [key for key, entry in cache.items() if entry[0]() is None or entry[3] < 0]
    for key in stale:
        cache.pop(key, None)


def _tensor_cache_lookup(
    cache: dict[int, _TensorCacheEntry],
    tensor: torch.Tensor,
    *,
    allow_inference_cache: bool = False,
) -> bool | None:
    is_inf = tensor.is_inference()
    if is_inf and not allow_inference_cache:
        return None
    entry = cache.get(id(tensor))
    if entry is None:
        return None
    ref, cached_ptr, cached_off, cached_ver, cached_shape, cached_stride, cached_dtype, cached_val = entry
    ptr = int(tensor.data_ptr())
    offset = int(tensor.storage_offset())
    version = -1 if is_inf else int(tensor._version)
    shape = tuple(int(dim) for dim in tensor.shape)
    stride = tuple(int(step) for step in tensor.stride())
    if (
        ref() is tensor
        and cached_ptr == ptr
        and cached_off == offset
        and cached_ver == version
        and cached_shape == shape
        and cached_stride == stride
        and cached_dtype == tensor.dtype
    ):
        return cached_val
    return None


def _tensor_cache_store(
    cache: dict[int, _TensorCacheEntry],
    tensor: torch.Tensor,
    value: bool,
    *,
    allow_inference_cache: bool = False,
) -> bool:
    is_inf = tensor.is_inference()
    if is_inf and not allow_inference_cache:
        return value
    if len(cache) >= 32:
        dead = [k for k, v in cache.items() if v[0]() is None]
        for k in dead:
            cache.pop(k, None)
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
    ptr = int(tensor.data_ptr())
    offset = int(tensor.storage_offset())
    version = -1 if is_inf else int(tensor._version)
    shape = tuple(int(dim) for dim in tensor.shape)
    stride = tuple(int(step) for step in tensor.stride())
    cache[id(tensor)] = (weakref.ref(tensor), ptr, offset, version, shape, stride, tensor.dtype, value)
    return value


def _is_uniform_denoise_mask(
    cache: dict[int, _TensorCacheEntry],
    mask: torch.Tensor,
    *,
    allow_inference_cache: bool = False,
) -> bool:
    """Return True when ``mask`` is constant along the token axis (``dim=1``), cached outside graph capture.

    When uniform (all T2V steps and full-video refinement steps), collapsing ``(B, T, 1)``
    to ``(B, 1, 1)`` avoids materializing ``T`` identical copies of AdaLN tables across all
    48 transformer blocks (saving ~60 GB/step of HBM traffic in Stage 2) while preserving
    exact per-token timesteps whenever ``denoise_mask`` varies across tokens (e.g. I2V).
    """
    if mask.ndim < 2 or mask.shape[1] <= 1 or mask.numel() == 0:
        return False
    if torch.is_grad_enabled() or (mask.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
        return False
    cached = _tensor_cache_lookup(cache, mask, allow_inference_cache=allow_inference_cache)
    if cached is not None:
        return cached
    uniform = bool((mask == mask[:, :1]).all().item())
    return _tensor_cache_store(cache, mask, uniform, allow_inference_cache=allow_inference_cache)


def _is_trivial_context_mask(
    cache: dict[int, _TensorCacheEntry],
    mask: torch.Tensor | None,
    *,
    allow_inference_cache: bool = False,
) -> bool:
    """Return True when ``mask`` leaves every key token unmasked (all-1s int/bool or all-0s float bias).

    Dropping a mathematically no-op mask unlocks unmasked Hopper FlashAttention-3 / Flash
    SDPA on text cross-attention instead of falling back to SM80 memory-efficient attention.
    """
    if mask is None or mask.numel() == 0:
        return False
    if torch.is_grad_enabled() or (mask.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
        return False
    cached = _tensor_cache_lookup(cache, mask, allow_inference_cache=allow_inference_cache)
    if cached is not None:
        return cached
    if mask.dtype == torch.bool:
        trivial = bool(mask.all().item())
    elif torch.is_floating_point(mask):
        trivial = bool((mask == 0).all().item())
    else:
        trivial = bool((mask == 1).all().item())
    return _tensor_cache_store(cache, mask, trivial, allow_inference_cache=allow_inference_cache)


def _resolve_teacache_max_tokens(context: ComponentBuildContext) -> int | None:
    raw = context.component_options.get("teacache_max_tokens", context.policy.options.get("teacache_max_tokens"))
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"teacache_max_tokens must be a positive integer, got {raw!r}")
    return int(raw)


def _parse_fbcache_config(
    raw: object,
    *,
    max_tokens_override: int | None = None,
) -> Any:
    """Normalize an opt-in LTX FirstBlockCache configuration (bool, float, mapping, or dataclass)."""
    if raw is None or raw is False:
        return None
    from ..networks.ltx.acceleration import LTXFirstBlockCacheConfig

    if isinstance(raw, LTXFirstBlockCacheConfig):
        cfg = raw
    elif raw is True:
        cfg = LTXFirstBlockCacheConfig()
    elif isinstance(raw, (int, float)):
        cfg = LTXFirstBlockCacheConfig(residual_diff_threshold=float(raw))
    elif isinstance(raw, Mapping):
        opts = dict(raw)
        if "threshold" in opts and "residual_diff_threshold" not in opts:
            opts["residual_diff_threshold"] = opts.pop("threshold")
        cfg = LTXFirstBlockCacheConfig(**opts)
    else:
        raise TypeError(f"fbcache option must be bool, float, Mapping, or LTXFirstBlockCacheConfig, got {raw!r}")
    if max_tokens_override is not None:
        cfg = replace(cfg, max_tokens=max_tokens_override)
    return cfg


def _resolve_fbcache_max_tokens(context: ComponentBuildContext) -> int | None:
    """Resolve optional ``fbcache_max_tokens`` override from component or policy options."""
    comp = context.component_options
    opts = context.policy.options
    raw_max = comp.get("fbcache_max_tokens", opts.get("fbcache_max_tokens"))
    if raw_max is None:
        return None
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max <= 0:
        raise ValueError(f"fbcache_max_tokens must be a positive integer, got {raw_max!r}")
    return int(raw_max)


def _resolve_fbcache_config(context: ComponentBuildContext) -> Any:
    """Resolve optional FirstBlockCache config from component/policy options."""
    comp = context.component_options
    opts = context.policy.options
    raw = comp.get("fbcache", comp.get("first_block_cache", opts.get("fbcache", opts.get("first_block_cache"))))
    max_tokens = _resolve_fbcache_max_tokens(context)
    if raw is None or raw is False:
        return None
    if isinstance(raw, (bool, int, float)):
        extra: dict[str, Any] = {}
        if not isinstance(raw, bool):
            extra["residual_diff_threshold"] = float(raw)
        for opt_key, field_name in (
            ("fbcache_dense_first", "dense_first"),
            ("fbcache_dense_last", "dense_last"),
            ("fbcache_max_consecutive_skips", "max_consecutive_skips"),
            ("fbcache_downsample_factor", "downsample_factor"),
            ("fbcache_cumulative", "cumulative"),
            ("fbcache_min_steps", "min_steps_to_enable"),
        ):
            val = comp.get(opt_key, opts.get(opt_key))
            if val is not None:
                extra[field_name] = val
        if max_tokens is not None:
            extra["max_tokens"] = max_tokens
        return _parse_fbcache_config(extra)
    return _parse_fbcache_config(raw, max_tokens_override=max_tokens)


_ProtectedCacheEntry = tuple[
    weakref.ref[torch.Tensor],
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    weakref.ref[torch.Tensor] | None,
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype | None,
    torch.Tensor | None,
    int,
]


def _evict_protected_cache_entries(cache: dict[tuple[int, int], _ProtectedCacheEntry]) -> None:
    stale = [
        key
        for key, entry in cache.items()
        if entry[0]() is None or entry[3] < 0 or (entry[7] is not None and (entry[7]() is None or entry[10] < 0))
    ]
    for key in stale:
        cache.pop(key, None)


def _resolve_protected_tokens(
    cache: dict[tuple[int, int], _ProtectedCacheEntry],
    denoise_mask: torch.Tensor,
    keyframes_mask: torch.Tensor | None,
    *,
    uniform_denoise_mask: bool,
    allow_inference_cache: bool = False,
) -> tuple[torch.Tensor | None, int]:
    """Resolve and cache ``(protected_mask, protected_count)`` for I2V / keyframe tokens outside graph capture."""
    token_count = int(denoise_mask.shape[1]) if denoise_mask.ndim >= 2 else 0
    if token_count <= 0:
        return None, 0
    if torch.is_grad_enabled() or (denoise_mask.device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
        return None, 0

    key = (id(denoise_mask), 0 if keyframes_mask is None else id(keyframes_mask))
    is_inf = denoise_mask.is_inference() or (keyframes_mask is not None and keyframes_mask.is_inference())
    ptr = int(denoise_mask.data_ptr())
    offset = int(denoise_mask.storage_offset())
    ver = -1 if is_inf else int(denoise_mask._version)
    shape = tuple(int(dim) for dim in denoise_mask.shape)
    stride = tuple(int(step) for step in denoise_mask.stride())
    if keyframes_mask is None:
        kf_ptr, kf_off, kf_ver, kf_shape, kf_stride, kf_dtype = 0, 0, 0, (), (), None
    else:
        kf_ptr = int(keyframes_mask.data_ptr())
        kf_off = int(keyframes_mask.storage_offset())
        kf_ver = -1 if keyframes_mask.is_inference() else int(keyframes_mask._version)
        kf_shape = tuple(int(dim) for dim in keyframes_mask.shape)
        kf_stride = tuple(int(step) for step in keyframes_mask.stride())
        kf_dtype = keyframes_mask.dtype

    if not is_inf or allow_inference_cache:
        entry = cache.get(key)
        if entry is not None:
            (
                ref,
                c_ptr,
                c_off,
                c_ver,
                c_shape,
                c_stride,
                c_dtype,
                kf_ref,
                c_kf_ptr,
                c_kf_off,
                c_kf_ver,
                c_kf_shape,
                c_kf_stride,
                c_kf_dtype,
                c_mask,
                c_cnt,
            ) = entry
            kf_ok = (keyframes_mask is None and kf_ref is None) or (
                keyframes_mask is not None
                and kf_ref is not None
                and kf_ref() is keyframes_mask
                and c_kf_ptr == kf_ptr
                and c_kf_off == kf_off
                and c_kf_ver == kf_ver
                and c_kf_shape == kf_shape
                and c_kf_stride == kf_stride
                and c_kf_dtype == kf_dtype
            )
            if (
                ref() is denoise_mask
                and c_ptr == ptr
                and c_off == offset
                and c_ver == ver
                and c_shape == shape
                and c_stride == stride
                and c_dtype == denoise_mask.dtype
                and kf_ok
            ):
                return c_mask, c_cnt

    if uniform_denoise_mask:
        # Even a constant denoise_mask can be < 1.0 (e.g. all-0 frozen/conditioned latents).
        all_conditioned = bool((denoise_mask.detach().float().reshape(-1)[0] < (1.0 - 1e-6)).item())
        if all_conditioned:
            mask = torch.ones(token_count, device=denoise_mask.device, dtype=torch.bool)
            count = token_count
            result_mask: torch.Tensor | None = mask
        elif keyframes_mask is None:
            count = 0
            result_mask = None
        else:
            mask = torch.zeros(token_count, device=denoise_mask.device, dtype=torch.bool)
            if keyframes_mask.ndim >= 2 and int(keyframes_mask.shape[1]) == token_count:
                mask = mask | (keyframes_mask.detach() > 0).movedim(1, 0).reshape(token_count, -1).any(dim=1).to(
                    device=denoise_mask.device
                )
            count = int(mask.sum().item())
            result_mask = mask if count > 0 else None
    else:
        mask = (denoise_mask.detach().float() < (1.0 - 1e-6)).movedim(1, 0).reshape(token_count, -1).any(dim=1)
        if keyframes_mask is not None and keyframes_mask.ndim >= 2 and int(keyframes_mask.shape[1]) == token_count:
            mask = mask | (keyframes_mask.detach() > 0).movedim(1, 0).reshape(token_count, -1).any(dim=1).to(
                device=denoise_mask.device
            )
        count = int(mask.sum().item())
        result_mask = mask if count > 0 else None

    if not is_inf or allow_inference_cache:
        if len(cache) >= 32:
            dead = [k for k, v in cache.items() if v[0]() is None or (v[7] is not None and v[7]() is None)]
            for k in dead:
                cache.pop(k, None)
            if len(cache) >= 32:
                cache.pop(next(iter(cache)))
        cache[key] = (
            weakref.ref(denoise_mask),
            ptr,
            offset,
            ver,
            shape,
            stride,
            denoise_mask.dtype,
            None if keyframes_mask is None else weakref.ref(keyframes_mask),
            kf_ptr,
            kf_off,
            kf_ver,
            kf_shape,
            kf_stride,
            kf_dtype,
            result_mask,
            count,
        )
    return result_mask, count


def _unwrap_velocity_model(model: Any) -> Any:
    """Return the inner velocity model whether wrapped or passed directly."""
    return getattr(model, "velocity_model", model)


def _token_pruning_kwargs(
    velocity_model: LTXModel,
    denoise_mask: torch.Tensor,
    keyframes_mask: torch.Tensor | None,
    *,
    step_index: int,
    total_steps: int | None,
    uniform_denoise_mask: bool,
    protected_cache: dict[tuple[int, int], _ProtectedCacheEntry],
) -> dict[str, Any]:
    """Compute graph-safe token pruning step eligibility and protected-token kwargs outside capture."""
    tp_cfg = getattr(velocity_model, "_worldfoundry_token_pruning_config", None)
    if tp_cfg is None:
        return {}
    step_ok, step_reason = tp_cfg.is_step_eligible(int(step_index), total_steps)
    if (
        not step_ok
        and step_reason == "distilled_short_step_guard"
        and float(tp_cfg.keep_ratio) < 1.0
        and denoise_mask.ndim >= 2
        and int(denoise_mask.shape[1]) >= int(tp_cfg.min_tokens)
    ):
        from ..networks.ltx.acceleration import record_distilled_pruning_guard

        tp_rt = getattr(velocity_model, "_worldfoundry_token_pruning_runtime", None)
        record_distilled_pruning_guard(tp_cfg, tp_rt, total_steps=total_steps)
    kwargs: dict[str, Any] = {"_token_prune_step_eligible": step_ok}
    if step_ok and tp_cfg.preserve_conditioning_tokens:
        pmask, pcount = _resolve_protected_tokens(
            protected_cache,
            denoise_mask,
            keyframes_mask,
            uniform_denoise_mask=uniform_denoise_mask,
            allow_inference_cache=True,
        )
        if pmask is not None:
            kwargs["_video_protected_mask"] = pmask
            kwargs["_video_protected_count"] = pcount
    return kwargs


def _sol_schedule_kwargs(
    velocity_model: LTXModel,
    positions: torch.Tensor,
    step_index: int,
    grid_cache: dict[int, Any],
) -> dict[str, Any]:
    """Resolve zero-sync sparse-attention (Sol-Attn / PISA) schedule kwargs outside CUDA Graph capture."""
    schedule = getattr(velocity_model, "_worldfoundry_sol_schedule", None)
    if not isinstance(schedule, Mapping):
        return {}
    kwargs: dict[str, Any] = {}
    if schedule.get("tile_3d") is not None:
        grid_shape = resolve_cached_video_grid_shape(grid_cache, positions)
        if grid_shape is not None:
            kwargs["_video_grid_shape"] = grid_shape
    dense_steps = int(schedule.get("dense_steps", 0))
    if dense_steps > 0:
        kwargs["_sol_force_dense"] = int(step_index) < dense_steps
    return kwargs


def _apply_ltx_post_load_accelerations(
    model: LTXAVTransformerModule | LTXVideoTransformerModule,
    context: ComponentBuildContext,
) -> None:
    """Apply LTX-specific post-load fused ops, FP8 FFN fusion, sparse attention, pruning, and FBCache."""
    options = context.policy.options
    comp = context.component_options
    if _resolve_ltx_fused_ops(context):
        model.velocity_model.enable_fused_ops()

    fused_fp8_ffn_blocks = 0
    if bool(context.policy.quantization.options.get("fuse_fp8_ffn", False)):
        fused_fp8_ffn_blocks = enable_ltx_fused_fp8_ffn(model.velocity_model)

    applied = getattr(model, "_worldfoundry_applied_optimizations", None)
    if applied is not None:
        fusion_stats = model.__dict__.get("_worldfoundry_ltx_fusion_stats")
        if isinstance(fusion_stats, Mapping):
            fused_qkv = int(fusion_stats.get("fused_qkv", 0))
            fused_kv = int(fusion_stats.get("fused_kv", 0))
            if fused_qkv > 0:
                applied.effective["fuse_qkv_blocks"] = fused_qkv
            if fused_kv > 0:
                applied.effective["fuse_cross_kv_blocks"] = fused_kv
        if fused_fp8_ffn_blocks > 0:
            applied.effective["fused_fp8_ffn_blocks"] = fused_fp8_ffn_blocks

    accelerations = comp["accelerations"] if "accelerations" in comp else options.get("accelerations")
    if accelerations is None:
        return
    if not isinstance(accelerations, Mapping):
        raise TypeError("accelerations must be a mapping of plugin names to options")
    active_plugins = {name: value for name, value in accelerations.items() if value is not None and value is not False}
    if not active_plugins:
        return

    allowed_plugins = {"attention_policy", "token_pruning", "fbcache", "first_block_cache"}
    unsupported = sorted(set(active_plugins) - allowed_plugins)
    if unsupported:
        raise ValueError(f"LTX denoiser does not support requested acceleration plugins: {unsupported}")
    if "fbcache" in active_plugins and "first_block_cache" in active_plugins:
        raise ValueError("fbcache and first_block_cache cannot both be specified in accelerations")

    installed_entries: list[dict[str, Any]] = []
    any_approximate = False

    if "attention_policy" in active_plugins:
        attn_policy = active_plugins["attention_policy"]
        if not isinstance(attn_policy, Mapping):
            raise TypeError("attention_policy must be a mapping of scope names to backend configs")
        if set(attn_policy) - {"self", "cross"} or not attn_policy:
            raise ValueError("attention_policy requires self and/or cross options")

        from worldfoundry.core.attention.backends.probe import resolve_attention_backend

        from ..networks.ltx.attention import _normalize_ltx_attention_backend, _resolve_pisa_default_min_sequence

        scopes_report: dict[str, dict[str, Any]] = {}
        for scope, raw in attn_policy.items():
            if not isinstance(raw, (str, Mapping)):
                raise TypeError("attention_policy scopes require a backend string or mapping")
            values = {"backend": raw} if isinstance(raw, str) else dict(raw)
            if "backend" not in values:
                raise ValueError("attention_policy scopes require a backend")
            requested = _normalize_ltx_attention_backend(str(values.pop("backend")))
            resolved = "pisa" if requested == "pisa" else resolve_attention_backend(requested, context.policy.device)

            dense_blocks: tuple[int, ...] = ()
            dense_backend: str | None = None
            schedule_report: dict[str, Any] | None = None
            if resolved in {"sol_attn", "pisa", "cudnn_fp8"} and "dense_backend" in values:
                raw_dense_backend = values.pop("dense_backend")
                if raw_dense_backend is not None:
                    dense_backend = resolve_attention_backend(
                        _normalize_ltx_attention_backend(str(raw_dense_backend)),
                        context.policy.device,
                    )
            if resolved in {"sol_attn", "pisa"}:
                backend_label = "Sol-Attn" if resolved == "sol_attn" else "PISA"
                if scope != "self":
                    raise ValueError(f"{backend_label} is supported only for unmasked self-attention")
                if resolved == "sol_attn" and context.policy.dtype != torch.bfloat16:
                    raise ValueError(f"Sol-Attn requires BF16 inference, got dtype={context.policy.dtype}")
                raw_dense_steps = values.pop("dense_steps", 0)
                if isinstance(raw_dense_steps, bool) or not isinstance(raw_dense_steps, int) or raw_dense_steps < 0:
                    raise ValueError(f"{backend_label} dense_steps must be a non-negative integer")
                num_blocks = len(model.velocity_model.transformer_blocks)
                if "dense_blocks" in values:
                    raw_dense_blocks = values.pop("dense_blocks")
                    dense_blocks = _parse_dense_blocks(raw_dense_blocks)
                elif resolved == "pisa" and num_blocks >= 4:
                    dense_blocks = (0, 1, -1)
                parsed_tile_3d = _parse_tile_3d(values.pop("tile_3d", None))
                min_tokens_explicit = "min_tokens" in values
                raw_min_tokens = values.get("min_tokens", 0)
                if isinstance(raw_min_tokens, bool) or not isinstance(raw_min_tokens, int) or raw_min_tokens < 0:
                    raise ValueError(f"{backend_label} min_tokens must be a non-negative integer")
                schedule_min_tokens = int(raw_min_tokens)
                if resolved == "pisa":
                    raw_min_seq = values.get("min_sequence_length")
                    if raw_min_seq is not None:
                        if isinstance(raw_min_seq, bool) or not isinstance(raw_min_seq, int) or raw_min_seq < 0:
                            raise ValueError(
                                f"pisa min_sequence_length must be a non-negative integer, got {raw_min_seq!r}"
                            )
                        schedule_min_tokens = max(schedule_min_tokens, int(raw_min_seq))
                    elif not min_tokens_explicit:
                        schedule_min_tokens = _resolve_pisa_default_min_sequence()
                if (
                    raw_dense_steps > 0
                    or parsed_tile_3d is not None
                    or dense_blocks
                    or schedule_min_tokens > 0
                    or dense_backend is not None
                ):
                    sol_schedule: dict[str, Any] = {
                        "dense_steps": int(raw_dense_steps),
                        "dense_blocks": dense_blocks,
                        "tile_3d": parsed_tile_3d,
                        "min_tokens": schedule_min_tokens,
                    }
                    model.velocity_model.__dict__["_worldfoundry_sol_schedule"] = sol_schedule
                    norm_dense = sorted(
                        {
                            idx + num_blocks if idx < 0 else idx
                            for idx in dense_blocks
                            if -num_blocks <= idx < num_blocks
                        }
                    )
                    schedule_report = {
                        "dense_steps": int(raw_dense_steps),
                        "dense_blocks": norm_dense,
                        "tile_3d": parsed_tile_3d,
                        "min_tokens": schedule_min_tokens,
                    }
                    if dense_backend is not None:
                        schedule_report["dense_backend"] = dense_backend
            elif resolved == "cudnn_fp8" and ("dense_blocks" in values or dense_backend is not None):
                if "dense_blocks" in values:
                    dense_blocks = _parse_dense_blocks(values.pop("dense_blocks"))
                num_blocks = len(model.velocity_model.transformer_blocks)
                norm_dense = sorted(
                    {idx + num_blocks if idx < 0 else idx for idx in dense_blocks if -num_blocks <= idx < num_blocks}
                )
                schedule_report = {"dense_blocks": norm_dense}
                if dense_backend is not None:
                    schedule_report["dense_backend"] = dense_backend

            configured = model.velocity_model.configure_attention_backend(
                resolved,
                self_only=(scope == "self"),
                cross_only=(scope == "cross"),
                backend_options=values or None,
                dense_blocks=dense_blocks,
                dense_backend=dense_backend,
            )
            scope_entry: dict[str, Any] = {
                "requested": requested,
                "resolved": resolved,
                "configured_modules": configured,
                "options": values,
            }
            if schedule_report is not None:
                scope_entry["schedule"] = schedule_report
            scopes_report[scope] = scope_entry

        attn_approx = any(
            row["resolved"] in _APPROXIMATE_ATTENTION_BACKENDS
            and not (row["resolved"] == "pisa" and float(row.get("options", {}).get("density", 0.1)) >= 1.0)
            for row in scopes_report.values()
        )
        any_approximate = any_approximate or attn_approx
        installed_entries.append(
            {
                "name": "attention_policy",
                "approximate": attn_approx,
                "scopes": scopes_report,
            }
        )

    if "token_pruning" in active_plugins:
        tp_rt = model.velocity_model.configure_token_pruning(active_plugins["token_pruning"])
        assert tp_rt is not None
        tp_approx = float(tp_rt["keep_ratio"]) < 1.0
        any_approximate = any_approximate or tp_approx
        installed_entries.append(
            {
                "name": "token_pruning",
                "approximate": tp_approx,
                "keep_ratio": tp_rt["keep_ratio"],
                "method": tp_rt["method"],
                "restore_mode": tp_rt["restore_mode"],
                "stratify_group_size": tp_rt["stratify_group_size"],
                "block_span": tp_rt["block_span"],
                "min_tokens": tp_rt["min_tokens"],
                "min_steps_to_enable": tp_rt["min_steps_to_enable"],
                "allow_distilled_pruning": tp_rt["allow_distilled_pruning"],
                "preserve_conditioning_tokens": tp_rt["preserve_conditioning_tokens"],
            }
        )

    fb_plugin_name = (
        "fbcache"
        if "fbcache" in active_plugins
        else ("first_block_cache" if "first_block_cache" in active_plugins else None)
    )
    if fb_plugin_name is not None:
        direct_fb = comp.get(
            "fbcache", comp.get("first_block_cache", options.get("fbcache", options.get("first_block_cache")))
        )
        if direct_fb is not None and direct_fb is not False:
            raise ValueError("fbcache cannot be specified in both runtime options and accelerations")
        fb_max_tokens = _resolve_fbcache_max_tokens(context)
        fb_cfg = _parse_fbcache_config(active_plugins[fb_plugin_name], max_tokens_override=fb_max_tokens)
        if fb_cfg is not None:
            model.__dict__["_worldfoundry_ltx_fbcache_config"] = fb_cfg
            fb_approx = fb_cfg.residual_diff_threshold > 0.0
            any_approximate = any_approximate or fb_approx
            installed_entries.append(
                {
                    "name": fb_plugin_name,
                    "approximate": fb_approx,
                    "config": fb_cfg.to_dict(),
                }
            )

    manifest: dict[str, Any] = {
        "installed": installed_entries,
        "execution_verified": False,
    }
    model.__dict__["_worldfoundry_ltx_accelerations"] = manifest
    if applied is not None:
        applied.requested["accelerations"] = dict(active_plugins)
        applied.effective["accelerations"] = manifest
        if any_approximate:
            applied.quality_tier = "approximate"


class _LTXFirstBlockCacheMixin:
    """Request- and CFG-branch-scoped FirstBlockCache ownership for LTX denoisers."""

    _fbcache_config: Any = None

    def _fbcache_lock(self) -> RLock:
        lock = self.__dict__.get("_fbcache_state_lock")
        if lock is None:
            lock = RLock()
            self.__dict__["_fbcache_state_lock"] = lock
        return lock

    def _fbcache_context_var(self) -> ContextVar[str | None]:
        context = self.__dict__.get("_fbcache_current_request")
        if isinstance(context, ContextVar):
            return context
        context = ContextVar(f"worldfoundry_fbcache_request_{id(self)}", default=None)
        self.__dict__["_fbcache_current_request"] = context
        return context

    def _init_fbcache(
        self,
        model: torch.nn.Module,
        *,
        graph_mode: bool | str,
        teacache_threshold: float | None,
        fbcache_threshold: float | None,
        fbcache_config: Any,
        fbcache_max_tokens: int | None,
    ) -> None:
        resolved = None
        if fbcache_config is not None:
            resolved = _parse_fbcache_config(fbcache_config, max_tokens_override=fbcache_max_tokens)
        elif fbcache_threshold is not None:
            resolved = _parse_fbcache_config(float(fbcache_threshold), max_tokens_override=fbcache_max_tokens)
        else:
            plugin_cfg = getattr(model, "_worldfoundry_ltx_fbcache_config", None)
            if plugin_cfg is not None:
                resolved = _parse_fbcache_config(plugin_cfg, max_tokens_override=fbcache_max_tokens)

        if resolved is not None:
            if teacache_threshold is not None:
                raise ValueError("LTX FirstBlockCache and TeaCache are mutually exclusive")
            if graph_mode is True:
                raise ValueError("LTX FirstBlockCache requires cuda_graph='piecewise' or 'off'")
            applied = getattr(model, "_worldfoundry_applied_optimizations", None)
            if resolved.residual_diff_threshold > 0.0 and applied is not None:
                applied.quality_tier = "approximate"

        self._fbcache_config = resolved
        self._fbcache_requests: dict[str, dict[str, Any]] = {}
        self._fbcache_last_request_id: str | None = None
        self._fbcache_epoch_counter: int = 0
        self._fbcache_receipt_snapshots: dict[str, dict[str, Any]] = {}
        self._fbcache_state_lock = RLock()
        self._fbcache_current_request: ContextVar[str | None] = ContextVar(
            f"worldfoundry_fbcache_request_{id(self)}",
            default=None,
        )

    def _fbcache_kwargs(
        self,
        model_input: MultiModalDenoiserInput,
        *,
        video_tokens: int,
    ) -> dict[str, Any]:
        cfg = self._fbcache_config
        if cfg is None:
            return {}
        if cfg.max_tokens is not None and video_tokens > int(cfg.max_tokens):
            return {}

        from ..networks.ltx.acceleration import LTXFirstBlockCache

        raw_req = getattr(model_input, "request_id", None)
        if not isinstance(raw_req, str) or not raw_req.strip():
            cond_req = model_input.conditioning.get("request_id")
            raw_req = cond_req if isinstance(cond_req, str) and cond_req.strip() else "__default__"
        request_id = str(raw_req)
        self._fbcache_context_var().set(request_id)

        step = int(model_input.step_index)
        branch = str(model_input.branch)
        with self._fbcache_lock():
            self._fbcache_last_request_id = request_id
            state: dict[str, Any] | None = self._fbcache_requests.get(request_id)
            if state is None:
                self._fbcache_epoch_counter += 1
                new_state: dict[str, Any] = {
                    "request_id": request_id,
                    "request_epoch": self._fbcache_epoch_counter,
                    "caches": {},
                    "complete": False,
                }
                state = new_state
                self._fbcache_requests[request_id] = state
                if len(self._fbcache_requests) > 64:
                    completed = sorted(
                        (
                            cand
                            for cand in self._fbcache_requests.values()
                            if bool(cand.get("complete")) and cand.get("request_id") != request_id
                        ),
                        key=lambda cand: int(cand["request_epoch"]),
                    )
                    for cand in completed[: max(len(self._fbcache_requests) - 32, 0)]:
                        self._fbcache_requests.pop(str(cand["request_id"]), None)

            caches: dict[str, LTXFirstBlockCache] = state["caches"]
            cache = caches.get(branch)
            if cache is None:
                cache = LTXFirstBlockCache(
                    cfg,
                    branch=branch,
                    request_id=request_id,
                    request_epoch=int(state["request_epoch"]),
                )
                caches[branch] = cache
            elif step == 0:
                if request_id == "__default__":
                    cache.reset()
                else:
                    # Explicit request_id starting a new denoising stage (e.g. Stage 2): clear
                    # cached probe/residual tensors without erasing Stage-1 request telemetry.
                    cache.clear_state()
            if model_input.total_steps is not None and step >= int(model_input.total_steps) - 1:
                state["complete"] = True

        return {
            "fbcache": cache,
            "feature_cache_step": step,
            "feature_cache_total_steps": model_input.total_steps,
        }

    def fbcache_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return FirstBlockCache telemetry for one request (or the most recent request)."""
        cfg = self._fbcache_config
        if cfg is None:
            return {
                "enabled": False,
                "algorithm": "none",
                "threshold": None,
                "effective": "disabled",
                "events": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "dense_block_calls": 0,
                "skipped_block_calls": 0,
                "branches": {},
            }
        with self._fbcache_lock():
            if isinstance(request_id, str):
                resolved_req: str | None = request_id
            else:
                ctx_req = self._fbcache_context_var().get()
                resolved_req = ctx_req if ctx_req is not None else self._fbcache_last_request_id
            state = self._fbcache_requests.get(resolved_req) if resolved_req is not None else None
            if state is None and resolved_req is not None and resolved_req in self._fbcache_receipt_snapshots:
                return dict(self._fbcache_receipt_snapshots[resolved_req])

            branches: dict[str, dict[str, Any]] = {}
            total_events = 0
            total_hits = 0
            total_dense = 0
            total_skipped = 0
            caches = state["caches"] if state is not None else {}
            for branch, cache in caches.items():
                rec = cache.receipt()
                branches[str(branch)] = rec
                total_events += int(rec["events"])
                total_hits += int(rec["hits"])
                total_dense += int(rec["dense_block_calls"])
                total_skipped += int(rec["skipped_block_calls"])

            if total_events == 0:
                effective = "pending"
            elif total_hits > 0:
                effective = "firstblock-reuse"
            else:
                effective = "dense-no-hit"
            return {
                "enabled": True,
                "algorithm": "firstblock",
                "threshold": cfg.residual_diff_threshold,
                "config": cfg.to_dict(),
                "request_id": resolved_req,
                "request_epoch": int(state["request_epoch"]) if state is not None else 0,
                "request_local": state is not None and bool(branches),
                "effective": effective,
                "events": total_events,
                "hits": total_hits,
                "misses": total_events - total_hits,
                "hit_rate": total_hits / total_events if total_events else 0.0,
                "dense_block_calls": total_dense,
                "skipped_block_calls": total_skipped,
                "branches": branches,
            }

    def _end_fbcache_request(self, request_id: str) -> None:
        with self._fbcache_lock():
            state = self._fbcache_requests.get(request_id)
            if state is None:
                return
            snapshot = self.fbcache_report(request_id)
            snapshot["finalized"] = True
            removed = self._fbcache_requests.pop(request_id, None)
            if removed is not None:
                for cache in removed["caches"].values():
                    cache.reset()
                removed["caches"].clear()
            self._fbcache_receipt_snapshots[request_id] = snapshot
            while len(self._fbcache_receipt_snapshots) > 32:
                self._fbcache_receipt_snapshots.pop(next(iter(self._fbcache_receipt_snapshots)), None)


class LTXJointDenoiser(_LTXFirstBlockCacheMixin, FeatureCacheDenoiserMixin, GraphWrappedDenoiserMixin):
    """Adapt the LTX joint velocity model to the canonical multi-modal contract.

    ``precision="autocast"`` runs the transformer under autocast to ``compute_dtype`` and
    returns the clean-sample prediction in the timestep dtype. ``precision="weights"`` runs it
    in its loaded weight dtype and rounds the prediction to the latent dtype, as the
    ``ltx_pipelines`` ``X0Model`` does.
    """

    def __init__(
        self,
        model: LTXAVTransformerModule,
        *,
        compute_dtype: torch.dtype,
        precision: str = "autocast",
        enable_cuda_graph: bool | str = False,
        teacache_threshold: float | None = None,
        teacache_max_tokens: int | None = None,
        fbcache_threshold: float | None = None,
        fbcache_config: Any = None,
        fbcache_max_tokens: int | None = None,
    ) -> None:
        if precision not in {"autocast", "weights"}:
            raise ValueError(f"unsupported LTX denoiser precision: {precision!r}")
        graph_mode = _normalize_ltx_cuda_graph_mode(enable_cuda_graph)
        _validate_ltx_graph_attention_compatibility(model, graph_mode)
        self.model = model
        self.compute_dtype = compute_dtype
        self.precision = precision
        self._teacache_threshold = teacache_threshold
        self._teacache_max_tokens = teacache_max_tokens
        self._init_fbcache(
            model,
            graph_mode=graph_mode,
            teacache_threshold=teacache_threshold,
            fbcache_threshold=fbcache_threshold,
            fbcache_config=fbcache_config,
            fbcache_max_tokens=fbcache_max_tokens,
        )
        self._video_grid_cache: dict[int, Any] = {}
        self._uniform_mask_cache: dict[int, Any] = {}
        self._trivial_context_mask_cache: dict[int, Any] = {}
        self._protected_token_cache: dict[tuple[int, int], _ProtectedCacheEntry] = {}
        self._init_graph_runner(
            model,
            enabled=graph_mode,
            extra_key=f"ltx_joint:{type(model).__module__}.{type(model).__qualname__}:{precision}",
        )

    @staticmethod
    def _modality(
        state,
        *,
        sigma: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        video_control: object | None = None,
        attention_cache=None,
        uniform_denoise_mask: bool = False,
    ) -> Modality:
        batch = state.latent.shape[0]
        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ValueError("LTX sigma must be scalar or contain one value per sample")
        effective_mask = state.denoise_mask[:, :1] if uniform_denoise_mask else state.denoise_mask
        mask_sigma = sigma.view(batch, *([1] * (effective_mask.ndim - 1)))
        return Modality(
            latent=state.latent,
            sigma=sigma,
            timesteps=effective_mask * mask_sigma,
            positions=state.positions,
            context=context,
            context_mask=context_mask,
            attention_mask=state.attention_mask,
            video_control=video_control,
            attention_cache=attention_cache,
            keyframes_mask=state.keyframes_mask,
        )

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        try:
            video_state = model_input.modalities["video"]
            audio_state = model_input.modalities["audio"]
            video_context = model_input.conditioning["video_context"]
            audio_context = model_input.conditioning["audio_context"]
        except KeyError as error:
            raise KeyError(f"LTX joint denoising is missing {error.args[0]!r}") from error
        if not isinstance(video_context, torch.Tensor) or not isinstance(audio_context, torch.Tensor):
            raise TypeError("LTX video_context and audio_context must be tensors")
        video_context_mask = self._context_mask(
            model_input.conditioning,
            "video_context_mask",
            token_count=video_state.latent.shape[1],
        )
        audio_context_mask = self._context_mask(
            model_input.conditioning,
            "audio_context_mask",
            token_count=audio_state.latent.shape[1],
        )
        if int(model_input.step_index) == 0:
            self._begin_graph_request_window()
            _evict_inference_cache_entries(self._uniform_mask_cache)
            _evict_inference_cache_entries(self._trivial_context_mask_cache)
            _evict_protected_cache_entries(self._protected_token_cache)
        if _is_trivial_context_mask(self._trivial_context_mask_cache, video_context_mask, allow_inference_cache=True):
            video_context_mask = None
        if _is_trivial_context_mask(self._trivial_context_mask_cache, audio_context_mask, allow_inference_cache=True):
            audio_context_mask = None

        video_uniform_mask = _is_uniform_denoise_mask(
            self._uniform_mask_cache,
            video_state.denoise_mask,
            allow_inference_cache=True,
        )
        video = self._modality(
            video_state,
            sigma=model_input.timestep,
            context=video_context,
            context_mask=video_context_mask,
            video_control=model_input.conditioning.get("video_control"),
            attention_cache=model_input.conditioning.get("attention_cache"),
            uniform_denoise_mask=video_uniform_mask,
        )
        audio = self._modality(
            audio_state,
            sigma=model_input.timestep,
            context=audio_context,
            context_mask=audio_context_mask,
            attention_cache=model_input.conditioning.get("attention_cache"),
            uniform_denoise_mask=_is_uniform_denoise_mask(
                self._uniform_mask_cache,
                audio_state.denoise_mask,
                allow_inference_cache=True,
            ),
        )
        extra_kwargs: dict[str, Any] = {}
        video_tokens = int(video_state.latent.shape[1])
        if self._fbcache_config is not None:
            fb_kwargs = self._fbcache_kwargs(model_input, video_tokens=video_tokens)
            if fb_kwargs:
                extra_kwargs.update(fb_kwargs)
        else:
            allow_cache = self._teacache_max_tokens is None or video_tokens <= self._teacache_max_tokens
            if allow_cache:
                cache_kwargs = self.feature_cache_kwargs(_feature_cache_input_view(model_input))
                if cache_kwargs:
                    extra_kwargs.update(cache_kwargs)
        if getattr(self, "_piecewise_graph", None) is not None:
            extra_kwargs["_worldfoundry_piecewise_graph"] = self._piecewise_graph
        velocity_model = _unwrap_velocity_model(self.model)
        sol_kwargs = _sol_schedule_kwargs(
            velocity_model,
            video_state.positions,
            int(model_input.step_index),
            self._video_grid_cache,
        )
        if sol_kwargs:
            extra_kwargs.update(sol_kwargs)
        tp_kwargs = _token_pruning_kwargs(
            velocity_model,
            video_state.denoise_mask,
            video_state.keyframes_mask,
            step_index=int(model_input.step_index),
            total_steps=model_input.total_steps,
            uniform_denoise_mask=video_uniform_mask,
            protected_cache=self._protected_token_cache,
        )
        if tp_kwargs:
            extra_kwargs.update(tp_kwargs)
        with torch.autocast(
            device_type=video_state.latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.precision == "autocast" and self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            video_velocity, audio_velocity = self._run_network(video, audio, None, **extra_kwargs)
        if video_velocity is None or audio_velocity is None:
            raise RuntimeError("LTX audio-video transformer returned an absent modality")
        return MultiModalDenoiserOutput(
            samples={
                "video": self._clean_prediction(video, video_velocity),
                "audio": self._clean_prediction(audio, audio_velocity),
            }
        )

    def end_feature_cache_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        if self._fbcache_config is not None:
            self._end_fbcache_request(request_id)
        super().end_feature_cache_request(request_id, error=error)

    def runtime_optimization_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return CUDA Graph, feature-cache, token-pruning, and load-time optimization telemetry."""
        applied = getattr(self.model, "_worldfoundry_applied_optimizations", None)
        cache_report = (
            self.fbcache_report(request_id)
            if self._fbcache_config is not None
            else self.feature_cache_report(request_id)
        )
        report: dict[str, Any] = {
            "cuda_graph": self.graph_report(),
            "feature_cache": cache_report,
            "applied_optimizations": None if applied is None else applied.to_optimization_snapshot(),
        }
        accelerations = getattr(self.model, "_worldfoundry_ltx_accelerations", None)
        if accelerations is not None:
            report["accelerations"] = accelerations
        tp_rt = getattr(_unwrap_velocity_model(self.model), "_worldfoundry_token_pruning_runtime", None)
        if isinstance(tp_rt, Mapping):
            report["token_pruning"] = {k: v for k, v in tp_rt.items() if not str(k).startswith("_")}
        return report

    def _clean_prediction(self, modality: Modality, velocity: torch.Tensor) -> torch.Tensor:
        if self.precision == "weights":
            return velocity_to_denoised(modality.latent, velocity, modality.timesteps)
        return modality.latent - modality.timesteps * velocity

    @staticmethod
    def _context_mask(
        conditioning: Mapping[str, object],
        modality_key: str,
        *,
        token_count: int,
    ) -> torch.Tensor | None:
        """Resolve a modality/stage mask while preserving the shared fallback."""

        value = conditioning.get(modality_key, conditioning.get("context_mask"))
        if isinstance(value, Mapping):
            value = value.get(token_count, value.get(str(token_count)))
            if value is None:
                raise KeyError(f"LTX {modality_key} has no mask for {token_count} query tokens")
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"LTX {modality_key} must be a tensor or token-count mapping")
        return value


class LTXVideoDenoiser(_LTXFirstBlockCacheMixin, FeatureCacheDenoiserMixin, GraphWrappedDenoiserMixin):
    """Adapt the video-only velocity model to the joint-state runner contract."""

    def __init__(
        self,
        model: LTXVideoTransformerModule,
        *,
        compute_dtype: torch.dtype,
        enable_cuda_graph: bool | str = False,
        teacache_threshold: float | None = None,
        teacache_max_tokens: int | None = None,
        fbcache_threshold: float | None = None,
        fbcache_config: Any = None,
        fbcache_max_tokens: int | None = None,
    ) -> None:
        graph_mode = _normalize_ltx_cuda_graph_mode(enable_cuda_graph)
        _validate_ltx_graph_attention_compatibility(model, graph_mode)
        self.model = model
        self.compute_dtype = compute_dtype
        self._teacache_threshold = teacache_threshold
        self._teacache_max_tokens = teacache_max_tokens
        self._init_fbcache(
            model,
            graph_mode=graph_mode,
            teacache_threshold=teacache_threshold,
            fbcache_threshold=fbcache_threshold,
            fbcache_config=fbcache_config,
            fbcache_max_tokens=fbcache_max_tokens,
        )
        self._video_grid_cache: dict[int, Any] = {}
        self._uniform_mask_cache: dict[int, Any] = {}
        self._trivial_context_mask_cache: dict[int, Any] = {}
        self._protected_token_cache: dict[tuple[int, int], _ProtectedCacheEntry] = {}
        self._init_graph_runner(
            model,
            enabled=graph_mode,
            extra_key=f"ltx_video:{type(model).__module__}.{type(model).__qualname__}",
        )

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        try:
            state = model_input.modalities["video"]
            context = model_input.conditioning["video_context"]
        except KeyError as error:
            raise KeyError(f"LTX-Video denoising is missing {error.args[0]!r}") from error
        if not isinstance(context, torch.Tensor):
            raise TypeError("LTX-Video context must be a tensor")
        context_mask = LTXJointDenoiser._context_mask(
            model_input.conditioning,
            "video_context_mask",
            token_count=state.latent.shape[1],
        )
        if int(model_input.step_index) == 0:
            self._begin_graph_request_window()
            _evict_inference_cache_entries(self._uniform_mask_cache)
            _evict_inference_cache_entries(self._trivial_context_mask_cache)
            _evict_protected_cache_entries(self._protected_token_cache)
        if _is_trivial_context_mask(self._trivial_context_mask_cache, context_mask, allow_inference_cache=True):
            context_mask = None
        video_uniform_mask = _is_uniform_denoise_mask(
            self._uniform_mask_cache,
            state.denoise_mask,
            allow_inference_cache=True,
        )
        video = LTXJointDenoiser._modality(
            state,
            sigma=model_input.timestep,
            context=context,
            context_mask=context_mask,
            uniform_denoise_mask=video_uniform_mask,
        )
        extra_kwargs: dict[str, Any] = {}
        video_tokens = int(state.latent.shape[1])
        if self._fbcache_config is not None:
            fb_kwargs = self._fbcache_kwargs(model_input, video_tokens=video_tokens)
            if fb_kwargs:
                extra_kwargs.update(fb_kwargs)
        else:
            allow_cache = self._teacache_max_tokens is None or video_tokens <= self._teacache_max_tokens
            if allow_cache:
                cache_kwargs = self.feature_cache_kwargs(_feature_cache_input_view(model_input))
                if cache_kwargs:
                    extra_kwargs.update(cache_kwargs)
        if getattr(self, "_piecewise_graph", None) is not None:
            extra_kwargs["_worldfoundry_piecewise_graph"] = self._piecewise_graph
        velocity_model = _unwrap_velocity_model(self.model)
        sol_kwargs = _sol_schedule_kwargs(
            velocity_model,
            state.positions,
            int(model_input.step_index),
            self._video_grid_cache,
        )
        if sol_kwargs:
            extra_kwargs.update(sol_kwargs)
        tp_kwargs = _token_pruning_kwargs(
            velocity_model,
            state.denoise_mask,
            state.keyframes_mask,
            step_index=int(model_input.step_index),
            total_steps=model_input.total_steps,
            uniform_denoise_mask=video_uniform_mask,
            protected_cache=self._protected_token_cache,
        )
        if tp_kwargs:
            extra_kwargs.update(tp_kwargs)
        with torch.autocast(
            device_type=state.latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            velocity, audio_velocity = self._run_network(video, None, None, **extra_kwargs)
        if velocity is None or audio_velocity is not None:
            raise RuntimeError("LTX-Video transformer returned an unexpected modality set")
        return MultiModalDenoiserOutput(samples={"video": video.latent - video.timesteps * velocity})

    def end_feature_cache_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        if self._fbcache_config is not None:
            self._end_fbcache_request(request_id)
        super().end_feature_cache_request(request_id, error=error)

    def runtime_optimization_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return CUDA Graph, feature-cache, token-pruning, and load-time optimization telemetry."""
        applied = getattr(self.model, "_worldfoundry_applied_optimizations", None)
        cache_report = (
            self.fbcache_report(request_id)
            if self._fbcache_config is not None
            else self.feature_cache_report(request_id)
        )
        report: dict[str, Any] = {
            "cuda_graph": self.graph_report(),
            "feature_cache": cache_report,
            "applied_optimizations": None if applied is None else applied.to_optimization_snapshot(),
        }
        accelerations = getattr(self.model, "_worldfoundry_ltx_accelerations", None)
        if accelerations is not None:
            report["accelerations"] = accelerations
        tp_rt = getattr(_unwrap_velocity_model(self.model), "_worldfoundry_token_pruning_runtime", None)
        if isinstance(tp_rt, Mapping):
            report["token_pruning"] = {k: v for k, v in tp_rt.items() if not str(k).startswith("_")}
        return report


def build_ltx_joint_denoiser(
    context: ComponentBuildContext,
    *,
    module_class: type[LTXAVTransformerModule] = LTXAVTransformerModule,
    denoiser_class: type[LTXJointDenoiser] = LTXJointDenoiser,
) -> LTXJointDenoiser:
    """Load one LTX-2.x AV transformer through the shared native loader."""

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )

    from ...optimizations.ltx.quantization import CHECKPOINT_QUANTIZATION_MODES, prepare_ltx_quantized_state_dict

    split_checkpoint = bool(context.component_options.get("split_checkpoint"))
    checkpoint_quantization = context.policy.quantization.mode in CHECKPOINT_QUANTIZATION_MODES
    ic_lora_checkpoint = context.checkpoint("ic_lora")
    lazy_state_dict = split_checkpoint or checkpoint_quantization or ic_lora_checkpoint is not None
    post_load_hook = None
    lora_checkpoint = context.checkpoint("lora")
    if lora_checkpoint is not None:
        materialized_lora = NativeCheckpointResolver().materialize(lora_checkpoint)
        from worldfoundry.core.model_loading import load_state_dict, merge_named_lora_

        lora_source: str | list[str]
        lora_paths = tuple(str(path) for path in materialized_lora.paths)
        lora_source = lora_paths[0] if len(lora_paths) == 1 else list(lora_paths)
        lora_state = load_state_dict(lora_source, torch_dtype=context.policy.dtype, device="cpu")
        expected_modules = context.component_options.get("expected_lora_modules")
        lora_alpha = float(context.component_options.get("lora_alpha", 1.0))

        def merge_lora(wrapper: torch.nn.Module) -> None:
            if not isinstance(wrapper, LTXAVTransformerModule):
                raise TypeError("LTX LoRA hook received an unexpected transformer wrapper")
            merge_named_lora_(
                wrapper.velocity_model,
                lora_state,
                alpha=lora_alpha,
                expected_modules=int(expected_modules) if expected_modules is not None else None,
                owner="LTX LoRA",
                device="cpu",
                torch_dtype=context.policy.dtype,
            )

        post_load_hook = merge_lora

    state_dict_adapter = (
        partial(prepare_ltx_quantized_state_dict, policy=context.policy.quantization, device=context.policy.device)
        if checkpoint_quantization
        else None
    )
    if ic_lora_checkpoint is not None:
        from worldfoundry.core.vram import DiskMap

        from .ltx_lora import convert_ltx_ic_lora_state_dict, prepare_ltx_ic_lora_state_dict

        materialized = NativeCheckpointResolver().materialize(ic_lora_checkpoint)
        lora = convert_ltx_ic_lora_state_dict(DiskMap([str(path) for path in materialized.paths], "cpu"))
        quantization_adapter = state_dict_adapter

        def adapt_ic_lora(
            module: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
        ) -> Mapping[str, torch.Tensor]:
            fused = prepare_ltx_ic_lora_state_dict(
                module,
                state_dict,
                lora=lora,
                strength=float(context.component_options.get("lora_strength", 1.0)),
                device=context.policy.device,
            )
            return quantization_adapter(module, fused) if quantization_adapter is not None else fused

        state_dict_adapter = adapt_ic_lora

    validate_cuda_graph_options(context)
    cuda_graph_mode = resolve_cuda_graph_mode(context)
    teacache_threshold = resolve_teacache_threshold(context)
    teacache_max_tokens = _resolve_teacache_max_tokens(context)
    fbcache_config = _resolve_fbcache_config(context)

    fuse_qkv = bool(context.component_options.get("fuse_qkv", context.policy.options.get("fuse_qkv", False)))
    fuse_cross_kv = bool(
        context.component_options.get("fuse_cross_kv", context.policy.options.get("fuse_cross_kv", fuse_qkv))
    )
    want_fused_ops = _resolve_ltx_fused_ops(context)
    want_fuse_proj = (
        (fuse_qkv or fuse_cross_kv)
        and context.policy.offload.mode is not OffloadMode.DISK
        and not checkpoint_quantization
    )
    if (want_fuse_proj or want_fused_ops) and context.policy.offload.mode is not OffloadMode.DISK:
        previous_hook = post_load_hook

        def prepare_after_load(wrapper: torch.nn.Module) -> None:
            if previous_hook is not None:
                previous_hook(wrapper)
            if isinstance(wrapper, LTXAVTransformerModule):
                if want_fuse_proj:
                    stats = wrapper.velocity_model.fuse_projections(
                        fuse_self_qkv=fuse_qkv,
                        fuse_cross_kv=fuse_cross_kv,
                    )
                    wrapper.__dict__["_worldfoundry_ltx_fusion_stats"] = stats
                if want_fused_ops:
                    wrapper.velocity_model.enable_fused_ops()

        post_load_hook = prepare_after_load

    load_policy = _with_ltx_default_fp8_excludes(context.policy)
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config=({"split_checkpoint": True} if split_checkpoint else {}),
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=(
                partial(convert_ltx_transformer_state_dict, lazy=True)
                if lazy_state_dict
                else convert_ltx_transformer_state_dict
            ),
            lazy_state_dict=lazy_state_dict,
            buffer_aware_offload=lazy_state_dict,
            state_dict_adapter=state_dict_adapter,
            vram_module_map={
                LTXModel: AutoWrappedNonRecurseModule,
                BasicAVTransformerBlock: AutoWrappedModule if lazy_state_dict else AutoWrappedNonRecurseModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="velocity_model.transformer_blocks",
            compile_regions=("velocity_model.transformer_blocks",),
            post_load_hook=post_load_hook,
        ),
        context.require_checkpoint("weights"),
        load_policy,
    )
    if not isinstance(model, LTXAVTransformerModule):
        raise TypeError(f"expected LTXAVTransformerModule, got {type(model).__name__}")
    model.velocity_model.refresh_preprocessor_bindings()
    _apply_ltx_post_load_accelerations(model, context)
    return denoiser_class(
        model,
        compute_dtype=context.policy.dtype,
        enable_cuda_graph=cuda_graph_mode,
        teacache_threshold=teacache_threshold,
        **({"teacache_max_tokens": teacache_max_tokens} if teacache_max_tokens is not None else {}),
        **({"fbcache_config": fbcache_config} if fbcache_config is not None else {}),
        **({"precision": context.component_options["precision"]} if "precision" in context.component_options else {}),
    )


def build_ltx_video_denoiser(context: ComponentBuildContext) -> LTXVideoDenoiser:
    """Load LTX-Video 0.9.x through the shared native loader."""

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )

    validate_cuda_graph_options(context)
    cuda_graph_mode = resolve_cuda_graph_mode(context)
    teacache_threshold = resolve_teacache_threshold(context)
    teacache_max_tokens = _resolve_teacache_max_tokens(context)
    fbcache_config = _resolve_fbcache_config(context)

    fuse_qkv = bool(context.component_options.get("fuse_qkv", context.policy.options.get("fuse_qkv", False)))
    fuse_cross_kv = bool(
        context.component_options.get("fuse_cross_kv", context.policy.options.get("fuse_cross_kv", fuse_qkv))
    )
    want_fused_ops = _resolve_ltx_fused_ops(context)
    want_fuse_proj = (fuse_qkv or fuse_cross_kv) and context.policy.offload.mode is not OffloadMode.DISK
    post_load_hook = None
    if (want_fuse_proj or want_fused_ops) and context.policy.offload.mode is not OffloadMode.DISK:

        def prepare_after_load(wrapper: torch.nn.Module) -> None:
            if isinstance(wrapper, LTXVideoTransformerModule):
                if want_fuse_proj:
                    stats = wrapper.velocity_model.fuse_projections(
                        fuse_self_qkv=fuse_qkv,
                        fuse_cross_kv=fuse_cross_kv,
                    )
                    wrapper.__dict__["_worldfoundry_ltx_fusion_stats"] = stats
                if want_fused_ops:
                    wrapper.velocity_model.enable_fused_ops()

        post_load_hook = prepare_after_load

    load_policy = _with_ltx_default_fp8_excludes(context.policy)
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXVideoTransformerModule,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=convert_ltx_video_transformer_state_dict,
            vram_module_map={
                LTXModel: AutoWrappedNonRecurseModule,
                BasicAVTransformerBlock: AutoWrappedNonRecurseModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="velocity_model.transformer_blocks",
            compile_regions=("velocity_model.transformer_blocks",),
            post_load_hook=post_load_hook,
        ),
        context.require_checkpoint("weights"),
        load_policy,
    )
    if not isinstance(model, LTXVideoTransformerModule):
        raise TypeError(f"expected LTXVideoTransformerModule, got {type(model).__name__}")
    model.velocity_model.refresh_preprocessor_bindings()
    _apply_ltx_post_load_accelerations(model, context)
    return LTXVideoDenoiser(
        model,
        compute_dtype=context.policy.dtype,
        enable_cuda_graph=cuda_graph_mode,
        teacache_threshold=teacache_threshold,
        teacache_max_tokens=teacache_max_tokens,
        fbcache_config=fbcache_config,
    )


__all__ = [
    "LTXAVTransformerModule",
    "LTXJointDenoiser",
    "LTXVideoDenoiser",
    "LTXVideoTransformerModule",
    "build_ltx_joint_denoiser",
    "build_ltx_video_denoiser",
    "convert_ltx_transformer_state_dict",
    "convert_ltx_video_transformer_state_dict",
]
