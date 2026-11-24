"""Native LTX audio-video denoiser component.

Adapts the official LTX velocity transformer to the multi-modal
``Denoiser`` contract used by the joint AV runner.  Config comes from
safetensors metadata via :func:`~...loaders.metadata.safetensors_json_metadata`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
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


_APPROXIMATE_ATTENTION_BACKENDS = frozenset({"sol_attn", "sage_attention", "sage_attention_3", "cudnn_fp8"})


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


def _sol_schedule_kwargs(
    velocity_model: LTXModel,
    positions: torch.Tensor,
    step_index: int,
    grid_cache: dict[int, Any],
) -> dict[str, Any]:
    """Resolve zero-sync Sol-Attn schedule kwargs outside CUDA Graph capture."""
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
    """Apply LTX-specific post-load fused ops, FP8 FFN fusion, and scoped attention routing."""
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

    unsupported = sorted(set(active_plugins) - {"attention_policy"})
    if unsupported:
        raise ValueError(f"LTX denoiser does not support requested acceleration plugins: {unsupported}")

    attn_policy = active_plugins["attention_policy"]
    if not isinstance(attn_policy, Mapping):
        raise TypeError("attention_policy must be a mapping of scope names to backend configs")
    if set(attn_policy) - {"self", "cross"} or not attn_policy:
        raise ValueError("attention_policy requires self and/or cross options")

    from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend

    scopes_report: dict[str, dict[str, Any]] = {}
    for scope, raw in attn_policy.items():
        if not isinstance(raw, (str, Mapping)):
            raise TypeError("attention_policy scopes require a backend string or mapping")
        values = {"backend": raw} if isinstance(raw, str) else dict(raw)
        if "backend" not in values:
            raise ValueError("attention_policy scopes require a backend")
        requested = normalize_attention_backend(str(values.pop("backend")))
        resolved = resolve_attention_backend(requested, context.policy.device)

        dense_blocks: tuple[int, ...] = ()
        schedule_report: dict[str, Any] | None = None
        if resolved == "sol_attn":
            if scope != "self":
                raise ValueError("Sol-Attn is supported only for unmasked self-attention")
            if context.policy.dtype != torch.bfloat16:
                raise ValueError(f"Sol-Attn requires BF16 inference, got dtype={context.policy.dtype}")
            raw_dense_steps = values.pop("dense_steps", 0)
            if isinstance(raw_dense_steps, bool) or not isinstance(raw_dense_steps, int) or raw_dense_steps < 0:
                raise ValueError("Sol-Attn dense_steps must be a non-negative integer")
            raw_dense_blocks = values.pop("dense_blocks", ())
            dense_blocks = _parse_dense_blocks(raw_dense_blocks)
            parsed_tile_3d = _parse_tile_3d(values.pop("tile_3d", None))
            raw_min_tokens = values.get("min_tokens", 0)
            if isinstance(raw_min_tokens, bool) or not isinstance(raw_min_tokens, int) or raw_min_tokens < 0:
                raise ValueError("Sol-Attn min_tokens must be a non-negative integer")
            if raw_dense_steps > 0 or parsed_tile_3d is not None or dense_blocks or raw_min_tokens > 0:
                sol_schedule: dict[str, Any] = {
                    "dense_steps": int(raw_dense_steps),
                    "dense_blocks": dense_blocks,
                    "tile_3d": parsed_tile_3d,
                    "min_tokens": int(raw_min_tokens),
                }
                model.velocity_model.__dict__["_worldfoundry_sol_schedule"] = sol_schedule
                num_blocks = len(model.velocity_model.transformer_blocks)
                norm_dense = sorted(
                    {idx + num_blocks if idx < 0 else idx for idx in dense_blocks if -num_blocks <= idx < num_blocks}
                )
                schedule_report = {
                    "dense_steps": int(raw_dense_steps),
                    "dense_blocks": norm_dense,
                    "tile_3d": parsed_tile_3d,
                    "min_tokens": int(raw_min_tokens),
                }
        elif resolved == "cudnn_fp8" and "dense_blocks" in values:
            dense_blocks = _parse_dense_blocks(values.pop("dense_blocks"))
            num_blocks = len(model.velocity_model.transformer_blocks)
            norm_dense = sorted(
                {idx + num_blocks if idx < 0 else idx for idx in dense_blocks if -num_blocks <= idx < num_blocks}
            )
            schedule_report = {"dense_blocks": norm_dense}

        configured = model.velocity_model.configure_attention_backend(
            resolved,
            self_only=(scope == "self"),
            cross_only=(scope == "cross"),
            backend_options=values or None,
            dense_blocks=dense_blocks,
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

    approximate = any(row["resolved"] in _APPROXIMATE_ATTENTION_BACKENDS for row in scopes_report.values())
    manifest: dict[str, Any] = {
        "installed": [
            {
                "name": "attention_policy",
                "approximate": approximate,
                "scopes": scopes_report,
            }
        ],
        "execution_verified": False,
    }
    model.__dict__["_worldfoundry_ltx_accelerations"] = manifest
    if applied is not None:
        applied.requested["accelerations"] = dict(active_plugins)
        applied.effective["accelerations"] = manifest
        if approximate:
            applied.quality_tier = "approximate"


class LTXJointDenoiser(FeatureCacheDenoiserMixin, GraphWrappedDenoiserMixin):
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
    ) -> None:
        if precision not in {"autocast", "weights"}:
            raise ValueError(f"unsupported LTX denoiser precision: {precision!r}")
        graph_mode = _normalize_ltx_cuda_graph_mode(enable_cuda_graph)
        _validate_ltx_graph_attention_compatibility(model, graph_mode)
        self.model = model
        self.compute_dtype = compute_dtype
        self.precision = precision
        self._teacache_threshold = teacache_threshold
        self._video_grid_cache: dict[int, Any] = {}
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
    ) -> Modality:
        batch = state.latent.shape[0]
        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ValueError("LTX sigma must be scalar or contain one value per sample")
        mask_sigma = sigma.view(batch, *([1] * (state.denoise_mask.ndim - 1)))
        return Modality(
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * mask_sigma,
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

        video = self._modality(
            video_state,
            sigma=model_input.timestep,
            context=video_context,
            context_mask=video_context_mask,
            video_control=model_input.conditioning.get("video_control"),
            attention_cache=model_input.conditioning.get("attention_cache"),
        )
        audio = self._modality(
            audio_state,
            sigma=model_input.timestep,
            context=audio_context,
            context_mask=audio_context_mask,
            attention_cache=model_input.conditioning.get("attention_cache"),
        )
        if int(model_input.step_index) == 0:
            self._begin_graph_request_window()
        extra_kwargs: dict[str, Any] = {}
        cache_kwargs = self.feature_cache_kwargs(_feature_cache_input_view(model_input))
        if cache_kwargs:
            extra_kwargs.update(cache_kwargs)
        if getattr(self, "_piecewise_graph", None) is not None:
            extra_kwargs["_worldfoundry_piecewise_graph"] = self._piecewise_graph
        sol_kwargs = _sol_schedule_kwargs(
            self.model.velocity_model,
            video_state.positions,
            int(model_input.step_index),
            self._video_grid_cache,
        )
        if sol_kwargs:
            extra_kwargs.update(sol_kwargs)
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

    def runtime_optimization_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return CUDA Graph, feature-cache, and load-time optimization telemetry."""
        applied = getattr(self.model, "_worldfoundry_applied_optimizations", None)
        report: dict[str, Any] = {
            "cuda_graph": self.graph_report(),
            "feature_cache": self.feature_cache_report(request_id),
            "applied_optimizations": None if applied is None else applied.to_optimization_snapshot(),
        }
        accelerations = getattr(self.model, "_worldfoundry_ltx_accelerations", None)
        if accelerations is not None:
            report["accelerations"] = accelerations
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


class LTXVideoDenoiser(FeatureCacheDenoiserMixin, GraphWrappedDenoiserMixin):
    """Adapt the video-only velocity model to the joint-state runner contract."""

    def __init__(
        self,
        model: LTXVideoTransformerModule,
        *,
        compute_dtype: torch.dtype,
        enable_cuda_graph: bool | str = False,
        teacache_threshold: float | None = None,
    ) -> None:
        graph_mode = _normalize_ltx_cuda_graph_mode(enable_cuda_graph)
        _validate_ltx_graph_attention_compatibility(model, graph_mode)
        self.model = model
        self.compute_dtype = compute_dtype
        self._teacache_threshold = teacache_threshold
        self._video_grid_cache: dict[int, Any] = {}
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
        video = LTXJointDenoiser._modality(
            state,
            sigma=model_input.timestep,
            context=context,
            context_mask=context_mask,
        )
        if int(model_input.step_index) == 0:
            self._begin_graph_request_window()
        extra_kwargs: dict[str, Any] = {}
        cache_kwargs = self.feature_cache_kwargs(_feature_cache_input_view(model_input))
        if cache_kwargs:
            extra_kwargs.update(cache_kwargs)
        if getattr(self, "_piecewise_graph", None) is not None:
            extra_kwargs["_worldfoundry_piecewise_graph"] = self._piecewise_graph
        sol_kwargs = _sol_schedule_kwargs(
            self.model.velocity_model,
            state.positions,
            int(model_input.step_index),
            self._video_grid_cache,
        )
        if sol_kwargs:
            extra_kwargs.update(sol_kwargs)
        with torch.autocast(
            device_type=state.latent.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            velocity, audio_velocity = self._run_network(video, None, None, **extra_kwargs)
        if velocity is None or audio_velocity is not None:
            raise RuntimeError("LTX-Video transformer returned an unexpected modality set")
        return MultiModalDenoiserOutput(samples={"video": video.latent - video.timesteps * velocity})

    def runtime_optimization_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return CUDA Graph, feature-cache, and load-time optimization telemetry."""
        applied = getattr(self.model, "_worldfoundry_applied_optimizations", None)
        report: dict[str, Any] = {
            "cuda_graph": self.graph_report(),
            "feature_cache": self.feature_cache_report(request_id),
            "applied_optimizations": None if applied is None else applied.to_optimization_snapshot(),
        }
        accelerations = getattr(self.model, "_worldfoundry_ltx_accelerations", None)
        if accelerations is not None:
            report["accelerations"] = accelerations
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
