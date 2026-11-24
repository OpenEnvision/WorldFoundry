"""Denoiser adapters for original and 1.5 HunyuanVideo checkpoints.

Each class implements :class:`~...contracts.Denoiser`.  Factories load
weights through :class:`~...loaders.module.NativeModuleLoader`.  Required
conditioning keys (``text_states``, masks, pooled embeds) raise
:exc:`KeyError` / :exc:`TypeError` when missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from ...optimizations import AttentionBackend
from ..networks.hunyuan_video.h15.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from ..networks.hunyuan_video.i2v.model import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer
from ..networks.hunyuan_video.original import HunyuanVideoDiT
from .graph_wrapped import (
    GraphWrappedDenoiserMixin,
    resolve_cuda_graph_mode,
    resolve_cuda_graph_option,
)
from .hunyuan_video_converter import HunyuanVideoDiTStateDictConverter


def _required_tensor(values: Mapping[str, object], key: str) -> torch.Tensor:
    try:
        value = values[key]
    except KeyError as error:
        raise KeyError(f"HunyuanVideo denoising requires conditioning {key!r}") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"HunyuanVideo conditioning {key!r} must be a tensor")
    return value


def _batch_timestep(value: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    timestep = value.to(device=latents.device, dtype=latents.dtype).reshape(-1)
    if timestep.numel() == 1:
        timestep = timestep.expand(latents.shape[0])
    if timestep.numel() != latents.shape[0]:
        raise ValueError("HunyuanVideo timestep must be scalar or have one value per sample")
    return timestep


def _embedded_guidance(model_input: DenoiserInput, *, multiply: float = 1.0) -> torch.Tensor:
    scale = float(model_input.conditioning.get("embedded_guidance_scale", 6.0)) * multiply
    return torch.full(
        (model_input.latents.shape[0],),
        scale,
        device=model_input.latents.device,
        dtype=model_input.latents.dtype,
    )


def _normalize_cuda_graph_arg(enabled: bool | str | None) -> bool | str:
    """Normalize direct constructor ``enable_cuda_graph`` arguments safely."""
    if enabled is None or enabled is False:
        return False
    if enabled is True:
        return True
    if isinstance(enabled, str):
        normalized = enabled.strip().casefold()
        if normalized in {"off", "none", "false", "disabled"}:
            return False
        if normalized in {"whole", "full", "true"}:
            return True
        if normalized == "piecewise":
            return "piecewise"
    raise ValueError(f"unsupported enable_cuda_graph mode {enabled!r}; expected bool, 'off', 'whole', or 'piecewise'")


def _resolve_compute_dtype(model: torch.nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Resolve the model's floating compute dtype without tripping on FP8 buffers."""
    for param in model.parameters():
        if param.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            return param.dtype
    return fallback


def _normalize_graph_report(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-safe CUDA Graph report guaranteed to carry top-level ``capture_failed``."""
    if report is None:
        return None
    normalized = dict(report)
    if "capture_failed" not in normalized:
        segments = normalized.get("segments")
        if isinstance(segments, Mapping):
            normalized["capture_failed"] = sum(
                int(seg.get("capture_failed", 0)) for seg in segments.values() if isinstance(seg, Mapping)
            )
        else:
            normalized["capture_failed"] = 0
    return normalized


class HunyuanVideoDenoiser(GraphWrappedDenoiserMixin):
    """Original HunyuanVideo T2V denoiser with embedded guidance, SOL-Engine, and CUDA Graph."""

    def __init__(self, model: HunyuanVideoDiT, *, enable_cuda_graph: bool | str = False) -> None:
        self.model = model
        self._rope: dict[tuple[tuple[int, ...], torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        graph_arg = _normalize_cuda_graph_arg(enable_cuda_graph)
        sol_runtime = getattr(model, "_sol_runtime", None)
        if sol_runtime is not None:
            # Whole-network CUDA Graph conflicts with dynamic text compaction / TeaCache /
            # Sol-Attn step gating; always route SOL-accelerated models through piecewise capture.
            if graph_arg is True:
                graph_arg = "piecewise"
            # Reuse the runtime's existing PiecewiseBlockGraphRunner instead of allocating a duplicate pool.
            if graph_arg == "piecewise" and getattr(sol_runtime, "piecewise_graph", None) is not None:
                graph_arg = False
        self._init_graph_runner(
            model,
            enabled=graph_arg,
            extra_key=f"hunyuan:{type(model).__module__}.{type(model).__qualname__}",
        )

    def close(self) -> None:
        """Release all CUDA Graph pools and SOL-Engine caches safely before process teardown."""
        if self._graph_runner is not None:
            self._graph_runner.invalidate()
        if self._piecewise_graph is not None:
            self._piecewise_graph.invalidate()
        if callable(getattr(self.model, "close_sol_acceleration", None)):
            self.model.close_sol_acceleration()

    def graph_report(self) -> Mapping[str, Any] | None:
        report = super().graph_report()
        if report is not None:
            return _normalize_graph_report(report)
        sol_runtime = getattr(self.model, "_sol_runtime", None)
        if sol_runtime is not None:
            sol_receipt = sol_runtime.receipt()
            return _normalize_graph_report(sol_receipt.get("cuda_graph"))
        return None

    def acceleration_receipt(self) -> dict[str, Any]:
        """Return auditable SOL-Engine and CUDA Graph telemetry for this denoiser."""
        sol_runtime = getattr(self.model, "_sol_runtime", None)
        active_graph = _normalize_graph_report(self.graph_report())
        if sol_runtime is not None:
            receipt = dict(sol_runtime.receipt())
            fallbacks = list(receipt.get("fallbacks", ()))
            if active_graph is not None:
                receipt["cuda_graph"] = active_graph
                capture_failed = int(active_graph.get("capture_failed", 0))
                if capture_failed > 0:
                    entry = f"cuda_graph_capture_failed={capture_failed}"
                    if entry not in fallbacks:
                        fallbacks.append(entry)
            receipt["fallbacks"] = fallbacks
            return receipt
        fallbacks = []
        if active_graph is not None and int(active_graph.get("capture_failed", 0)) > 0:
            fallbacks.append(f"cuda_graph_capture_failed={int(active_graph['capture_failed'])}")
        return {
            "engine": "worldfoundry-hunyuan13b-dense",
            "cuda_graph": active_graph,
            "fallbacks": fallbacks,
        }

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        shape_key = tuple(model_input.latents.shape[2:])
        cache_key = (shape_key, model_input.latents.device)
        freqs = self._rope.get(cache_key)
        if freqs is None:
            raw_cos, raw_sin = self.model.prepare_freqs(model_input.latents)
            freqs = (
                raw_cos.to(device=model_input.latents.device),
                raw_sin.to(device=model_input.latents.device),
            )
            if len(self._rope) >= 16:
                self._rope.clear()
            self._rope[cache_key] = freqs
        cos, sin = freqs
        compute_dtype = _resolve_compute_dtype(self.model, model_input.latents.dtype)
        extra: dict[str, Any] = {"piecewise_graph": self._piecewise_graph} if self._piecewise_graph is not None else {}
        if getattr(self.model, "_sol_runtime", None) is not None:
            extra.update(
                {
                    "step_index": model_input.step_index,
                    "total_steps": model_input.total_steps,
                    "request_id": model_input.request_id,
                    "branch": model_input.branch,
                }
            )
        with torch.autocast(
            device_type=model_input.latents.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            sample = self._run_network(
                x=model_input.latents,
                t=_batch_timestep(model_input.timestep, model_input.latents),
                prompt_emb=_required_tensor(model_input.conditioning, "text_states"),
                text_mask=_required_tensor(model_input.conditioning, "text_mask"),
                pooled_prompt_emb=_required_tensor(model_input.conditioning, "text_states_2"),
                freqs_cos=cos,
                freqs_sin=sin,
                guidance=_embedded_guidance(model_input),
                **extra,
            )
        return DenoiserOutput(sample=sample)


class HunyuanVideoI2VDenoiser(GraphWrappedDenoiserMixin):
    """Original token-replacement I2V denoiser with a frozen first latent frame."""

    def __init__(
        self,
        model: HYVideoDiffusionTransformer,
        *,
        rope_theta: float = 256.0,
        enable_cuda_graph: bool | str = False,
    ) -> None:
        self.model = model
        self.rope_theta = float(rope_theta)
        self._rope: dict[tuple[tuple[int, ...], torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_graph_runner(
            model,
            enabled=_normalize_cuda_graph_arg(enable_cuda_graph),
            extra_key=f"hunyuan_i2v:{type(model).__module__}.{type(model).__qualname__}",
        )

    def _freqs(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape_key = tuple(latents.shape[2:])
        cache_key = (shape_key, latents.device)
        values = self._rope.get(cache_key)
        if values is None:
            from worldfoundry.core.attention import get_nd_rotary_pos_embed

            sizes = tuple(size // patch for size, patch in zip(shape_key, self.model.patch_size, strict=False))
            raw_cos, raw_sin = get_nd_rotary_pos_embed(
                self.model.rope_dim_list,
                sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1.0,
            )
            values = (raw_cos.to(device=latents.device), raw_sin.to(device=latents.device))
            if len(self._rope) >= 16:
                self._rope.clear()
            self._rope[cache_key] = values
        return values

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        cos, sin = self._freqs(model_input.latents)
        compute_dtype = _resolve_compute_dtype(self.model, model_input.latents.dtype)
        self.model._active_piecewise_graph = self._piecewise_graph
        try:
            with torch.autocast(
                device_type=model_input.latents.device.type,
                dtype=compute_dtype,
                enabled=compute_dtype in {torch.float16, torch.bfloat16},
            ):
                output = self._run_network(
                    x=model_input.latents,
                    t=_batch_timestep(model_input.timestep, model_input.latents),
                    text_states=_required_tensor(model_input.conditioning, "text_states"),
                    text_mask=_required_tensor(model_input.conditioning, "text_mask"),
                    text_states_2=_required_tensor(model_input.conditioning, "text_states_2"),
                    freqs_cos=cos,
                    freqs_sin=sin,
                    guidance=_embedded_guidance(model_input, multiply=1000.0),
                    return_dict=False,
                )
        finally:
            model_any: Any = self.model
            model_any._active_piecewise_graph = None
        sample = output.clone()
        sample[:, :, :1] = 0
        return DenoiserOutput(sample=sample)


class HunyuanVideo15Denoiser(GraphWrappedDenoiserMixin):
    """HunyuanVideo 1.5 denoiser using native concat-condition inputs."""

    def __init__(
        self,
        model: HunyuanVideo_1_5_DiffusionTransformer,
        *,
        image_to_video: bool,
        enable_cuda_graph: bool | str = False,
    ) -> None:
        self.model = model
        self.image_to_video = bool(image_to_video)
        self._rope: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_graph_runner(
            model,
            enabled=_normalize_cuda_graph_arg(enable_cuda_graph),
            extra_key=f"hunyuan15:{type(model).__module__}.{type(model).__qualname__}",
        )

    def _freqs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, ot, oh, ow = hidden_states.shape
        tt = ot // self.model.patch_size[0]
        th = oh // self.model.patch_size[1]
        tw = ow // self.model.patch_size[2]
        key = (tt, th, tw, hidden_states.device)
        values = self._rope.get(key)
        if values is None:
            cos, sin = self.model.get_rotary_pos_embed((tt, th, tw))
            values = (cos.to(device=hidden_states.device), sin.to(device=hidden_states.device))
            if len(self._rope) >= 16:
                self._rope.clear()
            self._rope[key] = values
        return values

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        condition = _required_tensor(model_input.conditioning, "condition_latents").to(model_input.latents)
        if condition.shape[0] != model_input.latents.shape[0] or condition.shape[2:] != model_input.latents.shape[2:]:
            raise ValueError("HunyuanVideo 1.5 condition latent geometry must match the noise trajectory")
        hidden_states = torch.cat((model_input.latents, condition), dim=1)
        vision = model_input.conditioning.get("vision_states")
        if vision is None:
            vision = torch.zeros(
                hidden_states.shape[0],
                729,
                1152,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if not isinstance(vision, torch.Tensor):
            raise TypeError("HunyuanVideo 1.5 vision_states must be a tensor")
        extra_kwargs = {
            "byt5_text_states": _required_tensor(model_input.conditioning, "byt5_text_states"),
            "byt5_text_mask": _required_tensor(model_input.conditioning, "byt5_text_mask"),
        }
        timestep = _batch_timestep(model_input.timestep, model_input.latents)
        timestep_r = None
        if bool(self.model.config.use_meanflow):
            timestep_r = _batch_timestep(model_input.next_timestep, model_input.latents)
        freqs_cos, freqs_sin = self._freqs(hidden_states)
        compute_dtype = _resolve_compute_dtype(self.model, hidden_states.dtype)
        self.model._active_piecewise_graph = self._piecewise_graph
        try:
            with torch.autocast(
                device_type=hidden_states.device.type,
                dtype=compute_dtype,
                enabled=compute_dtype in {torch.float16, torch.bfloat16},
            ):
                output, _ = self._run_network(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    timestep_r=timestep_r,
                    text_states=_required_tensor(model_input.conditioning, "text_states"),
                    text_states_2=model_input.conditioning.get("text_states_2"),
                    encoder_attention_mask=_required_tensor(model_input.conditioning, "text_mask"),
                    vision_states=vision.to(device=hidden_states.device, dtype=hidden_states.dtype),
                    guidance=_embedded_guidance(model_input, multiply=1000.0),
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    mask_type="i2v" if self.image_to_video else "t2v",
                    extra_kwargs=extra_kwargs,
                    return_dict=False,
                )
        finally:
            model_any: Any = self.model
            model_any._active_piecewise_graph = None
        return DenoiserOutput(sample=output)


def _original_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return HunyuanVideoDiTStateDictConverter().from_civitai(state_dict)


def _resolve_graph_arg(context: ComponentBuildContext) -> bool | str:
    mode = resolve_cuda_graph_mode(context)
    if mode == "piecewise":
        return "piecewise"
    return resolve_cuda_graph_option(context)


def build_hunyuan_video_denoiser(context: ComponentBuildContext) -> HunyuanVideoDenoiser:
    sol_config = context.component_options.get(
        "sol_acceleration",
        context.component_options.get(
            "sol_profile",
            context.policy.options.get("sol_acceleration", context.policy.options.get("sol_profile")),
        ),
    )
    want_fuse_qkv = bool(context.component_options.get("fuse_qkv", context.policy.options.get("fuse_qkv", False)))

    def _post_load_hunyuan(loaded: torch.nn.Module) -> None:
        if not isinstance(loaded, HunyuanVideoDiT):
            return
        if sol_config is not None and sol_config is not False:
            loaded.enable_sol_acceleration(sol_config)
        elif want_fuse_qkv:
            from ..networks.hunyuan_video.original_acceleration import fuse_hunyuan13b_projections

            fuse_hunyuan13b_projections(loaded, release_originals=True)

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HunyuanVideoDiT,
            state_dict_converter=_original_state_dict,
            layer_container="double_blocks",
            compile_regions=("double_blocks", "single_blocks"),
            post_load_hook=_post_load_hunyuan if (sol_config or want_fuse_qkv) else None,
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HunyuanVideoDiT):
        raise TypeError(f"expected HunyuanVideoDiT, got {type(model).__name__}")
    graph_arg = _resolve_graph_arg(context)
    if getattr(model, "_sol_runtime", None) is not None and graph_arg is True:
        # Whole-network capture conflicts with dynamic TeaCache/Sol-Attn step gating;
        # route through PiecewiseBlockGraphRunner instead.
        graph_arg = "piecewise"
    return HunyuanVideoDenoiser(model, enable_cuda_graph=graph_arg)


def build_hunyuan_video_i2v_denoiser(context: ComponentBuildContext) -> HunyuanVideoI2VDenoiser:
    args = SimpleNamespace(
        text_states_dim=4096,
        text_states_dim_2=768,
        i2v_condition_type="token_replace",
        gradient_checkpoint=False,
        gradient_checkpoint_layers=-1,
    )
    config = dict(HUNYUAN_VIDEO_CONFIG["HYVideo-T/2-cfgdistill"])
    config.update({"args": args, "in_channels": 16, "out_channels": 16})
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HYVideoDiffusionTransformer,
            config=config,
            layer_container="double_blocks",
            compile_regions=("double_blocks", "single_blocks"),
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HYVideoDiffusionTransformer):
        raise TypeError(f"expected HYVideoDiffusionTransformer, got {type(model).__name__}")
    return HunyuanVideoI2VDenoiser(model, enable_cuda_graph=_resolve_graph_arg(context))


def _h15_config(checkpoint, relative_path: str, attention: AttentionBackend) -> Mapping[str, object]:
    config = dict(checkpoint_json_config(checkpoint, relative_path))
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    config["attn_mode"] = "flash" if attention is AttentionBackend.FLASH else "torch"
    return config


def build_hunyuan_video15_denoiser(context: ComponentBuildContext) -> HunyuanVideo15Denoiser:
    config_path = str(context.component_options.get("config_path", "transformer/480p_t2v_distilled/config.json"))
    want_fuse_qkv = bool(context.component_options.get("fuse_qkv", context.policy.options.get("fuse_qkv", False)))

    def _fuse_h15_qkv(loaded: torch.nn.Module) -> None:
        if isinstance(loaded, HunyuanVideo_1_5_DiffusionTransformer):
            loaded.fuse_qkv()

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HunyuanVideo_1_5_DiffusionTransformer,
            config_resolver=lambda checkpoint: _h15_config(checkpoint, config_path, context.policy.attention),
            layer_container="double_blocks",
            compile_regions=("double_blocks", "single_blocks"),
            post_load_hook=_fuse_h15_qkv if want_fuse_qkv else None,
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, HunyuanVideo_1_5_DiffusionTransformer):
        raise TypeError(f"expected HunyuanVideo 1.5 transformer, got {type(model).__name__}")
    return HunyuanVideo15Denoiser(
        model,
        image_to_video=bool(context.component_options.get("image_to_video", False)),
        enable_cuda_graph=_resolve_graph_arg(context),
    )


__all__ = [
    "HunyuanVideo15Denoiser",
    "HunyuanVideoDenoiser",
    "HunyuanVideoI2VDenoiser",
    "build_hunyuan_video15_denoiser",
    "build_hunyuan_video_denoiser",
    "build_hunyuan_video_i2v_denoiser",
]
