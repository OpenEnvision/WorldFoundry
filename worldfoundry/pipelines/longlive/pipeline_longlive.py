"""Native LongLive-1.3B interactive long-horizon streaming video pipeline.

Builds directly on :class:`NativeSelfForcingPipeline` and WorldFoundry's
``CausalWanModel`` (1.3B) + ``AutoregressiveWindowRunner`` stack, adding:

1. Automatic checkpoint layout resolution for the released LongLive-1.3B
   directory (``models/longlive_base.pt`` + optional ``models/lora.pt`` paired
   with Wan2.1-T2V-1.3B T5 encoder, tokenizer, and causal VAE).
2. Load-time zero-overhead FP32 LoRA merge (:func:`merge_longlive_lora`) into
   base ``nn.Linear`` layers *before* QKV projection fusion and Hopper FP8
   W8A8 / fused FP8 FFN quantization so LoRA adds zero inference latency.
3. Zero-recapture interactive prompt switching (:meth:`switch_prompt_realtime`
   and :meth:`stream_interactive_overlapped`) with in-place cross-attention K/V
   refresh (preserving tensor addresses so captured CUDA Graphs stay valid) and
   optional LongLive KV-recache over the most recent committed clean block.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.core.model_loading.optimize import apply_quantization_policy
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy

from ..forcing.pipeline_native_forcing import NativeSelfForcingPipeline


def _strip_lora_prefix(key: str) -> str:
    for prefix in ("base_model.model.", "generator.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _resolve_submodule(root: nn.Module, path: str) -> nn.Module | None:
    current: nn.Module = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        next_mod = getattr(current, part)
        if not isinstance(next_mod, nn.Module):
            return None
        current = next_mod
    return current


def merge_longlive_lora(
    model: nn.Module,
    lora_source: str | os.PathLike[str] | Mapping[str, torch.Tensor],
    *,
    alpha: float | None = None,
    scale: float | None = None,
) -> dict[str, Any]:
    """Merge LongLive ``generator_lora`` (rank-256) weights into ``nn.Linear`` in place.

    Must be called on pristine dense ``nn.Linear`` layers *before* QKV projection
    fusion or FP8 quantization so merged weights flow cleanly into both without
    any runtime LoRA overhead.
    """
    source_desc: str
    if isinstance(lora_source, Mapping):
        raw_state: Mapping[str, Any] = lora_source
        source_desc = "<in-memory-mapping>"
    else:
        path = Path(lora_source)
        if not path.is_file():
            raise FileNotFoundError(f"LongLive LoRA checkpoint not found: {path}")
        raw_state = torch.load(path, map_location="cpu", weights_only=False)
        source_desc = str(path)

    lora_dict: Mapping[str, Any]
    if (
        isinstance(raw_state, Mapping)
        and "generator_lora" in raw_state
        and isinstance(raw_state["generator_lora"], Mapping)
    ):
        lora_dict = raw_state["generator_lora"]
    elif isinstance(raw_state, Mapping):
        lora_dict = raw_state
    else:
        raise TypeError(f"expected mapping for LongLive LoRA checkpoint, got {type(raw_state).__name__}")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    for raw_key, tensor in lora_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        clean = _strip_lora_prefix(str(raw_key))
        if clean.endswith(".lora_A.weight"):
            target = clean[: -len(".lora_A.weight")]
            pairs.setdefault(target, {})["A"] = tensor
        elif clean.endswith(".lora_B.weight"):
            target = clean[: -len(".lora_B.weight")]
            pairs.setdefault(target, {})["B"] = tensor
        elif clean.endswith(".lora_A.default.weight"):
            target = clean[: -len(".lora_A.default.weight")]
            pairs.setdefault(target, {})["A"] = tensor
        elif clean.endswith(".lora_B.default.weight"):
            target = clean[: -len(".lora_B.default.weight")]
            pairs.setdefault(target, {})["B"] = tensor
        else:
            skipped.append(str(raw_key))

    merged_layers = 0
    observed_ranks: set[int] = set()
    used_scale: float | None = None

    with torch.no_grad():
        for target_path, ab in sorted(pairs.items()):
            if "A" not in ab or "B" not in ab:
                skipped.append(target_path)
                continue
            module = _resolve_submodule(model, target_path)
            if module is None:
                raise KeyError(f"LongLive LoRA target module not found in model: {target_path!r}")
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"LongLive LoRA target {target_path!r} is {type(module).__name__}, expected nn.Linear "
                    "(merge LoRA before QKV fusion or FP8 quantization)"
                )
            lora_a = ab["A"]
            lora_b = ab["B"]
            rank = int(lora_a.shape[0])
            if lora_b.shape[1] != rank:
                raise ValueError(
                    f"LoRA rank mismatch for {target_path!r}: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
                )
            if (lora_b.shape[0], lora_a.shape[1]) != tuple(module.weight.shape):
                raise ValueError(
                    f"LoRA shape mismatch for {target_path!r}: B@A={(lora_b.shape[0], lora_a.shape[1])} "
                    f"!= weight={tuple(module.weight.shape)}"
                )
            eff_scale = (
                float(scale) if scale is not None else (float(alpha) / float(rank) if alpha is not None else 1.0)
            )
            observed_ranks.add(rank)
            used_scale = eff_scale

            device = module.weight.device
            dtype = module.weight.dtype
            a_f32 = lora_a.to(device=device, dtype=torch.float32)
            b_f32 = lora_b.to(device=device, dtype=torch.float32)
            delta = (b_f32 @ a_f32) * eff_scale
            merged = (module.weight.data.float() + delta).to(dtype=dtype)
            module.weight.data.copy_(merged)
            merged_layers += 1

    return {
        "applied": merged_layers > 0,
        "source": source_desc,
        "merged_layers": merged_layers,
        "ranks": sorted(observed_ranks),
        "scale": used_scale if used_scale is not None else (1.0 if scale is None else float(scale)),
        "skipped_keys": len(skipped),
    }


def _coerce_quantization_policy(spec: Any) -> QuantizationPolicy | None:
    if spec is None or spec is False:
        return None
    if isinstance(spec, QuantizationPolicy):
        return spec if spec.mode is not QuantizationMode.NONE else None
    if isinstance(spec, str):
        mode = QuantizationMode(spec.strip().lower())
        return None if mode is QuantizationMode.NONE else QuantizationPolicy(mode=mode)
    if isinstance(spec, Mapping):
        raw_mode = spec.get("mode") or spec.get("quantization") or "fp8"
        mode = QuantizationMode(str(raw_mode).strip().lower())
        if mode is QuantizationMode.NONE:
            return None
        exclude = tuple(str(x) for x in spec.get("exclude", ()))
        options = dict(spec.get("options", {}))
        for opt_key in ("scaling", "use_fast_accum", "keep_dense_fallback", "fuse_fp8_ffn", "min_features"):
            if opt_key in spec and opt_key not in options:
                options[opt_key] = spec[opt_key]
        return QuantizationPolicy(mode=mode, exclude=exclude, options=options)
    raise TypeError(f"unsupported quantization specification: {type(spec).__name__}")


def _resolve_longlive_layout(
    model_path: str | os.PathLike[str] | Mapping[str, Any] | None,
    required_components: Mapping[str, Any] | None,
    *,
    wan_base_path: str | os.PathLike[str] | None,
    lora_path: str | os.PathLike[str] | None,
    apply_lora: bool,
) -> tuple[str | Mapping[str, Any] | None, dict[str, Any], str | None]:
    """Resolve official LongLive-1.3B directory layout into Self-Forcing component paths."""
    resolved_components = dict(required_components or {})
    resolved_lora: str | None = None if lora_path is None else os.fspath(lora_path)
    resolved_model_path: str | Mapping[str, Any] | None = (
        os.fspath(model_path) if isinstance(model_path, os.PathLike) else model_path
    )

    if isinstance(model_path, (str, os.PathLike)):
        root = Path(model_path)
        if root.is_dir():
            base_pt = root / "models" / "longlive_base.pt"
            if base_pt.is_file():
                resolved_model_path = str(base_pt)
            if apply_lora and resolved_lora is None:
                candidate_lora = root / "models" / "lora.pt"
                if candidate_lora.is_file():
                    resolved_lora = str(candidate_lora)

            wan_root: Path | None = None
            if wan_base_path is not None:
                wan_root = Path(wan_base_path)
            else:
                sibling_wan = root.parent / "Wan2.1-T2V-1.3B"
                if sibling_wan.is_dir():
                    wan_root = sibling_wan
            if wan_root is not None and wan_root.is_dir():
                if "text-encoder" not in resolved_components:
                    t5_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
                    if t5_path.is_file():
                        resolved_components["text-encoder"] = str(t5_path)
                if "tokenizer" not in resolved_components:
                    tok_dir = wan_root / "google" / "umt5-xxl"
                    if tok_dir.is_dir():
                        resolved_components["tokenizer"] = str(tok_dir)
                if "vae" not in resolved_components:
                    vae_path = wan_root / "Wan2.1_VAE.pth"
                    if vae_path.is_file():
                        resolved_components["vae"] = str(vae_path)
    elif wan_base_path is not None:
        wan_root = Path(wan_base_path)
        if wan_root.is_dir():
            resolved_components.setdefault("text-encoder", str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth"))
            resolved_components.setdefault("tokenizer", str(wan_root / "google" / "umt5-xxl"))
            resolved_components.setdefault("vae", str(wan_root / "Wan2.1_VAE.pth"))

    if not apply_lora:
        resolved_lora = None
    return resolved_model_path, resolved_components, resolved_lora


def load_interactive_prompt_jsonl(path: str | os.PathLike[str]) -> list[str]:
    """Load prompt lines from LongLive's ``prompts/interactive_example.jsonl`` format."""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict) and "prompts" in item and isinstance(item["prompts"], list):
                prompts.extend(str(p).strip() for p in item["prompts"] if str(p).strip())
            elif isinstance(item, dict) and "prompt" in item:
                text = str(item["prompt"]).strip()
                if text:
                    prompts.append(text)
            elif isinstance(item, str) and item.strip():
                prompts.append(item.strip())
    return prompts


class _LongLiveWindowedDenoiserProxy:
    """Transparent proxy routing denoiser calls through :class:`LongLiveFastDenoiserEngine`."""

    def __init__(self, base_denoiser: Any, engine: Any) -> None:
        self._base = base_denoiser
        self._engine = engine

    def __call__(self, model_input: Any) -> Any:
        return self._engine(model_input)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class NativeLongLivePipeline(NativeSelfForcingPipeline):
    """Accelerated native LongLive-1.3B interactive streaming video pipeline.

    Inherits the full native causal Wan 1.3B runner, chunkwise CUDA Graph
    support, real-valued RoPE, fused QKV, Hopper FP8 W8A8 + fused FP8 FFN,
    and overlapped dual-stream DiT/VAE execution from
    :class:`NativeSelfForcingPipeline`, and adds:

    - Load-time FP32 merge of LongLive's rank-256 ``generator_lora`` before
      QKV fusion and FP8 quantization (zero runtime LoRA cost).
    - Long-horizon ``sink5-window7-rerope`` sliding-window KV cache default.
    - Seamless interactive prompt switching (:meth:`switch_prompt_realtime` &
      :meth:`stream_interactive_overlapped`) with in-place cross-attention K/V
      refresh and optional KV-recache over the most recent committed block.
    """

    MODEL_ID = "self-forcing"
    OWNER = "LongLive"
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_FPS = 16

    def __init__(
        self,
        *,
        native_pipeline: Any,
        device: str,
        model_id: str | None = None,
        lora_report: dict[str, Any] | None = None,
        post_merge_qkv_blocks: int = 0,
        post_merge_quant_report: Any = None,
    ) -> None:
        super().__init__(native_pipeline=native_pipeline, device=device, model_id=model_id)
        self._lora_report: dict[str, Any] = dict(lora_report or {"applied": False, "merged_layers": 0})
        self._post_merge_qkv_blocks: int = int(post_merge_qkv_blocks)
        self._post_merge_quant_report: Any = post_merge_quant_report
        self._fast_engine: Any = None
        self._last_clean_latent: torch.Tensor | None = None
        self._prompt_switches: int = 0

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | os.PathLike[str] | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        *,
        wan_base_path: str | os.PathLike[str] | None = None,
        lora_path: str | os.PathLike[str] | None = None,
        apply_lora: bool = True,
        lora_alpha: float | None = None,
        lora_scale: float | None = None,
        kv_window: str | None = "sink5-window7-rerope",
        fast_longlive: bool = True,
        **kwargs: Any,
    ) -> NativeLongLivePipeline:
        """Load LongLive-1.3B with optional zero-overhead LoRA merge + SOL optimizations."""
        resolved_model_path, resolved_components, resolved_lora = _resolve_longlive_layout(
            model_path,
            required_components,
            wan_base_path=wan_base_path,
            lora_path=lora_path,
            apply_lora=apply_lora,
        )
        load_kwargs = dict(kwargs)
        load_kwargs.setdefault("offload_mode", "none")
        if kv_window is not None:
            load_kwargs["kv_window"] = kv_window

        # When streaming_compile is requested on CUDA, enable channels_last_3d on the
        # causal Wan VAE by default (unless caller explicitly opted out) to eliminate
        # ~30 ms/chunk of cuDNN NCHW<->NHWC layout conversions in lossless BF16.
        if (
            bool(load_kwargs.get("vae_streaming_compile", False))
            and str(device).startswith("cuda")
            and "vae_channels_last_3d" not in load_kwargs
        ):
            load_kwargs["vae_channels_last_3d"] = True
            load_kwargs.setdefault("vae_channels_last", True)

        requested_attention = load_kwargs.get("attention_backend") or load_kwargs.get("attention")
        if resolved_lora is not None and str(load_kwargs.get("offload_mode", "none")).strip().lower() != "none":
            raise ValueError(
                "LongLive load-time LoRA merge requires offload_mode='none' so base weights are materialized"
            )
        if resolved_lora is not None and bool(load_kwargs.get("compile", False)):
            raise ValueError(
                "LongLive load-time LoRA merge cannot combine with load-time compile=True; "
                "use cuda_graph='piecewise' or compile after merge"
            )

        # When merging LoRA, defer QKV fusion & FP8 quantization until AFTER
        # LoRA weights have been folded into dense nn.Linear parameters.
        deferred_fuse_qkv = False
        deferred_quant_policy: QuantizationPolicy | None = None
        qkv_strategy = str(load_kwargs.get("qkv_strategy", "auto"))
        qkv_split_threshold = int(load_kwargs.get("qkv_split_threshold", 8192))

        if resolved_lora is not None:
            deferred_fuse_qkv = bool(load_kwargs.pop("fuse_qkv", False))
            quant_raw = load_kwargs.pop("quantization", None)
            quant_exclude = load_kwargs.pop("quantization_exclude", None)
            quant_options = load_kwargs.pop("quantization_options", None)
            if quant_raw is not None:
                if isinstance(quant_raw, str):
                    spec: dict[str, Any] = {"mode": quant_raw}
                    if quant_exclude is not None:
                        spec["exclude"] = tuple(quant_exclude)
                    if quant_options is not None:
                        spec["options"] = dict(quant_options)
                    deferred_quant_policy = _coerce_quantization_policy(spec)
                else:
                    deferred_quant_policy = _coerce_quantization_policy(quant_raw)

        # Avoid rejecting flash_attention_3 in base loader if LongLiveFastDenoiserEngine handles it on CUDA.
        if fast_longlive and str(device).startswith("cuda") and requested_attention is not None:
            load_kwargs.pop("attention_backend", None)
            load_kwargs.pop("attention", None)

        base_pipe = NativeSelfForcingPipeline.from_pretrained(
            model_path=resolved_model_path,
            required_components=resolved_components or None,
            device=device,
            model_id=model_id or "self-forcing",
            **load_kwargs,
        )

        dit_model = base_pipe._window_runner.windowed_denoiser.model
        lora_report: dict[str, Any] = {"applied": False, "merged_layers": 0}
        if resolved_lora is not None:
            lora_report = merge_longlive_lora(
                dit_model,
                resolved_lora,
                alpha=lora_alpha,
                scale=lora_scale,
            )

        applied = getattr(dit_model, "_worldfoundry_applied_optimizations", None)
        fused_blocks = 0
        if deferred_fuse_qkv:
            from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections

            fused_blocks = fuse_qkv_projections(
                dit_model,
                strategy=qkv_strategy,
                split_threshold=qkv_split_threshold,
            )
            if applied is not None and hasattr(applied, "record_fusion"):
                applied.record_fusion(
                    requested=True,
                    fused_blocks=fused_blocks,
                    strategy=qkv_strategy,
                    split_threshold=qkv_split_threshold,
                )

        quant_report: Any = None
        if deferred_quant_policy is not None:
            quant_report = apply_quantization_policy(dit_model, deferred_quant_policy)
            if applied is not None and hasattr(applied, "record_quantization"):
                applied.record_quantization(quant_report)

        instance = cls(
            native_pipeline=base_pipe.native_pipeline,
            device=device,
            model_id=model_id or "self-forcing",
            lora_report=lora_report,
            post_merge_qkv_blocks=fused_blocks,
            post_merge_quant_report=quant_report,
        )
        if hasattr(base_pipe._codec, "streaming_compile"):
            instance._codec.streaming_compile = getattr(base_pipe._codec, "streaming_compile", False)

        # Install LongLive's O(1) bounded rolling-window graph + Hopper FA3 + fused Triton engine.
        if fast_longlive and str(device).startswith("cuda"):
            from .fast_kernels import LongLiveFastDenoiserEngine

            runner = instance._window_runner
            base_denoiser = runner.windowed_denoiser
            cg_enabled = bool(getattr(base_denoiser, "enable_cuda_graph", False))
            attn_choice = str(requested_attention) if requested_attention is not None else "torch"
            fast_engine = LongLiveFastDenoiserEngine(
                base_denoiser,
                attention_backend=attn_choice,
                enable_cuda_graph=cg_enabled,
            )
            proxy = _LongLiveWindowedDenoiserProxy(base_denoiser, fast_engine)
            object.__setattr__(runner.components, "denoiser", proxy)
            runner.windowed_denoiser = cast(Any, proxy)
            instance._fast_engine = fast_engine
        return instance

    @property
    def lora_report(self) -> dict[str, Any]:
        """Return load-time LongLive LoRA merge telemetry."""
        return dict(self._lora_report)

    def runtime_optimization_report(self) -> dict[str, Any]:
        """Expose combined denoiser (CUDA Graph / FP8 / QKV), LoRA merge, and interactive switch telemetry."""
        denoiser = self._window_runner.windowed_denoiser
        report_fn = getattr(denoiser, "runtime_optimization_report", None)
        base_report: dict[str, Any] = report_fn() if callable(report_fn) else {}
        base_report["longlive"] = {
            "lora": dict(self._lora_report),
            "post_merge_qkv_blocks": self._post_merge_qkv_blocks,
            "post_merge_quantization": (
                self._post_merge_quant_report.to_dict()
                if hasattr(self._post_merge_quant_report, "to_dict")
                else (
                    vars(self._post_merge_quant_report)
                    if self._post_merge_quant_report is not None and hasattr(self._post_merge_quant_report, "__dict__")
                    else None
                )
            ),
            "kv_window": getattr(denoiser, "kv_window", None),
            "prompt_switches": self._prompt_switches,
            "fast_engine": self._fast_engine.report() if self._fast_engine is not None else None,
        }
        return base_report

    @staticmethod
    @torch.no_grad()
    def _refresh_crossattn_cache_inplace_static(
        model: nn.Module,
        context: torch.Tensor,
        crossattn_cache: list[dict[str, Any]],
    ) -> bool:
        """Recompute cross-attention K/V for ``context`` and copy in-place to preserve CUDA Graph addresses."""
        wan_model: Any = model
        text_len = int(getattr(wan_model, "text_len", 512))
        compute_dtype = context.dtype
        for param in wan_model.text_embedding.parameters():
            if param.dtype in {torch.float16, torch.bfloat16, torch.float32}:
                compute_dtype = param.dtype
                break

        ctx_in = context.to(dtype=compute_dtype)
        with torch.autocast(
            device_type=ctx_in.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            if ctx_in.ndim == 3 and ctx_in.shape[1] == text_len:
                ctx_emb = wan_model.text_embedding(ctx_in)
            else:
                padded_rows = []
                for u in ctx_in:
                    if u.size(0) >= text_len:
                        padded_rows.append(u[:text_len])
                    else:
                        padded_rows.append(torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))], dim=0))
                ctx_emb = wan_model.text_embedding(torch.stack(padded_rows, dim=0))

            batch_size = int(ctx_emb.shape[0])
            in_place_preserved = True
            for block, layer_cache in zip(wan_model.blocks, crossattn_cache, strict=True):
                cross = block.cross_attn
                num_heads = int(cross.num_heads)
                head_dim = int(cross.head_dim)
                if getattr(cross, "_kv_fused", False) and hasattr(cross, "kv"):
                    k_proj, v_proj = cross.kv(ctx_emb).chunk(2, dim=-1)
                else:
                    k_proj = cross.k(ctx_emb)
                    v_proj = cross.v(ctx_emb)
                k_new = cross.norm_k(k_proj).view(batch_size, -1, num_heads, head_dim)
                v_new = v_proj.view(batch_size, -1, num_heads, head_dim)

                existing_k = layer_cache.get("k")
                existing_v = layer_cache.get("v")
                can_copy_inplace = (
                    isinstance(existing_k, torch.Tensor)
                    and isinstance(existing_v, torch.Tensor)
                    and existing_k.shape == k_new.shape
                    and existing_v.shape == v_new.shape
                    and existing_k.device == k_new.device
                    and existing_v.device == v_new.device
                    and (not existing_k.is_inference() or torch.is_inference_mode_enabled())
                    and (not existing_v.is_inference() or torch.is_inference_mode_enabled())
                )
                if can_copy_inplace:
                    assert isinstance(existing_k, torch.Tensor) and isinstance(existing_v, torch.Tensor)
                    existing_k.copy_(k_new)
                    existing_v.copy_(v_new)
                else:
                    # Allocate persistent cache buffers as normal (non-inference) tensors so later
                    # create_kv_cache() / switch_prompt_realtime() calls outside inference_mode()
                    # can safely mutate them in place.
                    with torch.inference_mode(False):
                        k_buf = torch.empty(k_new.shape, dtype=k_new.dtype, device=k_new.device)
                        v_buf = torch.empty(v_new.shape, dtype=v_new.dtype, device=v_new.device)
                        k_buf.copy_(k_new)
                        v_buf.copy_(v_new)
                    layer_cache["k"] = k_buf
                    layer_cache["v"] = v_buf
                    in_place_preserved = False
                layer_cache["is_init"] = True
        return in_place_preserved

    @torch.no_grad()
    def _refresh_crossattn_cache_inplace(
        self,
        model: nn.Module,
        context: torch.Tensor,
        crossattn_cache: list[dict[str, Any]],
    ) -> bool:
        return self._refresh_crossattn_cache_inplace_static(model, context, crossattn_cache)

    @torch.no_grad()
    def switch_prompt_realtime(
        self,
        prompt: str,
        *,
        recache_kv: bool = True,
    ) -> dict[str, Any]:
        """Switch active prompt mid-rollout without dropping frames or restarting the stream.

        1. Encodes ``prompt`` through the T5 conditioner.
        2. Updates ``stream._state.positive_base["context"]`` and refreshes every
           layer's ``crossattn_cache`` K/V in place (keeping buffer addresses
           identical so captured CUDA Graphs remain valid).
        3. When ``recache_kv=True`` and at least one clean block has been
           committed, performs a single ``t=0`` LongLive KV-recache pass over
           the most recent committed block so self-attention KV history adapts
           smoothly to the new prompt without advancing ``global_end_index``.
        """
        stream = self._realtime_stream
        if stream is None or getattr(stream, "_state", None) is None:
            raise RuntimeError("LongLive realtime session is not configured; call configure_realtime() first.")
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise ValueError("switch_prompt_realtime requires a non-empty prompt")

        t0 = time.perf_counter()
        runner = self._window_runner
        state = stream._state
        assert state is not None

        request, _ = self._prepare_call(
            prompt=clean_prompt,
            seed=self._realtime_seed,
            **self._realtime_options,
        )
        conditioning = runner.components.conditioner.encode(
            request,
            device=runner.device,
            dtype=runner.dtype,
        )
        new_context = conditioning.positive.get("context")
        if not isinstance(new_context, torch.Tensor):
            raise TypeError("conditioner did not return tensor 'context' for LongLive prompt switch")

        # Update stream positive conditioning in place.
        state.positive_base["context"] = new_context
        self._realtime_prompt = clean_prompt

        cache = state.positive_cache
        crossattn_cache = cache["crossattn_cache"] if isinstance(cache, Mapping) else None
        model = cast(Any, runner.windowed_denoiser).model
        in_place_preserved = False
        if isinstance(crossattn_cache, list) and len(crossattn_cache) == len(model.blocks):
            in_place_preserved = self._refresh_crossattn_cache_inplace(model, new_context, crossattn_cache)

        recached_self_kv = False
        if recache_kv and self._last_clean_latent is not None and stream.last_block_frames is not None:
            start, end = stream.last_block_frames
            recache_positive = runner._window_conditioning(
                state.positive_base,
                n_views=stream._n_views,
                frames_per_view=stream._frames_per_view,
                start=start,
                end=end,
                cache=state.positive_cache,
                frame_sequence_length=stream._frame_sequence_length,
                block_noise=self._last_clean_latent,
            )
            from worldfoundry.base_models.diffusion_model.contracts import SchedulerStep

            stream._context.step = SchedulerStep(
                index=0,
                timestep=torch.tensor(0, device=runner.device),
                next_timestep=torch.tensor(0, device=runner.device),
            )
            runner._call_denoiser_with_conditioning(
                stream._context,
                latents=self._last_clean_latent,
                branch="positive-kv-recache",
                conditioning=recache_positive,
            )
            recached_self_kv = True

        if runner.device.type == "cuda":
            torch.cuda.synchronize(runner.device)
        switch_ms = (time.perf_counter() - t0) * 1000.0
        self._prompt_switches += 1
        return {
            "prompt": clean_prompt,
            "switch_ms": switch_ms,
            "in_place_graph_safe": in_place_preserved,
            "recached_self_kv": recached_self_kv,
            "completed_blocks": stream.completed_blocks,
        }

    def stream_realtime(
        self,
        prompt: str = "",
        interactions: Any = None,
        *,
        recache_kv: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Advance one block; a changed non-empty prompt switches smoothly via KV-recache."""
        del interactions, kwargs
        stream = self._realtime_stream
        if stream is None:
            raise RuntimeError(f"{self.OWNER} realtime session is not configured; call configure_realtime() first.")
        clean_prompt = str(prompt or "").strip()
        switch_meta: dict[str, Any] | None = None
        if clean_prompt and clean_prompt != self._realtime_prompt:
            switch_meta = self.switch_prompt_realtime(clean_prompt, recache_kv=recache_kv)
        if stream.done:
            raise RuntimeError(
                f"{self.OWNER} realtime rollout already produced all {stream.total_blocks} chunks; "
                "reset to start a new rollout."
            )
        runner = self._window_runner
        synchronize = runner.device.type == "cuda"
        try:
            first_chunk = stream.completed_blocks == 0
            started = time.perf_counter()
            latents = stream.step()
            self._last_clean_latent = latents.detach().clone()
            if first_chunk and stream.first_frame_latent is not None:
                latents = torch.cat((stream.first_frame_latent, latents), dim=2)
            if synchronize:
                torch.cuda.synchronize(runner.device)
            decode_started = time.perf_counter()
            pixels = self._codec.decode_chunk(
                latents,
                state=cast(dict, self._realtime_decode_state),
                is_first_chunk=first_chunk,
            )
            if synchronize:
                torch.cuda.synchronize(runner.device)
            decode_finished = time.perf_counter()
            frames = video_tensor_to_uint8_frames(pixels, value_range="-1,1", layout="bcthw")
        except BaseException:
            self.reset_realtime()
            raise
        if stream.done:
            stream.close()
        metrics: dict[str, Any] = {
            "model_ms": (decode_started - started) * 1000.0,
            "decode_ms": (decode_finished - decode_started) * 1000.0,
            "chunk_index": stream.completed_blocks,
            "chunks_total": stream.total_blocks,
            "prompt": self._realtime_prompt,
        }
        if switch_meta is not None:
            metrics["prompt_switch"] = switch_meta
        return {
            "video": frames,
            "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255"),
            "realtime_spec": self._realtime_spec(stream.total_blocks).to_payload(),
            "realtime_metrics": metrics,
        }

    def stream_interactive_overlapped(
        self,
        prompt_schedule: Sequence[tuple[str, int]],
        *,
        images: Any = None,
        seed: int = 42,
        recache_kv: bool = True,
        **kwargs: Any,
    ):
        """Stream an interactive multi-prompt rollout with overlapped DiT/VAE + KV-recache.

        Args:
            prompt_schedule: Sequence of ``(prompt_text, num_chunks)`` pairs.
            images: Optional conditioning image.
            seed: Deterministic seed.
            recache_kv: Whether to run a single ``t=0`` self-attention KV-recache
                pass over the latest committed block on every prompt transition.
        """
        if not prompt_schedule:
            raise ValueError("prompt_schedule must contain at least one (prompt, num_chunks) pair")
        first_prompt, _ = prompt_schedule[0]
        total_chunks = sum(int(count) for _, count in prompt_schedule)
        if total_chunks <= 0:
            raise ValueError("prompt_schedule total chunk count must be positive")

        configure_kwargs = dict(kwargs)
        if not self._unbounded_rollout and "num_frames" not in configure_kwargs:
            block_size = int(self._window_runner.windowed_denoiser.block_size)
            configure_kwargs["num_frames"] = int(self._codec.get_pixel_num_frames(total_chunks * block_size))

        cfg = self.configure_realtime(images=images, prompt=first_prompt, seed=seed, **configure_kwargs)
        stream = self._realtime_stream
        if stream is None:
            raise RuntimeError("LongLive realtime stream failed to initialize.")
        runner = self._window_runner
        is_cuda = runner.device.type == "cuda"
        entry_stream = torch.cuda.current_stream(runner.device) if is_cuda else None
        compute_stream = torch.cuda.Stream(device=runner.device) if is_cuda else None
        decode_stream = torch.cuda.Stream(device=runner.device) if is_cuda else None
        if is_cuda and entry_stream is not None and compute_stream is not None and decode_stream is not None:
            init_ev = torch.cuda.Event()
            init_ev.record(entry_stream)
            compute_stream.wait_event(init_ev)
            decode_stream.wait_event(init_ev)

        pending_cpu_uint8: torch.Tensor | None = None
        pending_event: torch.cuda.Event | None = None
        pending_meta: dict[str, Any] = {}

        def _pixels_to_uint8_gpu(pixels: torch.Tensor) -> torch.Tensor:
            frames_t = pixels[0].permute(1, 2, 3, 0).float()
            return frames_t.clamp(-1.0, 1.0).add(1.0).mul(127.5).to(torch.uint8).contiguous()

        def _flush_pending():
            nonlocal pending_cpu_uint8, pending_event
            if pending_cpu_uint8 is None or pending_event is None:
                return None
            pending_event.synchronize()
            frames_np = pending_cpu_uint8.numpy()
            pending_meta["wall_ms"] = (time.perf_counter() - pending_meta["chunk_t0"]) * 1000.0
            payload = {
                "video": frames_np,
                "video_chunk": VideoChunk(frames=frames_np, layout="thwc", value_range="0,255"),
                "realtime_spec": cfg["realtime_spec"],
                "realtime_metrics": dict(pending_meta),
            }
            pending_cpu_uint8 = None
            pending_event = None
            return payload

        try:
            for seg_idx, (seg_prompt, seg_chunks) in enumerate(prompt_schedule):
                switch_meta: dict[str, Any] | None = None
                if seg_idx > 0:
                    if is_cuda and compute_stream is not None:
                        compute_stream.synchronize()
                    switch_meta = self.switch_prompt_realtime(seg_prompt, recache_kv=recache_kv)
                    if is_cuda and entry_stream is not None and compute_stream is not None:
                        sw_ev = torch.cuda.Event()
                        sw_ev.record(entry_stream)
                        compute_stream.wait_event(sw_ev)

                for local_idx in range(int(seg_chunks)):
                    if stream.done:
                        break
                    first_chunk = stream.completed_blocks == 0
                    t_step0 = time.perf_counter()
                    if is_cuda and compute_stream is not None and decode_stream is not None:
                        with torch.cuda.stream(compute_stream):
                            latents = stream.step()
                            self._last_clean_latent = latents.detach().clone()
                            if first_chunk and stream.first_frame_latent is not None:
                                latents = torch.cat((stream.first_frame_latent, latents), dim=2)
                            latents_for_decode = latents.clone()
                            latent_ready = torch.cuda.Event()
                            latent_ready.record(compute_stream)
                            latents_for_decode.record_stream(decode_stream)

                        prev_payload = _flush_pending()
                        if prev_payload is not None:
                            yield prev_payload

                        with torch.cuda.stream(decode_stream):
                            decode_stream.wait_event(latent_ready)
                            pixels_cur = self._codec.decode_chunk(
                                latents_for_decode,
                                state=cast(dict, self._realtime_decode_state),
                                is_first_chunk=first_chunk,
                            )
                            uint8_gpu = _pixels_to_uint8_gpu(pixels_cur)
                            cpu_buf = torch.empty(uint8_gpu.shape, dtype=torch.uint8, pin_memory=True)
                            cpu_buf.copy_(uint8_gpu, non_blocking=True)
                            done_ev = torch.cuda.Event()
                            done_ev.record(decode_stream)
                        pending_cpu_uint8 = cpu_buf
                        pending_event = done_ev
                        pending_meta = {
                            "chunk_t0": t_step0,
                            "chunk_index": stream.completed_blocks,
                            "chunks_total": total_chunks,
                            "overlapped": True,
                            "prompt": self._realtime_prompt,
                            "segment_index": seg_idx,
                        }
                        if local_idx == 0 and switch_meta is not None:
                            pending_meta["prompt_switch"] = switch_meta
                    else:
                        latents = stream.step()
                        self._last_clean_latent = latents.detach().clone()
                        if first_chunk and stream.first_frame_latent is not None:
                            latents = torch.cat((stream.first_frame_latent, latents), dim=2)
                        pixels_cur = self._codec.decode_chunk(
                            latents,
                            state=cast(dict, self._realtime_decode_state),
                            is_first_chunk=first_chunk,
                        )
                        frames_cur = video_tensor_to_uint8_frames(pixels_cur, value_range="-1,1", layout="bcthw")
                        meta = {
                            "wall_ms": (time.perf_counter() - t_step0) * 1000.0,
                            "chunk_index": stream.completed_blocks,
                            "chunks_total": total_chunks,
                            "overlapped": False,
                            "prompt": self._realtime_prompt,
                            "segment_index": seg_idx,
                        }
                        if local_idx == 0 and switch_meta is not None:
                            meta["prompt_switch"] = switch_meta
                        yield {
                            "video": frames_cur,
                            "video_chunk": VideoChunk(frames=frames_cur, layout="thwc", value_range="0,255"),
                            "realtime_spec": cfg["realtime_spec"],
                            "realtime_metrics": meta,
                        }

            final_payload = _flush_pending()
            if final_payload is not None:
                yield final_payload
        finally:
            if is_cuda and entry_stream is not None and compute_stream is not None and decode_stream is not None:
                entry_stream.wait_stream(compute_stream)
                entry_stream.wait_stream(decode_stream)
                compute_stream.synchronize()
                decode_stream.synchronize()
            self.reset_realtime()

    def reset_realtime(self) -> None:
        self._last_clean_latent = None
        super().reset_realtime()


__all__ = [
    "NativeLongLivePipeline",
    "load_interactive_prompt_jsonl",
    "merge_longlive_lora",
]
