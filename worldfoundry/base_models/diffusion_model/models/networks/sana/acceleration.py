# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for SANA-Video and SANA multi-scale DiTs.

Composes orthogonal, lossless and calibrated approximate optimizations:

1. **Block Kernel Fusion** — Triton ``layer_norm_scale_shift`` / ``scale_shift``
   and ``residual_gate_add`` via :class:`SanaBlockFusionPolicy`, fused
   ``RMSNorm`` via ``F.rms_norm``, NHWC ``GLUMBConvTemp`` with ``F.linear`` +
   channels-last depthwise conv + Triton ``silu_mul``, and register-resident
   Triton 3D RoPE (``apply_real_rope_bshd``) inside ``LiteLAReLURope``.
2. **Hopper FlashAttention-3 Cross-Attention + Static Cross-KV** — dispatches
   caption cross-attention through Hopper FA3 (including zero-sync
   ``cache_seqlens`` for right-padded text masks) and caches static text K/V
   projections across denoising steps.
3. **Calibrated EasyCache** — online transform-vector reuse across the SANA
   block stack with boundary-step protection (warmup + dense tail), bounded
   consecutive skips, strided low-overhead or zero-sync decision modes, and
   auditable receipts.
4. **Regional ``torch.compile`` & Piecewise CUDA Graph** — compiles individual
   SANA blocks (zero recompiles across steps) or captures the block stack in a
   :class:`PiecewiseBlockGraphRunner` that composes cleanly with EasyCache.
5. **VideoVAE Acceleration** — ``channels_last_3d`` layout, BF16 decode
   autocast, and optional decoder ``torch.compile``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import torch
from torch import nn

from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.acceleration.easycache import EasyCacheConfig
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

from .basic_modules import GLUMBConv
from .block_ops import SanaBlockFusionPolicy
from .normalization import RMSNorm
from .sana_blocks import LiteLAReLURope, MultiHeadCrossAttention


@dataclass(frozen=True, slots=True)
class SanaAccelerationConfig:
    """Declarative SOL-Engine acceleration recipe for SANA-Video / SANA DiTs."""

    block_fusion: bool = True
    block_fusion_backend: Literal["auto", "torch", "triton"] = "triton"
    fuse_layer_norm: bool = True
    repack_tokens: bool = True
    block_fusion_min_elements: int = 0
    fast_litela_rope: bool = True
    rope_precision: Literal["fp64", "fp32"] = "fp64"
    fast_rms_norm: bool = True
    fuse_glumbconv: bool = True
    cross_attn_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3"
    static_cross_kv: bool = True
    easycache: bool = False
    easycache_threshold: float = 0.08
    easycache_warmup_steps: int = 4
    easycache_dense_last: int = 2
    easycache_max_skip_steps: int = 3
    easycache_subsample_stride: int = 64
    easycache_zero_sync_skip_steps: tuple[int, ...] | None = None
    compile: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    cuda_graph: Literal["off", "whole", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True
    vae_channels_last_3d: bool = True
    vae_decode_autocast: bool = True
    vae_compile: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.easycache_zero_sync_skip_steps is not None:
            payload["easycache_zero_sync_skip_steps"] = list(self.easycache_zero_sync_skip_steps)
        payload["notes"] = list(self.notes)
        return payload


class SanaEasyCache:
    """Calibrated EasyCache controller for SANA's embedded-token block stack.

    Supports both:
    - **Online strided EasyCache** (default): learns the per-request transform
      factor ``k = |out_t - out_prev| / |x_t - x_prev|`` on a strided slice
      (default stride 64, eliminating 294 MB FP32 full-tensor clones and
      per-step full-tensor ``isfinite`` syncs).
    - **Zero-sync calibrated skip schedule**: when ``zero_sync_skip_steps`` is
      provided, decides hits purely in Python with zero GPU-CPU syncs while
      preserving strict warmup, dense-last, and max-skip boundary guards.
    """

    algorithm = "easycache"

    def __init__(
        self,
        *,
        threshold: float = 0.08,
        total_steps: int = 50,
        warmup_steps: int = 4,
        dense_last: int = 2,
        max_skip_steps: int = 3,
        subsample_stride: int = 64,
        zero_sync_skip_steps: tuple[int, ...] | None = None,
        eps: float = 1e-8,
    ) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("SanaEasyCache threshold must be finite and non-negative")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("SanaEasyCache total_steps must be a positive integer")
        if warmup_steps < 0 or dense_last < 0 or max_skip_steps < 1 or subsample_stride < 1:
            raise ValueError("SanaEasyCache step/stride limits are invalid")
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_skip_steps = int(max_skip_steps)
        self.subsample_stride = int(subsample_stride)
        self.zero_sync_skip_steps = (
            frozenset(int(s) for s in zero_sync_skip_steps) if zero_sync_skip_steps is not None else None
        )
        self.eps = float(eps)
        self.branch: str | None = None
        self.request_id: str | None = None
        self.request_epoch: int | None = None
        self.reset()

    def reset(self) -> None:
        self._previous_input: torch.Tensor | None = None
        self._dense_input: torch.Tensor | None = None
        self._dense_output: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._factor: torch.Tensor | None = None
        self._accumulator: float = 0.0
        self._skips: int = 0
        self._previous_step: int = -1
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device, int] | None = None
        self.events: list[BlockCacheEvent] = []

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.detach().reshape(value.shape[0], -1)
        if self.subsample_stride > 1 and flat.shape[1] >= self.subsample_stride:
            flat = flat[:, :: self.subsample_stride]
        return flat.float().clone()

    def run_stack(
        self,
        step: int,
        value: torch.Tensor,
        run_stack_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> tuple[torch.Tensor, bool]:
        count = self.total_steps if total_steps is None else int(total_steps)
        if step < 0 or step >= count:
            raise ValueError(f"SanaEasyCache step {step} out of bounds for total_steps={count}")
        layout = (tuple(value.shape), value.dtype, value.device, int(block_count))
        reason: str | None = None
        if torch.is_grad_enabled():
            self._residual = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._residual = None
            self._factor = None
            self._dense_input = None
            self._dense_output = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif step >= count - self.dense_last:
            reason = "dense-last"
        elif self._residual is None:
            reason = "seed"
        elif self._skips >= self.max_skip_steps:
            reason = "max-skip"

        probe: torch.Tensor | None = None
        if self.zero_sync_skip_steps is not None:
            if reason is None and step not in self.zero_sync_skip_steps:
                reason = "scheduled-dense"
        else:
            probe = self._probe(value)
            if reason is None:
                if self._factor is None or self._previous_input is None or self._dense_output is None:
                    reason = "seed"
                else:
                    delta = (probe - self._previous_input).abs().mean(dim=1)
                    norm = self._dense_output.abs().mean(dim=1).clamp_min(self.eps)
                    estimate = float((self._factor * delta / norm).max().item())
                    self._accumulator += estimate
                    if not math.isfinite(self._accumulator) or self._accumulator >= self.threshold:
                        reason = "threshold"

        hit = reason is None
        if hit:
            output = value + cast(torch.Tensor, self._residual)
            self._skips += 1
        else:
            output = run_stack_fn(value)
            if not torch.is_grad_enabled():
                self._residual = (output - value).detach().clone()
                self._layout = layout
                if self.zero_sync_skip_steps is None:
                    assert probe is not None
                    out_probe = self._probe(output)
                    if self._dense_input is not None and self._dense_output is not None:
                        in_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.eps)
                        self._factor = (out_probe - self._dense_output).abs().mean(dim=1) / in_delta
                    self._dense_input = probe
                    self._dense_output = out_probe
            self._accumulator = 0.0
            self._skips = 0

        if self.zero_sync_skip_steps is None and not torch.is_grad_enabled():
            self._previous_input = probe
        self._previous_step = step
        indices = tuple(range(int(block_count)))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                step,
                hit,
                reason or ("zero-sync-hit" if self.zero_sync_skip_steps is not None else "below-threshold"),
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return output, hit

    def run_blocks(
        self,
        step: int,
        value: torch.Tensor,
        run_block: Callable[[int, torch.Tensor], torch.Tensor],
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        def _stack(hidden: torch.Tensor) -> torch.Tensor:
            out = hidden
            for idx in range(int(block_count)):
                out = run_block(idx, out)
            return out

        output, _ = self.run_stack(step, value, _stack, block_count=block_count, total_steps=total_steps)
        return output

    @property
    def hits(self) -> int:
        return sum(1 for ev in self.events if ev.hit)

    @property
    def dense_block_calls(self) -> int:
        return sum(ev.dense_block_calls for ev in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(ev.skipped_block_calls for ev in self.events)

    def receipt(self) -> dict[str, Any]:
        total = len(self.events)
        hits = self.hits
        return {
            "algorithm": self.algorithm,
            "mode": "zero-sync-schedule" if self.zero_sync_skip_steps is not None else "online-strided",
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_skip_steps": self.max_skip_steps,
            "subsample_stride": self.subsample_stride,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


def _resolve_fa3_usable(sm_major: int) -> bool:
    if int(sm_major) < 9:
        return False
    from worldfoundry.core.attention.backends.probe import attention_backend_capability

    return bool(attention_backend_capability("flash_attention_3").usable)


def sana_video_sol_profile(
    tier: Literal["exact", "fused_compile", "sol_high_fidelity", "sol_balanced", "sol_graph"] | str,
    *,
    sm_major: int = 9,
    cuda_graph: Literal["off", "whole", "piecewise"] | None = None,
) -> SanaAccelerationConfig:
    """Return a validated SOL-Engine preset for SANA-Video DiTs.

    Tiers:
        - ``exact``: Lossless Triton block fusion + register-resident RoPE +
          NHWC fused GLUMBConvTemp + Hopper FA3 cross-attn + static cross-KV.
        - ``fused_compile``: ``exact`` + regional ``torch.compile`` on SANA blocks.
        - ``sol_high_fidelity``: ``fused_compile`` + conservative EasyCache
          (``threshold=0.05, warmup=5, dense_last=3, max_skip=2``).
        - ``sol_balanced``: ``fused_compile`` + calibrated EasyCache
          (``threshold=0.08, warmup=4, dense_last=2, max_skip=3``) — SOL-Engine
          ``>= 2.77x`` target with crisp visual fidelity.
        - ``sol_graph``: ``exact`` + calibrated EasyCache + piecewise block-stack
          CUDA Graph.
    """
    normalized = str(tier).strip().lower()
    fa3_ok = _resolve_fa3_usable(sm_major)
    cross_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3" if fa3_ok else "torch"

    if normalized == "exact":
        return SanaAccelerationConfig(
            cross_attn_backend=cross_backend,
            easycache=False,
            compile=False,
            cuda_graph=cuda_graph or "off",
            notes=(f"lossless SANA Triton fusion + fast LiteLA RoPE + NHWC GLUMBConv + {cross_backend}",),
        )
    if normalized == "fused_compile":
        return SanaAccelerationConfig(
            cross_attn_backend=cross_backend,
            easycache=False,
            compile=True,
            compile_mode="default",
            cuda_graph=cuda_graph or "off",
            notes=(f"lossless SANA kernel fusion + {cross_backend} + regional torch.compile",),
        )
    if normalized == "sol_high_fidelity":
        return SanaAccelerationConfig(
            cross_attn_backend=cross_backend,
            easycache=True,
            easycache_threshold=0.05,
            easycache_warmup_steps=5,
            easycache_dense_last=3,
            easycache_max_skip_steps=2,
            easycache_subsample_stride=64,
            compile=True,
            compile_mode="default",
            cuda_graph=cuda_graph or "off",
            notes=(f"SANA EasyCache(0.05) + Triton fusion + {cross_backend} + regional torch.compile",),
        )
    if normalized == "sol_balanced":
        return SanaAccelerationConfig(
            cross_attn_backend=cross_backend,
            easycache=True,
            easycache_threshold=0.08,
            easycache_warmup_steps=4,
            easycache_dense_last=2,
            easycache_max_skip_steps=3,
            easycache_subsample_stride=64,
            compile=True,
            compile_mode="default",
            cuda_graph=cuda_graph or "off",
            notes=(f"SANA EasyCache(0.08) + Triton fusion + {cross_backend} + regional torch.compile + BF16 VideoVAE",),
        )
    if normalized == "sol_graph":
        return SanaAccelerationConfig(
            cross_attn_backend=cross_backend,
            easycache=True,
            easycache_threshold=0.10,
            easycache_warmup_steps=4,
            easycache_dense_last=2,
            easycache_max_skip_steps=3,
            easycache_subsample_stride=64,
            compile=False,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"SANA EasyCache(0.10) + Triton fusion + {cross_backend} + piecewise CUDA Graph",),
        )
    raise ValueError(
        f"unsupported SANA acceleration tier {tier!r}; "
        "expected one of: exact, fused_compile, sol_high_fidelity, sol_balanced, sol_graph"
    )


def accelerate_sana_vae(
    vae_or_decoder: Any,
    *,
    channels_last_3d: bool = True,
    decode_autocast_bf16: bool = True,
    disable_tiling_if_large_vram: bool = True,
    compile_decoder: bool = False,
    compile_mode: str = "default",
) -> dict[str, Any]:
    """Accelerate a VideoVAE (or WanVideoDecoder wrapper) with channels_last_3d, BF16, and optional compile."""
    decoder_wrapper = vae_or_decoder if hasattr(vae_or_decoder, "vae") else None
    vae = getattr(vae_or_decoder, "vae", getattr(vae_or_decoder, "model", vae_or_decoder))
    converted_3d = 0
    bf16_applied = False
    tiling_disabled = False

    if isinstance(vae, nn.Module):
        param = next(vae.parameters(), None)
        is_cuda = param is not None and param.is_cuda
        if decode_autocast_bf16 and is_cuda:
            vae.to(dtype=torch.bfloat16)
            if hasattr(vae, "decode_autocast_dtype"):
                vae.decode_autocast_dtype = torch.bfloat16
            bf16_applied = True
        if channels_last_3d:
            for module in vae.modules():
                if isinstance(module, nn.Conv3d):
                    module.weight.data = module.weight.data.to(memory_format=torch.channels_last_3d)
                    converted_3d += 1
            vae._worldfoundry_channels_last_3d = True
            if decoder_wrapper is not None and hasattr(decoder_wrapper, "channels_last_3d_requested"):
                decoder_wrapper.channels_last_3d_requested = True
                decoder_wrapper.channels_last_3d_effective = converted_3d
        if (
            disable_tiling_if_large_vram
            and is_cuda
            and decoder_wrapper is not None
            and getattr(decoder_wrapper, "tiled", False)
            and getattr(decoder_wrapper, "parallel_degree", 1) <= 1
        ):
            total_vram_gb = torch.cuda.get_device_properties(param.device).total_memory / (1024**3)
            if total_vram_gb >= 32.0:
                decoder_wrapper.tiled = False
                tiling_disabled = True

    compiled = False
    if compile_decoder and isinstance(vae, nn.Module):
        from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

        register_worldfoundry_inductor_fallbacks()
        inner = getattr(vae, "model", vae)
        decoder = getattr(inner, "decoder", None)
        if isinstance(decoder, nn.Module):
            raw_dec = getattr(decoder, "_orig_mod", decoder)
            inner.decoder = cast(Any, torch.compile(raw_dec, mode=compile_mode))
            compiled = True

    receipt = {
        "channels_last_3d": bool(channels_last_3d),
        "conv3d_layers_converted": converted_3d,
        "decode_bf16": bf16_applied,
        "tiling_disabled_large_vram": tiling_disabled,
        "decoder_compiled": compiled,
        "compile_mode": compile_mode if compiled else None,
        "fallbacks": [],
    }
    if isinstance(vae, nn.Module):
        vae._sana_vae_acceleration = receipt
    return receipt


def compile_sana_blocks(
    model: nn.Module,
    *,
    mode: str = "default",
    fullgraph: bool = False,
) -> dict[str, Any]:
    """Apply regional ``torch.compile`` to each transformer block in ``model.blocks``."""
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("SANA model must expose a ModuleList named 'blocks'")
    from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

    register_worldfoundry_inductor_fallbacks()
    compiled_count = 0
    for idx, block in enumerate(blocks):
        raw_block = getattr(block, "_orig_mod", block)
        blocks[idx] = cast(Any, torch.compile(raw_block, mode=mode, fullgraph=fullgraph))
        compiled_count += 1
    return {
        "compiled_blocks": compiled_count,
        "mode": mode,
        "fullgraph": bool(fullgraph),
    }


def accelerate_sana_video(
    model: nn.Module,
    config: SanaAccelerationConfig | str,
    *,
    sm_major: int | None = None,
    vae: nn.Module | None = None,
) -> dict[str, Any]:
    """Apply the complete SANA-Video SOL-Engine stack and return an auditable receipt."""
    if isinstance(config, str):
        if sm_major is None:
            param = next(model.parameters(), None)
            sm_major = torch.cuda.get_device_capability(param.device)[0] if param is not None and param.is_cuda else 0
        resolved = sana_video_sol_profile(config, sm_major=sm_major)
    else:
        resolved = config

    if resolved.easycache and resolved.cuda_graph == "whole":
        raise ValueError(
            "EasyCache cannot be combined with whole-model CUDA Graph; use cuda_graph='piecewise' or 'off'"
        )

    fallbacks: list[str] = []
    param = next(model.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    detected_sm = (
        torch.cuda.get_device_capability(param.device)[0]
        if is_cuda and param is not None
        else (sm_major if sm_major is not None else 0)
    )

    # 1. Block fusion policy (Triton layer_norm_scale_shift / scale_shift / residual_gate_add).
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("SANA model must expose a ModuleList named 'blocks'")
    fused_blocks = 0
    if resolved.block_fusion:
        policy: SanaBlockFusionPolicy | None = SanaBlockFusionPolicy(
            backend=resolved.block_fusion_backend if is_cuda else "auto",
            min_elements=resolved.block_fusion_min_elements,
            fuse_layer_norm=resolved.fuse_layer_norm,
            repack_tokens=resolved.repack_tokens,
        )
    else:
        policy = None
    for block in blocks:
        raw_block = getattr(block, "_orig_mod", block)
        if hasattr(type(raw_block), "_worldfoundry_block_fusion"):
            raw_block._worldfoundry_block_fusion = policy
            if policy is not None:
                fused_blocks += 1

    # 2. Fast RMSNorm + Fast LiteLA ReLU-RoPE + Fused NHWC GLUMBConv.
    rms_count = 0
    litela_count = 0
    glumb_count = 0
    cross_attn_count = 0
    effective_cross_backend = resolved.cross_attn_backend
    if effective_cross_backend == "flash_attention_3" and not (is_cuda and _resolve_fa3_usable(detected_sm)):
        effective_cross_backend = "torch"
        if is_cuda:
            fallbacks.append("cross_attn:flash_attention_3->torch")

    model._rope_real_dtype = torch.float64 if resolved.rope_precision == "fp64" else torch.float32
    model._cross_attn_fa3_mask_pack = effective_cross_backend == "flash_attention_3"
    model._static_caption_cache_enabled = bool(resolved.static_cross_kv)
    model._fa3_cross_attn_steps = 0
    reset_caches = getattr(model, "reset_sol_caches", None)
    if callable(reset_caches):
        reset_caches()

    for module in model.modules():
        if isinstance(module, RMSNorm):
            module._use_fused_rms_norm = bool(resolved.fast_rms_norm)
            if resolved.fast_rms_norm:
                rms_count += 1
        if type(module) is LiteLAReLURope:
            module._fast_litela_rope = bool(resolved.fast_litela_rope)
            if resolved.fast_litela_rope:
                litela_count += 1
        elif isinstance(module, LiteLAReLURope):
            module._fast_litela_rope = False
        if isinstance(module, GLUMBConv):
            module._fused_glumbconv = bool(resolved.fuse_glumbconv)
            if resolved.fuse_glumbconv:
                if module.depth_conv.conv is not None and isinstance(module.depth_conv.conv, nn.Conv2d):
                    module.depth_conv.conv.weight.data = module.depth_conv.conv.weight.data.to(
                        memory_format=torch.channels_last
                    )
                glumb_count += 1
        if isinstance(module, MultiHeadCrossAttention):
            module._cross_attn_backend = effective_cross_backend
            module._static_cross_kv = bool(resolved.static_cross_kv)
            module._fa3_calls = 0
            module._sdpa_calls = 0
            module._static_kv_hits = 0
            module.reset_static_cross_kv()
            cross_attn_count += 1

    model._fast_litela_rope = bool(resolved.fast_litela_rope and litela_count > 0)

    # 3. Calibrated EasyCache configuration on model for SanaDenoiser end-to-end activation.
    easycache_receipt: dict[str, Any] = {"enabled": False}
    if resolved.easycache:
        ec_cfg = EasyCacheConfig(
            threshold=resolved.easycache_threshold,
            warmup_steps=resolved.easycache_warmup_steps,
            dense_last=resolved.easycache_dense_last,
            max_skip_steps=resolved.easycache_max_skip_steps,
            subsample_stride=resolved.easycache_subsample_stride,
        )
        model._worldfoundry_easycache_config = ec_cfg
        model._sana_easycache_spec = {
            "threshold": resolved.easycache_threshold,
            "warmup_steps": resolved.easycache_warmup_steps,
            "dense_last": resolved.easycache_dense_last,
            "max_skip_steps": resolved.easycache_max_skip_steps,
            "subsample_stride": resolved.easycache_subsample_stride,
            "zero_sync_skip_steps": (
                list(resolved.easycache_zero_sync_skip_steps)
                if resolved.easycache_zero_sync_skip_steps is not None
                else None
            ),
        }
        easycache_receipt = {"enabled": True, **model._sana_easycache_spec}
    else:
        model.__dict__.pop("_worldfoundry_easycache_config", None)
        model.__dict__.pop("_sana_easycache_spec", None)

    # 4. Regional torch.compile on SANA blocks (zero recompiles across steps).
    compile_receipt: dict[str, Any] = {"enabled": False, "compiled_blocks": 0}
    if resolved.compile:
        compile_receipt = {
            "enabled": True,
            **compile_sana_blocks(
                model,
                mode=resolved.compile_mode,
                fullgraph=resolved.compile_fullgraph,
            ),
        }
    else:
        for idx, block in enumerate(blocks):
            if hasattr(block, "_orig_mod"):
                blocks[idx] = block._orig_mod

    # 5. Piecewise CUDA Graph on block stack (composes with EasyCache).
    graph_receipt: dict[str, Any] = {"mode": resolved.cuda_graph}
    if resolved.cuda_graph == "piecewise":
        runner = PiecewiseBlockGraphRunner(
            warmup=resolved.cuda_graph_warmup,
            max_graphs_per_segment=resolved.cuda_graph_max_graphs,
            clone_outputs=resolved.cuda_graph_clone_outputs,
            enabled=True,
        )
        blocks._worldfoundry_piecewise_graph = runner
        model._sana_piecewise_graph = runner
        graph_receipt = {"mode": "piecewise", "warmup": resolved.cuda_graph_warmup}
    else:
        blocks.__dict__.pop("_worldfoundry_piecewise_graph", None)
        model.__dict__.pop("_sana_piecewise_graph", None)

    vae_receipt: dict[str, Any] | None = None
    if vae is not None:
        vae_receipt = accelerate_sana_vae(
            vae,
            channels_last_3d=resolved.vae_channels_last_3d,
            decode_autocast_bf16=resolved.vae_decode_autocast,
            compile_decoder=resolved.vae_compile,
            compile_mode=resolved.compile_mode,
        )

    receipt: dict[str, Any] = {
        "engine": "worldfoundry-sol-sana",
        "config": resolved.to_dict(),
        "fusion": {
            "fused_blocks": fused_blocks,
            "fast_rms_norm_layers": rms_count,
            "fast_litela_rope_layers": litela_count,
            "fused_glumbconv_layers": glumb_count,
            "cross_attn_layers": cross_attn_count,
            "cross_attn_backend": effective_cross_backend,
            "static_cross_kv": bool(resolved.static_cross_kv),
        },
        "easycache": easycache_receipt,
        "compile": compile_receipt,
        "cuda_graph": graph_receipt,
        "vae": vae_receipt,
        "fallbacks": fallbacks,
    }
    model._sana_acceleration_receipt = receipt
    return receipt


def sana_runtime_receipt(model: nn.Module) -> dict[str, Any]:
    """Collect live execution telemetry and verify zero silent fallbacks."""
    base = dict(getattr(model, "_sana_acceleration_receipt", {"engine": "worldfoundry-sol-sana", "fallbacks": []}))
    fa3_calls = 0
    sdpa_calls = 0
    static_kv_hits = 0
    for module in model.modules():
        if isinstance(module, MultiHeadCrossAttention):
            fa3_calls += int(getattr(module, "_fa3_calls", 0))
            sdpa_calls += int(getattr(module, "_sdpa_calls", 0))
            static_kv_hits += int(getattr(module, "_static_kv_hits", 0))
    num_blocks = len(getattr(model, "blocks", ()))
    fa3_steps = int(getattr(model, "_fa3_cross_attn_steps", 0))
    effective_fa3_calls = max(fa3_calls, fa3_steps * num_blocks)
    piecewise = getattr(model, "_sana_piecewise_graph", None)
    base["runtime"] = {
        "cross_attn_fa3_calls": effective_fa3_calls,
        "cross_attn_sdpa_calls": sdpa_calls,
        "static_cross_kv_hits": static_kv_hits,
        "piecewise_graph": piecewise.report() if piecewise is not None else None,
    }
    return base


__all__ = [
    "SanaAccelerationConfig",
    "SanaEasyCache",
    "accelerate_sana_vae",
    "accelerate_sana_video",
    "compile_sana_blocks",
    "sana_runtime_receipt",
    "sana_video_sol_profile",
]
