# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 Super Acceleration: 4-step H3 draft + 3-step LTX-2.5 refinement.

Composable, zero-fallback two-stage video synthesis architecture that achieves
>20x end-to-end acceleration over 50-step high-resolution MiniMax-H3 31B while
preserving crisp visual fidelity and synchronized audio:

- **Stage 1 (4-step MiniMax-H3 Draft)**: Runs a fast low-resolution (384p or
  512p) 4-step MiniMax-H3 31B draft with ``sol_attn(kv_splits=4)``, calibrated
  rowwise FP8 W8A8 (``use_fast_accum=False``), fused SwiGLU-FP8 FFN, piecewise
  CUDA Graph capture, and the accelerated 36-layer MiniMax-H3 VideoVAE
  (``bfloat16`` autocast + ``channels_last_3d`` + stacked tiling).
- **Handoff**: Spatially upscales and temporally aligns Stage-1 draft frames to
  LTX-2.5's target canvas (``H, W % 32 == 0``, ``F = 8*k + 1``) and encodes
  them into LTX-2.5 high-resolution latent space via the accelerated LTX-2.5
  VideoVAE encoder.
- **Stage 2 (3-step LTX-2.5 High-Res Video-to-Video Refinement)**: Runs 3-step
  distilled LTX-2.5 22B refinement (``ltx-2.5-22b-distilled``) with kernel
  fusion, Hopper FA3, 3D-tiled Sol-Attn, rowwise FP8 W8A8, CUDA Graph replay,
  and accelerated LTX-2.5 VideoVAE decoding to emit crisp 768p/720p 5s or 10s
  video alongside the synchronized MiniMax-H3 audio track.

This module composes :class:`NativeMiniMaxH3Pipeline` and
:class:`LTX25DistilledRuntime` without modifying ``pipeline_minimax_h3.py``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.acceleration import (
    MiniMaxH3AccelerationConfig,
    minimax_h3_sol_profile,
)
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import NativeMiniMaxH3Pipeline, _resolve_canvas
from worldfoundry.synthesis.visual_generation.ltx25.acceleration import LTX25AccelerationConfig
from worldfoundry.synthesis.visual_generation.ltx25.runtime import (
    LTX25_STAGE_2_SIGMAS,
    LTX25DistilledRuntime,
    _align_ltx25_num_frames,
)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _peak_vram_gib(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 3)


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _normalize_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _normalize_stereo_waveform(waveform: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize arbitrary audio VAE waveform shapes (e.g. H3 ``[2, 1, N]`` or mono ``[1, N]``) to ``[2, N]``."""
    if waveform is None or waveform.numel() == 0:
        return None
    wav = waveform.detach().float()
    while wav.ndim > 2:
        squeezed = False
        for dim in range(wav.ndim):
            if int(wav.shape[dim]) == 1:
                wav = wav.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            wav = wav.reshape(wav.shape[0], -1)
    if wav.ndim == 1:
        return wav.unsqueeze(0).repeat(2, 1).contiguous()
    if int(wav.shape[0]) == 1:
        return wav.repeat(2, 1).contiguous()
    if int(wav.shape[1]) == 1:
        return wav.transpose(0, 1).repeat(2, 1).contiguous()
    if int(wav.shape[0]) == 2:
        return wav.contiguous()
    if int(wav.shape[1]) == 2:
        return wav.transpose(0, 1).contiguous()
    return wav[:2].contiguous()


@dataclass(frozen=True)
class SuperAccelerationConfig:
    """Validated declarative configuration for MiniMax-H3 Super Acceleration."""

    # Stage 1: 4-step low-res MiniMax-H3 draft
    draft_short_edge: int = 384
    draft_steps: int = 4
    draft_flow_shift: float = 12.0
    draft_audio_flow_shift: float = 3.0
    draft_sigma_knots: tuple[float, ...] | None = None
    h3_sol_tau: float = 0.5
    h3_kv_splits: Literal[1, 2, 4] = 4
    h3_dense_initial_steps: int = 1
    h3_fp8_use_fast_accum: bool = False
    h3_fp8_exclude_edge_blocks: bool = True
    h3_cuda_graph: Literal["off", "whole", "piecewise"] = "piecewise"
    h3_vae_decode_autocast: bool = True
    h3_vae_channels_last_3d: bool = True
    h3_vae_stack_tiling: bool = True
    h3_vae_tile_batch_size: int | None = None
    h3_vae_compile: bool = False

    # Target high-res geometry & handoff
    target_short_edge: int = 768
    target_height: int | None = None
    target_width: int | None = None
    aspect_ratio: str = "16:9"
    duration_seconds: float = 5.0
    target_num_frames: int | None = None
    fps: float = 24.0
    upscale_mode: Literal["bicubic", "bilinear"] = "bicubic"

    # Stage 2: 3-step high-res LTX-2.5 refinement
    refinement_sigmas: tuple[float, ...] = LTX25_STAGE_2_SIGMAS
    ltx25_video_vae_variant: Literal["conv", "diffvae"] = "conv"
    ltx25_fp8_use_fast_accum: bool = False
    ltx25_fp8_exclude_edge_blocks: bool = True
    ltx25_sol_attn: bool = True
    ltx25_sol_tau: float = 0.5
    ltx25_sol_kv_splits: Literal[1, 2, 4] = 2
    ltx25_sol_tile_3d: tuple[int, int, int] | None = (1, 8, 8)
    ltx25_sol_dense_steps: int = 1
    ltx25_sol_min_tokens: int = 4096
    ltx25_cuda_graph: Literal["off", "whole", "piecewise"] = "piecewise"
    ltx25_vae_compile: bool = True
    ltx25_vae_tiled: bool = False

    # Memory & audio orchestration
    offload_text_encoders_after_encode: bool = True
    offload_between_stages: bool = False
    preserve_h3_audio: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.draft_short_edge, name="draft_short_edge")
        if self.draft_short_edge % 32 != 0:
            raise ValueError(f"draft_short_edge must be divisible by 32, got {self.draft_short_edge}")
        _validate_positive_int(self.draft_steps, name="draft_steps")
        if self.draft_sigma_knots is not None:
            if len(self.draft_sigma_knots) < 2:
                raise ValueError("draft_sigma_knots must contain at least 2 entries")
            if float(self.draft_sigma_knots[0]) != 1.0 or float(self.draft_sigma_knots[-1]) != 0.0:
                raise ValueError("draft_sigma_knots must start at 1.0 and end at 0.0")
        if isinstance(self.h3_kv_splits, bool) or self.h3_kv_splits not in (1, 2, 4):
            raise ValueError(f"h3_kv_splits must be 1, 2, or 4, got {self.h3_kv_splits!r}")
        if isinstance(self.ltx25_sol_kv_splits, bool) or self.ltx25_sol_kv_splits not in (1, 2, 4):
            raise ValueError(f"ltx25_sol_kv_splits must be 1, 2, or 4, got {self.ltx25_sol_kv_splits!r}")
        if not math.isfinite(float(self.draft_flow_shift)) or float(self.draft_flow_shift) <= 0:
            raise ValueError(f"draft_flow_shift must be finite and positive, got {self.draft_flow_shift!r}")
        if not math.isfinite(float(self.duration_seconds)) or float(self.duration_seconds) <= 0:
            raise ValueError(f"duration_seconds must be positive, got {self.duration_seconds!r}")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if (self.target_height is None) != (self.target_width is None):
            raise ValueError("target_height and target_width must be supplied together")
        if self.target_height is not None and self.target_width is not None:
            _validate_positive_int(self.target_height, name="target_height")
            _validate_positive_int(self.target_width, name="target_width")
            if self.target_height % 32 != 0 or self.target_width % 32 != 0:
                raise ValueError(
                    f"target_height and target_width must be multiples of 32, got {self.target_width}x{self.target_height}"
                )
        else:
            _validate_positive_int(self.target_short_edge, name="target_short_edge")
        if len(self.refinement_sigmas) < 2:
            raise ValueError("refinement_sigmas must contain at least 2 knots")
        if self.ltx25_video_vae_variant not in ("conv", "diffvae"):
            raise ValueError(f"unsupported ltx25_video_vae_variant: {self.ltx25_video_vae_variant!r}")
        if self.upscale_mode not in ("bicubic", "bilinear"):
            raise ValueError(f"unsupported upscale_mode: {self.upscale_mode!r}")

    def resolve_target_canvas(self) -> tuple[int, int]:
        """Return ``(target_height, target_width)`` aligned to 32-pixel boundaries."""
        if self.target_height is not None and self.target_width is not None:
            return int(self.target_height), int(self.target_width)
        return _resolve_canvas(int(self.target_short_edge), self.aspect_ratio)

    def resolve_target_frames(self, duration_seconds: float | None = None) -> int:
        """Return LTX-2.5-aligned frame count ``8 * k + 1`` for the requested clip duration."""
        if self.target_num_frames is not None:
            return _align_ltx25_num_frames(int(self.target_num_frames))
        dur = float(self.duration_seconds if duration_seconds is None else duration_seconds)
        raw_frames = max(9, round(dur * float(self.fps)) + 1)
        return _align_ltx25_num_frames(raw_frames)

    def build_h3_acceleration_config(self, *, sm_major: int = 9) -> MiniMaxH3AccelerationConfig:
        """Build Stage-1 MiniMax-H3 SOL-Engine DiT config (``sol_attn(kv_splits=4)`` + FP8 + piecewise graph)."""
        base = minimax_h3_sol_profile(
            "sol_balanced",
            sm_major=sm_major,
            keep_dense_fallback=False,
            cuda_graph=self.h3_cuda_graph,
            kv_splits=self.h3_kv_splits,
        )
        attn_opts = dict(base.attention_options or {})
        attn_opts["tau"] = float(self.h3_sol_tau)
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend=base.attention_backend,
            attention_options=attn_opts,
            dense_attention_backend=base.dense_attention_backend,
            dense_attention_layers=base.dense_attention_layers,
            dense_initial_steps=int(self.h3_dense_initial_steps),
            quantization=base.quantization,
            fp8_scaling="rowwise",
            fp8_use_fast_accum=bool(self.h3_fp8_use_fast_accum),
            fp8_keep_dense_fallback=False,
            fp8_exclude_edge_blocks=bool(self.h3_fp8_exclude_edge_blocks),
            fuse_fp8_ffn=True,
            cuda_graph=self.h3_cuda_graph,
            cuda_graph_warmup=1,
            piecewise_head_blocks=2,
            piecewise_tail_blocks=2,
            notes=(
                f"Stage-1 Super-Accel H3 draft: sol_attn(tau={self.h3_sol_tau:g}, kv_splits={self.h3_kv_splits}) "
                f"+ FP8(fast_accum={self.h3_fp8_use_fast_accum}) + SwiGLU-FP8 + {self.h3_cuda_graph} graph",
            ),
        )

    def build_ltx25_acceleration_config(self, *, sm_major: int = 9) -> LTX25AccelerationConfig:
        """Build Stage-2 LTX-2.5 3-step refinement SOL-Engine config (FA3 + 3D-tiled Sol-Attn + FP8 + Graph + VAE)."""
        fp8_ok = int(sm_major) >= 9
        dense_attn = "flash_attention_3" if int(sm_major) == 9 else "torch"
        return LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=True,
            attention_backend=dense_attn,
            sol_attn=bool(self.ltx25_sol_attn) and int(sm_major) == 9,
            sol_tau=float(self.ltx25_sol_tau),
            sol_thresh_type="exact",
            sol_kv_splits=self.ltx25_sol_kv_splits,
            sol_tile_3d=self.ltx25_sol_tile_3d,
            sol_dense_blocks=(0, 1, -1),
            sol_dense_steps=int(self.ltx25_sol_dense_steps),
            sol_min_tokens=int(self.ltx25_sol_min_tokens),
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=bool(self.ltx25_fp8_use_fast_accum),
            fp8_exclude_edge_blocks=bool(self.ltx25_fp8_exclude_edge_blocks),
            fp8_fuse_ffn=True,
            fp8_keep_dense_fallback=False,
            fbcache=None,  # 3-step refinement runs every step densely for crisp detail
            cuda_graph=self.ltx25_cuda_graph,
            cuda_graph_warmup=1,
            vae_channels_last_3d=True,
            vae_compile=bool(self.ltx25_vae_compile),
            vae_tiled=bool(self.ltx25_vae_tiled),
            diffvae_optimization="combined_compile",
            audio_cuda_graph=True,
            parallel_audio=True,
            compile_vocoder=True,
            notes=(
                f"Stage-2 Super-Accel LTX-2.5 3-step refinement: fused ops + {dense_attn} "
                f"+ 3D-tiled Sol-Attn(tau={self.ltx25_sol_tau:g}, kv_splits={self.ltx25_sol_kv_splits}) "
                f"+ FP8(fast_accum={self.ltx25_fp8_use_fast_accum}) + {self.ltx25_cuda_graph} graph + accelerated VAE",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.draft_sigma_knots is not None:
            payload["draft_sigma_knots"] = list(self.draft_sigma_knots)
        payload["refinement_sigmas"] = list(self.refinement_sigmas)
        if self.ltx25_sol_tile_3d is not None:
            payload["ltx25_sol_tile_3d"] = list(self.ltx25_sol_tile_3d)
        return payload


def _audit_h3_zero_fallbacks(h3_report: Mapping[str, Any]) -> bool:
    """Verify Stage-1 MiniMax-H3 ran with zero attention, FP8, fused-FFN, or CUDA-Graph fallbacks."""
    tx = h3_report.get("transformer")
    if not isinstance(tx, Mapping):
        return True
    attn_rt = tx.get("attention_provider_runtime")
    if isinstance(attn_rt, Mapping):
        for row in attn_rt.values():
            if isinstance(row, Mapping) and (int(row.get("errors", 0)) > 0 or int(row.get("fallbacks", 0)) > 0):
                return False
    for quant_key in ("quantization", "quantization_runtime"):
        quant_rt = tx.get(quant_key)
        if isinstance(quant_rt, Mapping):
            if int(quant_rt.get("dense_fallback_calls", 0)) > 0:
                return False
            if int(quant_rt.get("lifetime_dense_fallback_calls", 0)) > 0:
                return False
    fused_swiglu = tx.get("fused_fp8_swiglu")
    if isinstance(fused_swiglu, Mapping) and int(fused_swiglu.get("fallback_calls", 0)) > 0:
        return False
    cg = tx.get("cuda_graph")
    if isinstance(cg, Mapping) and cg.get("enabled"):
        for seg in (cg.get("segments") or {}).values():
            if isinstance(seg, Mapping):
                if int(seg.get("capture_failed", 0)) > 0 or int(seg.get("disabled_signatures", 0)) > 0:
                    return False
                lifetime = seg.get("lifetime")
                if isinstance(lifetime, Mapping) and int(lifetime.get("capture_failed", 0)) > 0:
                    return False
        if int(cg.get("capture_failed", 0)) > 0 or int(cg.get("disabled_signatures", 0)) > 0:
            return False
        top_lifetime = cg.get("lifetime")
        if isinstance(top_lifetime, Mapping) and int(top_lifetime.get("capture_failed", 0)) > 0:
            return False
    return True


class MiniMaxH3SuperAccelerationPipeline:
    """Two-stage MiniMax-H3 (4-step draft) + LTX-2.5 (3-step refinement) pipeline."""

    def __init__(
        self,
        *,
        h3_pipeline: NativeMiniMaxH3Pipeline | Any,
        ltx25_runtime: LTX25DistilledRuntime | Any,
        config: SuperAccelerationConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.h3_pipeline = h3_pipeline
        self.ltx25_runtime = ltx25_runtime
        self.config = config or SuperAccelerationConfig()
        self.device = str(device)
        self._configured_receipt: dict[str, Any] | None = None
        self._last_receipt: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        h3_checkpoint_dir: str | Path,
        ltx25_checkpoint_dir: str | Path,
        config: SuperAccelerationConfig | None = None,
        device: str = "cuda:0",
        h3_partition: str = "FL2VA",
        configure: bool = True,
    ) -> MiniMaxH3SuperAccelerationPipeline:
        """Load Stage-1 MiniMax-H3 31B and Stage-2 LTX-2.5 22B distilled checkpoints."""
        cfg = config or SuperAccelerationConfig()
        dev = torch.device(device)
        sm_major = torch.cuda.get_device_capability(dev)[0] if dev.type == "cuda" and torch.cuda.is_available() else 0

        # Keep Qwen3-VL on CPU until H3 transformer is quantized to FP8 (~32 GiB instead of ~62 GiB)
        # when offloading text encoders after encoding, cutting peak load VRAM by ~30 GiB!
        h3_devices = {
            "transformer": str(dev),
            "video_vae": str(dev),
            "audio_vae": str(dev) if cfg.preserve_h3_audio else "cpu",
            "text_encoder": "cpu" if cfg.offload_text_encoders_after_encode else str(dev),
        }
        h3_pipe = NativeMiniMaxH3Pipeline.from_pretrained(
            str(Path(h3_checkpoint_dir).expanduser().resolve()),
            device=str(dev),
            partition=h3_partition,
            component_devices=h3_devices,
        )

        ltx25_accel = cfg.build_ltx25_acceleration_config(sm_major=sm_major)
        ltx25_runtime = LTX25DistilledRuntime(
            checkpoint_dir=Path(ltx25_checkpoint_dir).expanduser().resolve(),
            device=str(dev),
            offload_mode="none",
            model_variant="distilled",
            video_vae_variant=cfg.ltx25_video_vae_variant,
            acceleration=ltx25_accel,
        )

        pipeline = cls(
            h3_pipeline=h3_pipe,
            ltx25_runtime=ltx25_runtime,
            config=cfg,
            device=str(dev),
        )
        if configure:
            pipeline.configure_acceleration(sm_major=sm_major)
        return pipeline

    def configure_acceleration(self, *, sm_major: int | None = None) -> dict[str, Any]:
        """Apply SOL-Engine acceleration across Stage-1 H3 and Stage-2 LTX-2.5."""
        dev = torch.device(self.device)
        resolved_sm = (
            int(sm_major)
            if sm_major is not None
            else (torch.cuda.get_device_capability(dev)[0] if dev.type == "cuda" and torch.cuda.is_available() else 0)
        )
        h3_cfg = self.config.build_h3_acceleration_config(sm_major=resolved_sm)
        h3_receipt: dict[str, Any] = {}
        if hasattr(self.h3_pipeline, "configure_acceleration"):
            h3_receipt = self.h3_pipeline.configure_acceleration(
                acceleration=h3_cfg,
                vae_decode_autocast=torch.bfloat16 if self.config.h3_vae_decode_autocast else False,
                vae_channels_last_3d=self.config.h3_vae_channels_last_3d,
                vae_stack_tiling=self.config.h3_vae_stack_tiling,
                vae_tile_batch_size=self.config.h3_vae_tile_batch_size,
                vae_compile=self.config.h3_vae_compile,
            )

        ltx25_cfg = self.config.build_ltx25_acceleration_config(sm_major=resolved_sm)
        if hasattr(self.ltx25_runtime, "configure_acceleration"):
            self.ltx25_runtime.configure_acceleration(ltx25_cfg)

        self._configured_receipt = {
            "config": self.config.to_dict(),
            "stage1_h3": h3_receipt,
            "stage2_ltx25": ltx25_cfg.to_dict(),
        }
        return dict(self._configured_receipt)

    def encode_prompts(
        self,
        prompt: str,
        *,
        offload_after: bool | None = None,
        encode_h3: bool = True,
        encode_ltx25: bool = True,
    ) -> dict[str, Any]:
        """Encode ``prompt`` for Stage-1 H3 (Qwen3-VL) and/or Stage-2 LTX-2.5 (Gemma-4)."""
        do_offload = self.config.offload_text_encoders_after_encode if offload_after is None else bool(offload_after)
        dev = _normalize_device(self.device)

        h3_embeds: torch.Tensor | None = None
        t_h3_te = 0.0
        if encode_h3:
            _cuda_sync(dev)
            t0 = time.perf_counter()
            if getattr(self.h3_pipeline, "text_encoder", None) is not None:
                te_dev = self.h3_pipeline.component_devices.get("text_encoder", str(dev))
                if str(te_dev) == "cpu" and dev.type == "cuda":
                    self.h3_pipeline.text_encoder.to(dev)
                    self.h3_pipeline.component_devices["text_encoder"] = str(dev)
            h3_embeds = self.h3_pipeline._encode_prompt(prompt).detach().to(device=dev)
            if do_offload and getattr(self.h3_pipeline, "text_encoder", None) is not None:
                self.h3_pipeline.text_encoder.to("cpu")
                self.h3_pipeline.component_devices["text_encoder"] = "cpu"
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            _cuda_sync(dev)
            t_h3_te = time.perf_counter() - t0

        ltx25_embeds: dict[str, dict[str, torch.Tensor]] | None = None
        t_ltx_te = 0.0
        if encode_ltx25:
            _cuda_sync(dev)
            t1 = time.perf_counter()
            ltx25_embeds = self.ltx25_runtime.encode_prompt(
                prompt,
                negative_prompt=None,
                offload_after=do_offload,
                device=str(dev),
            )
            _cuda_sync(dev)
            t_ltx_te = time.perf_counter() - t1

        return {
            "h3_prompt_embeds": h3_embeds,
            "ltx25_prompt_embeds": ltx25_embeds,
            "timings": {
                "h3_text_encode_s": round(t_h3_te, 4),
                "ltx25_text_encode_s": round(t_ltx_te, 4),
                "total_text_encode_s": round(t_h3_te + t_ltx_te, 4),
            },
        }

    def _move_module(
        self,
        module: Any,
        target: str,
        *,
        graph_owner: Any = None,
    ) -> None:
        """Move ``module`` to ``target`` safely, invalidating captured CUDA graphs first."""
        if not isinstance(module, torch.nn.Module):
            return
        target_dev = _normalize_device(target)
        first_p = next(module.parameters(), None)
        if first_p is not None and _normalize_device(first_p.device) == target_dev:
            return
        # Moving weights off or onto GPU reallocates storage addresses; captured CUDA graphs
        # would otherwise dereference freed weight pointers (cudaErrorIllegalAddress).
        for candidate in (graph_owner, module):
            inv = getattr(candidate, "invalidate_cuda_graphs", None)
            if callable(inv):
                inv()
        module.to(target_dev)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        h3_prompt_embeds: torch.Tensor | None = None,
        ltx25_prompt_embeds: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        duration_seconds: float | None = None,
        seed: int = 42,
        task: str = "t2va",
        return_draft: bool = True,
        output_path: str | Path | None = None,
        **h3_kwargs: Any,
    ) -> dict[str, Any]:
        """Run Stage-1 4-step MiniMax-H3 draft -> Handoff -> Stage-2 3-step LTX-2.5 refinement."""
        cfg = self.config
        dev = torch.device(self.device)
        eff_duration = float(cfg.duration_seconds if duration_seconds is None else duration_seconds)
        target_h, target_w = cfg.resolve_target_canvas()
        target_frames = cfg.resolve_target_frames(eff_duration)

        t_text_total = 0.0
        if h3_prompt_embeds is None or ltx25_prompt_embeds is None:
            encoded = self.encode_prompts(
                prompt,
                encode_h3=(h3_prompt_embeds is None),
                encode_ltx25=(ltx25_prompt_embeds is None),
            )
            if h3_prompt_embeds is None:
                h3_prompt_embeds = encoded["h3_prompt_embeds"]
            if ltx25_prompt_embeds is None:
                ltx25_prompt_embeds = encoded["ltx25_prompt_embeds"]
            t_text_total = float(encoded["timings"]["total_text_encode_s"])

        # ── Stage 1: 4-step low-res MiniMax-H3 draft ────────────────────────
        if cfg.offload_between_stages and dev.type == "cuda":
            self._move_module(
                getattr(self.ltx25_runtime, "_transformer_wrapper", None),
                "cpu",
                graph_owner=getattr(self.ltx25_runtime, "_executor", None),
            )
            self._move_module(getattr(self.h3_pipeline, "transformer", None), str(dev))

        _reset_peak_vram(dev)
        _cuda_sync(dev)
        t_s1_start = time.perf_counter()

        # MiniMax-H3 ``minimax_h3_time_shift_sigmas(num_steps=N+1)`` yields N Euler steps.
        h3_gen_kwargs: dict[str, Any] = {
            "prompt_embeds": h3_prompt_embeds,
            "task": task,
            "short_edge": int(cfg.draft_short_edge),
            "aspect_ratio": cfg.aspect_ratio,
            "duration_seconds": eff_duration,
            "seed": int(seed),
            "flow_shift": float(cfg.draft_flow_shift),
            "audio_flow_shift": float(cfg.draft_audio_flow_shift),
            **h3_kwargs,
        }
        if cfg.draft_sigma_knots is not None:
            h3_gen_kwargs["sigma_knots"] = cfg.draft_sigma_knots
            h3_gen_kwargs["num_inference_steps"] = len(cfg.draft_sigma_knots)
        else:
            h3_gen_kwargs["num_inference_steps"] = int(cfg.draft_steps) + 1

        h3_out = self.h3_pipeline.generate(**h3_gen_kwargs)
        _cuda_sync(dev)
        t_s1_total = time.perf_counter() - t_s1_start
        s1_peak_vram = _peak_vram_gib(dev)

        draft_video = h3_out.get("video")
        if not isinstance(draft_video, torch.Tensor):
            raise RuntimeError("Stage-1 MiniMax-H3 pipeline did not return a decoded 'video' tensor")
        draft_audio = h3_out.get("audio") if cfg.preserve_h3_audio else None
        draft_plan = dict(h3_out.get("plan") or {})

        if cfg.offload_between_stages and dev.type == "cuda":
            self._move_module(getattr(self.h3_pipeline, "transformer", None), "cpu")
            self._move_module(
                getattr(self.ltx25_runtime, "_transformer_wrapper", None),
                str(dev),
                graph_owner=getattr(self.ltx25_runtime, "_executor", None),
            )

        # ── Stage 2: Handoff (upscale + LTX VAE encode) + 3-step LTX-2.5 refinement ──
        _reset_peak_vram(dev)
        ltx_out = self.ltx25_runtime.refine_video_3step(
            draft_video,
            prompt=prompt,
            target_height=target_h,
            target_width=target_w,
            target_num_frames=target_frames,
            fps=cfg.fps,
            sigmas=cfg.refinement_sigmas,
            seed=int(seed),
            upscale_mode=cfg.upscale_mode,
            prompt_embeds=ltx25_prompt_embeds,
            decode_audio=not cfg.preserve_h3_audio,
            offload_text_encoder=cfg.offload_text_encoders_after_encode,
        )
        s2_peak_vram = _peak_vram_gib(dev)
        ltx_timings = dict(ltx_out["timings"])

        handoff_upscale_s = float(ltx_timings.get("upscale_s", 0.0))
        handoff_encode_s = float(ltx_timings.get("vae_encode_s", 0.0))
        handoff_total_s = handoff_upscale_s + handoff_encode_s
        s2_dit_s = float(ltx_timings.get("dit_refine_s", 0.0))
        s2_vae_decode_s = float(ltx_timings.get("vae_decode_s", 0.0))
        s2_total_s = s2_dit_s + s2_vae_decode_s
        total_e2e_s = t_s1_total + handoff_total_s + s2_total_s

        h3_report = self.h3_pipeline.acceleration_report() if hasattr(self.h3_pipeline, "acceleration_report") else {}
        ltx_report = dict(ltx_out.get("acceleration_report") or {})
        h3_zero_fb = _audit_h3_zero_fallbacks(h3_report)
        ltx_zero_fb = bool(ltx_report.get("zero_fallbacks_verified", False)) if ltx_report else True

        receipt: dict[str, Any] = {
            "pipeline": "minimax-h3-super-acceleration",
            "config": cfg.to_dict(),
            "geometry": {
                "duration_seconds": eff_duration,
                "draft_shape_bcthw": [int(v) for v in draft_video.shape],
                "draft_plan": draft_plan,
                "refined_shape_fhwc": [int(v) for v in ltx_out["video"].shape],
                "target_height": target_h,
                "target_width": target_w,
                "target_frames": target_frames,
                "draft_steps": int(cfg.draft_steps),
                "refinement_steps": len(cfg.refinement_sigmas) - 1,
            },
            "timings": {
                "text_encode_s": round(t_text_total, 4),
                "stage1_h3_draft_total_s": round(t_s1_total, 4),
                "handoff_upscale_s": round(handoff_upscale_s, 4),
                "handoff_ltx_vae_encode_s": round(handoff_encode_s, 4),
                "handoff_total_s": round(handoff_total_s, 4),
                "stage2_ltx25_dit_refine_s": round(s2_dit_s, 4),
                "stage2_ltx25_vae_decode_s": round(s2_vae_decode_s, 4),
                "stage2_ltx25_total_s": round(s2_total_s, 4),
                "total_e2e_s": round(total_e2e_s, 4),
                "total_e2e_with_text_s": round(total_e2e_s + t_text_total, 4),
            },
            "peak_vram_gib": {
                "stage1_h3": s1_peak_vram,
                "stage2_ltx25": s2_peak_vram,
                "max": max(s1_peak_vram, s2_peak_vram),
            },
            "stage1_h3_report": h3_report,
            "stage2_ltx25_report": ltx_report,
            "zero_fallbacks_verified": h3_zero_fb and ltx_zero_fb,
        }
        self._last_receipt = receipt

        refined_video = ltx_out["video"]
        final_audio = draft_audio if cfg.preserve_h3_audio else ltx_out.get("audio")
        final_audio_sr = 32000 if cfg.preserve_h3_audio else int(ltx_out.get("audio_sampling_rate", 48000))

        result: dict[str, Any] = {
            "video": refined_video,
            "video_latent": ltx_out.get("video_latent"),
            "audio": final_audio,
            "audio_sampling_rate": final_audio_sr,
            "timings": receipt["timings"],
            "receipt": receipt,
        }
        if return_draft:
            result["draft_video"] = draft_video

        if output_path is not None:
            from worldfoundry.core.media.codecs.pyav import save_audio_video_h264

            out_p = Path(output_path).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            mux_audio = _normalize_stereo_waveform(final_audio)
            save_audio_video_h264(
                refined_video,
                str(out_p),
                fps=cfg.fps,
                waveform=mux_audio,
                sample_rate=final_audio_sr if mux_audio is not None else None,
            )
            result["output_path"] = str(out_p)
        return result

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.generate(prompt=prompt, **kwargs)

    def acceleration_report(self) -> dict[str, Any]:
        """Return the most recent end-to-end Super Acceleration receipt."""
        if self._last_receipt is not None:
            return dict(self._last_receipt)
        return dict(self._configured_receipt or {"config": self.config.to_dict()})


__all__ = [
    "MiniMaxH3SuperAccelerationPipeline",
    "SuperAccelerationConfig",
]
