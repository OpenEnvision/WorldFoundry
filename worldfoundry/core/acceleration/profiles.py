# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Declarative acceleration profiles and multi-GPU topology planning.

Translates a concise, user-facing acceleration tier (``exact``, ``fast``,
``turbo``, ``sol``, ``realtime``) plus hardware geometry (``world_size``,
``sm_major``, ``num_heads``) into an explicit, auditable set of runtime options
accepted by native diffusion pipelines. Importing this module does not load
PyTorch or initialize CUDA.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

AccelerationTier = Literal["exact", "fast", "turbo", "sol", "realtime"]

_VALID_TIERS: frozenset[str] = frozenset({"exact", "fast", "turbo", "sol", "realtime"})


@dataclass(frozen=True, slots=True)
class ParallelTopologyPlan:
    """Validated multi-GPU parallelism mesh for one video generation workload.

    Attributes:
        world_size: Total participating GPU ranks.
        cfg_parallel: Classifier-free guidance replica axis (1 or 2).
        ulysses_degree: Head-sharded Ulysses sequence-parallel axis.
        ring_degree: Ring-attention sequence-parallel axis when heads cannot
            divide ``world_size // cfg_parallel`` evenly.
        vae_parallel: Parallel spatial tiled VAE decode degree.
        padded_ulysses: Whether Ulysses head padding is needed for exact head
            count alignment.
        notes: Human-readable planning rationale for receipts and logs.
    """

    world_size: int
    cfg_parallel: int
    ulysses_degree: int
    ring_degree: int = 1
    vae_parallel: int = 1
    padded_ulysses: bool = False
    notes: tuple[str, ...] = ()

    @property
    def sequence_parallel_degree(self) -> int:
        """Total sequence-parallel factor (``ulysses_degree * ring_degree``)."""
        return self.ulysses_degree * self.ring_degree

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology fields into a JSON-safe receipt."""
        return {
            "world_size": self.world_size,
            "cfg_parallel": self.cfg_parallel,
            "ulysses_degree": self.ulysses_degree,
            "ring_degree": self.ring_degree,
            "sequence_parallel_degree": self.sequence_parallel_degree,
            "vae_parallel": self.vae_parallel,
            "padded_ulysses": self.padded_ulysses,
            "notes": list(self.notes),
        }


def plan_parallel_topology(
    world_size: int,
    *,
    num_heads: int | None = None,
    supports_cfg: bool = True,
    prefer_cfg_parallel: bool = True,
    enable_vae_parallel: bool = True,
) -> ParallelTopologyPlan:
    """Plan an optimal ``(cfg_parallel, ulysses, ring, vae_parallel)`` mesh.

    For 2-branch CFG video DiTs on NVSwitch nodes (e.g. 2/4/8x H200), splitting
    positive and negative branches across ``cfg_parallel=2`` eliminates half of
    the per-block all-to-all traffic and resolves head-divisibility limits such
    as 12-head models on 8 GPUs (``CFG=2 x Ulysses=4``).

    Args:
        world_size: Number of GPUs in the distributed world (>= 1).
        num_heads: Attention head count of the DiT, if known.
        supports_cfg: Whether the workload executes 2-branch classifier-free
            guidance.
        prefer_cfg_parallel: Prefer dedicating a factor of 2 to CFG parallelism
            before scaling sequence parallelism.
        enable_vae_parallel: Enable multi-rank tiled VAE decode when
            ``world_size > 1``.

    Returns:
        A validated :class:`ParallelTopologyPlan`.

    Raises:
        ValueError: If ``world_size < 1`` or ``num_heads`` is non-positive.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if num_heads is not None and (isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1):
        raise ValueError(f"num_heads must be a positive integer when provided, got {num_heads!r}")

    if world_size == 1:
        return ParallelTopologyPlan(
            world_size=1,
            cfg_parallel=1,
            ulysses_degree=1,
            ring_degree=1,
            vae_parallel=1,
            notes=("single-gpu execution",),
        )

    notes: list[str] = []
    cfg_degree = 2 if (supports_cfg and prefer_cfg_parallel and world_size % 2 == 0) else 1
    sp_budget = world_size // cfg_degree
    if cfg_degree == 2:
        notes.append("assigned 2-way CFG parallelism before sequence sharding")

    if num_heads is None or num_heads % sp_budget == 0:
        ulysses = sp_budget
        ring = 1
        padded = False
    else:
        # Select the largest exact divisor of both sp_budget and num_heads for
        # Ulysses, delegating the remainder to Ring (2D USP) when possible.
        ulysses = math.gcd(sp_budget, num_heads)
        ring = sp_budget // ulysses
        padded = False
        if ulysses == 1 and sp_budget > 1:
            # No common head divisor: prefer padded Ulysses on high-bandwidth
            # intra-node fabrics over pure ring P2P.
            ulysses = sp_budget
            ring = 1
            padded = True
            notes.append(f"num_heads={num_heads} is coprime to sp={sp_budget}; requires padded Ulysses")
        else:
            notes.append(f"2D USP split for num_heads={num_heads}: ulysses={ulysses} x ring={ring}")

    vae_parallel = world_size if enable_vae_parallel else 1
    return ParallelTopologyPlan(
        world_size=world_size,
        cfg_parallel=cfg_degree,
        ulysses_degree=ulysses,
        ring_degree=ring,
        vae_parallel=vae_parallel,
        padded_ulysses=padded,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class AccelerationProfileSpec:
    """Resolved acceleration preset ready for pipeline construction."""

    tier: AccelerationTier
    topology: ParallelTopologyPlan
    options: Mapping[str, Any]
    quality_contract: str

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Return a mutable copy of kwargs for ``Pipeline.from_pretrained``."""
        return dict(self.options)

    def to_receipt(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of requested preset behavior."""
        return {
            "tier": self.tier,
            "quality_contract": self.quality_contract,
            "topology": self.topology.to_dict(),
            "options": dict(self.options),
        }


def resolve_acceleration_profile(
    tier: AccelerationTier | str,
    *,
    world_size: int = 1,
    num_heads: int | None = None,
    height: int | None = None,
    supports_cfg: bool = True,
    is_causal_ar: bool = False,
    sm_major: int = 9,
    enable_compile: bool = False,
    enable_cuda_graph: bool = True,
    enable_vae_parallel: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> AccelerationProfileSpec:
    """Build a coherent, non-conflicting acceleration configuration.

    Tiers:
        - ``exact``: Resident BF16 weights + lossless operator fusions (QKV,
          static cross-KV, inplace residual) + VAE ``channels_last_3d``/autocast
          + whole CUDA Graph + multi-GPU CFG/Ulysses topology (bitwise parity).
        - ``fast``: ``exact`` plus FP64 Triton fused QK-RMSNorm+3D-RoPE and
          Hopper/Blackwell rowwise FP8 W8A8 GEMMs with FP32 accumulation
          (``use_fast_accum=False``, ``keep_dense_fallback=False``).
        - ``turbo``: ``fast`` plus TeaCache cross-step residual caching composed
          with piecewise block-stack CUDA Graphs.
        - ``sol``: ``turbo`` plus scheduled ``1x8x8`` spatial-tiled Sol-Attn
          (dense warmup steps + dense edge blocks) under piecewise CUDA Graphs.
        - ``realtime``: Optimized for causal autoregressive streaming world
          models (zero-sync KV pointers, fused QKV, Triton FP64 Real-RoPE,
          chunk-level piecewise CUDA Graph, compiled streaming VAE).

    Args:
        tier: Preset tier name.
        world_size: Active GPU rank count.
        num_heads: Transformer attention head count for exact SP planning.
        height: Optional target video height in pixels; when ``<= 480`` spatial
            VAE tiling is skipped because untiled autocast decode is faster.
        supports_cfg: True for 2-branch CFG models; False for embedded guidance
            or distilled AR models.
        is_causal_ar: True for causal chunk-streaming models (Self-Forcing,
            Causal-Forcing, LongLive, Matrix-Game).
        sm_major: CUDA compute capability major version (9 for H100/H200, 10 for
            Blackwell).
        enable_compile: Opt into ``torch.compile`` on compatible regions.
        enable_cuda_graph: Enable whole-model or piecewise CUDA Graph replay.
        enable_vae_parallel: Enable multi-GPU tiled VAE decode when
            ``world_size > 1`` and resolution benefits from spatial tiling.
        overrides: Explicit user overrides applied last.

    Returns:
        :class:`AccelerationProfileSpec` containing resolved kwargs and audit
        metadata.
    """
    normalized = str(tier).strip().lower()
    if normalized not in _VALID_TIERS:
        allowed = ", ".join(sorted(_VALID_TIERS))
        raise ValueError(f"unsupported acceleration tier {tier!r}; expected one of: {allowed}")
    resolved_tier = cast(AccelerationTier, normalized)

    if resolved_tier == "realtime":
        is_causal_ar = True
        supports_cfg = False

    resolution_benefits_from_tiling = height is None or int(height) > 480
    topology = plan_parallel_topology(
        world_size,
        num_heads=num_heads,
        supports_cfg=supports_cfg and not is_causal_ar,
        prefer_cfg_parallel=not is_causal_ar,
        enable_vae_parallel=enable_vae_parallel and not is_causal_ar and resolution_benefits_from_tiling,
    )

    options: dict[str, Any] = {
        "torch_dtype": "bfloat16",
        "offload_mode": "resident",
        "attention_backend": "torch",
        "dit_weight_dtype": "bfloat16",
        "vae_weight_dtype": "bfloat16",
        "vae_channels_last_3d": True,
        "vae_decode_autocast": True,
    }

    if topology.cfg_parallel > 1:
        options["cfg_parallel"] = topology.cfg_parallel
    if topology.ulysses_degree > 1:
        options["sequence_parallel"] = topology.ulysses_degree
        if topology.padded_ulysses:
            options["sequence_parallel_pad_heads"] = True
    if topology.ring_degree > 1:
        options["ring_degree"] = topology.ring_degree
    if topology.vae_parallel > 1:
        options["vae_parallel"] = topology.vae_parallel
        options["vae_spatial_tiling"] = True

    if is_causal_ar:
        options["fuse_qkv"] = True
        options["rope_precision"] = "fp64"
        options["vae_channels_last_3d"] = True
        options["vae_streaming_compile"] = True
        if enable_cuda_graph:
            options["cuda_graph"] = True
        if enable_compile:
            options["compile"] = True
            options["compile_scope"] = "regional"
        quality_contract = "realtime-streaming-parity"
    else:
        options["fuse_qkv"] = True
        options["static_cross_kv"] = True
        options["inplace_residual"] = True
        # Whole/piecewise CUDA Graph runs cleanly on 1 GPU and CFG-only 2 GPU;
        # under Ulysses SP > 1 NCCL collective capture stays off by default.
        graph_allowed = enable_cuda_graph and topology.ulysses_degree == 1
        if graph_allowed:
            options["cuda_graph"] = True
        if enable_compile:
            options["compile"] = True
            options["compile_scope"] = "regional" if graph_allowed else "full"
            if not graph_allowed:
                options["compile_mode"] = "default"

        if resolved_tier == "exact":
            quality_contract = "lossless-bf16-parity (bitwise latent parity, PSNR>=50dB)"
        elif resolved_tier in {"fast", "turbo", "sol"}:
            options["fused_rope"] = True
            options["rope_precision"] = "fp64"
            if sm_major >= 9:
                # Exclude condition/head projections plus cross_attn (already cached once per
                # request via static_cross_kv). Full FP32 accumulation (use_fast_accum=False)
                # avoids Hopper FP22 norm drift across deep 30-40 block DiTs, and
                # keep_dense_fallback=False releases duplicate BF16 weight copies.
                exclude: tuple[str, ...] = (
                    "time_embedding",
                    "time_projection",
                    "text_embedding",
                    "head",
                    "img_emb",
                    "cross_attn",
                )
                options["quantization"] = {
                    "mode": "fp8",
                    "exclude": exclude,
                    "options": {
                        "min_features": 1024,
                        "scaling": "rowwise",
                        "use_fast_accum": False,
                        "fuse_fp8_ffn": False,
                        "keep_dense_fallback": False,
                    },
                }
            if resolved_tier == "fast":
                quality_contract = "fp8-exact-accum (perceptual parity, trajectory-shifted vs BF16)"
            elif resolved_tier == "turbo":
                options["teacache"] = 0.15
                options["inplace_residual"] = False
                if graph_allowed:
                    options["cuda_graph"] = "piecewise"
                quality_contract = "turbo-cached-fp8 (perceptual parity, TeaCache 0.15 + FP8)"
            else:
                # Sol-Engine 5-pillar stack: scheduled 1x8x8 spatial-tiled Sol-Attn +
                # FP8 W8A8 + TeaCache + Fused QKV + Static Cross-KV + Piecewise CUDA Graph.
                options["teacache"] = 0.15
                options["inplace_residual"] = False
                options["accelerations"] = {
                    "attention_policy": {
                        "self": {
                            "backend": "sol_attn",
                            "tau": 1.0,
                            "dense_steps": 10,
                            "dense_blocks": (0, 1, -1),
                            "tile_3d": (1, 8, 8),
                        }
                    },
                }
                if graph_allowed:
                    options["cuda_graph"] = "piecewise"
                quality_contract = "sol-full-stack (Scheduled 1x8x8 Sol-Attn + FP8 + TeaCache + Piecewise Graph)"
        else:  # pragma: no cover - guarded by _VALID_TIERS
            quality_contract = "custom"

    if overrides:
        options.update(dict(overrides))

    return AccelerationProfileSpec(
        tier=resolved_tier,
        topology=topology,
        options=options,
        quality_contract=quality_contract,
    )


__all__ = [
    "AccelerationProfileSpec",
    "AccelerationTier",
    "ParallelTopologyPlan",
    "plan_parallel_topology",
    "resolve_acceleration_profile",
]
