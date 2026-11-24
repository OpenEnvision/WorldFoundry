# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Sol-Engine-inspired 5-pillar acceleration stack and conflict-resolving integrator.

Composes five orthogonal inference optimization pillars plus multi-GPU
parallelism and graph execution into one validated, auditable recipe:

1. **Caching** (TeaCache / MagCache / EasyCache / TaylorSeer)
2. **Sparse / Fast Attention** (Sol-Attn on-chip block thresholding, cuDNN9 SDPA,
   Triton TMA, cuDNN FP8)
3. **Token Pruning** (feature-norm / velocity token pruning with previous-step
   compensation across middle blocks)
4. **Quantization** (Hopper/Blackwell rowwise FP8 W8A8 with sensitivity-aware
   layer and edge-block exclusions)
5. **Kernel Fusion & Execution Graphs** (Fused QKV, Fused QK-RMSNorm+3D RoPE,
   Static Cross-KV, Piecewise CUDA Graph, Async Overlapped VAE, 3D CFG-P x
   Ulysses SP x Parallel Tiled VAE)

Importing this module does not import PyTorch or initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

from worldfoundry.core.acceleration.profiles import ParallelTopologyPlan, plan_parallel_topology

QualityBudgetTier = Literal["lossless", "high_fidelity", "balanced", "aggressive", "realtime"]
CacheAlgorithm = Literal["none", "teacache", "magcache", "easycache", "taylorseer"]
AttentionPolicyBackend = Literal["torch", "flash_attention_3", "sol_attn", "triton_tma", "cudnn_fp8"]
CudaGraphMode = Literal["off", "whole", "piecewise"]

_DEFAULT_FP8_EXCLUDES: tuple[str, ...] = (
    "time_embedding",
    "time_projection",
    "text_embedding",
    "head",
    "img_emb",
    "cross_attn",
)

_BUDGET_DEFAULTS: dict[str, tuple[float | None, float | None, float | None]] = {
    # tier -> (min_psnr_db, min_ssim, max_latent_rel_l2)
    "lossless": (45.0, 0.995, 0.02),
    "high_fidelity": (33.0, 0.96, 0.12),
    "balanced": (28.0, 0.90, 0.25),
    "aggressive": (25.0, 0.84, 0.35),
    "realtime": (40.0, 0.99, 0.05),
}


@dataclass(frozen=True, slots=True)
class QualityBudget:
    """Quantitative quality gate constraining Sol-Engine stack composition."""

    tier: QualityBudgetTier = "balanced"
    min_psnr_db: float | None = 28.0
    min_ssim: float | None = 0.90
    max_latent_rel_l2: float | None = 0.25

    @classmethod
    def for_tier(cls, tier: QualityBudgetTier | str) -> QualityBudget:
        """Construct the default quantitative quality contract for ``tier``."""
        normalized = str(tier).strip().lower()
        if normalized not in _BUDGET_DEFAULTS:
            allowed = ", ".join(sorted(_BUDGET_DEFAULTS))
            raise ValueError(f"unsupported quality budget tier {tier!r}; expected one of: {allowed}")
        psnr, ssim, rel_l2 = _BUDGET_DEFAULTS[normalized]
        return cls(
            tier=cast(QualityBudgetTier, normalized),
            min_psnr_db=psnr,
            min_ssim=ssim,
            max_latent_rel_l2=rel_l2,
        )

    def accepts(self, quality: Mapping[str, Any]) -> bool:
        """Return whether a benchmark quality dict satisfies every threshold."""
        if not quality:
            return False
        if quality.get("reference") is True:
            return True
        if quality.get("finite") is False:
            return False
        raw_psnr = quality.get("video_psnr_db")
        if (
            self.min_psnr_db is not None
            and raw_psnr is not None
            and raw_psnr != "infinity"
            and float(raw_psnr) < self.min_psnr_db
        ):
            return False
        raw_ssim = quality.get("video_ssim")
        if self.min_ssim is not None and raw_ssim is not None and float(raw_ssim) < self.min_ssim:
            return False
        raw_l2 = quality.get("latent_relative_l2")
        return not (
            self.max_latent_rel_l2 is not None and raw_l2 is not None and float(raw_l2) > self.max_latent_rel_l2
        )


@dataclass(frozen=True, slots=True)
class ModelHardwareProfile:
    """Static architecture and hardware target facts for stack integration."""

    model_id: str
    num_heads: int
    head_dim: int = 128
    num_layers: int = 30
    world_size: int = 1
    sm_major: int = 9
    supports_cfg: bool = True
    is_causal_ar: bool = False
    available_attention_backends: tuple[str, ...] = ("torch",)


@dataclass(frozen=True, slots=True)
class CachePillar:
    """Pillar 1: Cross-step residual/feature caching configuration."""

    algorithm: CacheAlgorithm = "none"
    threshold: float = 0.15


@dataclass(frozen=True, slots=True)
class SparseAttentionPillar:
    """Pillar 2: Sparse or specialized attention provider configuration."""

    backend: AttentionPolicyBackend = "torch"
    tau: float = 1.0
    thresh_type: Literal["diag", "exact"] = "diag"
    compress_kv: bool = True
    kv_splits: int = 1
    allow_multi_split: bool = False
    sink_tokens: int | None = None
    sink_start: int | None = None
    dense_steps: int = 10
    dense_blocks: tuple[int, ...] = (0, 1, -1)
    tile_3d: tuple[int, int, int] | None = (1, 8, 8)


@dataclass(frozen=True, slots=True)
class TokenPruningPillar:
    """Pillar 3: Mid-stack feature-norm token pruning with compensation."""

    enabled: bool = False
    keep_ratio: float = 0.75
    method: str = "feat_norm"
    start_block: int = 2
    end_block: int = -2
    dense_first_steps: int = 5
    dense_last_steps: int = 5


@dataclass(frozen=True, slots=True)
class QuantizationPillar:
    """Pillar 4: Hardware-native weight/activation quantization configuration."""

    mode: Literal["none", "fp8"] = "none"
    scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    use_fast_accum: bool = False
    fuse_fp8_ffn: bool = True
    keep_dense_fallback: bool = False
    min_features: int = 1024
    exclude_edge_blocks: bool = False
    exclude: tuple[str, ...] = _DEFAULT_FP8_EXCLUDES


@dataclass(frozen=True, slots=True)
class KernelFusionPillar:
    """Pillar 5a: Lossless/near-lossless operator and memory layout fusions."""

    fuse_qkv: bool = True
    fused_rope: bool = False
    static_cross_kv: bool = True
    inplace_residual: bool = True
    vae_channels_last_3d: bool = True
    vae_decode_autocast: bool = True
    vae_weight_dtype: str = "bfloat16"
    rope_precision: Literal["fp32", "fp64"] = "fp64"


@dataclass(frozen=True, slots=True)
class ExecutionPillar:
    """Pillar 5b: CUDA Graph capture, compilation, and async VAE execution."""

    cuda_graph: CudaGraphMode = "whole"
    compile: bool = False
    compile_scope: Literal["regional", "full"] = "regional"
    overlapped_vae: bool = False


@dataclass(frozen=True, slots=True)
class SolEngineStack:
    """Validated, conflict-free 5-pillar acceleration stack ready for execution."""

    hardware: ModelHardwareProfile
    budget: QualityBudget
    topology: ParallelTopologyPlan
    cache: CachePillar
    sparse_attention: SparseAttentionPillar
    token_pruning: TokenPruningPillar
    quantization: QuantizationPillar
    kernel_fusion: KernelFusionPillar
    execution: ExecutionPillar
    composition_notes: tuple[str, ...] = field(default_factory=tuple)

    def active_pillars(self) -> tuple[str, ...]:
        """Return names of all active optimization pillars for concise summaries."""
        active: list[str] = []
        if self.cache.algorithm != "none":
            active.append(f"cache:{self.cache.algorithm}({self.cache.threshold:g})")
        if self.sparse_attention.backend != "torch":
            if self.sparse_attention.backend == "sol_attn":
                active.append(f"sparse_attn:sol_attn(tau={self.sparse_attention.tau:g})")
            else:
                active.append(f"sparse_attn:{self.sparse_attention.backend}")
        if self.token_pruning.enabled:
            active.append(f"token_pruning:{self.token_pruning.method}({self.token_pruning.keep_ratio:g})")
        if self.quantization.mode != "none":
            active.append(f"quant:{self.quantization.mode}-{self.quantization.scaling}")
        fusions: list[str] = []
        if self.kernel_fusion.fuse_qkv:
            fusions.append("qkv")
        if self.kernel_fusion.fused_rope:
            fusions.append("rope")
        if self.kernel_fusion.static_cross_kv and not self.hardware.is_causal_ar:
            fusions.append("xkv")
        if fusions:
            active.append(f"fusion:{'+'.join(fusions)}")
        if self.execution.cuda_graph != "off":
            active.append(f"graph:{self.execution.cuda_graph}")
        if self.execution.overlapped_vae:
            active.append("async_vae")
        if self.topology.world_size > 1:
            parts = []
            if self.topology.cfg_parallel > 1:
                parts.append(f"cfg{self.topology.cfg_parallel}")
            if self.topology.ulysses_degree > 1:
                parts.append(f"sp{self.topology.ulysses_degree}")
            if self.topology.vae_parallel > 1:
                parts.append(f"vaep{self.topology.vae_parallel}")
            active.append(f"parallel:{'x'.join(parts)}")
        return tuple(active)

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Compile this validated stack into native WorldFoundry pipeline kwargs."""
        opts: dict[str, Any] = {
            "torch_dtype": "bfloat16",
            "offload_mode": "resident",
            "attention_backend": "torch",
            "dit_weight_dtype": "bfloat16",
            "vae_weight_dtype": self.kernel_fusion.vae_weight_dtype,
            "vae_channels_last_3d": self.kernel_fusion.vae_channels_last_3d,
            "vae_decode_autocast": self.kernel_fusion.vae_decode_autocast,
        }
        if self.topology.cfg_parallel > 1:
            opts["cfg_parallel"] = self.topology.cfg_parallel
        if self.topology.ulysses_degree > 1:
            opts["sequence_parallel"] = self.topology.ulysses_degree
            if self.topology.padded_ulysses:
                opts["sequence_parallel_pad_heads"] = True
        if self.topology.ring_degree > 1:
            opts["ring_degree"] = self.topology.ring_degree
        if self.topology.vae_parallel > 1:
            opts["vae_parallel"] = self.topology.vae_parallel
            opts["vae_spatial_tiling"] = True
        if self.execution.overlapped_vae:
            opts["overlap_vae_decode"] = True

        if self.kernel_fusion.fuse_qkv:
            opts["fuse_qkv"] = True

        if self.hardware.is_causal_ar:
            opts["rope_precision"] = self.kernel_fusion.rope_precision
            opts["vae_streaming_compile"] = True
            if self.execution.cuda_graph != "off":
                opts["cuda_graph"] = True
        else:
            if self.kernel_fusion.fused_rope:
                opts["fused_rope"] = True
                opts["rope_precision"] = self.kernel_fusion.rope_precision
            if self.kernel_fusion.static_cross_kv:
                opts["static_cross_kv"] = True
            opts["inplace_residual"] = bool(self.kernel_fusion.inplace_residual)
            if self.execution.cuda_graph == "whole":
                opts["cuda_graph"] = True
            elif self.execution.cuda_graph == "piecewise":
                opts["cuda_graph"] = "piecewise"

        if self.execution.compile:
            opts["compile"] = True
            opts["compile_scope"] = self.execution.compile_scope
            if self.execution.cuda_graph == "off":
                opts["compile_mode"] = "default"

        if self.quantization.mode == "fp8":
            excludes = list(self.quantization.exclude)
            if self.quantization.exclude_edge_blocks and self.hardware.num_layers >= 2:
                excludes.extend(["blocks.0.", f"blocks.{self.hardware.num_layers - 1}."])
            opts["quantization"] = {
                "mode": "fp8",
                "exclude": tuple(dict.fromkeys(excludes)),
                "options": {
                    "min_features": self.quantization.min_features,
                    "scaling": self.quantization.scaling,
                    "use_fast_accum": self.quantization.use_fast_accum,
                    "fuse_fp8_ffn": self.quantization.fuse_fp8_ffn,
                    "keep_dense_fallback": self.quantization.keep_dense_fallback,
                },
            }

        accelerations: dict[str, Any] = {}
        if self.cache.algorithm == "teacache":
            opts["teacache"] = float(self.cache.threshold)
        elif self.cache.algorithm == "taylorseer":
            opts["taylorseer"] = True
        elif self.cache.algorithm == "magcache":
            opts["magcache"] = float(self.cache.threshold)
        elif self.cache.algorithm == "easycache":
            accelerations["easycache"] = {"threshold": float(self.cache.threshold)}

        if self.sparse_attention.backend == "sol_attn":
            sol_entry: dict[str, Any] = {
                "backend": "sol_attn",
                "tau": float(self.sparse_attention.tau),
                "thresh_type": self.sparse_attention.thresh_type,
                "compress_kv": self.sparse_attention.compress_kv,
            }
            if self.sparse_attention.kv_splits > 1:
                sol_entry["kv_splits"] = int(self.sparse_attention.kv_splits)
                sol_entry["allow_multi_split"] = bool(self.sparse_attention.allow_multi_split)
            if self.sparse_attention.sink_tokens is not None:
                sol_entry["sink_tokens"] = int(self.sparse_attention.sink_tokens)
            if self.sparse_attention.sink_start is not None:
                sol_entry["sink_start"] = int(self.sparse_attention.sink_start)
            if self.sparse_attention.dense_steps > 0:
                sol_entry["dense_steps"] = int(self.sparse_attention.dense_steps)
            if self.sparse_attention.dense_blocks:
                sol_entry["dense_blocks"] = tuple(int(b) for b in self.sparse_attention.dense_blocks)
            if self.sparse_attention.tile_3d is not None:
                sol_entry["tile_3d"] = tuple(int(v) for v in self.sparse_attention.tile_3d)
            accelerations["attention_policy"] = {"self": sol_entry}
        elif self.sparse_attention.backend == "flash_attention_3":
            opts["attention_backend"] = "flash_attention_3"
        elif self.sparse_attention.backend != "torch":
            accelerations["attention_policy"] = {"self": {"backend": self.sparse_attention.backend}}

        if self.token_pruning.enabled:
            accelerations["token_pruning"] = {
                "keep_ratio": float(self.token_pruning.keep_ratio),
                "method": self.token_pruning.method,
                "start_block": int(self.token_pruning.start_block),
                "end_block": int(self.token_pruning.end_block),
                "dense_first_steps": int(self.token_pruning.dense_first_steps),
                "dense_last_steps": int(self.token_pruning.dense_last_steps),
            }

        if accelerations:
            opts["accelerations"] = accelerations
            if len(accelerations) > 1 or (
                self.cache.algorithm != "none"
                and (self.sparse_attention.backend != "torch" or self.token_pruning.enabled)
            ):
                opts["allow_approximate_composition"] = True
        return opts

    def to_recipe_dict(self) -> dict[str, Any]:
        """Serialize the full stack into a versioned JSON-compatible recipe."""
        return {
            "schema_version": 1,
            "engine": "worldfoundry-sol-stack",
            "active_pillars": list(self.active_pillars()),
            "hardware": asdict(self.hardware),
            "budget": asdict(self.budget),
            "topology": self.topology.to_dict(),
            "pillars": {
                "cache": asdict(self.cache),
                "sparse_attention": asdict(self.sparse_attention),
                "token_pruning": asdict(self.token_pruning),
                "quantization": asdict(self.quantization),
                "kernel_fusion": asdict(self.kernel_fusion),
                "execution": asdict(self.execution),
            },
            "composition_notes": list(self.composition_notes),
            "pipeline_kwargs": self.to_pipeline_kwargs(),
        }

    @classmethod
    def from_recipe_dict(cls, payload: Mapping[str, Any]) -> SolEngineStack:
        """Reconstruct a validated :class:`SolEngineStack` from a saved recipe."""
        hw_raw = dict(payload["hardware"])
        hw_raw["available_attention_backends"] = tuple(hw_raw.get("available_attention_backends", ("torch",)))
        hw = ModelHardwareProfile(**hw_raw)
        budget = QualityBudget(**payload["budget"])
        pillars = payload["pillars"]
        quant_raw = dict(pillars["quantization"])
        quant_raw["exclude"] = tuple(quant_raw.get("exclude", _DEFAULT_FP8_EXCLUDES))
        sparse_raw = dict(pillars["sparse_attention"])
        if "dense_blocks" in sparse_raw and sparse_raw["dense_blocks"] is not None:
            sparse_raw["dense_blocks"] = tuple(int(b) for b in sparse_raw["dense_blocks"])
        if "tile_3d" in sparse_raw and sparse_raw["tile_3d"] is not None:
            sparse_raw["tile_3d"] = tuple(int(v) for v in sparse_raw["tile_3d"])
        topo_raw = payload.get("topology") or {}
        enable_vae_p = bool(int(topo_raw.get("vae_parallel", 1)) > 1) if hw.world_size > 1 else True
        return compose_sol_stack(
            hw,
            budget=budget,
            cache=CachePillar(**pillars["cache"]),
            sparse_attention=SparseAttentionPillar(**sparse_raw),
            token_pruning=TokenPruningPillar(**pillars["token_pruning"]),
            quantization=QuantizationPillar(**quant_raw),
            kernel_fusion=KernelFusionPillar(**pillars["kernel_fusion"]),
            execution=ExecutionPillar(**pillars["execution"]),
            enable_vae_parallel=enable_vae_p,
        )


def compose_sol_stack(
    hardware: ModelHardwareProfile,
    *,
    budget: QualityBudget | QualityBudgetTier | str = "balanced",
    cache: CachePillar | None = None,
    sparse_attention: SparseAttentionPillar | None = None,
    token_pruning: TokenPruningPillar | None = None,
    quantization: QuantizationPillar | None = None,
    kernel_fusion: KernelFusionPillar | None = None,
    execution: ExecutionPillar | None = None,
    enable_vae_parallel: bool = True,
    height: int | None = None,
) -> SolEngineStack:
    """Integrate the 5 Sol-Engine pillars into one conflict-free execution stack.

    When individual pillars are omitted, selects hardware- and budget-appropriate
    defaults. When pillars interact (for example TeaCache + CUDA Graph, or
    Ulysses SP + CUDA Graph, or Sol-Attn + head_dim), resolves safe compositions
    explicitly and records every decision in ``composition_notes``. Fails closed
    on genuinely impossible explicit requests instead of degrading silently.
    """
    resolved_budget = budget if isinstance(budget, QualityBudget) else QualityBudget.for_tier(budget)
    notes: list[str] = []

    is_causal = hardware.is_causal_ar or resolved_budget.tier == "realtime"
    resolution_benefits_from_tiling = height is None or int(height) > 480
    topology = plan_parallel_topology(
        hardware.world_size,
        num_heads=hardware.num_heads,
        supports_cfg=hardware.supports_cfg and not is_causal,
        prefer_cfg_parallel=not is_causal,
        enable_vae_parallel=enable_vae_parallel and not is_causal and resolution_benefits_from_tiling,
    )

    sol_available = (
        "sol_attn" in hardware.available_attention_backends
        and hardware.head_dim == 128
        and hardware.sm_major >= 8
        and not is_causal
    )
    fp8_available = hardware.sm_major >= 9
    small_dit = hardware.num_heads <= 16

    # 1. Resolve per-pillar defaults from quality budget + hardware target
    if cache is None:
        if is_causal or resolved_budget.tier in {"lossless", "high_fidelity"}:
            cache = CachePillar(algorithm="none")
        elif resolved_budget.tier == "balanced":
            cache = CachePillar(algorithm="teacache", threshold=0.15)
        else:
            cache = CachePillar(algorithm="teacache", threshold=0.20)

    if sparse_attention is None:
        if sol_available and resolved_budget.tier in {"balanced", "aggressive"}:
            tau = 1.0 if resolved_budget.tier == "balanced" else 1.25
            sparse_attention = SparseAttentionPillar(backend="sol_attn", tau=tau)
            notes.append(f"enabled Sol-Attn (tau={tau}) for {resolved_budget.tier} budget on SM{hardware.sm_major}x")
        else:
            sparse_attention = SparseAttentionPillar(backend="torch")

    if token_pruning is None:
        if resolved_budget.tier == "aggressive" and not is_causal and topology.sequence_parallel_degree == 1:
            token_pruning = TokenPruningPillar(enabled=True, keep_ratio=0.75)
        else:
            token_pruning = TokenPruningPillar(enabled=False)

    if quantization is None:
        if resolved_budget.tier == "lossless" or not fp8_available:
            quantization = QuantizationPillar(mode="none")
        elif is_causal:
            # Small causal DiTs (<=16 heads, e.g. Self-Forcing 1.3B) are faster and
            # higher fidelity in compiled resident BF16 than dynamic rowwise FP8.
            quantization = (
                QuantizationPillar(mode="none")
                if small_dit
                else QuantizationPillar(mode="fp8", exclude_edge_blocks=True, fuse_fp8_ffn=True)
            )
        else:
            quantization = QuantizationPillar(
                mode="fp8",
                exclude_edge_blocks=small_dit,
                fuse_fp8_ffn=True,
            )

    if kernel_fusion is None:
        if is_causal:
            kernel_fusion = KernelFusionPillar(
                fuse_qkv=True,
                fused_rope=False,
                static_cross_kv=False,
                inplace_residual=False,
                rope_precision="fp64",
            )
        elif resolved_budget.tier == "lossless":
            kernel_fusion = KernelFusionPillar(
                fuse_qkv=True,
                fused_rope=False,
                static_cross_kv=True,
                inplace_residual=True,
                rope_precision="fp64",
            )
        else:
            kernel_fusion = KernelFusionPillar(
                fuse_qkv=True,
                fused_rope=True,
                static_cross_kv=True,
                inplace_residual=True,
                rope_precision="fp64",
            )

    if execution is None:
        execution = ExecutionPillar(
            cuda_graph="whole",
            compile=is_causal and small_dit,
            compile_scope="regional",
            overlapped_vae=is_causal,
        )

    # 2. Cross-pillar Integrator validation & conflict resolution
    if sparse_attention.backend == "sol_attn":
        if is_causal:
            raise ValueError("Sol-Attn is non-causal self-attention only and cannot be used on causal AR models")
        if hardware.head_dim != 128:
            raise ValueError(f"Sol-Attn requires head_dim=128, got {hardware.head_dim}")
        if hardware.sm_major < 8:
            raise ValueError(f"Sol-Attn requires SM80+, got SM{hardware.sm_major}0")
        if execution.compile:
            raise ValueError("Sol-Attn cannot be combined with torch.compile; use CUDA Graph instead")

    if quantization.mode == "fp8" and not fp8_available:
        raise ValueError(f"FP8 quantization requires SM90+ (Hopper/Blackwell), got SM{hardware.sm_major}0")

    if token_pruning.enabled and topology.sequence_parallel_degree > 1:
        raise ValueError("token_pruning cannot be combined with sequence_parallel > 1")

    resolved_inplace = kernel_fusion.inplace_residual
    resolved_graph: CudaGraphMode = execution.cuda_graph

    if cache.algorithm != "none":
        if resolved_inplace:
            resolved_inplace = False
            notes.append(f"disabled inplace_residual because cache={cache.algorithm} retains block inputs")
        if resolved_graph == "whole":
            resolved_graph = "piecewise"
            notes.append(f"upgraded cuda_graph from 'whole' to 'piecewise' to compose with cache={cache.algorithm}")

    if token_pruning.enabled and resolved_graph == "whole":
        resolved_graph = "piecewise"
        notes.append("upgraded cuda_graph from 'whole' to 'piecewise' for token-pruning head/mid/tail segments")

    if topology.sequence_parallel_degree > 1 and resolved_graph != "off":
        resolved_graph = "off"
        notes.append(
            f"disabled cuda_graph under sequence_parallel={topology.ulysses_degree} "
            "(NCCL all-to-all graph capture requires explicit opt-in)"
        )

    final_fusion = KernelFusionPillar(
        fuse_qkv=kernel_fusion.fuse_qkv,
        fused_rope=kernel_fusion.fused_rope,
        static_cross_kv=kernel_fusion.static_cross_kv,
        inplace_residual=resolved_inplace,
        vae_channels_last_3d=kernel_fusion.vae_channels_last_3d,
        vae_decode_autocast=kernel_fusion.vae_decode_autocast,
        vae_weight_dtype=kernel_fusion.vae_weight_dtype,
        rope_precision=kernel_fusion.rope_precision,
    )
    final_execution = ExecutionPillar(
        cuda_graph=resolved_graph,
        compile=execution.compile,
        compile_scope=execution.compile_scope,
        overlapped_vae=execution.overlapped_vae,
    )

    return SolEngineStack(
        hardware=hardware,
        budget=resolved_budget,
        topology=topology,
        cache=cache,
        sparse_attention=sparse_attention,
        token_pruning=token_pruning,
        quantization=quantization,
        kernel_fusion=final_fusion,
        execution=final_execution,
        composition_notes=tuple(notes),
    )


__all__ = [
    "AttentionPolicyBackend",
    "CacheAlgorithm",
    "CachePillar",
    "CudaGraphMode",
    "ExecutionPillar",
    "KernelFusionPillar",
    "ModelHardwareProfile",
    "QualityBudget",
    "QualityBudgetTier",
    "QuantizationPillar",
    "SolEngineStack",
    "SparseAttentionPillar",
    "TokenPruningPillar",
    "compose_sol_stack",
]
