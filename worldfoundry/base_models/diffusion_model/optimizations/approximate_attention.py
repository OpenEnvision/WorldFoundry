"""Opt-in, user-selected *approximate* (lossy) self-attention for Wan denoise.

Video self-attention over the full 3D latent (T·H·W tokens) is the dominant cost
at long sequence / high resolution. Published sparse kernels from FastVideo and
LightX2V trade quality for lower attention cost on supported accelerator shapes:

- **STA** (Sliding Tile Attention): each query tile attends only to a local 3D
  window of key tiles. Best for spatially-local video where distant tokens
  contribute little.
- **VSA** (Video Sparse Attention): block-level top-k — a coarse pass scores
  KV blocks and each query attends only to its top-k blocks (plus a learned,
  compressed global term). ``sparsity=0`` keeps every sparse-branch block, but
  it is still the VSA-QAT architecture and is *not* exact SDPA.
- **VMoBA** (Video Mixture-of-Block Attention): FastVideo's weight-free dynamic
  router alternates temporal, spatial, and spatiotemporal chunk layouts by Wan
  block index, then selects chunks by top-k or similarity-mass threshold.
- **DynamicSparse / Sparge / NBHD / LightX2V SLA-mask / FlexBlock / SPAS-Sage**:
  LightX2V's training-free sparse-mask families, with an explicit upstream CUDA
  operator choice. Its ``sla`` compatibility alias is only a QK-derived sparse
  mask; it is not FastVideo's separately trained Sparse-Linear Attention with
  a learnable ``proj_l`` branch.

**This is lossy and OFF by default.** It is never auto-selected; a user opts in
explicitly via ``RuntimePolicy.options["approximate_attention"]`` and picks the
kind + parameters. When engaged, the audit snapshot is marked
``quality_tier="approximate"`` so a manifest never hides that a lossy path ran.

Design (mirrors :mod:`static_cross_kv`): install a replacement
:class:`SelfAttentionProcessor` on every Wan ``SelfAttention``. The processor
computes q/k/v + RoPE exactly like the dense path, then routes the attention
through the sparse provider **only on the scheduled sparse steps**. VSA follows
FastVideo's full contract: it projects the checkpoint-trained compression gate,
tiles Q/K/V/gate in 3D, pads partial boundary tiles, passes the gate as
``compress_attn_weight``, and untile-restores raster order. The
``fastvideo_kernel`` package owns its internal TK/Triton/CuTe selection.
LightX2V owns its mask generators and CUDA operators. WorldFoundry has no
independent sparse implementation. If required metadata/weights are unavailable,
STA/VSA run exact attention and record the fallback. VMoBA and every LightX2V
lane are stricter: because their providers are the selected algorithms, a
requested path raises instead of silently running dense. Installation alone is
never reported as runtime-effective.
"""

from __future__ import annotations

import json
import math
import operator
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch
from torch import nn

from .sparse_linear_attention import (
    PINNED_FASTVIDEO_COMMIT,
    FastVideoSLAAdapter,
    FastVideoSLAConfig,
    FastVideoSLALayerSpec,
)
from .sparse_mask_attention import (
    PINNED_LIGHTX2V_COMMIT,
    LightX2VSparseAdapter,
    LightX2VSparseConfig,
    lightx2v_kernel_symbol,
    lightx2v_provider_family,
)

# STA ships a hardcoded set of latent grids it has tuned tile windows for. A
# token-count-only lookup is unsafe: different 3D grids can have the same S.
_STA_GRID_SHAPES = {
    (36, 48, 48): "36x48x48",
    (18, 48, 80): "18x48x80",
}
_SUPPORTED_SPARSE_HEAD_DIMS = frozenset((64, 128))
_SUPPORTED_VSA_BLOCK_VOLUMES = frozenset((64, 256))
_MAX_REQUEST_RECEIPTS = 32
_LIGHTX2V_SPARSE_KINDS = frozenset(
    {
        "dynamic_sparse",
        "sparge",
        "nbhd",
        "lightx2v_sla_mask",
        "flexblock",
        "lightx2v_spas_sage",
        "draft_attn",
        "radial_attn",
        "rainfusion_attn",
        "svg_attn",
        "svg2_attn",
        "lightx2v_svg_mask",
    }
)
_FASTVIDEO_SLA_KINDS = frozenset({"fastvideo_sla", "fastvideo_sagesla"})
_GRID_REQUIRED_STRICT_KINDS = frozenset({"vmoba", *_LIGHTX2V_SPARSE_KINDS})
_STRICT_SPARSE_KINDS = frozenset(
    {*_GRID_REQUIRED_STRICT_KINDS, *_FASTVIDEO_SLA_KINDS}
)


def _positive_int_triplet(name: str, values: Any) -> tuple[int, int, int]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of three positive integers, got {values!r}") from exc
    if len(raw_values) != 3:
        raise ValueError(f"{name} must contain three positive integers, got {raw_values!r}")
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain three positive integers, got {raw_values!r}")
        try:
            normalized.append(operator.index(value))
        except TypeError as exc:
            raise ValueError(f"{name} must contain three positive integers, got {raw_values!r}") from exc
    result = tuple(normalized)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain three positive integers, got {result!r}")
    return result


def _positive_int_pair(name: str, values: Any) -> tuple[int, int]:
    """Normalize one two-dimensional VMoBA chunk shape."""

    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of two positive integers, got {values!r}"
        ) from exc
    if len(raw_values) != 2:
        raise ValueError(
            f"{name} must contain two positive integers, got {raw_values!r}"
        )
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must contain two positive integers, got {raw_values!r}"
            )
        try:
            normalized.append(operator.index(value))
        except TypeError as exc:
            raise ValueError(
                f"{name} must contain two positive integers, got {raw_values!r}"
            ) from exc
    result = tuple(normalized)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain two positive integers, got {result!r}")
    return result  # type: ignore[return-value]


def _positive_int(name: str, value: Any) -> int:
    """Normalize a non-bool positive integer used by VMoBA routing."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer, got {normalized}")
    return normalized


def _nonnegative_int(name: str, value: Any) -> int:
    """Normalize a non-bool non-negative integer used by VMoBA scheduling."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {normalized}")
    return normalized


@dataclass
class ApproximateAttentionConfig:
    """User-facing config for the approximate self-attention lane."""

    kind: str = "vsa"
    # VSA: fraction of KV blocks omitted from the sparse branch. Zero is still
    # VSA (compressed global branch + learned gate), not exact dense SDPA.
    sparsity: float = 0.9
    window: tuple[int, int, int] = (3, 3, 3)  # STA: per-head tile window (t,h,w).
    dense_steps: int = 0  # first/last N denoise steps stay dense (boundary).
    # VSA tile shape (t,h,w); provider supports volumes 64 and 256. Internal
    # TK/Triton/CuTe selection remains the fastvideo_kernel package's contract.
    block_tile: tuple[int, int, int] = (4, 4, 4)

    # Pinned LightX2V training-free sparse lanes. ``sparsity`` above maps to
    # the upstream sparsity ratio; the remaining fields select the exact mask
    # and CUDA operator without mutating LightX2V's class-global defaults.
    lightx2v_operator: str | None = None
    nbhd_coefficient: tuple[float, ...] = (1.0, 0.5, 0.056)
    nbhd_min_width: float = 1.0
    attnmap_frame_num: int | None = None
    lightx2v_per_block_mean: bool = False
    lightx2v_pool_size: int = 128
    lightx2v_skip_timesteps: int = -1
    lightx2v_dense_attn_type: str = "flash_attn3"
    svg_sample_mse_max_row: int = 10000
    svg_num_sampled_rows: int = 64
    svg_context_length: int = 0

    # FastVideo's learned Sparse-Linear Attention is deliberately distinct
    # from LightX2V's SLA-mask alias. Every Wan layer requires checkpoint
    # trained proj_l weights; official dense checkpoints fail closed.
    fastvideo_topk_ratio: float | None = None
    fastvideo_feature_map: str = "softmax"

    # FastVideo VMoBA routing for Wan2.2 TI2V's real post-patchify token grid
    # (31, 22, 40). Every default chunk divides that grid exactly.
    temporal_chunk_size: int = 1
    temporal_topk: int = 3
    spatial_chunk_size: tuple[int, int] = (2, 5)
    spatial_topk: int = 20
    st_chunk_size: tuple[int, int, int] = (1, 2, 5)
    st_topk: int = 15
    moba_select_mode: str = "threshold"
    moba_threshold: float = 0.25
    moba_threshold_type: str = "query_head"
    first_full_step: int = 12
    first_full_layer: int = 0
    temporal_layer: int = 1
    spatial_layer: int = 1
    st_layer: int = 1

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().casefold().replace("-", "_")
        if self.kind not in {
            "sta",
            "vsa",
            "vmoba",
            *_LIGHTX2V_SPARSE_KINDS,
            *_FASTVIDEO_SLA_KINDS,
        }:
            try:
                self.kind = LightX2VSparseConfig(kind=self.kind).kind
            except ValueError as exc:
                allowed = sorted(
                    {
                        "sta",
                        "vsa",
                        "vmoba",
                        *_LIGHTX2V_SPARSE_KINDS,
                        *_FASTVIDEO_SLA_KINDS,
                    }
                )
                raise ValueError(
                    f"unsupported approximate attention kind {self.kind!r}; "
                    f"expected one of {allowed}"
                ) from exc
        if self.kind not in {
            "sta",
            "vsa",
            "vmoba",
            *_LIGHTX2V_SPARSE_KINDS,
            *_FASTVIDEO_SLA_KINDS,
        }:
            raise ValueError(
                f"unsupported approximate attention kind {self.kind!r}"
            )
        if not (0.0 <= self.sparsity < 1.0):
            raise ValueError(f"sparsity must be in [0.0, 1.0), got {self.sparsity}")
        if isinstance(self.dense_steps, bool):
            raise ValueError(f"dense_steps must be a non-negative integer, got {self.dense_steps!r}")
        try:
            self.dense_steps = operator.index(self.dense_steps)
        except TypeError as exc:
            raise ValueError(f"dense_steps must be a non-negative integer, got {self.dense_steps!r}") from exc
        if self.dense_steps < 0:
            raise ValueError(f"dense_steps must be >= 0, got {self.dense_steps}")
        self.window = _positive_int_triplet("window", self.window)
        self.block_tile = _positive_int_triplet("block_tile", self.block_tile)
        block_volume = math.prod(self.block_tile)
        if block_volume not in _SUPPORTED_VSA_BLOCK_VOLUMES:
            raise ValueError(
                f"unsupported VSA block_tile volume {block_volume} for {self.block_tile!r}; "
                f"supported volumes are {sorted(_SUPPORTED_VSA_BLOCK_VOLUMES)}"
            )
        self.temporal_chunk_size = _positive_int(
            "temporal_chunk_size", self.temporal_chunk_size
        )
        self.temporal_topk = _positive_int("temporal_topk", self.temporal_topk)
        self.spatial_chunk_size = _positive_int_pair(
            "spatial_chunk_size", self.spatial_chunk_size
        )
        self.spatial_topk = _positive_int("spatial_topk", self.spatial_topk)
        self.st_chunk_size = _positive_int_triplet(
            "st_chunk_size", self.st_chunk_size
        )
        self.st_topk = _positive_int("st_topk", self.st_topk)
        self.first_full_step = _nonnegative_int(
            "first_full_step", self.first_full_step
        )
        self.first_full_layer = _nonnegative_int(
            "first_full_layer", self.first_full_layer
        )
        self.temporal_layer = _positive_int("temporal_layer", self.temporal_layer)
        self.spatial_layer = _positive_int("spatial_layer", self.spatial_layer)
        self.st_layer = _positive_int("st_layer", self.st_layer)
        self.moba_select_mode = str(self.moba_select_mode).strip().lower()
        if self.moba_select_mode not in {"threshold", "topk"}:
            raise ValueError(
                "moba_select_mode must be 'threshold' or 'topk', got "
                f"{self.moba_select_mode!r}"
            )
        self.moba_threshold_type = str(self.moba_threshold_type).strip().lower()
        allowed_threshold_types = (
            {"query_head", "block", "overall", "head_global"}
            if self.moba_select_mode == "threshold"
            else {"query_head", "overall", "head_global"}
        )
        if self.moba_threshold_type not in allowed_threshold_types:
            raise ValueError(
                "unsupported moba_threshold_type for "
                f"{self.moba_select_mode}: {self.moba_threshold_type!r}; "
                f"expected one of {sorted(allowed_threshold_types)}"
            )
        self.moba_threshold = float(self.moba_threshold)
        if not math.isfinite(self.moba_threshold) or not (
            0.0 <= self.moba_threshold <= 1.0
        ):
            raise ValueError(
                "moba_threshold must be finite and in [0, 1], got "
                f"{self.moba_threshold!r}"
            )
        if self.kind in _LIGHTX2V_SPARSE_KINDS:
            lightx2v = LightX2VSparseConfig(
                kind=self.kind,
                sparsity_ratio=self.sparsity,
                operator=self.lightx2v_operator,
                nbhd_coefficient=self.nbhd_coefficient,
                nbhd_min_width=self.nbhd_min_width,
                attnmap_frame_num=self.attnmap_frame_num,
                per_block_mean=self.lightx2v_per_block_mean,
                pool_size=self.lightx2v_pool_size,
                skip_timesteps=self.lightx2v_skip_timesteps,
                dense_attn_type=self.lightx2v_dense_attn_type,
                svg_sample_mse_max_row=self.svg_sample_mse_max_row,
                svg_num_sampled_rows=self.svg_num_sampled_rows,
                svg_context_length=self.svg_context_length,
            )
            self.kind = lightx2v.kind
            self.lightx2v_operator = lightx2v.operator
            self.nbhd_coefficient = lightx2v.nbhd_coefficient
            self.nbhd_min_width = lightx2v.nbhd_min_width
            self.attnmap_frame_num = lightx2v.attnmap_frame_num
            self.lightx2v_per_block_mean = lightx2v.per_block_mean
            self.lightx2v_pool_size = lightx2v.pool_size
            self.lightx2v_skip_timesteps = lightx2v.skip_timesteps
            self.lightx2v_dense_attn_type = lightx2v.dense_attn_type
            self.svg_sample_mse_max_row = lightx2v.svg_sample_mse_max_row
            self.svg_num_sampled_rows = lightx2v.svg_num_sampled_rows
            self.svg_context_length = lightx2v.svg_context_length
        elif self.lightx2v_operator is not None or self.attnmap_frame_num is not None:
            raise ValueError(
                "lightx2v_operator/attnmap_frame_num require a LightX2V sparse kind"
            )
        if self.kind in _FASTVIDEO_SLA_KINDS:
            fastvideo = FastVideoSLAConfig(
                kind=self.kind,
                topk_ratio=self.fastvideo_topk_ratio,
                feature_map=self.fastvideo_feature_map,
            )
            self.fastvideo_topk_ratio = fastvideo.topk_ratio
            self.fastvideo_feature_map = fastvideo.feature_map
        elif self.fastvideo_topk_ratio is not None or self.fastvideo_feature_map != "softmax":
            raise ValueError(
                "fastvideo_topk_ratio/fastvideo_feature_map require "
                "fastvideo_sla or fastvideo_sagesla"
            )


def _lightx2v_config(config: ApproximateAttentionConfig) -> LightX2VSparseConfig:
    if config.kind not in _LIGHTX2V_SPARSE_KINDS:
        raise ValueError(f"{config.kind!r} is not a LightX2V sparse kind")
    return LightX2VSparseConfig(
        kind=config.kind,
        sparsity_ratio=config.sparsity,
        operator=config.lightx2v_operator,
        nbhd_coefficient=config.nbhd_coefficient,
        nbhd_min_width=config.nbhd_min_width,
        attnmap_frame_num=config.attnmap_frame_num,
        per_block_mean=config.lightx2v_per_block_mean,
        pool_size=config.lightx2v_pool_size,
        skip_timesteps=config.lightx2v_skip_timesteps,
        dense_attn_type=config.lightx2v_dense_attn_type,
        svg_sample_mse_max_row=config.svg_sample_mse_max_row,
        svg_num_sampled_rows=config.svg_num_sampled_rows,
        svg_context_length=config.svg_context_length,
    )


@dataclass(frozen=True)
class _VSAMetadata:
    """Read-only, per-grid routing metadata matching FastVideo's VSA builder."""

    tile_partition_indices: torch.Tensor
    untile_combined_index: torch.Tensor
    variable_block_sizes: torch.Tensor
    non_pad_index: torch.Tensor
    pad_index: torch.Tensor
    num_blocks: int
    block_elements: int
    padded_sequence: int


@dataclass(frozen=True)
class _ApproxInvocation:
    """Context-local owner for one denoiser branch invocation."""

    request_id: str
    request_epoch: int
    branch: str
    step: int
    total_steps: int
    request_state: Any = field(repr=False, compare=False)


@dataclass
class _ApproxRequestState:
    """JSON-safe telemetry owned by exactly one generation request."""

    request_id: str
    request_epoch: int
    total_steps: int
    routed_steps: bool = False
    grid_size: tuple[int, int, int] | None = None
    branch_steps: dict[str, int] = field(default_factory=dict)
    branch_step_history: dict[str, list[int]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    event_keys: set[tuple[str, int, int]] = field(default_factory=set, repr=False)
    sparse_calls: int = 0
    provider_dense_calls: int = 0
    kernel_attempts: int = 0
    dense_fallback_calls: int = 0
    scheduled_dense_calls: int = 0
    kernel_fallbacks: int = 0
    provider_paths: set[str] = field(default_factory=set)
    provider_dense_paths: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def record_note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)


@dataclass
class _ApproxState:
    """Installation plus isolated request-owned runtime telemetry.

    The scalar counters remain as a compatibility view for direct processor
    callers. Runner-backed inference is never certified from those scalars:
    every call is owned by ``request_id -> CFG branch -> step -> layer`` and
    finalized into a bounded immutable JSON receipt.
    """

    config: ApproximateAttentionConfig
    grid_size: tuple[int, int, int] | None = None
    total_steps: int | None = None
    step: int = 0
    wrapped_blocks: int = 0
    sparse_calls: int = 0
    provider_dense_calls: int = 0
    kernel_attempts: int = 0
    dense_fallback_calls: int = 0
    scheduled_dense_calls: int = 0
    kernel_fallbacks: int = 0
    effective_kernel: str = "exact (sparse provider not executed)"
    provider_path: str | None = None
    notes: list[str] = field(default_factory=list)
    install_notes: list[str] = field(default_factory=list, repr=False)
    install_reason: str | None = field(default=None, repr=False)
    default_grid_size: tuple[int, int, int] | None = field(default=None, repr=False)
    fastvideo_sla_adapter: FastVideoSLAAdapter | None = field(
        default=None,
        repr=False,
    )
    lightx2v_load_preparation: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )
    vsa_metadata_cache: dict[tuple[Any, ...], _VSAMetadata] = field(default_factory=dict, repr=False)
    vmoba_events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    vmoba_chunk_calls: dict[str, int] = field(
        default_factory=lambda: {"temporal": 0, "spatial": 0, "spatiotemporal": 0},
        repr=False,
    )
    layer_indices: tuple[int, ...] = field(default_factory=tuple, repr=False)
    _request_epoch_counter: int = field(default=0, repr=False)
    _requests: dict[str, _ApproxRequestState] = field(default_factory=dict, repr=False)
    _receipt_snapshots: dict[str, str] = field(default_factory=dict, repr=False)
    _current_invocation: ContextVar[_ApproxInvocation | None] = field(
        default_factory=lambda: ContextVar(
            f"worldfoundry_approximate_attention_request_{id(object())}",
            default=None,
        ),
        repr=False,
    )
    _last_finalized_request_id: ContextVar[str | None] = field(
        default_factory=lambda: ContextVar(
            f"worldfoundry_approximate_attention_finalized_{id(object())}",
            default=None,
        ),
        repr=False,
    )
    _provider_attempted: ContextVar[bool] = field(
        default_factory=lambda: ContextVar(
            f"worldfoundry_approximate_attention_attempt_{id(object())}",
            default=False,
        ),
        repr=False,
    )
    _lock: RLock = field(default_factory=RLock, repr=False)

    def is_dense_step(
        self,
        *,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> bool:
        """Boundary schedule: first/last ``dense_steps`` steps run dense."""
        invocation = self._current_invocation.get()
        active_step = (
            invocation.step
            if step is None and invocation is not None
            else self.step
            if step is None
            else step
        )
        active_total_steps = (
            invocation.total_steps
            if total_steps is None and invocation is not None
            else self.total_steps
            if total_steps is None
            else total_steps
        )
        if self.config.kind == "vmoba" and active_step < self.config.first_full_step:
            return True
        d = self.config.dense_steps
        if d <= 0:
            return False
        if active_step < d:
            return True
        if active_total_steps is not None and active_step >= active_total_steps - d:
            return True
        return False

    def current_invocation(self) -> _ApproxInvocation | None:
        return self._current_invocation.get()

    def begin_provider_window(self) -> None:
        self._provider_attempted.set(False)

    def mark_kernel_attempt(self) -> None:
        self.kernel_attempts += 1
        self._provider_attempted.set(True)

    def provider_attempted(self) -> bool:
        return self._provider_attempted.get()

    def request_state(
        self,
        request_id: str,
        *,
        create: bool,
        total_steps: int | None = None,
        routed_steps: bool | None = None,
    ) -> _ApproxRequestState | None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None and create:
                if total_steps is None or total_steps <= 0:
                    raise ValueError(
                        "approximate-attention requests require positive total_steps"
                    )
                self._request_epoch_counter += 1
                request = _ApproxRequestState(
                    request_id=request_id,
                    request_epoch=self._request_epoch_counter,
                    total_steps=total_steps,
                    routed_steps=bool(routed_steps),
                )
                self._requests[request_id] = request
            elif (
                request is not None
                and total_steps is not None
                and request.total_steps != total_steps
            ):
                raise ValueError(
                    "approximate-attention total_steps changed within request "
                    f"{request_id!r}: {request.total_steps} -> {total_steps}"
                )
            elif (
                request is not None
                and routed_steps is not None
                and request.routed_steps != routed_steps
            ):
                raise ValueError(
                    "approximate-attention routed-step ownership changed within "
                    f"request {request_id!r}"
                )
            return request

    def begin_invocation(
        self,
        *,
        request_id: str,
        branch: str,
        step: int,
        total_steps: int,
        routed_steps: bool = False,
    ) -> None:
        """Select one request branch and reject ambiguous step histories."""

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(
                "approximate attention requires an explicit non-empty request_id"
            )
        normalized_branch = str(branch).strip()
        if not normalized_branch:
            raise ValueError("approximate-attention branch cannot be empty")
        if step < 0 or step >= total_steps:
            raise ValueError(
                f"approximate-attention step must be in [0, {total_steps}), got {step}"
            )
        request = self.request_state(
            request_id,
            create=True,
            total_steps=total_steps,
            routed_steps=routed_steps,
        )
        assert request is not None
        with self._lock:
            previous_step = request.branch_steps.get(normalized_branch)
            if previous_step is None:
                if step != 0 and not request.routed_steps:
                    raise ValueError(
                        "an approximate-attention request branch must begin at step 0"
                    )
            elif (
                step <= previous_step
                or (not request.routed_steps and step != previous_step + 1)
            ):
                raise ValueError(
                    "approximate-attention branch steps must be unique and "
                    + ("increasing" if request.routed_steps else "contiguous")
                )
            request.branch_steps[normalized_branch] = step
            request.branch_step_history.setdefault(normalized_branch, []).append(step)
        self._current_invocation.set(
            _ApproxInvocation(
                request_id=request_id,
                request_epoch=request.request_epoch,
                branch=normalized_branch,
                step=step,
                total_steps=total_steps,
                request_state=request,
            )
        )

    def record_event(self, event: Mapping[str, Any]) -> None:
        """Append one layer event to its context-local request, or legacy view."""

        frozen = dict(event)
        invocation = self.current_invocation()
        if invocation is None:
            if (
                self.config.kind == "vmoba"
                and frozen.get("execution") == "sparse"
            ) or self.config.kind in _LIGHTX2V_SPARSE_KINDS:
                self.vmoba_events.append(frozen)
            return
        key = (invocation.branch, invocation.step, int(frozen["layer_idx"]))
        with self._lock:
            request = invocation.request_state
            if request is None or request.request_epoch != invocation.request_epoch:
                raise RuntimeError(
                    "approximate-attention invocation refers to a stale request epoch"
                )
            if key in request.event_keys:
                raise RuntimeError(
                    "duplicate approximate-attention event for "
                    f"request={invocation.request_id!r}, branch={invocation.branch!r}, "
                    f"step={invocation.step}, layer={key[2]}"
                )
            request.event_keys.add(key)
            request.events.append(frozen)
            execution = str(frozen["execution"])
            if bool(frozen.get("provider_attempted")):
                request.kernel_attempts += 1
            if execution == "sparse":
                request.sparse_calls += 1
                provider_path = frozen.get("provider_path")
                if isinstance(provider_path, str):
                    request.provider_paths.add(provider_path)
            elif execution == "provider_dense":
                request.provider_dense_calls += 1
                provider_path = frozen.get("provider_path")
                if isinstance(provider_path, str):
                    request.provider_dense_paths.add(provider_path)
            elif execution == "scheduled_dense":
                request.scheduled_dense_calls += 1
            elif execution == "fallback":
                request.kernel_fallbacks += 1
                reason = frozen.get("fallback_reason")
                if isinstance(reason, str):
                    request.record_note(reason)

    def record_grid(self, grid: tuple[int, int, int]) -> None:
        """Bind dynamic shape metadata to the active request exactly once."""

        self.grid_size = grid
        invocation = self.current_invocation()
        if invocation is None:
            return
        request = invocation.request_state
        if request.grid_size == grid:
            return
        with self._lock:
            if request is None or request.request_epoch != invocation.request_epoch:
                raise RuntimeError(
                    "approximate-attention grid refers to a stale request epoch"
                )
            if request.grid_size is not None and request.grid_size != grid:
                raise ValueError(
                    "approximate-attention grid changed within request "
                    f"{invocation.request_id!r}: {request.grid_size} -> {grid}"
                )
            request.grid_size = grid

    def current_grid(self) -> tuple[int, int, int] | None:
        invocation = self.current_invocation()
        if invocation is None:
            return self.grid_size
        return invocation.request_state.grid_size

    def mark_dense_fallback(self, *, layer_idx: int) -> None:
        """Mark that a fallback event actually executed exact attention."""

        invocation = self.current_invocation()
        if invocation is None:
            self.dense_fallback_calls += 1
            return
        key = (invocation.branch, invocation.step, layer_idx)
        with self._lock:
            request = invocation.request_state
            if request is None:
                raise RuntimeError("approximate-attention request ended during forward")
            matches = [
                event
                for event in request.events
                if (
                    str(event.get("branch")) == key[0]
                    and int(event.get("step", -1)) == key[1]
                    and int(event.get("layer_idx", -1)) == key[2]
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "approximate-attention fallback receipt could not resolve its event"
                )
            event = matches[0]
            if event.get("execution") == "fallback" and not bool(
                event.get("dense_fallback_executed")
            ):
                event["dense_fallback_executed"] = True
                request.dense_fallback_calls += 1

    def record_note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)
        invocation = self.current_invocation()
        if invocation is not None:
            request = invocation.request_state
            if request is not None:
                with self._lock:
                    request.record_note(note)

    def record_kernel_fallback(self, reason: str, *, note: str | None = None) -> None:
        self.kernel_fallbacks += 1
        self.record_note(note or reason)
        if self.sparse_calls:
            self.effective_kernel = f"exact/{self.config.kind} mixed (sparse provider fallback)"
        else:
            self.effective_kernel = f"exact ({reason})"

    def record_sparse_success(self, provider_path: str) -> None:
        self.sparse_calls += 1
        self.provider_path = provider_path
        if self.kernel_fallbacks:
            self.effective_kernel = f"exact/{self.config.kind} mixed (sparse provider fallback)"
        else:
            self.effective_kernel = self.config.kind

    def record_provider_dense_success(self, provider_path: str) -> None:
        """Record a dense phase intentionally implemented by the provider."""

        self.provider_dense_calls += 1
        self.provider_path = provider_path
        if self.sparse_calls:
            self.effective_kernel = self.config.kind
        else:
            self.effective_kernel = (
                f"{self.config.kind} (provider-designed dense phase only)"
            )

    def record_vmoba_success(self, receipt: Mapping[str, Any]) -> None:
        """Record one provider-complete VMoBA call in the current request window."""

        frozen_receipt = dict(receipt)
        chunk_kind = str(frozen_receipt["chunk_kind"])
        if chunk_kind not in self.vmoba_chunk_calls:
            raise RuntimeError(f"invalid VMoBA chunk kind in receipt: {chunk_kind!r}")
        self.vmoba_chunk_calls[chunk_kind] += 1
        self.record_sparse_success("fastvideo_kernel.moba_attn_varlen")


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    """Return the JSON-safe identity contract at one provider boundary."""

    return {
        "shape": list(value.shape),
        "device": str(value.device),
        "dtype": str(value.dtype),
    }


def _load_sparse_ops() -> Any:
    """Return the optional ``fastvideo_kernel`` provider, or ``None``.

    The package may internally choose a compiled TK/CuTe kernel or its bundled
    Triton path. This loader does not manufacture a second sparse fallback.
    """
    try:
        import fastvideo_kernel  # noqa: F401

        return fastvideo_kernel
    except (ImportError, OSError):
        return None


_VMOBA_PROVIDER_SYMBOLS = (
    "moba_attn_varlen",
    "process_moba_input",
    "process_moba_output",
)
_WAN_BLOCK_PATH = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


def _require_vmoba_ops(ops: Any | None = None) -> Any:
    """Return a complete VMoBA provider or fail before a dense run can masquerade."""

    provider = _load_sparse_ops() if ops is None else ops
    if provider is None:
        raise RuntimeError(
            "VMoBA was requested but fastvideo_kernel could not be imported"
        )
    missing = [
        name
        for name in _VMOBA_PROVIDER_SYMBOLS
        if not callable(getattr(provider, name, None))
    ]
    if missing:
        raise RuntimeError(
            "VMoBA was requested but fastvideo_kernel is missing callable symbols: "
            + ", ".join(missing)
        )
    return provider


def _wan_block_index(module_path: str) -> int | None:
    """Extract the Wan block index used by FastVideo's VMoBA layer cycle."""

    match = _WAN_BLOCK_PATH.search(module_path)
    return None if match is None else int(match.group(1))


def _vmoba_route(
    config: ApproximateAttentionConfig,
    layer_idx: int,
) -> tuple[str, int | tuple[int, int] | tuple[int, int, int], int]:
    """Select temporal → spatial → 3D routing exactly from the layer cycle."""

    if layer_idx < config.first_full_layer:
        raise ValueError(
            f"VMoBA layer {layer_idx} precedes first_full_layer={config.first_full_layer}"
        )
    relative_layer = layer_idx - config.first_full_layer
    period = config.temporal_layer + config.spatial_layer + config.st_layer
    cycle_position = relative_layer % period
    if cycle_position < config.temporal_layer:
        return "temporal", config.temporal_chunk_size, config.temporal_topk
    if cycle_position < config.temporal_layer + config.spatial_layer:
        return "spatial", config.spatial_chunk_size, config.spatial_topk
    return "spatiotemporal", config.st_chunk_size, config.st_topk


def _validate_vmoba_grid_profile(
    config: ApproximateAttentionConfig,
    grid: tuple[int, int, int],
) -> dict[str, Any]:
    """Validate every scheduled VMoBA layout before the first provider seam."""

    _, height, width = grid
    spatial_height, spatial_width = config.spatial_chunk_size
    if height % spatial_height or width % spatial_width:
        raise RuntimeError(
            f"VMoBA grid {grid} is incompatible with spatial_chunk_size="
            f"{config.spatial_chunk_size}; h and w must be divisible respectively"
        )
    if any(size % chunk for size, chunk in zip(grid, config.st_chunk_size)):
        raise RuntimeError(
            f"VMoBA grid {grid} is incompatible with st_chunk_size="
            f"{config.st_chunk_size}; t, h, and w must be divisible respectively"
        )
    return {
        "validated_grid": list(grid),
        "validated_profile": {
            "temporal_chunk_size": config.temporal_chunk_size,
            "spatial_chunk_size": list(config.spatial_chunk_size),
            "st_chunk_size": list(config.st_chunk_size),
        },
    }


def _require_vmoba_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate every provider boundary without coercing a malformed result."""

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"VMoBA {name} must be a Tensor, got {type(value).__name__}"
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"VMoBA {name} returned shape {tuple(value.shape)}, expected {shape}"
        )
    if value.device != device or value.dtype != dtype:
        raise RuntimeError(
            f"VMoBA {name} must preserve device/dtype; got "
            f"device={value.device}, dtype={value.dtype}, expected "
            f"device={device}, dtype={dtype}"
        )
    return value


_RECOVERABLE_SPARSE_KERNEL_MARKERS = (
    "not supported",
    "unsupported",
    "only supports",
    "must be divisible",
    "requires sm",
    "no kernel image is available",
    "invalid device function",
    "not compiled with",
)


def _recoverable_sparse_kernel_error(exc: Exception) -> bool:
    """Classify only capability/shape failures that may use exact attention."""

    if isinstance(exc, (ImportError, OSError, ValueError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    if any(marker in message for marker in ("out of memory", "alloc_failed", "illegal memory", "device-side assert")):
        return False
    return any(marker in message for marker in _RECOVERABLE_SPARSE_KERNEL_MARKERS)


def _sparse_runtime_ineligibility(q: torch.Tensor, head_dim: int) -> str | None:
    """Return a fail-closed capability reason before invoking CUDA-only ops."""

    if q.device.type != "cuda":
        return f"sparse provider requires CUDA, got {q.device.type}"
    if q.dtype is not torch.bfloat16:
        return f"sparse provider requires bfloat16 Q/K/V, got {q.dtype}"
    if head_dim not in _SUPPORTED_SPARSE_HEAD_DIMS:
        return f"sparse provider requires head_dim in {sorted(_SUPPORTED_SPARSE_HEAD_DIMS)}, got {head_dim}"
    return None


def _normalize_grid(grid: Any) -> tuple[int, int, int]:
    return _positive_int_triplet("sparse attention grid", grid)


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda" and device.index is None:
        return device.type, torch.cuda.current_device()
    return device.type, device.index


def _build_vsa_metadata(
    grid: tuple[int, int, int],
    tile: tuple[int, int, int],
    device: torch.device,
) -> _VSAMetadata:
    """Build FastVideo-compatible 3D tile/pad/untile routing metadata."""

    t, h, w = grid
    tile_t, tile_h, tile_w = tile
    block_elements = math.prod(tile)
    num_tiles = (
        math.ceil(t / tile_t),
        math.ceil(h / tile_h),
        math.ceil(w / tile_w),
    )

    raster = torch.arange(math.prod(grid), dtype=torch.long, device=device).reshape(grid)
    tile_parts: list[torch.Tensor] = []
    block_sizes: list[int] = []
    for tile_t_index in range(num_tiles[0]):
        t_slice = slice(tile_t_index * tile_t, min((tile_t_index + 1) * tile_t, t))
        for tile_h_index in range(num_tiles[1]):
            h_slice = slice(tile_h_index * tile_h, min((tile_h_index + 1) * tile_h, h))
            for tile_w_index in range(num_tiles[2]):
                w_slice = slice(tile_w_index * tile_w, min((tile_w_index + 1) * tile_w, w))
                part = raster[t_slice, h_slice, w_slice].flatten()
                tile_parts.append(part)
                block_sizes.append(part.numel())

    tile_partition_indices = torch.cat(tile_parts)
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)
    variable_block_sizes = torch.tensor(block_sizes, dtype=torch.int32, device=device)
    num_blocks = len(block_sizes)
    starts = torch.arange(num_blocks, dtype=torch.long, device=device) * block_elements
    offsets = torch.arange(block_elements, dtype=torch.long, device=device)
    block_positions = starts[:, None] + offsets[None, :]
    valid = offsets[None, :] < variable_block_sizes.to(torch.long)[:, None]
    non_pad_index = block_positions[valid]
    pad_index = block_positions[~valid]
    untile_combined_index = non_pad_index[reverse_tile_partition_indices]
    return _VSAMetadata(
        tile_partition_indices=tile_partition_indices,
        untile_combined_index=untile_combined_index,
        variable_block_sizes=variable_block_sizes,
        non_pad_index=non_pad_index,
        pad_index=pad_index,
        num_blocks=num_blocks,
        block_elements=block_elements,
        padded_sequence=num_blocks * block_elements,
    )


def _vsa_metadata(
    state: _ApproxState,
    grid: tuple[int, int, int],
    device: torch.device,
) -> _VSAMetadata:
    tile = tuple(state.config.block_tile)
    key = (grid, tile, *_device_cache_key(device))
    cached = state.vsa_metadata_cache.get(key)
    if cached is not None:
        return cached
    if len(state.vsa_metadata_cache) >= 8:
        state.vsa_metadata_cache.clear()
    metadata = _build_vsa_metadata(grid, tile, device)
    state.vsa_metadata_cache[key] = metadata
    return metadata


def _gate_projection(attention: nn.Module) -> Any | None:
    projection = getattr(attention, "gate_compress", None)
    return projection if callable(projection) else None


def _project_vsa_gate(attention: nn.Module, x: torch.Tensor) -> torch.Tensor:
    projection = _gate_projection(attention)
    if projection is None:
        raise RuntimeError("VSA-QAT gate_compress projection is unavailable")
    gate = projection(x)
    # FastVideo's tensor-parallel ReplicatedLinear returns (output, bias), while
    # native torch projections return the output tensor directly.
    if isinstance(gate, tuple):
        gate = gate[0]
    if not isinstance(gate, torch.Tensor):
        raise RuntimeError(f"gate_compress must return a Tensor, got {type(gate).__name__}")
    if gate.shape != x.shape:
        raise RuntimeError(f"gate_compress returned shape {tuple(gate.shape)}, expected {tuple(x.shape)}")
    if gate.device != x.device or gate.dtype != x.dtype:
        raise RuntimeError(
            "gate_compress output must match hidden-state device/dtype; "
            f"got device={gate.device}, dtype={gate.dtype}, expected device={x.device}, dtype={x.dtype}"
        )
    return gate


class ApproximateSelfAttentionProcessor:
    """Drop-in replacement for Wan ``SelfAttentionProcessor`` with a sparse lane.

    Delegates to the module's own exact attention (``attention.attn``) for dense
    boundary steps, unsupported contracts, or unavailable providers. STA consumes
    BHSD directly; VSA uses FastVideo's BSHD 3D tile/pad/gate/untile contract.
    """

    # QKV fusion may retain processor dispatch when this stable capability is
    # present. The processor consumes ``attention.qkv`` directly when fused.
    supports_fused_qkv = True

    def __init__(
        self,
        inner: Any,
        state: _ApproxState,
        *,
        layer_idx: int | None = None,
        module_path: str | None = None,
    ) -> None:
        self._inner = inner
        self._state = state
        self._layer_idx = layer_idx
        self._module_path = module_path
        self._lightx2v_adapter: LightX2VSparseAdapter | None = None
        self._event_identity = {
            "algorithm": state.config.kind,
            "layer_idx": self._receipt_layer_idx,
            "module_path": module_path,
        }

    def _get_lightx2v_adapter(self) -> LightX2VSparseAdapter:
        if self._state.config.kind not in _LIGHTX2V_SPARSE_KINDS:
            raise RuntimeError(
                "a LightX2V sparse adapter was requested for a non-LightX2V lane"
            )
        if self._lightx2v_adapter is None:
            self._lightx2v_adapter = LightX2VSparseAdapter(
                _lightx2v_config(self._state.config)
            )
        return self._lightx2v_adapter

    @property
    def _receipt_layer_idx(self) -> int:
        return -1 if self._layer_idx is None else self._layer_idx

    def _record_event(
        self,
        *,
        execution: str,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor | None = None,
        provider_path: str | None = None,
        provider_attempted: bool = False,
        fallback_reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        state = self._state
        invocation = state.current_invocation()
        current_grid = state.current_grid()
        input_contract = _tensor_contract(input_tensor)
        output_contract = (
            _tensor_contract(output_tensor) if output_tensor is not None else None
        )
        event: dict[str, Any] = {
            **self._event_identity,
            "request_id": invocation.request_id if invocation is not None else None,
            "request_epoch": invocation.request_epoch if invocation is not None else 0,
            "request_local": invocation is not None,
            "branch": invocation.branch if invocation is not None else "legacy",
            "step": invocation.step if invocation is not None else state.step,
            "step_index": invocation.step if invocation is not None else state.step,
            "total_steps": (
                invocation.total_steps
                if invocation is not None
                else state.total_steps
            ),
            "grid_size": (
                list(current_grid) if current_grid is not None else None
            ),
            "execution": execution,
            "provider_attempted": provider_attempted,
            "provider_path": provider_path,
            "input": input_contract,
            "output": output_contract,
            # Flat aliases make provider-boundary checks easy for manifests
            # while the nested objects remain the canonical receipt.
            "input_shape": input_contract["shape"],
            "input_device": input_contract["device"],
            "input_dtype": input_contract["dtype"],
            "output_shape": output_contract["shape"] if output_contract else None,
            "output_device": output_contract["device"] if output_contract else None,
            "output_dtype": output_contract["dtype"] if output_contract else None,
            "fallback_reason": fallback_reason,
            "dense_fallback_executed": False,
        }
        if extra is not None:
            event.update(dict(extra))
        state.record_event(event)

    def _record_fallback(
        self,
        q: torch.Tensor,
        reason: str,
        *,
        provider_path: str | None = None,
        provider_attempted: bool = False,
    ) -> None:
        self._record_event(
            execution="fallback",
            input_tensor=q,
            provider_path=provider_path,
            provider_attempted=provider_attempted,
            fallback_reason=reason,
        )

    def __call__(self, attention: nn.Module, x: torch.Tensor, freqs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
            apply_wan_qk_norm_rope,
        )

        # q/k/v + RoPE are identical to the dense path and compose with both
        # merged-QKV and fused QK-norm/RoPE runtime transforms.
        fused_qkv = getattr(attention, "qkv", None)
        if callable(fused_qkv):
            from .qkv_fusion import project_fused_qkv

            q, k, v = project_fused_qkv(fused_qkv, x)
        else:
            q = attention.q(x)
            k = attention.k(x)
            v = attention.v(x)
        q, k = apply_wan_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=kwargs.pop("_worldfoundry_rope_table", None),
            fused_grid=kwargs.pop("_worldfoundry_rope_grid", None),
            precision=kwargs.pop("_worldfoundry_rope_precision", "fp64"),
        )

        invocation = self._state.current_invocation()
        scheduled_dense_before = self._state.scheduled_dense_calls
        out = self._sparse_attention(attention, x, q, k, v, **kwargs)
        if out is None:
            # Exact fallback: reuse the module's own attention module so the
            # numerics match the dense path exactly. Planned first/last/full
            # steps already have a dedicated receipt and are not fallbacks.
            if (
                invocation is None
                and self._state.scheduled_dense_calls == scheduled_dense_before
            ):
                self._state.dense_fallback_calls += 1
            out = attention.attn(q, k, v)
            if invocation is not None:
                self._state.mark_dense_fallback(
                    layer_idx=self._receipt_layer_idx,
                )
        return attention.o(out)

    def _sparse_attention(
        self,
        attention: nn.Module,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        state = self._state
        state.begin_provider_window()
        explicit_grid = kwargs.pop("_worldfoundry_sparse_grid", None)
        if kwargs:
            if state.config.kind in _STRICT_SPARSE_KINDS:
                self._record_event(
                    execution="error",
                    input_tensor=q,
                    fallback_reason=(
                        f"unsupported {state.config.kind} self-attention kwargs: "
                        f"{sorted(kwargs)}"
                    ),
                )
                raise ValueError(
                    f"{state.config.kind} cannot compose with unsupported "
                    "self-attention kwargs: "
                    f"{sorted(kwargs)}"
                )
            state.record_kernel_fallback(
                "unsupported sparse-attention composition",
                note=f"unsupported sparse-attention kwargs: {sorted(kwargs)}",
            )
            self._record_fallback(
                q,
                f"unsupported sparse-attention kwargs: {sorted(kwargs)}",
            )
            return None
        normalized_grid: tuple[int, int, int] | None = None
        grid_profile_receipt: dict[str, Any] | None = None
        grid_error: ValueError | None = None
        grid_candidate = (
            explicit_grid
            if explicit_grid is not None
            else state.default_grid_size
        )
        if grid_candidate is not None:
            try:
                normalized_grid = _normalize_grid(grid_candidate)
            except ValueError as exc:
                if state.config.kind in _STRICT_SPARSE_KINDS:
                    self._record_event(
                        execution="error",
                        input_tensor=q,
                        fallback_reason=(
                            f"invalid {state.config.kind} patch grid: {exc}"
                        ),
                    )
                    raise RuntimeError(
                        f"invalid {state.config.kind} patch grid: {exc}"
                    ) from exc
                grid_error = exc
            else:
                if state.config.kind == "vmoba":
                    try:
                        grid_profile_receipt = _validate_vmoba_grid_profile(
                            state.config,
                            normalized_grid,
                        )
                    except RuntimeError as exc:
                        self._record_event(
                            execution="error",
                            input_tensor=q,
                            fallback_reason=str(exc),
                        )
                        raise
                state.record_grid(normalized_grid)
        elif state.config.kind in _GRID_REQUIRED_STRICT_KINDS:
            self._record_event(
                execution="error",
                input_tensor=q,
                fallback_reason=(
                    f"{state.config.kind} requires the current Wan 3D patch grid "
                    "on every dynamic-shape call"
                ),
            )
            raise RuntimeError(
                f"{state.config.kind} requires the current Wan 3D patch grid on "
                "every dynamic-shape call"
            )
        if state.is_dense_step():
            state.scheduled_dense_calls += 1
            self._record_event(
                execution="scheduled_dense",
                input_tensor=q,
                output_tensor=q,
                extra=grid_profile_receipt,
            )
            return None
        if state.config.kind in _FASTVIDEO_SLA_KINDS:
            try:
                return self._run_fastvideo_sla(attention, q, k, v)
            except Exception as exc:
                self._record_event(
                    execution="error",
                    input_tensor=q,
                    provider_attempted=state.provider_attempted(),
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
                raise
        if state.config.kind == "vmoba":
            try:
                return self._run_vmoba(attention, q, k, v, normalized_grid)
            except Exception as exc:
                self._record_event(
                    execution="error",
                    input_tensor=q,
                    provider_path="fastvideo_kernel.moba_attn_varlen",
                    provider_attempted=state.provider_attempted(),
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
                raise

        if grid_error is not None:
            state.record_kernel_fallback(
                "invalid 3D sparse grid",
                note=str(grid_error),
            )
            self._record_fallback(q, str(grid_error))
            return None
        grid = normalized_grid
        if grid is None:
            state.record_kernel_fallback(
                "3D sparse grid unavailable",
                note="sparse attention requires the current patch grid on every dynamic-shape call",
            )
            self._record_fallback(q, "3D sparse grid unavailable")
            return None
        b, sequence, dim = q.shape
        if math.prod(grid) != sequence:
            state.record_kernel_fallback(
                "3D sparse grid/token mismatch",
                note=f"sparse grid {grid} has {math.prod(grid)} tokens, but attention received {sequence}",
            )
            self._record_fallback(
                q,
                f"sparse grid {grid} has {math.prod(grid)} tokens, but attention received {sequence}",
            )
            return None
        num_heads = int(attention.num_heads)
        if dim % num_heads:
            raise RuntimeError(f"attention dim {dim} is not divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        capability_reason = _sparse_runtime_ineligibility(q, head_dim)
        if capability_reason is not None:
            if state.config.kind in _LIGHTX2V_SPARSE_KINDS:
                self._record_event(
                    execution="error",
                    input_tensor=q,
                    provider_path=lightx2v_kernel_symbol(
                        _lightx2v_config(state.config)
                    ),
                    fallback_reason=capability_reason,
                )
                raise RuntimeError(
                    f"{state.config.kind} cannot execute: {capability_reason}"
                )
            state.record_kernel_fallback(capability_reason)
            self._record_fallback(q, capability_reason)
            return None
        if state.config.kind in _LIGHTX2V_SPARSE_KINDS:
            try:
                return self._run_lightx2v(q, k, v, grid, num_heads, head_dim)
            except Exception as exc:
                provider_path = lightx2v_kernel_symbol(
                    _lightx2v_config(state.config)
                )
                self._record_event(
                    execution="error",
                    input_tensor=q,
                    provider_path=provider_path,
                    provider_attempted=state.provider_attempted(),
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
                raise
        if state.config.kind == "vsa" and _gate_projection(attention) is None:
            state.record_kernel_fallback(
                "VSA-QAT gate weights unavailable",
                note=(
                    "vsa requires checkpoint-trained gate_compress projections; "
                    "official dense Wan checkpoints do not contain them"
                ),
            )
            self._record_fallback(q, "VSA-QAT gate weights unavailable")
            return None

        ops = _load_sparse_ops()
        if ops is None:
            state.record_kernel_fallback(
                "no fastvideo_kernel",
                note="fastvideo_kernel unavailable; using exact attention",
            )
            self._record_fallback(q, "no fastvideo_kernel")
            return None

        try:
            if state.config.kind == "sta":
                qh = q.view(b, sequence, num_heads, head_dim).transpose(1, 2).contiguous()
                kh = k.view(b, sequence, num_heads, head_dim).transpose(1, 2).contiguous()
                vh = v.view(b, sequence, num_heads, head_dim).transpose(1, 2).contiguous()
                out, provider_receipt = self._run_sta(ops, qh, kh, vh, grid)
                expected_shape = qh.shape
                provider_path = "fastvideo_kernel.sliding_tile_attention"
                if not isinstance(out, torch.Tensor) or out.shape != expected_shape:
                    state.record_kernel_fallback(
                        "invalid STA provider output",
                        note=(
                            "sliding_tile_attention returned "
                            f"{type(out).__name__} shape={getattr(out, 'shape', None)}, "
                            f"expected Tensor shape={tuple(expected_shape)}"
                        ),
                    )
                    self._record_fallback(
                        provider_receipt["input_tensor"],
                        "invalid STA provider output",
                        provider_path=provider_path,
                        provider_attempted=True,
                    )
                    return None
                packed_out = out.transpose(1, 2).reshape(b, sequence, dim)
            else:
                gate = _project_vsa_gate(attention, x)
                q_bshd = q.view(b, sequence, num_heads, head_dim)
                k_bshd = k.view(b, sequence, num_heads, head_dim)
                v_bshd = v.view(b, sequence, num_heads, head_dim)
                gate_bshd = gate.view(b, sequence, num_heads, head_dim)
                packed_out, provider_path, provider_receipt = self._run_vsa(
                    ops,
                    q_bshd,
                    k_bshd,
                    v_bshd,
                    gate_bshd,
                    grid,
                )
        except Exception as exc:
            if not _recoverable_sparse_kernel_error(exc):
                raise
            state.record_kernel_fallback(
                f"{state.config.kind} provider error: {type(exc).__name__}",
                note=f"{state.config.kind} provider error: {exc}",
            )
            self._record_fallback(
                q,
                f"{state.config.kind} provider error: {type(exc).__name__}: {exc}",
                provider_path=(
                    "fastvideo_kernel.sliding_tile_attention"
                    if state.config.kind == "sta"
                    else None
                ),
                provider_attempted=state.provider_attempted(),
            )
            return None
        state.record_sparse_success(provider_path)
        provider_input = provider_receipt.pop("input_tensor")
        provider_output = provider_receipt.pop("output_tensor")
        self._record_event(
            execution="sparse",
            input_tensor=provider_input,
            output_tensor=provider_output,
            provider_path=provider_path,
            provider_attempted=True,
            extra=provider_receipt,
        )
        return packed_out

    def _run_lightx2v(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid: tuple[int, int, int],
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Execute one exact pinned-LightX2V provider contract, fail-closed."""

        state = self._state
        batch, sequence, dim = q.shape
        adapter = self._get_lightx2v_adapter()
        q_bshd = q.reshape(batch, sequence, num_heads, head_dim)
        k_bshd = k.reshape(batch, sequence, num_heads, head_dim)
        v_bshd = v.reshape(batch, sequence, num_heads, head_dim)
        invocation = state.current_invocation()
        result = adapter(
            q_bshd,
            k_bshd,
            v_bshd,
            grid=grid,
            softmax_scale=head_dim**-0.5,
            layer_idx=self._layer_idx,
            step_index=(
                invocation.step if invocation is not None else state.step
            ),
            request_key=(
                f"{invocation.request_id}:{invocation.request_epoch}:"
                f"{invocation.branch}"
                if invocation is not None
                else None
            ),
            on_provider_attempt=state.mark_kernel_attempt,
        )
        receipt = dict(result.receipt)
        provider_path = str(receipt["provider_path"])
        execution = str(receipt.get("execution", "sparse"))
        if execution == "sparse":
            state.record_sparse_success(provider_path)
        elif execution == "provider_dense":
            state.record_provider_dense_success(provider_path)
        else:
            raise RuntimeError(
                f"LightX2V adapter returned invalid execution {execution!r}"
            )
        event_extra = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "algorithm",
                "provider_path",
                "grid",
                "input",
                "output",
                "execution",
            }
        }
        self._record_event(
            execution=execution,
            input_tensor=q_bshd,
            output_tensor=result.output,
            provider_path=provider_path,
            provider_attempted=True,
            extra=event_extra,
        )
        return result.output.reshape(batch, sequence, dim)

    def _run_fastvideo_sla(
        self,
        attention: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute FastVideo's learned SLA/SageSLA provider, fail-closed."""

        state = self._state
        adapter = state.fastvideo_sla_adapter
        if adapter is None:
            raise RuntimeError(
                f"{state.config.kind} requested without a learned-SLA adapter"
            )
        if self._layer_idx is None:
            raise RuntimeError(
                f"{state.config.kind} requires a Wan blocks.<index> module path"
            )
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError(
                "FastVideo learned SLA requires matching packed [B,S,C] Q/K/V"
            )
        batch, sequence, dim = q.shape
        num_heads = int(attention.num_heads)
        if dim % num_heads:
            raise RuntimeError(
                f"attention dim {dim} is not divisible by num_heads={num_heads}"
            )
        head_dim = dim // num_heads
        capability_reason = _sparse_runtime_ineligibility(q, head_dim)
        if capability_reason is not None:
            raise RuntimeError(
                f"{state.config.kind} cannot execute: {capability_reason}"
            )
        q_bshd = q.reshape(batch, sequence, num_heads, head_dim).contiguous()
        k_bshd = k.reshape(batch, sequence, num_heads, head_dim).contiguous()
        v_bshd = v.reshape(batch, sequence, num_heads, head_dim).contiguous()
        invocation = state.current_invocation()
        current_timestep = invocation.step if invocation is not None else state.step
        result = adapter(
            self._layer_idx,
            q_bshd,
            k_bshd,
            v_bshd,
            current_timestep=current_timestep,
            on_provider_attempt=state.mark_kernel_attempt,
        )
        receipt = dict(result.receipt)
        provider_path = str(receipt["provider_path"])
        state.record_sparse_success(provider_path)
        event_extra = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "algorithm",
                "provider_path",
                "q",
                "k",
                "v",
                "output",
                "layer_index",
            }
        }
        self._record_event(
            execution="sparse",
            input_tensor=q_bshd,
            output_tensor=result.output,
            provider_path=provider_path,
            provider_attempted=True,
            extra=event_extra,
        )
        return result.output.reshape(batch, sequence, dim)

    def _run_vmoba(
        self,
        attention: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        explicit_grid: Any,
    ) -> torch.Tensor | None:
        """Run all three FastVideo VMoBA provider seams or raise fail-closed."""

        state = self._state
        layer_idx = self._layer_idx
        if layer_idx is None:
            raise RuntimeError(
                "VMoBA requires a Wan blocks.<index>.self_attn module path; got "
                f"{self._module_path!r}"
            )
        if layer_idx < state.config.first_full_layer:
            state.scheduled_dense_calls += 1
            self._record_event(
                execution="scheduled_dense",
                input_tensor=q,
                output_tensor=q,
                extra={"dense_reason": "first_full_layer"},
            )
            return None
        if explicit_grid is None:
            grid = state.default_grid_size
            if grid is None:
                raise RuntimeError(
                    "VMoBA requires the current Wan 3D patch grid on every dynamic-shape call"
                )
        else:
            try:
                grid = _normalize_grid(explicit_grid)
            except ValueError as exc:
                raise RuntimeError(f"invalid VMoBA patch grid: {exc}") from exc
        state.record_grid(grid)
        grid_profile_receipt = _validate_vmoba_grid_profile(state.config, grid)

        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise RuntimeError(
                "VMoBA Q/K/V must use packed [B,S,C] layout; got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError(
                "VMoBA Q/K/V shapes must match exactly; got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if q.device != k.device or q.device != v.device:
            raise RuntimeError(
                "VMoBA Q/K/V devices must match exactly; got "
                f"q={q.device}, k={k.device}, v={v.device}"
            )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise RuntimeError(
                "VMoBA Q/K/V dtypes must match exactly; got "
                f"q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )

        batch, sequence, dim = q.shape
        if math.prod(grid) != sequence:
            raise RuntimeError(
                f"VMoBA grid {grid} has {math.prod(grid)} tokens, but Q/K/V have {sequence}"
            )
        attention_heads = getattr(attention, "num_heads", None)
        if not isinstance(attention_heads, int) or attention_heads <= 0:
            raise RuntimeError("VMoBA requires a positive integer Wan num_heads")
        if dim % attention_heads:
            raise RuntimeError(
                f"VMoBA projection width {dim} is not divisible by num_heads={attention_heads}"
            )
        head_dim = dim // attention_heads
        capability_reason = _sparse_runtime_ineligibility(q, head_dim)
        if capability_reason is not None:
            raise RuntimeError(f"VMoBA requested but {capability_reason}")

        chunk_kind, chunk_layout, requested_topk = _vmoba_route(
            state.config,
            layer_idx,
        )
        if chunk_kind == "spatial":
            chunk_h, chunk_w = chunk_layout  # type: ignore[misc]
            if grid[1] % chunk_h or grid[2] % chunk_w:
                raise RuntimeError(
                    f"VMoBA spatial grid {grid[1:]} must be divisible by chunk "
                    f"{chunk_layout} at layer {layer_idx}"
                )
        elif chunk_kind == "spatiotemporal":
            chunk_t, chunk_h, chunk_w = chunk_layout  # type: ignore[misc]
            if grid[0] % chunk_t or grid[1] % chunk_h or grid[2] % chunk_w:
                raise RuntimeError(
                    f"VMoBA grid {grid} must be divisible by 3D chunk "
                    f"{chunk_layout} at layer {layer_idx}"
                )

        ops = _require_vmoba_ops()
        q_bshd = q.reshape(batch, sequence, attention_heads, head_dim).contiguous()
        k_bshd = k.reshape(batch, sequence, attention_heads, head_dim).contiguous()
        v_bshd = v.reshape(batch, sequence, attention_heads, head_dim).contiguous()
        expected_bshd = tuple(q_bshd.shape)

        state.mark_kernel_attempt()
        processed: list[torch.Tensor] = []
        provider_chunk_sizes: list[int] = []
        for name, tensor in (("processed Q", q_bshd), ("processed K", k_bshd), ("processed V", v_bshd)):
            result = ops.process_moba_input(tensor, grid, chunk_layout)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "fastvideo_kernel.process_moba_input must return (Tensor, chunk_size)"
                )
            transformed = _require_vmoba_tensor(
                result[0],
                name=name,
                shape=expected_bshd,
                device=q.device,
                dtype=q.dtype,
            )
            provider_chunk_sizes.append(_positive_int("provider moba_chunk_size", result[1]))
            processed.append(transformed)
        if len(set(provider_chunk_sizes)) != 1:
            raise RuntimeError(
                "VMoBA provider returned inconsistent Q/K/V chunk sizes: "
                f"{provider_chunk_sizes}"
            )
        provider_chunk_size = provider_chunk_sizes[0]
        chunks_per_sample = math.ceil(sequence / provider_chunk_size)
        effective_topk = min(requested_topk, chunks_per_sample)

        q_flat, k_flat, v_flat = (
            tensor.reshape(batch * sequence, attention_heads, head_dim)
            for tensor in processed
        )
        cu_seqlens = torch.arange(
            0,
            batch * sequence + 1,
            sequence,
            dtype=torch.int32,
            device=q.device,
        )
        output_flat = ops.moba_attn_varlen(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens=cu_seqlens,
            max_seqlen=sequence,
            moba_chunk_size=provider_chunk_size,
            moba_topk=effective_topk,
            select_mode=state.config.moba_select_mode,
            simsum_threshold=state.config.moba_threshold,
            threshold_type=state.config.moba_threshold_type,
        )
        expected_flat = (batch * sequence, attention_heads, head_dim)
        output_flat = _require_vmoba_tensor(
            output_flat,
            name="moba_attn_varlen output",
            shape=expected_flat,
            device=q.device,
            dtype=q.dtype,
        )
        restored = ops.process_moba_output(
            output_flat.reshape(expected_bshd),
            grid,
            chunk_layout,
        )
        restored = _require_vmoba_tensor(
            restored,
            name="process_moba_output",
            shape=expected_bshd,
            device=q.device,
            dtype=q.dtype,
        )
        packed = restored.reshape(batch, sequence, dim)
        vmoba_receipt = {
                "algorithm": "vmoba",
                "grid_size": list(grid),
                "chunk_kind": chunk_kind,
                "chunk_layout": (
                    list(chunk_layout)
                    if isinstance(chunk_layout, tuple)
                    else chunk_layout
                ),
                "provider_chunk_size": provider_chunk_size,
                "requested_topk": requested_topk,
                "effective_topk": effective_topk,
                "select_mode": state.config.moba_select_mode,
                "threshold": state.config.moba_threshold,
                "threshold_type": state.config.moba_threshold_type,
                "provider_path": "fastvideo_kernel.moba_attn_varlen",
                "provider_symbols": list(_VMOBA_PROVIDER_SYMBOLS),
                "packed_input": _tensor_contract(q_bshd),
                "restored_output": _tensor_contract(restored),
                **grid_profile_receipt,
            }
        state.record_vmoba_success(vmoba_receipt)
        self._record_event(
            execution="sparse",
            input_tensor=q_flat,
            output_tensor=output_flat,
            provider_path="fastvideo_kernel.moba_attn_varlen",
            provider_attempted=True,
            extra=vmoba_receipt,
        )
        return packed

    def _run_sta(
        self,
        ops: Any,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        seq_shape = _STA_GRID_SHAPES.get(grid)
        if seq_shape is None:
            raise ValueError(f"STA has no tuned provider plan for 3D grid {grid}")
        num_heads = qh.shape[1]
        window = [tuple(self._state.config.window)] * num_heads
        self._state.mark_kernel_attempt()
        output = ops.sliding_tile_attention(
            qh,
            kh,
            vh,
            window,
            text_length=0,
            has_text=False,
            seq_shape=seq_shape,
        )
        return output, {
            "input_tensor": qh,
            "output_tensor": output if isinstance(output, torch.Tensor) else qh,
            "grid_size": list(grid),
            "window": [list(value) for value in window],
            "provider_layout": "BHSD",
        }

    def _run_vsa(
        self,
        ops: Any,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
        gate_bshd: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> tuple[torch.Tensor, str, dict[str, Any]]:
        state = self._state
        tile = tuple(state.config.block_tile)
        metadata = _vsa_metadata(state, grid, q_bshd.device)
        topk = max(
            1,
            min(
                metadata.num_blocks,
                math.ceil((1.0 - state.config.sparsity) * metadata.num_blocks),
            ),
        )

        # FastVideo tiles Q/K/V/gate together in BSHD. Each partial boundary
        # tile occupies a full fixed-size block; invalid slots must be zero.
        qkvg = torch.cat((q_bshd, k_bshd, v_bshd, gate_bshd), dim=0)
        padded = torch.empty(
            (qkvg.shape[0], metadata.padded_sequence, qkvg.shape[2], qkvg.shape[3]),
            dtype=qkvg.dtype,
            device=qkvg.device,
        )
        padded[:, metadata.non_pad_index] = qkvg[:, metadata.tile_partition_indices]
        if metadata.pad_index.numel():
            padded[:, metadata.pad_index] = 0
        q_tiled, k_tiled, v_tiled, gate_tiled = padded.chunk(4, dim=0)

        bshd_provider = getattr(ops, "video_sparse_attn_bshd", None)
        if metadata.block_elements == 256 and callable(bshd_provider):
            state.mark_kernel_attempt()
            provider_q = q_tiled
            provider_k = k_tiled
            provider_v = v_tiled
            provider_gate = gate_tiled
            output = bshd_provider(
                provider_q,
                provider_k,
                provider_v,
                variable_block_sizes=metadata.variable_block_sizes,
                q_variable_block_sizes=metadata.variable_block_sizes,
                topk=topk,
                block_size=tile,
                compress_attn_weight=provider_gate,
            )
            provider_path = "fastvideo_kernel.video_sparse_attn_bshd"
            provider_layout = "BSHD"
        else:
            provider = getattr(ops, "video_sparse_attn", None)
            if not callable(provider):
                raise RuntimeError("fastvideo_kernel.video_sparse_attn is unavailable")
            state.mark_kernel_attempt()
            provider_q = q_tiled.transpose(1, 2).contiguous()
            provider_k = k_tiled.transpose(1, 2).contiguous()
            provider_v = v_tiled.transpose(1, 2).contiguous()
            provider_gate = gate_tiled.transpose(1, 2).contiguous()
            output_bhsd = provider(
                provider_q,
                provider_k,
                provider_v,
                variable_block_sizes=metadata.variable_block_sizes,
                q_variable_block_sizes=metadata.variable_block_sizes,
                topk=topk,
                block_size=tile,
                compress_attn_weight=provider_gate,
            )
            if not isinstance(output_bhsd, torch.Tensor):
                raise ValueError(f"video_sparse_attn returned {type(output_bhsd).__name__}, expected Tensor")
            output = output_bhsd.transpose(1, 2)
            provider_path = "fastvideo_kernel.video_sparse_attn"
            provider_layout = "BHSD"

        if not isinstance(output, torch.Tensor) or output.shape != q_tiled.shape:
            raise ValueError(
                "VSA provider returned invalid output: "
                f"{type(output).__name__} shape={getattr(output, 'shape', None)}, "
                f"expected Tensor shape={tuple(q_tiled.shape)}"
            )
        raster_output = output[:, metadata.untile_combined_index]
        batch, sequence, heads, head_dim = raster_output.shape
        packed = raster_output.reshape(batch, sequence, heads * head_dim)
        provider_output = output if provider_layout == "BSHD" else output_bhsd
        return packed, provider_path, {
            "input_tensor": provider_q,
            "output_tensor": provider_output,
            "grid_size": list(grid),
            "block_tile": list(tile),
            "provider_layout": provider_layout,
            "topk": topk,
            "num_blocks": metadata.num_blocks,
            "padded_sequence": metadata.padded_sequence,
        }


def parse_approximate_attention(option: Any) -> ApproximateAttentionConfig:
    """Normalize a user ``options["approximate_attention"]`` value to a config.

    Accepts (boundary parser, mirrors ``parse_offload_policy``):
    - an :class:`ApproximateAttentionConfig` (passed through);
    - a string kind (FastVideo ``sta``/``vsa``/``vmoba`` or a LightX2V kind);
    - a mapping: ``{"kind": "vsa", "sparsity": 0.9, "window": [3,3,3],
      "dense_steps": 2, "block_tile": [4,4,4]}``.
    """
    if isinstance(option, ApproximateAttentionConfig):
        return option
    if isinstance(option, str):
        return ApproximateAttentionConfig(kind=option.lower())
    if isinstance(option, Mapping):
        known = {
            "kind",
            "sparsity",
            "window",
            "dense_steps",
            "block_tile",
            "lightx2v_operator",
            "nbhd_coefficient",
            "nbhd_min_width",
            "attnmap_frame_num",
            "lightx2v_per_block_mean",
            "lightx2v_pool_size",
            "lightx2v_skip_timesteps",
            "lightx2v_dense_attn_type",
            "svg_sample_mse_max_row",
            "svg_num_sampled_rows",
            "svg_context_length",
            "fastvideo_topk_ratio",
            "fastvideo_feature_map",
            "temporal_chunk_size",
            "temporal_topk",
            "spatial_chunk_size",
            "spatial_topk",
            "st_chunk_size",
            "st_topk",
            "moba_select_mode",
            "moba_threshold",
            "moba_threshold_type",
            "first_full_step",
            "first_full_layer",
            "temporal_layer",
            "spatial_layer",
            "st_layer",
        }
        unknown = sorted(str(key) for key in set(option) - known)
        if unknown:
            hint = "; use block_tile=[t,h,w], not block_size" if "block_size" in unknown else ""
            raise ValueError(f"unknown approximate_attention options: {unknown}{hint}")
        fields: dict[str, Any] = {}
        if "kind" in option:
            fields["kind"] = str(option["kind"]).lower()
        if "sparsity" in option:
            fields["sparsity"] = float(option["sparsity"])
        if "window" in option:
            fields["window"] = option["window"]
        if "dense_steps" in option:
            fields["dense_steps"] = option["dense_steps"]
        if "block_tile" in option:
            fields["block_tile"] = option["block_tile"]
        for name in (
            "lightx2v_operator",
            "nbhd_coefficient",
            "nbhd_min_width",
            "attnmap_frame_num",
            "lightx2v_per_block_mean",
            "lightx2v_pool_size",
            "lightx2v_skip_timesteps",
            "lightx2v_dense_attn_type",
            "svg_sample_mse_max_row",
            "svg_num_sampled_rows",
            "svg_context_length",
            "fastvideo_topk_ratio",
            "fastvideo_feature_map",
        ):
            if name in option:
                fields[name] = option[name]
        for name in (
            "temporal_chunk_size",
            "temporal_topk",
            "spatial_chunk_size",
            "spatial_topk",
            "st_chunk_size",
            "st_topk",
            "moba_select_mode",
            "moba_threshold",
            "moba_threshold_type",
            "first_full_step",
            "first_full_layer",
            "temporal_layer",
            "spatial_layer",
            "st_layer",
        ):
            if name in option:
                fields[name] = option[name]
        return ApproximateAttentionConfig(**fields)
    raise TypeError(
        f"approximate_attention option must be a str, mapping, or ApproximateAttentionConfig, got {type(option).__name__}"
    )


def install_approximate_attention(
    model: nn.Module,
    config: ApproximateAttentionConfig,
    *,
    grid_size: tuple[int, int, int] | None = None,
    total_steps: int | None = None,
    checkpoint_state_dict: Mapping[str, Any] | None = None,
) -> _ApproxState:
    """Wrap every Wan ``SelfAttention`` with the approximate processor.

    Returns a shared state handle. Idempotent: re-installing unwraps a prior
    approximate processor first. The default numerical path is unchanged for any
    module that is not matched.
    """
    normalized_grid = _normalize_grid(grid_size) if grid_size is not None else None
    state = _ApproxState(
        config=config,
        grid_size=normalized_grid,
        default_grid_size=normalized_grid,
        total_steps=total_steps,
    )
    matching_attention = [
        (module_path, module)
        for module_path, module in model.named_modules()
        if type(module).__name__ == "SelfAttention"
        and hasattr(module, "get_processor")
        and hasattr(module, "set_processor")
    ]
    if config.kind == "vmoba":
        _require_vmoba_ops()
        if not matching_attention:
            raise RuntimeError("VMoBA requested but no Wan SelfAttention blocks matched")
        missing_paths = [
            module_path
            for module_path, _ in matching_attention
            if _wan_block_index(module_path) is None
        ]
        if missing_paths:
            raise RuntimeError(
                "VMoBA requires every attention module to live under blocks.<index>; "
                f"unmatched paths={missing_paths}"
            )
        layer_indices = [
            _wan_block_index(module_path)
            for module_path, _ in matching_attention
        ]
        if len(set(layer_indices)) != len(layer_indices):
            raise RuntimeError(
                f"VMoBA found duplicate Wan block indices: {layer_indices}"
            )
    elif config.kind in {
        *_LIGHTX2V_SPARSE_KINDS,
        *_FASTVIDEO_SLA_KINDS,
    } and not matching_attention:
        raise RuntimeError(
            f"{config.kind} requested but no Wan SelfAttention blocks matched"
        )
    if config.kind in _FASTVIDEO_SLA_KINDS:
        indexed_attention: dict[int, tuple[str, nn.Module]] = {}
        for ordinal, (module_path, module) in enumerate(matching_attention):
            layer_idx = _wan_block_index(module_path)
            if layer_idx is None:
                layer_idx = ordinal
            if layer_idx in indexed_attention:
                raise RuntimeError(
                    "FastVideo learned SLA found duplicate Wan block index "
                    f"{layer_idx}"
                )
            indexed_attention[layer_idx] = (module_path, module)
        expected_layers = list(range(len(indexed_attention)))
        if sorted(indexed_attention) != expected_layers:
            raise RuntimeError(
                "FastVideo learned SLA requires every zero-based Wan block; "
                f"expected={expected_layers}, actual={sorted(indexed_attention)}"
            )
        if checkpoint_state_dict is None:
            raise RuntimeError(
                f"{config.kind} requires learned proj_l checkpoint tensors"
            )
        layer_specs = []
        for layer_idx in expected_layers:
            module_path, module = indexed_attention[layer_idx]
            num_heads = int(getattr(module, "num_heads", 0))
            head_size = int(getattr(module, "head_dim", 0))
            layer_specs.append(
                FastVideoSLALayerSpec(
                    layer_index=layer_idx,
                    num_heads=num_heads,
                    head_size=head_size,
                    softmax_scale=head_size**-0.5,
                    prefix=module_path.replace(".self_attn", ".attn1"),
                )
            )
        state.fastvideo_sla_adapter = FastVideoSLAAdapter(
            FastVideoSLAConfig(
                kind=config.kind,
                topk_ratio=config.fastvideo_topk_ratio,
                feature_map=config.fastvideo_feature_map,
            ),
            layer_specs=layer_specs,
            checkpoint_state_dict=checkpoint_state_dict,
        )
        setattr(
            model,
            "_worldfoundry_fastvideo_sla_adapter",
            state.fastvideo_sla_adapter,
        )
    wrapped = 0
    installed_layer_indices: list[int] = []
    for ordinal, (module_path, module) in enumerate(matching_attention):
        inner = module.get_processor()
        if isinstance(inner, ApproximateSelfAttentionProcessor):
            inner = inner._inner
        layer_idx = _wan_block_index(module_path)
        if layer_idx is None:
            layer_idx = ordinal
        module.set_processor(
            ApproximateSelfAttentionProcessor(
                inner,
                state,
                layer_idx=layer_idx,
                module_path=module_path,
            )
        )
        if bool(getattr(module, "_qkv_fused", False)):
            from .qkv_fusion import restore_fused_qkv_processor_dispatch

            restore_fused_qkv_processor_dispatch(module)
        installed_layer_indices.append(layer_idx)
        wrapped += 1
    state.wrapped_blocks = wrapped
    state.layer_indices = tuple(installed_layer_indices)
    install_reasons: list[str] = []
    if wrapped == 0:
        install_reasons.append("no Wan SelfAttention blocks matched")
    if config.kind in {"sta", "vsa"} and _load_sparse_ops() is None:
        install_reasons.append("no fastvideo_kernel")
        state.notes.append("fastvideo_kernel unavailable; exact attention remains active")
    if (
        config.kind == "vsa"
        and matching_attention
        and not any(
            _gate_projection(module) is not None
            for _, module in matching_attention
        )
    ):
        install_reasons.append("VSA-QAT gate weights unavailable")
        state.notes.append(
            "vsa requires checkpoint-trained gate_compress projections; "
            "official dense Wan checkpoints do not contain them"
        )
    state.install_reason = "; ".join(install_reasons) or None
    if state.install_reason is not None:
        state.effective_kernel = f"exact ({state.install_reason})"
    # Even with all prerequisites present, a wrapper has not executed a sparse
    # provider yet. Never turn availability into runtime-effectiveness.
    state.install_notes = list(state.notes)
    place_fastvideo_sla_providers(model, state)
    return state


def place_fastvideo_sla_providers(model: nn.Module, state: _ApproxState) -> None:
    """Place each learned provider with its owning Wan block, preserving FP32."""

    adapter = state.fastvideo_sla_adapter
    if adapter is None:
        return
    placed_layers: set[int] = set()
    for module_path, module in model.named_modules():
        if type(module).__name__ != "SelfAttention":
            continue
        processor = getattr(module, "processor", None)
        if not isinstance(processor, ApproximateSelfAttentionProcessor):
            continue
        layer_idx = processor._layer_idx
        if layer_idx is None or str(layer_idx) not in adapter.providers:
            continue
        q_projection = getattr(module, "q", None)
        weight = getattr(q_projection, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type == "meta":
            raise RuntimeError(
                f"cannot place {state.config.kind} provider for {module_path}: "
                "Wan Q projection has no materialized device"
            )
        adapter.providers[str(layer_idx)].to(device=weight.device)
        placed_layers.add(layer_idx)
    expected_layers = set(state.layer_indices)
    if placed_layers != expected_layers:
        raise RuntimeError(
            f"{state.config.kind} provider placement incomplete: "
            f"expected={sorted(expected_layers)}, placed={sorted(placed_layers)}"
        )


def prepare_lightx2v_providers(
    model: nn.Module,
    state: _ApproxState,
    *,
    fallback_device: torch.device | str,
) -> None:
    """Move LightX2V provider imports/preflight out of the first denoise step.

    Public LightX2V constructs its attention providers as part of model loading.
    WorldFoundry uses the same timing boundary here. Installation remains only
    an availability receipt, while actual ``apply`` calls remain the sole
    runtime-effectiveness proof.
    """

    if state.config.kind not in _LIGHTX2V_SPARSE_KINDS:
        return
    default_device = torch.device(fallback_device)
    if default_device.type != "cuda":
        state.lightx2v_load_preparation = {
            "attempted": False,
            "preflight_blocks": 0,
            "materialized_blocks": 0,
            "deferred_blocks": state.wrapped_blocks,
            "reason": "CUDA provider preparation deferred on a non-CUDA runtime",
        }
        return

    receipts: list[dict[str, Any]] = []
    prepared_layers: set[int] = set()
    for module_path, module in model.named_modules():
        if type(module).__name__ != "SelfAttention":
            continue
        processor = getattr(module, "processor", None)
        if not isinstance(processor, ApproximateSelfAttentionProcessor):
            continue
        layer_idx = processor._layer_idx
        if layer_idx is None:
            raise RuntimeError(
                f"LightX2V provider preparation needs a layer index for {module_path}"
            )
        projection = getattr(module, "qkv", None)
        if projection is None:
            projection = getattr(module, "q", None)
        weight = getattr(projection, "weight", None)
        layer_device = (
            weight.device
            if isinstance(weight, torch.Tensor)
            and weight.device.type == "cuda"
            else default_device
        )
        receipt = processor._get_lightx2v_adapter().prepare(
            device=layer_device,
            heads=int(getattr(module, "num_heads", 0)),
            head_dim=int(getattr(module, "head_dim", 0)),
            grid=state.default_grid_size,
        )
        receipts.append(receipt)
        prepared_layers.add(layer_idx)

    expected_layers = set(state.layer_indices)
    if prepared_layers != expected_layers:
        raise RuntimeError(
            f"{state.config.kind} load preparation incomplete: "
            f"expected={sorted(expected_layers)}, prepared={sorted(prepared_layers)}"
        )
    materialized = sum(
        receipt.get("provider_materialized") is True for receipt in receipts
    )
    preflight = sum(
        receipt.get("preflight_completed") is True for receipt in receipts
    )
    state.lightx2v_load_preparation = {
        "attempted": True,
        "preflight_blocks": preflight,
        "materialized_blocks": materialized,
        "deferred_blocks": len(receipts) - materialized,
        "devices": sorted(
            {
                str(receipt["device"])
                for receipt in receipts
                if isinstance(receipt.get("device"), str)
            }
        ),
    }


def advance_approximate_step(
    state: _ApproxState,
    step: int | None = None,
    total_steps: int | None = None,
    *,
    request_id: str | None = None,
    branch: str | None = None,
    routed_steps: bool = False,
) -> None:
    """Select the denoise invocation driving schedule and receipt ownership.

    ``request_id`` and ``branch`` are optional only for backwards-compatible
    direct processor calls. Native Wan runners always provide both and thereby
    get concurrency-safe ownership and strict zero-based contiguous steps.
    """

    if total_steps is not None:
        state.total_steps = total_steps
    resolved_step = state.step + 1 if step is None else step
    state.step = resolved_step
    if request_id is None and branch is None:
        state._current_invocation.set(None)
        state._last_finalized_request_id.set(None)
        return
    if request_id is None or branch is None:
        raise ValueError(
            "request_id and branch must be supplied together for approximate attention"
        )
    resolved_total_steps = state.total_steps
    if resolved_total_steps is None:
        raise ValueError(
            "request-local approximate attention requires explicit total_steps"
        )
    state.begin_invocation(
        request_id=request_id,
        branch=branch,
        step=resolved_step,
        total_steps=resolved_total_steps,
        routed_steps=routed_steps,
    )


def reset_approximate_attention(
    state: _ApproxState,
    request_id: str | None = None,
) -> None:
    """Reset one active request, or the legacy direct-call telemetry window."""

    invocation = state.current_invocation()
    selected_request = request_id
    if selected_request is None and invocation is not None:
        selected_request = invocation.request_id
    if selected_request is not None:
        with state._lock:
            state._requests.pop(selected_request, None)
        if invocation is not None and invocation.request_id == selected_request:
            state._current_invocation.set(None)
        if state._last_finalized_request_id.get() == selected_request:
            state._last_finalized_request_id.set(None)
        return
    state._last_finalized_request_id.set(None)
    state.step = 0
    state.sparse_calls = 0
    state.provider_dense_calls = 0
    state.kernel_attempts = 0
    state.dense_fallback_calls = 0
    state.scheduled_dense_calls = 0
    state.kernel_fallbacks = 0
    state.provider_path = None
    state.vmoba_events.clear()
    state.vmoba_chunk_calls = {
        "temporal": 0,
        "spatial": 0,
        "spatiotemporal": 0,
    }
    state.notes = list(state.install_notes)
    if state.install_reason is not None:
        state.effective_kernel = f"exact ({state.install_reason})"
    else:
        state.effective_kernel = "exact (sparse provider not executed)"


def _vmoba_report_view(
    state: _ApproxState,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if state.config.kind != "vmoba":
        return None
    sparse_events = [event for event in events if event.get("execution") == "sparse"]
    chunk_calls = {"temporal": 0, "spatial": 0, "spatiotemporal": 0}
    for event in sparse_events:
        chunk_kind = event.get("chunk_kind")
        if isinstance(chunk_kind, str) and chunk_kind in chunk_calls:
            chunk_calls[chunk_kind] += 1
    return {
        "first_full_step": state.config.first_full_step,
        "first_full_layer": state.config.first_full_layer,
        "layer_cycle": {
            "temporal": state.config.temporal_layer,
            "spatial": state.config.spatial_layer,
            "spatiotemporal": state.config.st_layer,
        },
        "chunk_calls": chunk_calls,
        "provider_symbols": list(_VMOBA_PROVIDER_SYMBOLS),
        # Compatibility: historically this list contained provider-complete
        # events only. The complete dense/fallback ledger lives at top level.
        "events": sparse_events,
    }


def _lightx2v_report_view(
    state: _ApproxState,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate immutable evidence for one selected LightX2V sparse lane."""

    if state.config.kind not in _LIGHTX2V_SPARSE_KINDS:
        return None
    config = _lightx2v_config(state.config)
    canonical_symbol = lightx2v_kernel_symbol(config)
    sparse_events = [
        event for event in events if event.get("execution") == "sparse"
    ]
    provider_dense_events = [
        event for event in events if event.get("execution") == "provider_dense"
    ]
    provider_events = [*sparse_events, *provider_dense_events]
    # Provider events are normalized receipts. Tensor contracts are
    # stored once by ``_record_event`` instead of embedding a duplicate nested
    # receipt for every layer and denoise step.
    provider_receipts = [
        event
        for event in provider_events
        if isinstance(event.get("provider_family"), str)
    ]
    source_commits = sorted(
        {
            str(commit)
            for receipt in provider_receipts
            if isinstance(
                (commit := receipt.get("provider_source_commit")), str
            )
        }
    )
    source_fingerprints = sorted(
        {
            str(fingerprint)
            for receipt in provider_receipts
            if isinstance(
                (
                    fingerprint := receipt.get(
                        "provider_source_fingerprint"
                    )
                ),
                str,
            )
        }
    )
    provider_paths = sorted(
        {
            str(path)
            for event in sparse_events
            if isinstance((path := event.get("provider_path")), str)
        }
    )
    complete_receipts = len(provider_receipts) == len(provider_events)
    return {
        "algorithm": config.kind,
        "provider_family": lightx2v_provider_family(config),
        "provider_families": sorted(
            {
                str(family)
                for receipt in provider_receipts
                if isinstance(
                    (family := receipt.get("provider_family")), str
                )
            }
        ),
        "operator": config.operator,
        "canonical_provider_symbol": canonical_symbol,
        "canonical_provider_executed": bool(sparse_events)
        and provider_paths == [canonical_symbol],
        "reference_lightx2v_commit": PINNED_LIGHTX2V_COMMIT,
        "provider_source_commit": (
            source_commits[0] if len(source_commits) == 1 else None
        ),
        "provider_source_commits": source_commits,
        "provider_source_fingerprints": source_fingerprints,
        "provider_commit_complete": bool(provider_receipts)
        and all(
            isinstance(receipt.get("provider_source_commit"), str)
            for receipt in provider_receipts
        ),
        "provider_source_clean": bool(provider_receipts)
        and all(
            receipt.get("provider_source_clean") is True
            for receipt in provider_receipts
        ),
        "reference_parity_verified": bool(provider_receipts)
        and complete_receipts
        and all(
            receipt.get("reference_parity_verified") is True
            for receipt in provider_receipts
        ),
        "provider_calls": sum(
            int(receipt.get("provider_calls", 0))
            for receipt in provider_receipts
        ),
        "receipt_count": len(provider_receipts),
        "sparse_event_count": len(sparse_events),
        "provider_dense_event_count": len(provider_dense_events),
        "provider_dense_events": provider_dense_events,
        "load_preparation": dict(state.lightx2v_load_preparation),
        "events": sparse_events,
    }


def _fastvideo_sla_report_view(
    state: _ApproxState,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate learned-SLA provider and checkpoint evidence per request."""

    if state.config.kind not in _FASTVIDEO_SLA_KINDS:
        return None
    sparse_events = [
        event for event in events if event.get("execution") == "sparse"
    ]
    provider_paths = sorted(
        {
            str(path)
            for event in sparse_events
            if isinstance((path := event.get("provider_path")), str)
        }
    )
    reference_paths = sorted(
        {
            str(path)
            for event in sparse_events
            if isinstance((path := event.get("reference_provider_path")), str)
        }
    )
    source_commits = sorted(
        {
            str(commit)
            for event in sparse_events
            if isinstance((commit := event.get("provider_source_commit")), str)
        }
    )
    source_fingerprints = sorted(
        {
            str(fingerprint)
            for event in sparse_events
            if isinstance(
                (fingerprint := event.get("provider_source_fingerprint")),
                str,
            )
        }
    )
    provider_fingerprints = sorted(
        {
            str(fingerprint)
            for event in sparse_events
            if isinstance((fingerprint := event.get("provider_fingerprint")), str)
        }
    )
    projection_fingerprints = sorted(
        {
            str(fingerprint)
            for event in sparse_events
            if isinstance(
                (
                    fingerprint := event.get(
                        "all_projection_weights_fingerprint"
                    )
                ),
                str,
            )
        }
    )
    checkpoint_layouts = sorted(
        {
            str(layout)
            for event in sparse_events
            if isinstance((layout := event.get("checkpoint_layout")), str)
        }
    )
    source_roots = sorted(
        {
            str(root)
            for event in sparse_events
            if isinstance((root := event.get("provider_source_root")), str)
        }
    )
    source_files = sorted(
        {
            str(path)
            for event in sparse_events
            if isinstance((path := event.get("provider_source_file")), str)
        }
    )
    layer_projection_values: dict[int, set[str]] = {}
    layer_source_key_values: dict[int, set[tuple[str, ...]]] = {}
    for event in sparse_events:
        layer_index = event.get("layer_idx")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            continue
        fingerprint = event.get("projection_weight_fingerprint")
        if isinstance(fingerprint, str):
            layer_projection_values.setdefault(layer_index, set()).add(fingerprint)
        source_keys = event.get("projection_source_keys")
        if isinstance(source_keys, (list, tuple)) and all(
            isinstance(key, str) for key in source_keys
        ):
            layer_source_key_values.setdefault(layer_index, set()).add(
                tuple(source_keys)
            )
    layer_projection_fingerprints = {
        str(layer): next(iter(values))
        for layer, values in sorted(layer_projection_values.items())
        if len(values) == 1
    }
    projection_source_keys = {
        str(layer): list(next(iter(values)))
        for layer, values in sorted(layer_source_key_values.items())
        if len(values) == 1
    }
    provider_families = sorted(
        {
            str(family)
            for event in sparse_events
            if isinstance((family := event.get("provider_family")), str)
        }
    )
    expected_family = (
        "fastvideo/sparse-linear-attention"
        if state.config.kind == "fastvideo_sla"
        else "fastvideo/sage-sparse-linear-attention"
    )
    expected_provider_path = (
        "fastvideo.attention.backends.sla.SLAAttentionImpl"
        if state.config.kind == "fastvideo_sla"
        else "fastvideo.attention.backends.sla.SageSLAAttentionImpl"
    )
    expected_layers = set(state.layer_indices)
    complete = bool(sparse_events) and all(
        event.get("provider_calls") == 1
        and event.get("provider_family") == expected_family
        and event.get("provider_path") == expected_provider_path
        and event.get("reference_provider_path") == expected_provider_path
        and event.get("injected_test_provider") is False
        and event.get("reference_fastvideo_commit") == PINNED_FASTVIDEO_COMMIT
        and event.get("provider_source_commit") == PINNED_FASTVIDEO_COMMIT
        and event.get("provider_source_clean") is True
        and event.get("reference_parity_verified") is True
        and event.get("topk_ratio") == state.config.fastvideo_topk_ratio
        and event.get("feature_map") == state.config.fastvideo_feature_map
        and event.get("runtime_effective") is True
        for event in sparse_events
    ) and set(layer_projection_values) == expected_layers and all(
        len(values) == 1 for values in layer_projection_values.values()
    ) and set(layer_source_key_values) == expected_layers and all(
        len(values) == 1 for values in layer_source_key_values.values()
    )
    return {
        "algorithm": state.config.kind,
        "provider_family": expected_family,
        "provider_families": provider_families,
        "expected_layers": list(state.layer_indices),
        "canonical_provider_path": (
            reference_paths[0] if len(reference_paths) == 1 else None
        ),
        "canonical_provider_executed": bool(sparse_events)
        and len(reference_paths) == 1
        and provider_paths == reference_paths,
        "reference_fastvideo_commit": PINNED_FASTVIDEO_COMMIT,
        "provider_source_commit": (
            source_commits[0] if len(source_commits) == 1 else None
        ),
        "provider_source_commits": source_commits,
        "provider_source_fingerprints": source_fingerprints,
        "provider_fingerprints": provider_fingerprints,
        "provider_source_root": (
            source_roots[0] if len(source_roots) == 1 else None
        ),
        "provider_source_roots": source_roots,
        "provider_source_file": (
            source_files[0] if len(source_files) == 1 else None
        ),
        "provider_source_files": source_files,
        "provider_commit_complete": bool(sparse_events)
        and all(
            event.get("provider_source_commit") == PINNED_FASTVIDEO_COMMIT
            for event in sparse_events
        ),
        "provider_source_clean": bool(sparse_events)
        and all(
            event.get("provider_source_clean") is True
            for event in sparse_events
        ),
        "reference_parity_verified": complete,
        "checkpoint_layout": (
            checkpoint_layouts[0] if len(checkpoint_layouts) == 1 else None
        ),
        "checkpoint_layouts": checkpoint_layouts,
        "all_projection_weights_fingerprint": (
            projection_fingerprints[0]
            if len(projection_fingerprints) == 1
            else None
        ),
        "projection_fingerprints": projection_fingerprints,
        "layer_projection_fingerprints": layer_projection_fingerprints,
        "projection_source_keys": projection_source_keys,
        "topk_ratio": state.config.fastvideo_topk_ratio,
        "feature_map": state.config.fastvideo_feature_map,
        "provider_calls": sum(
            int(event.get("provider_calls", 0)) for event in sparse_events
        ),
        "receipt_count": len(sparse_events),
        "sparse_event_count": len(sparse_events),
        "events": sparse_events,
    }


def _request_coverage(
    state: _ApproxState,
    request: _ApproxRequestState,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_layers = list(state.layer_indices)
    expected_layer_set = set(expected_layers)
    branch_reports: dict[str, Any] = {}
    for branch, last_step in sorted(request.branch_steps.items()):
        step_layers: dict[int, list[int]] = {}
        for event in events:
            if event.get("branch") != branch:
                continue
            event_step = int(event["step"])
            step_layers.setdefault(event_step, []).append(int(event["layer_idx"]))
        completed_steps = sorted(
            step
            for step, layers in step_layers.items()
            if len(layers) == len(expected_layers)
            and len(set(layers)) == len(layers)
            and set(layers) == expected_layer_set
        )
        observed_steps = sorted(step_layers)
        routed_history = list(request.branch_step_history.get(branch, ()))
        expected_steps = (
            routed_history if request.routed_steps else list(range(request.total_steps))
        )
        steps_valid = (
            len(routed_history) == len(set(routed_history))
            and routed_history == sorted(routed_history)
            if request.routed_steps
            else observed_steps == list(range(observed_steps[-1] + 1))
            if observed_steps
            else False
        )
        branch_reports[branch] = {
            "last_step": last_step,
            "observed_steps": observed_steps,
            "expected_steps": expected_steps,
            "completed_steps": completed_steps,
            "steps_contiguous": steps_valid,
            "routed_steps": request.routed_steps,
            "all_steps_complete": completed_steps == expected_steps,
            "layer_events": sum(len(layers) for layers in step_layers.values()),
            "missing_layers": {
                str(step): sorted(expected_layer_set - set(step_layers.get(step, ())))
                for step in expected_steps
                if set(step_layers.get(step, ())) != expected_layer_set
            },
        }
    expected_step_calls = (
        sum(len(steps) for steps in request.branch_step_history.values())
        if request.routed_steps
        else len(request.branch_steps) * request.total_steps
    )
    expected_calls = expected_step_calls * len(expected_layers)
    execution_counts: dict[str, int] = {}
    for event in events:
        execution = str(event.get("execution", "unknown"))
        execution_counts[execution] = execution_counts.get(execution, 0) + 1
    counter_event_total = sum(
        execution_counts.get(name, 0)
        for name in (
            "sparse",
            "provider_dense",
            "scheduled_dense",
            "fallback",
            "error",
        )
    )
    event_totals_match_counters = (
        counter_event_total == len(events)
        and execution_counts.get("sparse", 0) == request.sparse_calls
        and execution_counts.get("provider_dense", 0)
        == request.provider_dense_calls
        and execution_counts.get("scheduled_dense", 0)
        == request.scheduled_dense_calls
        and execution_counts.get("fallback", 0) == request.kernel_fallbacks
        and sum(bool(event.get("provider_attempted")) for event in events)
        == request.kernel_attempts
        and sum(bool(event.get("dense_fallback_executed")) for event in events)
        == request.dense_fallback_calls
    )
    provider_contract_complete = all(
        isinstance(event.get("provider_path"), str)
        and all(
            event.get(name) is not None
            for name in (
                "input_shape",
                "input_device",
                "input_dtype",
                "output_shape",
                "output_device",
                "output_dtype",
            )
        )
        for event in events
        if event.get("execution") in {"sparse", "provider_dense"}
    )
    request_local = all(
        event.get("request_id") == request.request_id
        and int(event.get("request_epoch", -1)) == request.request_epoch
        and bool(event.get("request_local"))
        for event in events
    )
    complete = (
        bool(branch_reports)
        and len(expected_layers) == state.wrapped_blocks
        and expected_layers == list(range(state.wrapped_blocks))
        and len(events) == expected_calls
        and event_totals_match_counters
        and request_local
        and provider_contract_complete
        and all(report["all_steps_complete"] for report in branch_reports.values())
    )
    return {
        "complete": complete,
        "routed_steps": request.routed_steps,
        "expected_layers": expected_layers,
        "expected_layer_count": len(expected_layers),
        "expected_calls": expected_calls,
        "observed_calls": len(events),
        "event_totals_match_counters": event_totals_match_counters,
        "provider_contract_complete": provider_contract_complete,
        "request_local": request_local,
        "branches": branch_reports,
        "execution_counts": execution_counts,
    }


def _active_request_report(
    state: _ApproxState,
    request: _ApproxRequestState,
) -> dict[str, Any]:
    # A JSON round-trip makes nested event metadata independent from the live
    # mutable ledger even before finalization.
    events = json.loads(json.dumps(request.events))
    coverage = _request_coverage(state, request, events)
    if request.sparse_calls and not request.kernel_fallbacks:
        effective_kernel = state.config.kind
    elif request.sparse_calls:
        effective_kernel = (
            f"exact/{state.config.kind} mixed (sparse provider fallback)"
        )
    elif request.kernel_fallbacks:
        effective_kernel = "exact (sparse provider runtime fallback)"
    elif request.provider_dense_calls:
        effective_kernel = (
            f"{state.config.kind} (provider-designed dense phase only)"
        )
    elif request.scheduled_dense_calls:
        effective_kernel = "exact (scheduled dense)"
    else:
        effective_kernel = "exact (sparse provider not executed)"
    provider_paths = sorted(request.provider_paths)
    provider_dense_paths = sorted(request.provider_dense_paths)
    error_calls = sum(event.get("execution") == "error" for event in events)
    return {
        "kind": state.config.kind,
        "request_id": request.request_id,
        "request_epoch": request.request_epoch,
        "request_local": coverage["request_local"],
        "routed_steps": request.routed_steps,
        "finalized": False,
        "completed": False,
        "release_reason": None,
        "wrapped_blocks": state.wrapped_blocks,
        "expected_layers": list(state.layer_indices),
        "expected_calls": coverage["expected_calls"],
        "runtime_effective": (
            request.sparse_calls > 0
            and request.kernel_fallbacks == 0
            and error_calls == 0
        ),
        "kernel_attempts": request.kernel_attempts,
        "sparse_calls": request.sparse_calls,
        "provider_dense_calls": request.provider_dense_calls,
        "dense_fallback_calls": request.dense_fallback_calls,
        "scheduled_dense_calls": request.scheduled_dense_calls,
        "kernel_fallbacks": request.kernel_fallbacks,
        "effective_kernel": effective_kernel,
        "provider_path": provider_paths[0] if len(provider_paths) == 1 else None,
        "provider_paths": provider_paths,
        "provider_dense_path": (
            provider_dense_paths[0] if len(provider_dense_paths) == 1 else None
        ),
        "provider_dense_paths": provider_dense_paths,
        "grid_size": request.grid_size,
        "branches": coverage["branches"],
        "events": events,
        "event_count": len(events),
        "coverage": coverage,
        "vmoba": _vmoba_report_view(state, events),
        "lightx2v": _lightx2v_report_view(state, events),
        "fastvideo_sla": _fastvideo_sla_report_view(state, events),
        "notes": list(request.notes),
    }


def finalize_approximate_attention_request(
    state: _ApproxState,
    request_id: str,
    *,
    error: BaseException | None = None,
) -> None:
    """Freeze one immutable receipt, then release its active event ledger."""

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError(
            "finalize_approximate_attention_request requires a non-empty request_id"
        )
    request = state.request_state(request_id, create=False)
    if request is None:
        return
    report = _active_request_report(state, request)
    report["finalized"] = True
    report["completed"] = error is None
    report["release_reason"] = "error" if error is not None else "completed"
    if error is not None:
        report["error_type"] = type(error).__name__
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    with state._lock:
        removed = state._requests.pop(request_id, None)
        if removed is None:
            return
        # Dropping ``removed`` releases the full active event list. Only the
        # tensor-free JSON receipt is retained, with a hard history bound.
        state._receipt_snapshots[request_id] = encoded
        while len(state._receipt_snapshots) > _MAX_REQUEST_RECEIPTS:
            oldest_request_id = next(iter(state._receipt_snapshots))
            state._receipt_snapshots.pop(oldest_request_id, None)
    invocation = state.current_invocation()
    if invocation is not None and invocation.request_id == request_id:
        state._current_invocation.set(None)
    # Retain only the identifier needed by the default reporting path. The
    # invocation itself owns the mutable request ledger, so it must never
    # survive finalization through a ContextVar reference.
    state._last_finalized_request_id.set(request_id)


def approximate_attention_lifecycle_report(state: _ApproxState) -> dict[str, int]:
    """Expose bounded ownership without retaining provider tensors."""

    with state._lock:
        return {
            "live_requests": len(state._requests),
            "receipt_snapshots": len(state._receipt_snapshots),
            "max_receipt_snapshots": _MAX_REQUEST_RECEIPTS,
        }


def approximate_attention_report(
    state: _ApproxState,
    request_id: str | None = None,
) -> dict[str, Any]:
    invocation = state.current_invocation()
    if request_id is None and invocation is not None:
        request_id = invocation.request_id
    elif request_id is None:
        request_id = state._last_finalized_request_id.get()
    if isinstance(request_id, str) and request_id.strip():
        request = state.request_state(request_id, create=False)
        if request is not None:
            report = _active_request_report(state, request)
            report["lifecycle"] = approximate_attention_lifecycle_report(state)
            return report
        with state._lock:
            encoded = state._receipt_snapshots.get(request_id)
        if isinstance(encoded, str):
            report = json.loads(encoded)
            if not isinstance(report, dict):
                raise TypeError(
                    "approximate-attention receipt snapshot must decode to an object"
                )
            report["lifecycle"] = approximate_attention_lifecycle_report(state)
            return report

    # Compatibility report for direct processor calls without request IDs.
    runtime_effective = state.sparse_calls > 0 and state.kernel_fallbacks == 0
    legacy_events = json.loads(json.dumps(state.vmoba_events))
    if any(event.get("execution") == "error" for event in legacy_events):
        runtime_effective = False
    return {
        "kind": state.config.kind,
        "request_id": None,
        "request_epoch": 0,
        "request_local": False,
        "finalized": False,
        "completed": False,
        "release_reason": None,
        "wrapped_blocks": state.wrapped_blocks,
        "expected_layers": list(state.layer_indices),
        "expected_calls": None,
        "runtime_effective": runtime_effective,
        "kernel_attempts": state.kernel_attempts,
        "sparse_calls": state.sparse_calls,
        "provider_dense_calls": state.provider_dense_calls,
        "dense_fallback_calls": state.dense_fallback_calls,
        "scheduled_dense_calls": state.scheduled_dense_calls,
        "kernel_fallbacks": state.kernel_fallbacks,
        "effective_kernel": state.effective_kernel,
        "provider_path": state.provider_path,
        "provider_paths": [state.provider_path] if state.provider_path else [],
        "grid_size": state.grid_size,
        "branches": {},
        "events": legacy_events,
        "event_count": len(legacy_events),
        "coverage": None,
        "vmoba": _vmoba_report_view(state, legacy_events),
        "lightx2v": _lightx2v_report_view(state, legacy_events),
        "fastvideo_sla": _fastvideo_sla_report_view(state, legacy_events),
        "notes": list(state.notes),
        "lifecycle": approximate_attention_lifecycle_report(state),
    }


def lightx2v_attention_report(
    state: _ApproxState,
    request_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the concise LightX2V evidence view for one request receipt."""

    report = approximate_attention_report(state, request_id)
    lightx2v = report.get("lightx2v")
    return dict(lightx2v) if isinstance(lightx2v, Mapping) else None


def fastvideo_sla_attention_report(
    state: _ApproxState,
    request_id: str | None = None,
) -> dict[str, Any] | None:
    """Return concise FastVideo learned-SLA evidence for one request."""

    report = approximate_attention_report(state, request_id)
    fastvideo_sla = report.get("fastvideo_sla")
    return (
        dict(fastvideo_sla)
        if isinstance(fastvideo_sla, Mapping)
        else None
    )


__all__ = [
    "ApproximateAttentionConfig",
    "ApproximateSelfAttentionProcessor",
    "parse_approximate_attention",
    "install_approximate_attention",
    "advance_approximate_step",
    "finalize_approximate_attention_request",
    "approximate_attention_lifecycle_report",
    "reset_approximate_attention",
    "approximate_attention_report",
    "lightx2v_attention_report",
    "fastvideo_sla_attention_report",
    "place_fastvideo_sla_providers",
    "prepare_lightx2v_providers",
    "_recoverable_sparse_kernel_error",
]
