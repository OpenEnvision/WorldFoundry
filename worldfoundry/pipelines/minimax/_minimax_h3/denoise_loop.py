"""MiniMax H3 CFG-distilled full denoise loop (ported from SGLang Apache-2.0).

Per step, the positive presentation is forwarded exactly once. Video and audio
target rows chain through the Euler-eta0 update while visual and audio
condition rows stay pinned to their noised step-0 anchors.

Single-GPU port: SGLang's Ulysses sequence-parallel context collapses to
``(world_size=1, rank=0)``, so the per-rank row slicing is the identity and the
``local_embedding_layout`` covers the full sequence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
    MiniMaxH3ContextParallelConfig,
    _coerce_context_parallel_config,
    _cp_all_gather_stats,
    _resolve_probe_bounds,
    _ResolvedContextParallel,
    resolve_minimax_h3_context_parallel,
)

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32


@dataclass(frozen=True)
class MiniMaxH3StepCacheConfig:
    """Opt-in calibrated cross-step hidden-state residual cache (EasyCache / TeaCache / FirstBlockCache).

    Modes:
    - ``"easycache"``: estimates full 50-layer output drift from Block 0's
      AdaLN-modulated input and reuses the full block-stack residual when
      calibrated accumulated drift remains below ``threshold`` (default
      ``0.40``).
    - ``"teacache"``: maps relative L1 drift of Block 0's AdaLN-modulated input
      through ``teacache_coefficients`` and reuses the full block-stack residual.
    - ``"firstblock"`` (SOL-Engine ``FirstBlockCache`` / ``FBCache``): always
      executes Block 0 (BF16 dense), measures relative L1 drift of Block 0's
      residual delta ``(h1 - h0)`` on active denoised tokens (``probe_start:``)
      against the last computed anchor, and when drift < ``threshold`` reuses
      the cached ``blocks[1..49]`` residual delta to skip all 49 remaining
      blocks. Composes cleanly with :class:`PiecewiseBlockGraphRunner`
      (``head = blocks[0:1]``, ``body = blocks[1:48]``, ``tail = blocks[48:50]``)
      and Ulysses Context Parallel (via synchronized global L1 reductions).

    Warmup steps, final boundary steps, attention mode transitions (e.g.
    ``dense_initial_steps``), and consecutive-skip caps always execute full
    compute.
    """

    enabled: bool = True
    threshold: float = 0.40
    warmup_steps: int = 3
    dense_last_steps: int = 2
    max_consecutive_skips: int = 2
    probe_stride: int = 16
    mode: Literal["easycache", "teacache", "firstblock"] = "easycache"
    teacache_coefficients: tuple[float, ...] | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"step_cache enabled must be bool, got {self.enabled!r}")
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError(f"step_cache threshold must be finite and non-negative, got {self.threshold!r}")
        for name in ("warmup_steps", "dense_last_steps", "max_consecutive_skips"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"step_cache {name} must be a non-negative integer, got {value!r}")
        if isinstance(self.probe_stride, bool) or not isinstance(self.probe_stride, int) or self.probe_stride < 1:
            raise ValueError(f"step_cache probe_stride must be a positive integer, got {self.probe_stride!r}")
        if self.mode not in ("easycache", "teacache", "firstblock"):
            raise ValueError(f"step_cache mode must be 'easycache', 'teacache', or 'firstblock', got {self.mode!r}")
        if self.teacache_coefficients is not None and (
            not self.teacache_coefficients
            or any(
                isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c)
                for c in self.teacache_coefficients
            )
        ):
            raise ValueError("step_cache teacache_coefficients must be a non-empty tuple of finite floats")
        if isinstance(self.eps, bool) or not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"step_cache eps must be finite and positive, got {self.eps!r}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.teacache_coefficients is not None:
            payload["teacache_coefficients"] = list(self.teacache_coefficients)
        return payload


def _coerce_step_cache_config(
    config: MiniMaxH3StepCacheConfig | Mapping[str, Any] | None,
) -> MiniMaxH3StepCacheConfig | None:
    if config is None:
        return None
    if isinstance(config, MiniMaxH3StepCacheConfig):
        return config if config.enabled else None
    if isinstance(config, Mapping):
        raw = dict(config)
        if "teacache_coefficients" in raw and raw["teacache_coefficients"] is not None:
            raw["teacache_coefficients"] = tuple(float(c) for c in raw["teacache_coefficients"])
        parsed = MiniMaxH3StepCacheConfig(**raw)
        return parsed if parsed.enabled else None
    raise TypeError(f"step_cache must be MiniMaxH3StepCacheConfig, Mapping, or None, got {type(config).__name__}")


class MiniMaxH3ResidualCache:
    """Per-branch calibrated cross-step residual cache for MiniMax H3 DiT blocks."""

    def __init__(
        self,
        config: MiniMaxH3StepCacheConfig,
        *,
        total_steps: int,
        branch_name: str = "positive",
        probe_start: int = 0,
    ) -> None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(f"total_steps must be a positive integer, got {total_steps!r}")
        if isinstance(probe_start, bool) or not isinstance(probe_start, int) or probe_start < 0:
            raise ValueError(f"probe_start must be a non-negative integer, got {probe_start!r}")
        self.config = config
        self.total_steps = int(total_steps)
        self.branch_name = str(branch_name)
        self.probe_stride = int(config.probe_stride)
        self.probe_start = int(probe_start)
        self.reset()

    def _clear_state(self) -> None:
        self._cached_residual: torch.Tensor | None = None
        self._prev_probe: torch.Tensor | None = None
        self._dense_probe: torch.Tensor | None = None
        self._dense_out_probe: torch.Tensor | None = None
        self._dense_out_norm: float | None = None
        self._transform_factor: float | None = None
        self._layout: tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device, int, int, int, int] | None = (
            None
        )
        self._last_force_compute: bool | None = None
        self._accumulator: float = 0.0
        self._consecutive_skips: int = 0

    def reset(self) -> None:
        self._clear_state()
        self.decisions: list[dict[str, Any]] = []

    def run(
        self,
        *,
        step: int,
        hidden_in: torch.Tensor,
        probe: torch.Tensor,
        active_tokens: int,
        force_compute: bool,
        compute_fn: Callable[[], torch.Tensor],
        on_hit: Callable[[], None] | None = None,
        sp_ctx: _ResolvedContextParallel | None = None,
        total_tokens: int | None = None,
        row_start: int = 0,
        row_stop: int | None = None,
    ) -> torch.Tensor:
        """Compute full/remaining transformer blocks or reuse cached hidden-state residual."""

        total_tok = int(hidden_in.shape[0]) if total_tokens is None else int(total_tokens)
        eff_row_stop = total_tok if row_stop is None else int(row_stop)
        start_idx, n_active, local_start, local_stop = _resolve_probe_bounds(
            active_tokens=active_tokens,
            total_tokens=total_tok,
            probe_start=self.probe_start,
            probe_stride=self.probe_stride,
            row_start=row_start,
            row_stop=eff_row_stop,
        )
        layout = (
            tuple(hidden_in.shape),
            tuple(probe.shape),
            hidden_in.dtype,
            hidden_in.device,
            start_idx,
            n_active,
            int(row_start),
            eff_row_stop,
        )
        probe_fp32 = probe.detach().to(device=hidden_in.device, dtype=torch.float32)
        mode_changed = self._last_force_compute is not None and bool(force_compute) != self._last_force_compute

        reason: str | None = None
        step_drift: float = 0.0
        if not self.config.enabled or self.config.threshold == 0.0 or self.config.max_consecutive_skips == 0:
            reason = "disabled"
        elif torch.is_grad_enabled():
            self._clear_state()
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._clear_state()
            reason = "layout_change"
        elif step < self.config.warmup_steps:
            reason = "warmup"
        elif step >= self.total_steps - self.config.dense_last_steps:
            reason = "dense_last"
        elif force_compute:
            reason = "force_dense"
        elif mode_changed:
            # Never extrapolate residual or transformation factor across dense->sparse switches.
            self._transform_factor = None
            reason = "attention_mode_transition"
        elif (
            self._cached_residual is None
            or self._prev_probe is None
            or (self.config.mode == "firstblock" and self._dense_probe is None)
        ):
            reason = "seed"
        elif self.config.mode == "easycache" and (self._transform_factor is None or self._dense_out_norm is None):
            reason = "calibrate_factor"
        elif self._consecutive_skips >= self.config.max_consecutive_skips:
            reason = "max_consecutive_skips"

        zero_in_f64 = torch.zeros((), dtype=torch.float64, device=hidden_in.device)
        probe_nf_t = (
            (~torch.isfinite(probe_fp32).all()).to(torch.float64)
            if (probe_fp32.numel() > 0 and reason not in ("disabled", "autograd"))
            else zero_in_f64
        )

        if reason is None:
            if self.config.mode == "firstblock":
                assert self._dense_probe is not None
                ref_tensor = self._dense_probe
            else:
                assert self._prev_probe is not None
                ref_tensor = self._prev_probe

            local_count = float(probe_fp32.numel())
            if local_count > 0.0:
                diff_sum_t = (probe_fp32 - ref_tensor).abs().sum(dtype=torch.float64)
                ref_sum_t = (
                    ref_tensor.abs().sum(dtype=torch.float64)
                    if self.config.mode in ("firstblock", "teacache")
                    else zero_in_f64
                )
            else:
                diff_sum_t = zero_in_f64
                ref_sum_t = zero_in_f64

            # Pack all pre-check scalars on-device for a single optional collective + 1 D2H sync.
            count_t = torch.full((), local_count, dtype=torch.float64, device=hidden_in.device)
            pre_stats = torch.stack((diff_sum_t, ref_sum_t, count_t, probe_nf_t))
            if sp_ctx is not None:
                total_diff, total_ref, total_count, total_nonfinite = _cp_all_gather_stats(pre_stats, sp_ctx).tolist()
            else:
                total_diff, total_ref, total_count, total_nonfinite = pre_stats.tolist()

            if total_nonfinite > 0.0:
                self._clear_state()
                reason = "nonfinite_probe"
            else:
                denom_count = max(float(total_count), 1.0)
                probe_delta = float(total_diff) / denom_count
                if self.config.mode == "firstblock":
                    anchor_norm = max(float(total_ref) / denom_count, self.config.eps)
                    step_drift = probe_delta / anchor_norm
                    candidate_acc = step_drift
                elif self.config.mode == "easycache":
                    assert self._transform_factor is not None and self._dense_out_norm is not None
                    step_drift = self._transform_factor * probe_delta / max(self._dense_out_norm, self.config.eps)
                    candidate_acc = self._accumulator + step_drift
                else:
                    prev_norm = max(float(total_ref) / denom_count, self.config.eps)
                    rel_l1 = probe_delta / prev_norm
                    coeffs = self.config.teacache_coefficients or (1.0, 0.0)
                    poly = 0.0
                    for coeff in coeffs:
                        poly = poly * rel_l1 + float(coeff)
                    step_drift = max(0.0, poly)
                    candidate_acc = self._accumulator + step_drift

                if not math.isfinite(candidate_acc) or candidate_acc >= self.config.threshold:
                    self._accumulator = candidate_acc
                    reason = "threshold"
                else:
                    assert self._cached_residual is not None
                    self._accumulator = candidate_acc
                    self._consecutive_skips += 1
                    self._prev_probe = probe_fp32
                    self._last_force_compute = bool(force_compute)
                    if on_hit is not None:
                        on_hit()
                    self.decisions.append(
                        {
                            "step": int(step),
                            "hit": True,
                            "reason": "cache_hit",
                            "step_drift": float(step_drift),
                            "accumulator": float(candidate_acc),
                        }
                    )
                    return hidden_in + self._cached_residual

        hidden_out = compute_fn()
        if reason in ("disabled", "autograd") or torch.is_grad_enabled():
            if reason == "autograd" or torch.is_grad_enabled():
                self._clear_state()
            self.decisions.append(
                {
                    "step": int(step),
                    "hit": False,
                    "reason": reason,
                    "step_drift": float(step_drift),
                    "accumulator": float(self._accumulator),
                }
            )
            self._accumulator = 0.0
            self._consecutive_skips = 0
            return hidden_out

        out_sub = (
            hidden_out[local_start : local_stop : self.probe_stride].detach().float()
            if local_start < local_stop
            else hidden_out[:0].detach().float()
        )
        can_calibrate = (
            self.config.mode == "easycache"
            and self._dense_probe is not None
            and self._dense_out_probe is not None
            and not mode_changed
            and self._last_force_compute == bool(force_compute)
        )
        zero_out_f64 = torch.zeros((), dtype=torch.float64, device=hidden_out.device)
        local_out_count = float(out_sub.numel())
        if local_out_count > 0.0:
            out_nf_t = (~torch.isfinite(out_sub).all()).to(torch.float64)
            out_sum_t = out_sub.abs().sum(dtype=torch.float64) if self.config.mode != "firstblock" else zero_out_f64
            if can_calibrate:
                assert self._dense_probe is not None and self._dense_out_probe is not None
                in_diff_t = (probe_fp32 - self._dense_probe).abs().sum(dtype=torch.float64)
                out_diff_t = (out_sub - self._dense_out_probe).abs().sum(dtype=torch.float64)
            else:
                in_diff_t = zero_out_f64
                out_diff_t = zero_out_f64
        else:
            out_nf_t = zero_out_f64
            out_sum_t = zero_out_f64
            in_diff_t = zero_out_f64
            out_diff_t = zero_out_f64

        probe_nf_out = probe_nf_t if probe_nf_t.device == hidden_out.device else probe_nf_t.to(hidden_out.device)
        out_count_t = torch.full((), local_out_count, dtype=torch.float64, device=hidden_out.device)
        post_stats = torch.stack((out_sum_t, in_diff_t, out_diff_t, out_count_t, probe_nf_out, out_nf_t))
        if sp_ctx is not None:
            tot_out_sum, tot_in_diff, tot_out_diff, tot_out_count, tot_probe_nf, tot_out_nf = _cp_all_gather_stats(
                post_stats, sp_ctx
            ).tolist()
        else:
            tot_out_sum, tot_in_diff, tot_out_diff, tot_out_count, tot_probe_nf, tot_out_nf = post_stats.tolist()
        global_probe_finite = tot_probe_nf == 0.0
        global_out_finite = tot_out_nf == 0.0

        if not global_probe_finite:
            reason = "nonfinite_probe"

        if global_probe_finite and global_out_finite:
            denom_out = max(float(tot_out_count), 1.0)
            if self.config.mode == "firstblock":
                self._dense_probe = probe_fp32
                self._dense_out_probe = None
                self._dense_out_norm = None
                self._transform_factor = None
            else:
                out_norm = max(float(tot_out_sum) / denom_out, self.config.eps)
                if can_calibrate:
                    in_delta = float(tot_in_diff) / denom_out
                    out_delta = float(tot_out_diff) / denom_out
                    if in_delta > self.config.eps and math.isfinite(out_delta / in_delta):
                        self._transform_factor = out_delta / in_delta
                elif mode_changed:
                    self._transform_factor = None
                self._dense_probe = probe_fp32
                self._dense_out_probe = out_sub
                self._dense_out_norm = out_norm
            self._cached_residual = (hidden_out - hidden_in).detach()
            self._layout = layout
            self._prev_probe = probe_fp32
            self._last_force_compute = bool(force_compute)
        else:
            self._clear_state()

        self.decisions.append(
            {
                "step": int(step),
                "hit": False,
                "reason": reason,
                "step_drift": float(step_drift),
                "accumulator": float(self._accumulator),
            }
        )
        self._accumulator = 0.0
        self._consecutive_skips = 0
        return hidden_out

    def receipt(self) -> dict[str, Any]:
        total = len(self.decisions)
        cached = sum(1 for item in self.decisions if item["hit"])
        computed = total - cached
        return {
            "branch": self.branch_name,
            "probe_start": self.probe_start,
            "config": self.config.to_dict(),
            "total_steps": total,
            "computed_steps": computed,
            "cached_steps": cached,
            "skip_ratio": (cached / total) if total > 0 else 0.0,
            "decisions": list(self.decisions),
        }


@dataclass(frozen=True)
class MiniMaxH3CFGParallelConfig:
    """2-GPU Classifier-Free Guidance parallel execution context.

    Rank 0 evaluates the positive branch while Rank 1 evaluates the negative
    branch concurrently, then a single FP32 ``all_gather`` exchanges both video
    and audio target velocities before synchronized Euler updates.
    """

    enabled: bool = False
    process_group: Any = None
    rank: int | None = None
    world_size: int | None = None
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"cfg_parallel enabled must be bool, got {self.enabled!r}")
        if self.rank is not None and (isinstance(self.rank, bool) or not isinstance(self.rank, int)):
            raise ValueError(f"cfg_parallel rank must be an integer or None, got {self.rank!r}")
        if self.world_size is not None and (isinstance(self.world_size, bool) or not isinstance(self.world_size, int)):
            raise ValueError(f"cfg_parallel world_size must be an integer or None, got {self.world_size!r}")
        if (self.rank is None) != (self.world_size is None):
            raise ValueError("cfg_parallel rank and world_size must be specified together or both omitted")


def _coerce_cfg_parallel_config(
    cfg_parallel: MiniMaxH3CFGParallelConfig | bool | None,
) -> MiniMaxH3CFGParallelConfig | None:
    if cfg_parallel is None:
        return None
    if isinstance(cfg_parallel, bool):
        return MiniMaxH3CFGParallelConfig(enabled=cfg_parallel)
    if isinstance(cfg_parallel, MiniMaxH3CFGParallelConfig):
        return cfg_parallel
    raise TypeError(
        f"cfg_parallel must be MiniMaxH3CFGParallelConfig, bool, or None, got {type(cfg_parallel).__name__}"
    )


def _resolve_cfg_parallel(
    cfg_parallel: MiniMaxH3CFGParallelConfig | bool | None,
    *,
    use_cfg: bool = True,
) -> tuple[bool, int, int, Any, Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None]:
    """Resolve and validate 2-rank CFG parallel execution state."""

    config = _coerce_cfg_parallel_config(cfg_parallel)
    if config is None or not config.enabled or not use_cfg:
        return False, 0, 1, None, None

    group = config.process_group
    if config.world_size is None or config.rank is None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "MiniMax H3 CFG parallel requires an initialized torch.distributed process group "
                "(or explicit rank/world_size/all_gather_fn for testing)."
            )
        if group is None and int(torch.distributed.get_world_size()) > 2:
            from worldfoundry.core.distributed.sequence_parallel.runtime import get_cfg_parallel_group

            cfg_grp = get_cfg_parallel_group()
            if cfg_grp is not None:
                group = cfg_grp
        world_size = int(torch.distributed.get_world_size(group=group))
        rank = int(torch.distributed.get_rank(group=group))
    else:
        world_size = int(config.world_size)
        rank = int(config.rank)

    if world_size == 1:
        return False, 0, 1, None, None
    if world_size != 2:
        raise ValueError(f"MiniMax H3 CFG parallel requires cfg_world_size == 2 (got {world_size})")
    if rank not in (0, 1):
        raise ValueError(f"MiniMax H3 CFG parallel rank must be 0 or 1 (got {rank})")

    gather_fn = config.all_gather_fn
    if gather_fn is None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "MiniMax H3 CFG parallel requires an initialized torch.distributed process group "
                "when all_gather_fn is not provided."
            )

        def _dist_all_gather(out_list: list[torch.Tensor], src: torch.Tensor, grp: Any) -> None:
            torch.distributed.all_gather(out_list, src, group=grp)

        gather_fn = _dist_all_gather
    return True, rank, world_size, group, gather_fn


_MINIMAX_H3_2D_GROUPS: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}


def _get_or_init_2d_cfg_sp_groups(sp_size: int) -> tuple[Any, Any]:
    """Return cached ``(sp_group, cfg_group)`` for 2D ``CFG=2 x SP=sp_size`` without leaking communicators."""

    world_size = int(torch.distributed.get_world_size())
    world_group = getattr(torch.distributed.group, "WORLD", None)
    key = (id(world_group), world_size, int(sp_size), 2)
    cached = _MINIMAX_H3_2D_GROUPS.get(key)
    if cached is not None:
        sp_group, cfg_group = cached
        try:
            if (
                int(torch.distributed.get_world_size(group=sp_group)) == int(sp_size)
                and int(torch.distributed.get_world_size(group=cfg_group)) == 2
            ):
                return sp_group, cfg_group
        except Exception:
            _MINIMAX_H3_2D_GROUPS.pop(key, None)

    from worldfoundry.core.distributed.sequence_parallel.runtime import (
        get_cfg_parallel_group,
        get_sequence_parallel_group,
        initialize_cfg_parallel_group,
        initialize_sequence_parallel_group,
    )

    sp_group = get_sequence_parallel_group()
    cfg_group = get_cfg_parallel_group()
    sp_ok = sp_group is not None and int(torch.distributed.get_world_size(group=sp_group)) == int(sp_size)
    cfg_ok = cfg_group is not None and int(torch.distributed.get_world_size(group=cfg_group)) == 2
    if not sp_ok:
        initialize_sequence_parallel_group(int(sp_size))
        sp_group = get_sequence_parallel_group()
    if not cfg_ok:
        initialize_cfg_parallel_group(int(sp_size), 2)
        cfg_group = get_cfg_parallel_group()
    assert sp_group is not None and cfg_group is not None
    _MINIMAX_H3_2D_GROUPS[key] = (sp_group, cfg_group)
    return sp_group, cfg_group


def _resolve_parallel_topology(
    *,
    cfg_parallel: MiniMaxH3CFGParallelConfig | bool | None,
    context_parallel: MiniMaxH3ContextParallelConfig | _ResolvedContextParallel | Mapping[str, Any] | int | bool | None,
    model: Any,
    use_cfg: bool,
) -> tuple[
    MiniMaxH3CFGParallelConfig | None,
    bool,
    int,
    int,
    Any,
    Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None,
    MiniMaxH3ContextParallelConfig | None,
    _ResolvedContextParallel | None,
    bool,
]:
    """Validate and resolve 1D CFG, 1D Ulysses CP, or 2D (CFG=2 x SP=P) process groups."""

    cfg_parallel_cfg = _coerce_cfg_parallel_config(cfg_parallel)
    explicit_cp_disable = context_parallel is False or context_parallel == 1
    if explicit_cp_disable:
        raw_cp: Any = None
    elif context_parallel is not None:
        raw_cp = context_parallel
    else:
        raw_cp = getattr(model, "_context_parallel", None)
    cp_cfg = _coerce_context_parallel_config(raw_cp)

    cfg_enabled = bool(cfg_parallel_cfg is not None and cfg_parallel_cfg.enabled)
    cfg_wants = bool(cfg_enabled and use_cfg)
    cp_wants = bool(cp_cfg is not None and cp_cfg.sp_size > 1)

    if (
        cfg_enabled
        and cp_wants
        and cfg_parallel_cfg is not None
        and cp_cfg is not None
        and cfg_parallel_cfg.process_group is None
        and cp_cfg.process_group is None
        and cfg_parallel_cfg.all_gather_fn is None
        and cp_cfg.all_to_all_fn is None
    ):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if cfg_wants:
                raise RuntimeError(
                    "MiniMax H3 CFG=2 x Context Parallel requires an initialized torch.distributed process group "
                    "(or explicit rank/process_group/collective callbacks for testing)."
                )
        else:
            sp_size = int(cp_cfg.sp_size)
            world_size = int(torch.distributed.get_world_size())
            if cfg_wants and world_size != 2 * sp_size:
                raise ValueError(
                    f"MiniMax H3 CFG=2 x Context Parallel SP={sp_size} requires world_size == {2 * sp_size} "
                    f"(got {world_size})"
                )
            # Support both active CFG (use_cfg=True) and cfg_scale=1.0 requests on a 2D
            # CFG=2 x SP=P deployment (world_size == 2 * sp_size) regardless of call order.
            if world_size == 2 * sp_size:
                sp_group, cfg_group = _get_or_init_2d_cfg_sp_groups(sp_size)
                if cfg_wants:
                    cfg_parallel_cfg = MiniMaxH3CFGParallelConfig(
                        enabled=True,
                        process_group=cfg_group,
                        rank=cfg_parallel_cfg.rank,
                        world_size=cfg_parallel_cfg.world_size,
                        all_gather_fn=cfg_parallel_cfg.all_gather_fn,
                    )
                cp_cfg = MiniMaxH3ContextParallelConfig(
                    sp_size=sp_size,
                    process_group=sp_group,
                    rank=cp_cfg.rank,
                    all_to_all_fn=cp_cfg.all_to_all_fn,
                    all_gather_fn=cp_cfg.all_gather_fn,
                    allow_cuda_graph=cp_cfg.allow_cuda_graph,
                )
                raw_cp = cp_cfg

    cfg_parallel_active, cfg_rank, cfg_world_size, cfg_group, cfg_all_gather = _resolve_cfg_parallel(
        cfg_parallel_cfg,
        use_cfg=use_cfg,
    )
    model_num_heads = getattr(model, "num_attention_heads", None)
    resolved_cp = resolve_minimax_h3_context_parallel(raw_cp, num_heads=model_num_heads) if cp_wants else None
    return (
        cfg_parallel_cfg,
        cfg_parallel_active,
        cfg_rank,
        cfg_world_size,
        cfg_group,
        cfg_all_gather,
        cp_cfg,
        resolved_cp,
        explicit_cp_disable,
    )


@torch.inference_mode()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_curr: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
    renoise: bool = False,
    noise_generator: torch.Generator | None = None,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if float(sigma_curr) == 0.0:
        return
    if renoise:
        noise = torch.randn(
            state.shape,
            device=state.device,
            dtype=state.dtype,
            generator=noise_generator,
        )
        state.copy_((1.0 - float(sigma_next)) * denoised_scratch + float(sigma_next) * noise)
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def _ulysses_ctx() -> tuple[int, int]:
    """Single-GPU: no sequence parallelism."""

    return 1, 0


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if seq_len % world_size:
        raise ValueError(f"packed seq_len {seq_len} not divisible by Ulysses world size {world_size}")
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero((pos >= row_start) & (pos < row_stop), as_tuple=False).view(-1)
        return source_ids.to(device), pos.index_select(0, source_ids).to(device)

    text_source_start = min(row_start, int(text_pos.shape[0]))
    text_source_stop = min(row_stop, int(text_pos.shape[0]))
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


class MiniMaxH3DenoiseBranch:
    """Static per-branch state: packed layout + fixed forward kwargs.

    ``packed`` is a :func:`minimax_h3_packed_sequence` result (or equivalent
    layout dict); ``text_embeddings`` is the branch's ``[text_len, 5120]``
    hidden states; ``token_tags`` must already carry any fl2va vision-span
    overrides.
    """

    def __init__(
        self,
        *,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(self.audio_pos.shape[0], dtype=torch.bool)
        # Packed layouts place reference rows before the generated suffix for
        # both modalities. Keep the suffix boundary once so the hot path can
        # update contiguous views instead of scattering static index tensors.
        self.video_target_start = int((~self.update_mask).sum())
        self.audio_target_start = int((~self.audio_update_mask).sum())
        self.video_target_slice = slice(self.video_target_start, None)
        self.audio_target_slice = slice(self.audio_target_start, None)
        text_pos = packed["text_pos"].view(-1).to(torch.long)
        text_len = int(text_pos.shape[0])
        if text_embeddings.shape[0] != text_len:
            raise ValueError(f"text_embeddings rows {list(text_embeddings.shape)} != packed text_len {text_len}")
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(f"token_tags length {int(token_tags.view(-1).shape[0])} != seq_len {seq_len}")
        cu = packed["cu_seqlens"].to(torch.int32)
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        # Resolve step-static packed-sequence and anchor row sets once.
        self.img_cond_seq_idx = self.img_pos_dev[~self.update_mask_dev]
        self.img_target_seq_idx = self.img_pos_dev[self.update_mask_dev]
        self.audio_target_seq_idx = self.audio_pos_dev[self.audio_update_mask_dev]
        self.audio_ref_seq_idx = self.audio_pos_dev[~self.audio_update_mask_dev]
        self.cond_row_idx = torch.nonzero(~self.update_mask_dev).view(-1)
        self.audio_ref_row_idx = torch.nonzero(~self.audio_update_mask_dev).view(-1)
        # rows that keep the video timestep each step: text, padding, video target.
        self.n_video_timestep_rows = (
            seq_len
            - int(self.img_cond_seq_idx.numel())
            - int(self.audio_target_seq_idx.numel())
            - int(self.audio_ref_seq_idx.numel())
        )
        # persistent packed-row buffers; every img/audio position is fully
        # rewritten on the first forward_kwargs() call, then only target rows.
        self._x_buffer_primed = False
        self.x_buffer = torch.zeros(1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.audio_x_buffer = torch.zeros(1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
        text_pos_dev = text_pos.to(device)
        ulysses_world_size, ulysses_rank = _ulysses_ctx()
        token_tags_host = token_tags.view(-1).to(dtype=torch.long)
        local_seq_len = seq_len // ulysses_world_size
        local_row_start = ulysses_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        self.local_row_slice = slice(local_row_start, local_row_stop)
        self.block_token_tags = token_tags_host[local_row_start:local_row_stop].clamp(min=0).to(device)
        refiner_bounds = [0, text_len]
        refiner_max = text_len
        self.static_kwargs: dict[str, Any] = {
            # Cast the fp64 position grid on the host; MiniMaxH3Rope casts to
            # fp32 anyway, and some devices cannot execute fp64.
            "img_position_ids": packed["img_position_ids"][None].to(torch.float32).to(device),
            "update_mask": self.update_mask_dev,
            "block_token_tags": self.block_token_tags,
            "skip_mask_out_condition": True,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_target_seq_idx},
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "cu_seqlens_q_host": tuple(int(value) for value in cu.tolist()),
                "max_seqlen_q": int(cu[1]),
                "ignore_trailing_pad": bool(int(cu[1]) < seq_len),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor(refiner_bounds, dtype=torch.int32, device=device),
                "cu_seqlens_q_host": tuple(refiner_bounds),
                "max_seqlen_q": refiner_max,
            },
        }

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        step_timesteps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        x = self.x_buffer
        audio_x = self.audio_x_buffer
        if not self._x_buffer_primed:
            # First step: condition/reference rows have been pinned into
            # video_rows/audio_rows and never change again, so write the full
            # img/audio extent once.
            x[0].index_copy_(0, self.img_pos_dev, video_rows)
            audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
            self._x_buffer_primed = True
        else:
            # Later steps: only the target-row subset changed.
            x[0].index_copy_(0, self.img_target_seq_idx, video_rows[self.video_target_slice])
            audio_x[0].index_copy_(0, self.audio_target_seq_idx, audio_rows[self.audio_target_slice])
        unique_timesteps, inverse_indices, block_combined_indices = step_timesteps
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "block_combined_indices": block_combined_indices,
        }

    def _expand_step_timesteps(
        self,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor],
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build step-local timestep and AdaLN index tensors.

        Non-media rows (text and padding) inherit the current video timestep;
        condition rows pin their noise-aug timesteps. The row->timestep layout
        is step-static, so the at-most-four candidate values are deduped in
        fp32 on the host and inverse indices are index_fill'ed from the static
        position sets.
        """

        candidates: list[float] = []
        fill_groups: list[tuple[torch.Tensor, int]] = []
        base_slot = -1
        if self.n_video_timestep_rows > 0:
            base_slot = len(candidates)
            candidates.append(float(t_video))
        for seq_idx, value in (
            (self.img_cond_seq_idx, imgvid_cond_timestep),
            (self.audio_target_seq_idx, t_audio),
            (self.audio_ref_seq_idx, audio_ref_cond_timestep),
        ):
            if seq_idx.numel() > 0:
                fill_groups.append((seq_idx, len(candidates)))
                candidates.append(float(value))
        unique_cpu, slot_to_unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[base_slot]) if base_slot >= 0 else 0
        pattern = tuple(slot_to_unique.tolist())
        inverse_indices = inverse_indices_by_pattern.get(pattern)
        if inverse_indices is None:
            inverse_indices = torch.full((self.seq_len,), base_index, dtype=torch.long, device=device)
            for seq_idx, slot in fill_groups:
                inverse_indices.index_fill_(0, seq_idx, int(slot_to_unique[slot]))
            inverse_indices_by_pattern[pattern] = inverse_indices
        block_combined = block_combined_by_pattern.get(pattern)
        if block_combined is None:
            block_combined = torch.add(
                self.block_token_tags,
                inverse_indices[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            block_combined_by_pattern[pattern] = block_combined
        return unique_cpu.to(device), inverse_indices, block_combined

    def prepare_timestep_plan(
        self,
        *,
        video_timesteps: list[float],
        audio_timesteps: list[float],
        imgvid_cond_noise_aug: float,
        audio_ref_cond_noise_aug: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Stage every step's packed timestep state before denoising."""

        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video/audio timestep plans must have equal length")
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        return [
            self._expand_step_timesteps(
                t_video=t_video,
                t_audio=t_audio,
                imgvid_cond_timestep=max(t_video, imgvid_cond_noise_aug),
                audio_ref_cond_timestep=max(t_audio, audio_ref_cond_noise_aug),
                inverse_indices_by_pattern=inverse_indices_by_pattern,
                block_combined_by_pattern=block_combined_by_pattern,
            )
            for t_video, t_audio in zip(video_timesteps, audio_timesteps, strict=False)
        ]


def minimax_h3_denoise_loop(
    *,
    model: Any,
    model_forward: (Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]] | None) = None,
    positive: MiniMaxH3DenoiseBranch,
    negative: MiniMaxH3DenoiseBranch | None = None,
    cfg_scale: float = 1.0,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    renoise: bool = False,
    noise_generator: torch.Generator | None = None,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
    step_cache: MiniMaxH3StepCacheConfig | Mapping[str, Any] | None = None,
    step_cache_report: dict[str, Any] | None = None,
    cfg_parallel: MiniMaxH3CFGParallelConfig | bool | None = None,
    cfg_parallel_report: dict[str, Any] | None = None,
    context_parallel: (
        MiniMaxH3ContextParallelConfig | _ResolvedContextParallel | Mapping[str, Any] | int | bool | None
    ) = None,
    context_parallel_report: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoise loop; returns final (video_rows, audio_rows).

    ``initial_video_rows`` covers all image rows of the positive layout. For a
    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. When ``negative`` is provided with
    ``cfg_scale != 1.0``, classifier-free guidance combines the positive and
    negative velocity predictions; otherwise the raw positive velocity is used.

    Optional SOL-Engine accelerations:
    - ``step_cache``: calibrated cross-step hidden-state residual cache
      (:class:`MiniMaxH3StepCacheConfig`) supporting ``easycache``,
      ``teacache``, and ``firstblock`` (SOL-Engine FirstBlockCache / FBCache).
    - ``cfg_parallel``: 2-GPU CFG=2 parallel execution
      (:class:`MiniMaxH3CFGParallelConfig` or ``True``) evaluating positive on
      rank 0 and negative on rank 1 with one fused FP32 ``all_gather`` per step.
    - ``context_parallel``: Ulysses sequence/head parallelism
      (:class:`MiniMaxH3ContextParallelConfig` or ``sp_size`` int such as ``2``
      or ``4``), working even when ``cfg_scale == 1.0`` and composable with
      ``cfg_parallel``, ``step_cache``, ``Sol-Attn``, and Hopper ``FP8``.
    """

    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    use_cfg = negative is not None and float(cfg_scale) != 1.0
    (
        cfg_parallel_cfg,
        cfg_parallel_active,
        cfg_rank,
        cfg_world_size,
        cfg_group,
        cfg_all_gather,
        cp_cfg,
        resolved_cp,
        explicit_cp_disable,
    ) = _resolve_parallel_topology(
        cfg_parallel=cfg_parallel,
        context_parallel=context_parallel,
        model=model,
        use_cfg=use_cfg,
    )
    cp_start_a2a = int(resolved_cp.all_to_all_calls) if resolved_cp is not None else 0
    cp_start_gather = int(resolved_cp.gather_calls) if resolved_cp is not None else 0
    cp_start_stats = int(resolved_cp.stats_gather_calls) if resolved_cp is not None else 0
    if use_cfg:
        assert negative is not None
        if negative.video_target_start != positive.video_target_start:
            raise ValueError("negative branch video condition row count must match positive branch")
        if negative.audio_target_start != positive.audio_target_start:
            raise ValueError("negative branch audio reference row count must match positive branch")
        if cfg_parallel_active:
            if int(negative.img_target_seq_idx.numel()) != int(positive.img_target_seq_idx.numel()):
                raise ValueError("negative branch video target row count must match positive branch for CFG parallel")
            if int(negative.audio_target_seq_idx.numel()) != int(positive.audio_target_seq_idx.numel()):
                raise ValueError("negative branch audio target row count must match positive branch for CFG parallel")

    n_cond = positive.video_target_start
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(f"layout has {n_cond} cond rows but keyframe_cond_rows is None")
    else:
        if int(keyframe_cond_rows.shape[0]) != n_cond:
            raise ValueError(f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != layout cond rows {n_cond}")
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32, copy=True)
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32, copy=True)
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = positive.audio_target_start
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None")
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout audio ref rows {n_audio_ref}")
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32) if keyframe_cond_rows is not None else None
    if cond_anchor is not None:
        video_rows.index_copy_(0, positive.cond_row_idx, cond_anchor)
    if audio_anchor is not None:
        audio_rows.index_copy_(0, positive.audio_ref_row_idx, audio_anchor)

    num_steps = len(sigmas_video) - 1
    video_target_slice = positive.video_target_slice
    audio_target_slice = positive.audio_target_slice
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
    audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)

    need_positive = (not cfg_parallel_active) or cfg_rank == 0
    need_negative = use_cfg and negative is not None and ((not cfg_parallel_active) or cfg_rank == 1)

    timestep_plan = (
        positive.prepare_timestep_plan(
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
            audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
        )
        if need_positive
        else None
    )
    negative_timestep_plan = (
        negative.prepare_timestep_plan(
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
            audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
        )
        if need_negative and negative is not None
        else None
    )
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
    audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
    audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
    video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])
    audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])

    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import MiniMaxH3DiTModel

    is_native_dit = isinstance(model, MiniMaxH3DiTModel)
    branch_cached_extras: dict[int, dict[str, Any]] = {}
    if is_native_dit:
        branches_to_prep: list[MiniMaxH3DenoiseBranch] = []
        if need_positive:
            branches_to_prep.append(positive)
        if need_negative and negative is not None:
            branches_to_prep.append(negative)
        with torch.inference_mode():
            for br in branches_to_prep:
                extras: dict[str, Any] = {}
                if "rope_cache" not in br.static_kwargs:
                    extras["rope_cache"] = model.build_rope_cache(
                        br.static_kwargs["img_position_ids"],
                        device=device,
                    )
                if "refined_prompt_embeds_length" not in br.static_kwargs:
                    refiner_psp = br.static_kwargs["refiner_packed_seq_params"]
                    refiner_host = refiner_psp.get("cu_seqlens_q_host")
                    refined = model.refine_prompt_embeds(
                        br.static_kwargs["prompt_embeds"],
                        refiner_psp["cu_seqlens_q"],
                        device=device,
                        refiner_cu_seqlens_host=refiner_host,
                    )
                    extras["prompt_embeds"] = refined
                    extras["refined_prompt_embeds_length"] = int(refined.shape[0])
                branch_cached_extras[id(br)] = extras

    dense_initial_steps = int(getattr(model, "_dense_initial_steps", 0)) if is_native_dit else 0

    resolved_cache_cfg = _coerce_step_cache_config(step_cache)
    branch_step_caches: dict[int, MiniMaxH3ResidualCache] = {}
    if resolved_cache_cfg is not None and is_native_dit:

        def _branch_probe_start(br: MiniMaxH3DenoiseBranch) -> int:
            text_len = int(br.static_kwargs["text_pos_info"]["position_ids"].shape[0])
            return text_len + int(br.video_target_start) + int(br.audio_target_start)

        if need_positive:
            branch_step_caches[id(positive)] = MiniMaxH3ResidualCache(
                resolved_cache_cfg,
                total_steps=num_steps,
                branch_name="positive",
                probe_start=_branch_probe_start(positive),
            )
        if need_negative and negative is not None:
            branch_step_caches[id(negative)] = MiniMaxH3ResidualCache(
                resolved_cache_cfg,
                total_steps=num_steps,
                branch_name="negative",
                probe_start=_branch_probe_start(negative),
            )

    # Pre-allocate reusable FP32 payload buffers for single-collective 2-GPU CFG exchange.
    cfg_video_elems = int(video_rows[video_target_slice].numel())
    cfg_audio_elems = int(audio_rows[audio_target_slice].numel())
    cfg_local_payload: torch.Tensor | None = None
    cfg_gathered_payloads: list[torch.Tensor] | None = None
    cfg_collectives = 0
    if cfg_parallel_active:
        cfg_local_payload = torch.empty(cfg_video_elems + cfg_audio_elems, dtype=torch.float32, device=device)
        cfg_gathered_payloads = [torch.empty_like(cfg_local_payload) for _ in range(cfg_world_size)]

    def _run_branch(branch: MiniMaxH3DenoiseBranch, step_t: tuple[torch.Tensor, torch.Tensor, torch.Tensor], step: int):
        fk = branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            step_timesteps=step_t,
        )
        extras = branch_cached_extras.get(id(branch))
        if extras:
            fk.update(extras)
        if dense_initial_steps > 0 and step < dense_initial_steps:
            fk["force_dense_attention"] = True
        cache_obj = branch_step_caches.get(id(branch))
        if cache_obj is not None:
            fk["step_cache"] = cache_obj
            fk["step_index"] = step
        if resolved_cp is not None:
            fk["context_parallel"] = resolved_cp
        elif explicit_cp_disable and is_native_dit:
            fk["context_parallel"] = False
        if model_forward is None:
            return model(**fk)
        return model_forward(model, fk, step)

    for step in range(num_steps):
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v = sigmas_video[step]
            s_a = sigmas_audio[step]

            with torch.inference_mode():
                video_target = video_rows[video_target_slice]
                audio_target = audio_rows[audio_target_slice]
                if use_cfg and negative is not None and cfg_parallel_active:
                    assert cfg_all_gather is not None
                    assert cfg_local_payload is not None and cfg_gathered_payloads is not None
                    if cfg_rank == 0:
                        assert timestep_plan is not None
                        local_branch = positive
                        local_step_t = timestep_plan[step]
                    else:
                        assert negative_timestep_plan is not None
                        local_branch = negative
                        local_step_t = negative_timestep_plan[step]
                    v_video_local, v_audio_local = _run_branch(local_branch, local_step_t, step)
                    cfg_local_payload[:cfg_video_elems].copy_(v_video_local.float().reshape(-1))
                    cfg_local_payload[cfg_video_elems:].copy_(v_audio_local[audio_target_slice].float().reshape(-1))
                    cfg_all_gather(cfg_gathered_payloads, cfg_local_payload, cfg_group)
                    cfg_collectives += 1
                    pos_payload, neg_payload = cfg_gathered_payloads[0], cfg_gathered_payloads[1]
                    v_video_pos = pos_payload[:cfg_video_elems].view_as(video_target)
                    v_video_neg = neg_payload[:cfg_video_elems].view_as(video_target)
                    mv_audio_pos = pos_payload[cfg_video_elems:].view_as(audio_target)
                    mv_audio_neg = neg_payload[cfg_video_elems:].view_as(audio_target)
                    scale = float(cfg_scale)
                    mv_video_t = v_video_neg + scale * (v_video_pos - v_video_neg)
                    mv_audio_t = mv_audio_neg + scale * (mv_audio_pos - mv_audio_neg)
                elif use_cfg and negative is not None and negative_timestep_plan is not None:
                    assert timestep_plan is not None
                    v_video_pos, v_audio_pos = _run_branch(positive, timestep_plan[step], step)
                    v_video_neg, v_audio_neg = _run_branch(negative, negative_timestep_plan[step], step)
                    scale = float(cfg_scale)
                    mv_video_t = v_video_neg.float() + scale * (v_video_pos.float() - v_video_neg.float())
                    mv_audio_pos = v_audio_pos[audio_target_slice].float()
                    mv_audio_neg = v_audio_neg[audio_target_slice].float()
                    mv_audio_t = mv_audio_neg + scale * (mv_audio_pos - mv_audio_neg)
                else:
                    assert timestep_plan is not None
                    v_video, v_audio = _run_branch(positive, timestep_plan[step], step)
                    mv_video_t = v_video.float()
                    mv_audio_t = v_audio[audio_target_slice].float()

                _minimax_h3_update_target_rows_(
                    video_target,
                    mv_video_t,
                    sigma_t=video_sigma_t[step],
                    sigma_curr=s_v,
                    sigma_next=sigmas_video[step + 1],
                    sigma_ratio=video_sigma_ratios[step],
                    one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
                    denoised_scratch=video_denoised_scratch,
                    renoise=renoise and step < num_steps - 1,
                    noise_generator=noise_generator,
                )

                _minimax_h3_update_target_rows_(
                    audio_target,
                    mv_audio_t,
                    sigma_t=audio_sigma_t[step],
                    sigma_curr=s_a,
                    sigma_next=sigmas_audio[step + 1],
                    sigma_ratio=audio_sigma_ratios[step],
                    one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
                    denoised_scratch=audio_denoised_scratch,
                    renoise=renoise and step < num_steps - 1,
                    noise_generator=noise_generator,
                )
            if on_step is not None:
                on_step(step, video_rows, audio_rows)

    if step_cache_report is not None:
        step_cache_report.clear()
        if branch_step_caches:
            branches_rep = {cache.branch_name: cache.receipt() for cache in branch_step_caches.values()}
            tot = sum(int(rep["total_steps"]) for rep in branches_rep.values())
            comp = sum(int(rep["computed_steps"]) for rep in branches_rep.values())
            cached = sum(int(rep["cached_steps"]) for rep in branches_rep.values())
            step_cache_report.update(
                {
                    "enabled": True,
                    "total_steps": tot,
                    "computed_steps": comp,
                    "cached_steps": cached,
                    "skip_ratio": (cached / tot) if tot > 0 else 0.0,
                    "branches": branches_rep,
                }
            )
        else:
            step_cache_report["enabled"] = False

    if cfg_parallel_report is not None:
        cfg_parallel_report.clear()
        cfg_parallel_report.update(
            {
                "enabled": bool(cfg_parallel_cfg is not None and cfg_parallel_cfg.enabled),
                "active": bool(cfg_parallel_active),
                "rank": int(cfg_rank),
                "world_size": int(cfg_world_size),
                "collectives": int(cfg_collectives),
            }
        )

    if context_parallel_report is not None:
        context_parallel_report.clear()
        model_num_heads = getattr(model, "num_attention_heads", None)
        context_parallel_report.update(
            {
                "enabled": bool(cp_cfg is not None and cp_cfg.sp_size > 1),
                "active": bool(resolved_cp is not None),
                "sp_size": int(resolved_cp.sp_size) if resolved_cp is not None else 1,
                "rank": int(resolved_cp.rank) if resolved_cp is not None else 0,
                "local_heads": (
                    int(model_num_heads // resolved_cp.sp_size)
                    if (resolved_cp is not None and model_num_heads is not None)
                    else (int(model_num_heads) if model_num_heads is not None else None)
                ),
                "all_to_all_calls": (
                    int(resolved_cp.all_to_all_calls) - cp_start_a2a if resolved_cp is not None else 0
                ),
                "gather_calls": (int(resolved_cp.gather_calls) - cp_start_gather if resolved_cp is not None else 0),
                "stats_gather_calls": (
                    int(resolved_cp.stats_gather_calls) - cp_start_stats if resolved_cp is not None else 0
                ),
            }
        )

    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3CFGParallelConfig",
    "MiniMaxH3ContextParallelConfig",
    "MiniMaxH3DenoiseBranch",
    "MiniMaxH3ResidualCache",
    "MiniMaxH3StepCacheConfig",
    "minimax_h3_denoise_loop",
]
