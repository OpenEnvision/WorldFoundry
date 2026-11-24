"""SolarWM auto-tiling memory budget for the canonical diffusion decoder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_tiling import (
    _axis_candidates,
    _round_up,
    recommended_pixel_overlaps,
    stage4_feature_bytes,
    stage5_tokens_for_pixel_tile,
    volumetric_overlap_waste,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.apply import (
    resolve_attention_for_host,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import (
    DiffVAEMode,
    NAttentionKind,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.tiling import DimensionSizeConfig
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.tiling import TileSizeConfig

from ...types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors

# Peak-activation heuristic (bytes):
#   hard (resident for whole tiled decode):
#     stage-4 input feature (stages 1-3 output): s4_txs4_hxs4_wxstage4_channelsxbf16
#   per temporal group + stage-5 tile:
#     accumulator: H x W x (2 x tile_t) x out_channels x accum_elem
#       (full-frame spatial; ~current group buffer + retained overlap stub / still-
#        live emitted exclusive chunk heading to encode; RGB by default; accum_elem
#        matches decode: fp16 when features are bf16, else feature dtype)
#     stage-5: stage5_tokens x stage5_channels x bf16 x coef
# where stage5_tokens = F x (H/patch) x (W/patch) and coef folds NA working-set multiplicity.
_MEM_COEF_BY_MODE: dict[DiffVAEMode, float] = {
    DiffVAEMode.COMBINED_COMPILE: 11,
    DiffVAEMode.CHUNKED_COMPILE: 7,  # natten backend; Triton/eager use CHUNKED_EAGER
    DiffVAEMode.CHUNKED_EAGER: 5,
    DiffVAEMode.BLACKWELL_DSL: 2.5,
}
_DEFAULT_ELEMENT_SIZE: int = 2  # bf16 features → fp16 accumulator / bf16 stage-5
_ACCUMULATOR_CHANNELS: int = 3  # RGB pixel blend buffer (decoder out_channels)
_MIN_MODEL_BYTES_FLOOR: int = 1 << 30  # never assume a free DiffVAE weight footprint
_BUDGET_SAFETY_BYTES_EAGER: int = 1 << 30
_BUDGET_SAFETY_BYTES_COMPILED: int = 2 << 30


def stage5_mem_coef(mode: DiffVAEMode) -> float:
    """Stage-5 working-set multiplicity for auto tiling, after host NA resolve.
    ``CHUNKED_COMPILE``'s coef 7 assumes natten. When the host remaps chunked
    modes to Triton/eager fallback, use the ``CHUNKED_EAGER`` coef (5) instead.
    ``COMBINED_COMPILE`` requires natten and keeps coef 11.
    """
    try:
        base = _MEM_COEF_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported DiffVAEMode for tiling budget: {mode!r}") from exc
    resolved = resolve_attention_for_host(mode.resolve())
    if resolved.attention in (NAttentionKind.TRITON, NAttentionKind.EAGER_SDPA):
        return _MEM_COEF_BY_MODE[DiffVAEMode.CHUNKED_EAGER]
    return base


def budget_safety_bytes(mode: DiffVAEMode) -> int:
    """Extra bytes withheld from the recommend budget (eager 1 GiB, compiled 2 GiB)."""
    if mode is DiffVAEMode.CHUNKED_EAGER:
        return _BUDGET_SAFETY_BYTES_EAGER
    # Chunked compile that remaps to fallback NA is effectively eager for peak VRAM.
    resolved = resolve_attention_for_host(mode.resolve())
    if resolved.attention in (NAttentionKind.TRITON, NAttentionKind.EAGER_SDPA):
        return _BUDGET_SAFETY_BYTES_EAGER
    return _BUDGET_SAFETY_BYTES_COMPILED


def accumulator_element_size(feature_dtype: torch.dtype) -> int:
    """Bytes per accumulator element; mirrors ``_decode_temporal_group_isolated``.
    ``accum_dtype = float16 if feat_s4.dtype == bfloat16 else feat_s4.dtype``.
    """
    if feature_dtype is torch.bfloat16:
        return 2  # stored as fp16
    return int(torch.tensor([], dtype=feature_dtype).element_size())


# ---------------------------------------------------------------------------
# Public entry points (pipeline recommend / decode schedule)
# ---------------------------------------------------------------------------


def recommended_decode_tiling_config(
    *,
    tile_halos: tuple[tuple[int, int, int], tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
    min_tile_size_s4: tuple[int, int, int],
    patch_size: int,
    height: int,
    width: int,
    num_frames: int,
    mode: DiffVAEMode,
    free_bytes: int,
    stage5_channels: int,
    stage4_channels: int,
    upsample_strides: Sequence[tuple[int, int, int]],
    model_bytes: int = 0,
    element_size: int = _DEFAULT_ELEMENT_SIZE,
    natten_trailing_pad_latent_frames: int = 0,
    out_channels: int = _ACCUMULATOR_CHANNELS,
) -> TileSizeConfig:
    """Pick DiffVAE decode tiling from stage-4/5 halos and free VRAM.
    Always enables both spatial and temporal tiling (temporal-only full-frame slabs
    are unsafe on Hopper / some natten builds).
    Selection (size-grid, accumulator-aware):
      1. Enumerate legal tile **sizes** on the LCM of DiffVAE ``pixel_scale`` and
         :data:`~ltx_core.types.VIDEO_SCALE_FACTORS` (so configs also pass
         :class:`~ltx_core.tiling.TileSizeConfig` construction); derive tile
         counts from :func:`~ltx_core.tiling.split_by_size` (same as decode).
      2. Drop triples whose peak-bytes estimate exceeds ``usable`` bytes
         (``free - max(model, 1 GiB) - safety - stage4_feature``; safety is
         1 GiB eager / 2 GiB compiled). Stage-4 input features stay resident
         for the whole tiled decode.
      3. Among feasible triples, pick minimal :func:`volumetric_overlap_waste`.
    Peak-bytes estimate::
        stage4_feature_bytes(...)                         # hard, full volume
        + H * W * (2 * tile_t) * out_channels * element_size
        + stage5_tokens * stage5_channels * element_size * coef
    Accumulator is full output HxW (not spatially tiled) with temporal extent
    ``2 * tile_t``: current group buffer plus the still-live previous exclusive
    emit / overlap stub during handoff (not merely ``tile_t + overlap_t``).
    RGBx``element_size`` by default. ``element_size`` is the activation width:
    production bf16 features use fp16 accumulators (2), matching
    :func:`accumulator_element_size`. Stage-5 uses the same element size x
    ``stage5_channels`` x ``coef`` (11 / 7 / 5 / 2.5 by mode).
    """
    if height < 1 or width < 1 or num_frames < 1:
        raise ValueError(f"height/width/num_frames must be >= 1, got {height}x{width}x{num_frames}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if stage5_channels < 1:
        raise ValueError(f"stage5_channels must be >= 1, got {stage5_channels}")
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    if element_size < 1:
        raise ValueError(f"element_size must be >= 1, got {element_size}")

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)

    ft, fh, fw = pixel_scale.time, pixel_scale.height, pixel_scale.width
    # Construction validates fixed 8/32/32; to_splitters uses pixel_scale - step both.
    step_t = math.lcm(ft, VIDEO_SCALE_FACTORS.time)
    step_h = math.lcm(fh, VIDEO_SCALE_FACTORS.height)
    step_w = math.lcm(fw, VIDEO_SCALE_FACTORS.width)
    min_t_px = _round_up(
        # ``2 * overlap`` so left+right ramps fit (else masks are not complementary and
        # decode allocates a full weights buffer ≈ another accumulator).
        max(2 * ft, 2 * overlap_t, _round_up(min_tile_size_s4[0] * ft, ft), 16),
        step_t,
    )
    min_h_px = _round_up(
        max(2 * fh, 2 * overlap_hw, _round_up(min_tile_size_s4[1] * fh, fh), 64),
        step_h,
    )
    min_w_px = _round_up(
        max(2 * fw, 2 * overlap_hw, _round_up(min_tile_size_s4[2] * fw, fw), 64),
        step_w,
    )

    model_cost = max(int(model_bytes), _MIN_MODEL_BYTES_FLOOR)
    coef = stage5_mem_coef(mode)
    s4_feat_bytes = stage4_feature_bytes(
        height=height,
        width=width,
        num_frames=num_frames,
        upsample_strides=upsample_strides,
        stage4_channels=stage4_channels,
        element_size=element_size,
        natten_trailing_pad_latent_frames=natten_trailing_pad_latent_frames,
    )
    usable = max(0, int(free_bytes) - model_cost - budget_safety_bytes(mode) - s4_feat_bytes)
    s5_bytes_per_token = max(1.0, float(stage5_channels) * float(element_size) * coef)
    acc_bytes_per_pixel = int(out_channels) * int(element_size)

    t_cands = _axis_candidates(num_frames, overlap_t, min_t_px, step_t)
    h_cands = _axis_candidates(height, overlap_hw, min_h_px, step_h)
    w_cands = _axis_candidates(width, overlap_hw, min_w_px, step_w)

    scored: list[tuple[float, int, int, int, int, int]] = []
    # (waste, -volume, n_t*n_h*n_w, tile_t, tile_h, tile_w) - minimize waste, then launches.
    for tile_t, n_t in t_cands:
        # Current group buffer + still-live emit/stub during temporal handoff.
        acc_frames = 2 * int(tile_t)
        acc_bytes = acc_frames * int(height) * int(width) * acc_bytes_per_pixel
        if acc_bytes >= usable:
            continue
        s5_budget_bytes = usable - acc_bytes
        max_s5_tokens = int(s5_budget_bytes // s5_bytes_per_token)
        for tile_h, n_h in h_cands:
            for tile_w, n_w in w_cands:
                if stage5_tokens_for_pixel_tile(tile_t, tile_h, tile_w, patch_size=patch_size) > max_s5_tokens:
                    continue
                waste = volumetric_overlap_waste(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tile_frames=tile_t,
                    tile_height=tile_h,
                    tile_width=tile_w,
                    n_t=n_t,
                    n_h=n_h,
                    n_w=n_w,
                )
                scored.append((waste, -tile_t * tile_h * tile_w, n_t * n_h * n_w, tile_t, tile_h, tile_w))

    if not scored:
        raise ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            f"min tile ~{min_t_px}f x {min_h_px}x{min_w_px}px "
            f"(overlaps T={overlap_t}, HW={overlap_hw}), "
            f"mode={mode.value}, coef={coef}, stage5_channels={stage5_channels}, "
            f"stage4_feature_bytes={s4_feat_bytes}, usable_bytes={usable}. "
            "Reduce resolution or free GPU memory."
        )

    scored.sort()
    _waste, _vol, _ntiles, tile_t, tile_h, tile_w = scored[0]
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=tile_t, overlap=overlap_t),
        height=DimensionSizeConfig(tile_size=tile_h, overlap=overlap_hw),
        width=DimensionSizeConfig(tile_size=tile_w, overlap=overlap_hw),
    )


# ---------------------------------------------------------------------------
# Common helpers (geometry, pad/crop, stage floors)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
