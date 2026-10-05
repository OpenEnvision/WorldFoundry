"""Pure-tensor packing of Cosmos3 text / vision / sound / action tokens.

No modules or weights: this file only computes index tensors.  The joint
denoiser calls :func:`build_cosmos3_sequence_layout` with tokenizer
``input_ids`` and a ``ModalityState`` map.  Returned
:class:`Cosmos3SequenceLayout.values` keys match
:class:`~...networks.cosmos3.model.Cosmos3OmniTransformer` kwargs
(``text_indexes``, ``vision_sequence_indexes``, per-modality timesteps,
MSE-loss indexes, RoPE ``position_ids``).

Temporal positions can be integer or FPS-scaled floats.  Spatial indices
optionally reset per modality.  Noisy-frame indexes come from
``ModalityState.denoise_mask`` so frozen context tokens stay clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import torch

from ....contracts import ModalityState


def _text_positions(
    num_tokens: int,
    temporal_offset: int | float,
    *,
    use_float_positions: bool,
) -> tuple[torch.Tensor, int | float]:
    dtype = torch.float32 if use_float_positions else torch.long
    ids = torch.arange(num_tokens, dtype=dtype) + temporal_offset
    return ids.unsqueeze(0).expand(3, -1).contiguous(), temporal_offset + num_tokens


def _modality_positions(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    *,
    reset_spatial_indices: bool,
    fps: float | None,
    base_fps: float,
    temporal_compression_factor: int,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> torch.Tensor:
    base_tcf = base_temporal_compression_factor or temporal_compression_factor
    if fps is not None and grid_t > 1:
        tokens_per_second = fps / temporal_compression_factor
        base_tokens_per_second = base_fps / base_tcf
        frames = torch.arange(grid_t, dtype=torch.float32)
        temporal = (frames + start_frame_offset) / tokens_per_second * base_tokens_per_second + temporal_offset
        temporal = temporal.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        temporal = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )
    height = torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    width = torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if not reset_spatial_indices:
        height = height + int(temporal_offset)
        width = width + int(temporal_offset)
    if temporal.is_floating_point():
        height = height.float()
        width = width.float()
    return torch.stack((temporal, height, width))


def _noisy_frames(state: ModalityState, *, temporal_dim: int) -> torch.Tensor:
    mask = state.denoise_mask
    if mask.shape != state.latent.shape:
        mask = torch.broadcast_to(mask, state.latent.shape)
    reduce_dims = tuple(index for index in range(mask.ndim) if index != temporal_dim)
    activity = mask.amax(dim=reduce_dims)
    if activity.ndim != 1:
        raise ValueError("Cosmos3 currently requires one sample per joint sequence")
    return torch.nonzero(activity > 0, as_tuple=False).flatten()


@dataclass(frozen=True, slots=True)
class Cosmos3SequenceLayout:
    """Static transformer arguments for one prompt and modality shape set."""

    values: Mapping[str, object]


def build_cosmos3_sequence_layout(
    input_ids: torch.Tensor,
    states: Mapping[str, ModalityState],
    config: SimpleNamespace,
    *,
    fps: float,
    temporal_compression_factor: int = 4,
) -> Cosmos3SequenceLayout:
    """Build text, vision, sound, and action indexes without owning model layers."""

    if input_ids.ndim != 1:
        raise ValueError("Cosmos3 token IDs must be a one-dimensional single-sample sequence")
    try:
        video = states["video"]
    except KeyError as error:
        raise KeyError("Cosmos3 joint state requires a video modality") from error
    if video.latent.ndim != 5 or video.latent.shape[0] != 1:
        raise ValueError("Cosmos3 video latents must have shape [1, C, T, H, W]")

    device = video.latent.device
    input_ids = input_ids.to(device=device, dtype=torch.long)
    und_len = int(input_ids.numel())
    text_mrope, offset = _text_positions(
        und_len,
        0,
        use_float_positions=bool(config.enable_fps_modulation),
    )
    offset += int(config.unified_3d_mrope_temporal_modality_margin)
    values: dict[str, object] = {
        "input_ids": input_ids,
        "text_indexes": torch.arange(und_len, device=device),
        "und_len": und_len,
    }
    mrope = [text_mrope]
    cursor = und_len

    patch = int(config.latent_patch_size)
    _, _, frames, height, width = video.latent.shape
    patch_h = math.ceil(height / patch)
    patch_w = math.ceil(width / patch)
    frame_stride = patch_h * patch_w
    noisy_video = _noisy_frames(video, temporal_dim=2).to(device)
    video_count = frames * frame_stride
    video_indexes = torch.arange(cursor, cursor + video_count, device=device)
    video_mse = (
        torch.cat(
            [video_indexes[index * frame_stride : (index + 1) * frame_stride] for index in noisy_video.tolist()],
            dim=0,
        )
        if noisy_video.numel()
        else video_indexes[:0]
    )
    values.update(
        vision_token_shapes=[(frames, patch_h, patch_w)],
        vision_sequence_indexes=video_indexes,
        vision_mse_loss_indexes=video_mse,
        vision_noisy_frame_indexes=[noisy_video],
        num_noisy_vision_tokens=int(video_mse.numel()),
    )
    mrope.append(
        _modality_positions(
            frames,
            patch_h,
            patch_w,
            offset,
            reset_spatial_indices=bool(config.unified_3d_mrope_reset_spatial_ids),
            fps=fps if config.enable_fps_modulation else None,
            base_fps=float(config.base_fps),
            temporal_compression_factor=temporal_compression_factor,
        )
    )
    cursor += video_count

    sound = states.get("sound")
    if sound is not None:
        if sound.latent.ndim != 2:
            raise ValueError("Cosmos3 sound latents must have shape [C, T]")
        sound_frames = int(sound.latent.shape[1])
        sound_indexes = torch.arange(cursor, cursor + sound_frames, device=device)
        noisy_sound = _noisy_frames(sound, temporal_dim=1).to(device)
        values.update(
            sound_token_shapes=[(sound_frames, 1, 1)],
            sound_sequence_indexes=sound_indexes,
            sound_mse_loss_indexes=sound_indexes[noisy_sound],
            sound_noisy_frame_indexes=[noisy_sound],
            num_noisy_sound_tokens=int(noisy_sound.numel()),
        )
        mrope.append(
            _modality_positions(
                sound_frames,
                1,
                1,
                offset,
                reset_spatial_indices=bool(config.unified_3d_mrope_reset_spatial_ids),
                fps=float(config.sound_latent_fps) if config.enable_fps_modulation else None,
                base_fps=float(config.base_fps),
                temporal_compression_factor=1,
            )
        )
        cursor += sound_frames

    action = states.get("action")
    if action is not None:
        if action.latent.ndim != 2:
            raise ValueError("Cosmos3 action latents must have shape [T, D]")
        action_frames = int(action.latent.shape[0])
        action_indexes = torch.arange(cursor, cursor + action_frames, device=device)
        noisy_action = _noisy_frames(action, temporal_dim=0).to(device)
        values.update(
            action_token_shapes=[(action_frames, 1, 1)],
            action_sequence_indexes=action_indexes,
            action_mse_loss_indexes=action_indexes[noisy_action],
            action_noisy_frame_indexes=[noisy_action],
            num_noisy_action_tokens=int(noisy_action.numel()),
        )
        mrope.append(
            _modality_positions(
                action_frames,
                1,
                1,
                offset,
                reset_spatial_indices=bool(config.unified_3d_mrope_reset_spatial_ids),
                fps=fps if config.enable_fps_modulation else None,
                base_fps=float(config.base_fps),
                temporal_compression_factor=1,
                base_temporal_compression_factor=temporal_compression_factor,
                start_frame_offset=1,
            )
        )
        cursor += action_frames

    values["position_ids"] = torch.cat(mrope, dim=1).to(device)
    values["sequence_length"] = cursor
    return Cosmos3SequenceLayout(values)


__all__ = ["Cosmos3SequenceLayout", "build_cosmos3_sequence_layout"]
