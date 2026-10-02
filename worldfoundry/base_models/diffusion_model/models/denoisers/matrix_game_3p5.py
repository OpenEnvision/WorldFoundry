"""Matrix Game 3.5 denoiser component for the native diffusion runtime.

The tensor preparation below is model computation: it assembles Mosaic memory,
PRoPE camera features, subject-reference tokens, and the Matrix-specific token
sequence before evaluating the DiT.  Sampling, checkpoint discovery, placement,
and offload remain framework-owned.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import torch
from einops import rearrange

from worldfoundry.core.attention.rotary.projective_rope import invert_k, invert_se3, lift_k
from worldfoundry.core.nn import RMSNorm, sinusoidal_embedding_1d

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.matrix_game_3p5.model import WanModel

MATRIX_GAME_35_DIT_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "subject_ref_memory_enabled": True,
    "subject_ref_memory_max_refs": 2,
}


def _resolve_mosaic_frame_indices(
    mosaic_frame_indices,
    *,
    noisy_frame_count: int,
    mosaic_frame_count: int,
    device,
):
    if mosaic_frame_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if mosaic_frame_indices is None:
        if mosaic_frame_count != noisy_frame_count:
            raise ValueError(
                "mosaic_frame_indices is required when mosaic_frame_count "
                f"({mosaic_frame_count}) differs from noisy_frame_count ({noisy_frame_count})."
            )
        return torch.arange(noisy_frame_count, dtype=torch.long, device=device)
    if not torch.is_tensor(mosaic_frame_indices):
        mosaic_frame_indices = torch.as_tensor(mosaic_frame_indices)
    indices = mosaic_frame_indices.to(device=device, dtype=torch.long).reshape(-1)
    if int(indices.numel()) != int(mosaic_frame_count):
        raise ValueError(
            "mosaic_frame_indices length must match mosaic_frame_count, got "
            f"{int(indices.numel())} vs {int(mosaic_frame_count)}."
        )
    if indices.numel() and (int(indices.min().item()) < 0 or int(indices.max().item()) >= int(noisy_frame_count)):
        raise ValueError(f"mosaic_frame_indices must be within the noisy latent range [0, {int(noisy_frame_count)}).")
    return indices


def _resolve_latent_rope_time_indices(
    latent_rope_time_indices,
    *,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    mosaic_frame_indices,
    device,
):
    total_count = int(first_frame_count) + int(mosaic_frame_count) + int(noisy_frame_count)
    if latent_rope_time_indices is not None:
        if not torch.is_tensor(latent_rope_time_indices):
            latent_rope_time_indices = torch.as_tensor(latent_rope_time_indices)
        indices = latent_rope_time_indices.to(device=device, dtype=torch.long).reshape(-1)
        if int(indices.numel()) != int(total_count):
            raise ValueError(
                "latent_rope_time_indices length must match "
                "first_frame_count + mosaic_frame_count + noisy_frame_count, got "
                f"{int(indices.numel())} vs {int(total_count)}."
            )
        if indices.numel() and int(indices.min().item()) < 0:
            raise ValueError("latent_rope_time_indices must be non-negative.")
        return indices

    first_times = torch.arange(int(first_frame_count), dtype=torch.long, device=device)
    noisy_times = int(first_frame_count) + torch.arange(int(noisy_frame_count), dtype=torch.long, device=device)
    if int(mosaic_frame_count) > 0:
        mosaic_times = int(first_frame_count) + _resolve_mosaic_frame_indices(
            mosaic_frame_indices,
            noisy_frame_count=int(noisy_frame_count),
            mosaic_frame_count=int(mosaic_frame_count),
            device=device,
        )
    else:
        mosaic_times = torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat([first_times, mosaic_times, noisy_times], dim=0)


def _build_mosaic_cross_attn_keep_mask(
    *,
    prefix_memory_token_count: int = 0,
    reference_token_count: int,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    tokens_per_frame: int,
    device,
):
    prefix_memory_token_count = int(prefix_memory_token_count)
    total = (
        prefix_memory_token_count
        + int(reference_token_count)
        + (int(first_frame_count) + int(mosaic_frame_count) + int(noisy_frame_count)) * int(tokens_per_frame)
    )
    mask = torch.ones(total, dtype=torch.bool, device=device)
    if prefix_memory_token_count > 0:
        mask[:prefix_memory_token_count] = False
    if mosaic_frame_count > 0:
        start = prefix_memory_token_count + int(reference_token_count) + int(first_frame_count) * int(tokens_per_frame)
        end = start + int(mosaic_frame_count) * int(tokens_per_frame)
        mask[start:end] = False
    return mask


def _build_subject_ref_memory_tokens(
    dit: WanModel,
    subject_ref_latents: Optional[torch.Tensor],
    *,
    batch_size: int,
    video_h: int,
    video_w: int,
    subject_ref_slot_ratio: float,
    subject_ref_time_gap: int,
    device,
    dtype,
):
    if subject_ref_latents is None:
        return None
    if not getattr(dit, "subject_ref_memory_enabled", False):
        return None
    required_attrs = (
        "subject_ref_index_embedding",
        "subject_ref_type_embedding",
        "subject_ref_local_h_embedding",
        "subject_ref_local_w_embedding",
    )
    missing_attrs = [name for name in required_attrs if not hasattr(dit, name)]
    if missing_attrs:
        raise ValueError(
            "subject_ref_latents were provided, but the DiT has no "
            f"{missing_attrs}. Enable subject ref memory before loading."
        )
    refs = subject_ref_latents
    if not torch.is_tensor(refs):
        refs = torch.as_tensor(refs)
    if refs.ndim == 5:
        # Materialization stores (R, C, 1, H, W), while direct callers may
        # pass (B, C, R, H, W). Batch size > 1 is intentionally unsupported
        # for the current variable-ref-count path.
        if int(refs.shape[2]) == 1 and int(refs.shape[0]) != 1:
            refs = refs.permute(2, 1, 0, 3, 4).contiguous()
        elif int(refs.shape[0]) in (1, int(batch_size)):
            refs = refs.contiguous()
        else:
            raise ValueError(f"subject_ref_latents expects (R,C,1,H,W) or (1,C,R,H,W), got {tuple(refs.shape)}.")
    elif refs.ndim == 4:
        refs = refs.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    else:
        raise ValueError(f"subject_ref_latents expects 4 or 5 dims, got {tuple(refs.shape)}.")
    if int(refs.shape[0]) not in (1, int(batch_size)):
        raise ValueError(
            "subject_ref_latents batch size must be 1 or match model batch, got "
            f"{int(refs.shape[0])} vs {int(batch_size)}."
        )
    ref_count = int(refs.shape[2])
    if ref_count <= 0:
        return None
    max_refs = int(dit.subject_ref_index_embedding.shape[0])
    if ref_count > max_refs:
        refs = refs[:, :, :max_refs]
        ref_count = max_refs
    refs = refs.to(device=device, dtype=dtype)
    ref_x = dit.patchify(refs)
    if int(ref_x.shape[0]) == 1 and int(batch_size) > 1:
        ref_x = ref_x.expand(int(batch_size), -1, -1, -1, -1)
    _, _, _, ref_h, ref_w = ref_x.shape

    ratio = min(1.0, max(0.01, float(subject_ref_slot_ratio)))
    slot_size = int(round(min(int(video_h), int(video_w)) * ratio))
    if slot_size <= 0:
        return None
    slot_h = max(1, int(round(ref_h * slot_size / float(video_h))))
    slot_w = max(1, int(round(ref_w * slot_size / float(video_w))))
    slot_h = min(slot_h, ref_h)
    slot_w = min(slot_w, ref_w)
    h_start = int(ref_h - slot_h)
    w_start = int(ref_w - slot_w)

    ref_x = ref_x[:, :, :, h_start:ref_h, w_start:ref_w]
    ref_x = rearrange(ref_x, "b c r h w -> b r h w c").contiguous()

    ref_index_pos = dit.subject_ref_index_embedding[:ref_count].to(device=device, dtype=ref_x.dtype)
    ref_type_pos = dit.subject_ref_type_embedding.to(device=device, dtype=ref_x.dtype)
    local_h_pos = _subject_ref_local_pos(dit.subject_ref_local_h_embedding, slot_h, device=device, dtype=ref_x.dtype)
    local_w_pos = _subject_ref_local_pos(dit.subject_ref_local_w_embedding, slot_w, device=device, dtype=ref_x.dtype)
    local_pos = local_h_pos.view(1, 1, slot_h, 1, -1) + local_w_pos.view(1, 1, 1, slot_w, -1)
    ref_x = ref_x + ref_type_pos.view(1, 1, 1, 1, -1) + ref_index_pos.view(1, ref_count, 1, 1, -1) + local_pos
    ref_x = rearrange(ref_x, "b r h w c -> b (r h w) c").contiguous()

    ref_freqs = _build_subject_ref_time_freqs(
        dit,
        ref_count=ref_count,
        slot_h=slot_h,
        slot_w=slot_w,
        subject_ref_time_gap=subject_ref_time_gap,
        device=device,
    )
    return {
        "x": ref_x,
        "freqs": ref_freqs,
        "token_count": int(ref_x.shape[1]),
        "ref_count": ref_count,
        "slot_grid": (int(slot_h), int(slot_w)),
        "slot_start": (int(h_start), int(w_start)),
    }


def _subject_ref_local_pos(table, length: int, *, device, dtype):
    table = table.to(device=device, dtype=dtype)
    length = int(length)
    if length <= int(table.shape[0]):
        return table[:length]
    pos = torch.nn.functional.interpolate(
        table.float().transpose(0, 1).unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=True,
    )
    return pos.squeeze(0).transpose(0, 1).to(dtype=dtype)


def _build_subject_ref_time_freqs(
    dit: WanModel,
    *,
    ref_count: int,
    slot_h: int,
    slot_w: int,
    subject_ref_time_gap: int,
    device,
):
    freq_dev = dit.freqs[0].device
    time_gap = max(1, int(subject_ref_time_gap))
    ref_time_indices = (torch.arange(1, ref_count + 1, device=freq_dev, dtype=torch.long) * time_gap).clamp(
        max=int(dit.freqs[0].shape[0]) - 1
    )
    time_freqs = dit.freqs[0][ref_time_indices].conj()
    h_freqs = dit.freqs[1][:1].expand(slot_h, -1)
    w_freqs = dit.freqs[2][:1].expand(slot_w, -1)
    ref_freqs = torch.cat(
        [
            time_freqs.view(ref_count, 1, 1, -1).expand(ref_count, slot_h, slot_w, -1),
            h_freqs.view(1, slot_h, 1, -1).expand(ref_count, slot_h, slot_w, -1),
            w_freqs.view(1, 1, slot_w, -1).expand(ref_count, slot_h, slot_w, -1),
        ],
        dim=-1,
    )
    return ref_freqs.reshape(ref_count * slot_h * slot_w, 1, -1).to(device)


# Single source of truth for PRoPE camera input keys. Keeping the pipeline unit
# and inference pass-through aligned prevents accidental bf16 quantization of
# absolute extrinsics and cancellation of small relative motion.
WAN_VIDEO_PROPE_CAMERA_KEYS = (
    "clean_latent_indices_prope_intrinsic",
    "clean_latent_indices_prope_extrinsic",
    "noisy_latent_indices_prope_intrinsic",
    "noisy_latent_indices_prope_extrinsic",
)
WAN_VIDEO_PROPE_CLEAN_CAMERA_KEYS = tuple(key for key in WAN_VIDEO_PROPE_CAMERA_KEYS if key.startswith("clean_"))


def _negative_no_context_inputs_shared(inputs_shared):
    """Build negative CFG shared inputs with only the latest clean context."""
    negative_shared = dict(inputs_shared)
    first_frame_latents = negative_shared.get("first_frame_latents")
    if torch.is_tensor(first_frame_latents) and int(first_frame_latents.shape[2]) > 1:
        negative_shared["first_frame_latents"] = first_frame_latents[:, :, -1:]

    for name in WAN_VIDEO_PROPE_CLEAN_CAMERA_KEYS:
        value = negative_shared.get(name)
        if torch.is_tensor(value) and int(value.shape[0]) > 4:
            negative_shared[name] = value[-4:]

    # The full-context RoPE table has the wrong length after trimming the
    # clean prefix. Let matrix_game_35_forward rebuild the single-clean timeline.
    negative_shared["latent_rope_time_indices"] = None
    return negative_shared


def _drop_holes_reindex_prope_camera_info(
    camera_info,
    *,
    full_frame_count: int,
    tokens_per_frame: int,
    keep_idx_latent: torch.Tensor,
):
    if camera_info is None:
        return None
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info

    w2c_info = camera_info[0]
    viewmats_st = camera_info[1]
    view_change_positions = camera_info[2] if len(camera_info) > 2 else None
    p_st, pt_st, pinv_st = viewmats_st
    b_cam, s_cam = p_st.shape[0], p_st.shape[1]
    if int(s_cam) != int(full_frame_count):
        raise ValueError(
            "PROPE viewmats temporal length "
            f"{int(s_cam)} does not match expected "
            f"{int(full_frame_count)} (first+mosaic+noisy)."
        )

    def _broadcast_then_select(mat):
        rest = mat.shape[2:]
        return (
            mat.unsqueeze(2)
            .expand(b_cam, s_cam, int(tokens_per_frame), *rest)
            .reshape(b_cam, s_cam * int(tokens_per_frame), *rest)
            .index_select(1, keep_idx_latent)
            .contiguous()
        )

    viewmats_pt = (
        _broadcast_then_select(p_st),
        _broadcast_then_select(pt_st),
        _broadcast_then_select(pinv_st),
    )
    if view_change_positions is None:
        return (w2c_info, viewmats_pt)

    expected_view_change_tokens = int(full_frame_count) * int(tokens_per_frame)
    if int(view_change_positions.shape[1]) != expected_view_change_tokens:
        raise ValueError(
            "PROPE view_change_positions length "
            f"{int(view_change_positions.shape[1])} does not match expected "
            f"{expected_view_change_tokens} (first+mosaic+noisy tokens)."
        )
    view_change_positions = view_change_positions.index_select(1, keep_idx_latent).contiguous()
    return (w2c_info, viewmats_pt, view_change_positions)


def _reindex_token_prope_camera_info(camera_info, keep_idx: torch.Tensor):
    if camera_info is None:
        return None
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info
    w2c_info = camera_info[0]
    p, p_t, p_inv = camera_info[1]
    viewmats = (
        p.index_select(1, keep_idx).contiguous(),
        p_t.index_select(1, keep_idx).contiguous(),
        p_inv.index_select(1, keep_idx).contiguous(),
    )
    if len(camera_info) <= 2 or camera_info[2] is None:
        return (w2c_info, viewmats)
    view_change_positions = camera_info[2].index_select(1, keep_idx).contiguous()
    return (w2c_info, viewmats, view_change_positions)


def _prepend_subject_ref_prope_camera_info(
    camera_info,
    *,
    prefix_token_count: int,
    tokens_per_frame: int,
    frame_count: Optional[int] = None,
    mode: str = "identity",
    clean_anchor_token_index: Optional[int] = None,
):
    if camera_info is None or int(prefix_token_count) <= 0:
        return camera_info
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info

    w2c_info = camera_info[0]
    viewmats = camera_info[1]
    view_change_positions = camera_info[2] if len(camera_info) > 2 else None
    p, p_t, p_inv = viewmats
    b_cam = int(p.shape[0])
    token_count = int(prefix_token_count)
    mode = str(mode or "identity").strip().lower()
    if mode not in {"identity", "clean_anchor"}:
        raise ValueError(f"subject_ref_prope_mode must be 'identity' or 'clean_anchor', got {mode!r}.")

    def _as_token_viewmats(mat):
        rest = mat.shape[2:]
        if frame_count is not None and int(mat.shape[1]) == int(frame_count):
            return (
                mat.unsqueeze(2)
                .expand(b_cam, int(mat.shape[1]), int(tokens_per_frame), *rest)
                .reshape(b_cam, int(mat.shape[1]) * int(tokens_per_frame), *rest)
                .contiguous()
            )
        if frame_count is not None and int(mat.shape[1]) == (int(frame_count) * int(tokens_per_frame)):
            return mat
        return mat

    def _prepend_refs(mat):
        mat_pt = _as_token_viewmats(mat)
        if mode == "clean_anchor":
            if int(mat_pt.shape[1]) <= 0:
                raise ValueError("Cannot use subject_ref_prope_mode='clean_anchor' with empty PROPE viewmats.")
            anchor_idx = int(clean_anchor_token_index or 0)
            if anchor_idx < 0 or anchor_idx >= int(mat_pt.shape[1]):
                raise ValueError(
                    "subject_ref_prope_mode='clean_anchor' anchor token index "
                    f"{anchor_idx} is outside PROPE token length "
                    f"{int(mat_pt.shape[1])}."
                )
            prefix = mat_pt[:, anchor_idx : anchor_idx + 1].expand(b_cam, token_count, *mat_pt.shape[2:])
        else:
            eye = torch.eye(
                mat_pt.shape[-1],
                device=mat_pt.device,
                dtype=mat_pt.dtype,
            )
            view_shape = (1, 1) + tuple(1 for _ in mat_pt.shape[2:-2]) + (mat_pt.shape[-2], mat_pt.shape[-1])
            prefix = eye.view(view_shape).expand(b_cam, token_count, *mat_pt.shape[2:])
        return torch.cat([prefix, mat_pt], dim=1).contiguous()

    viewmats_pt = (
        _prepend_refs(p),
        _prepend_refs(p_t),
        _prepend_refs(p_inv),
    )
    if view_change_positions is None:
        return (w2c_info, viewmats_pt)

    if mode == "clean_anchor":
        if int(view_change_positions.shape[1]) <= 0:
            raise ValueError("Cannot use subject_ref_prope_mode='clean_anchor' with empty PROPE view_change_positions.")
        anchor_idx = int(clean_anchor_token_index or 0)
        if anchor_idx < 0 or anchor_idx >= int(view_change_positions.shape[1]):
            raise ValueError(
                "subject_ref_prope_mode='clean_anchor' anchor token index "
                f"{anchor_idx} is outside PROPE view-change token length "
                f"{int(view_change_positions.shape[1])}."
            )
        prefix_view_change = view_change_positions[:, anchor_idx : anchor_idx + 1].expand(b_cam, token_count, 3)
    else:
        prefix_view_change = torch.zeros(
            b_cam,
            token_count,
            3,
            device=view_change_positions.device,
            dtype=view_change_positions.dtype,
        )
        prefix_view_change[..., 0] = 1.0
    view_change_positions = torch.cat([prefix_view_change, view_change_positions], dim=1).contiguous()
    return (w2c_info, viewmats_pt, view_change_positions)


class MatrixLatentSequence:
    def process(
        self,
        pipe,
        latents,
        first_frame_latents=None,
        mosaic_latent=None,
        mosaic_frame_indices=None,
    ):
        first_frame_count = 0
        mosaic_frame_count = 0
        latent_sequence = []

        if first_frame_latents is not None:
            first_frame_latents = first_frame_latents.to(device=latents.device, dtype=latents.dtype)
            if (
                first_frame_latents.shape[:2] != latents.shape[:2]
                or first_frame_latents.shape[-2:] != latents.shape[-2:]
            ):
                raise ValueError(
                    "first_frame_latents must share batch/channel/spatial shape with latents, got "
                    f"{tuple(first_frame_latents.shape)} vs {tuple(latents.shape)}."
                )
            first_frame_count = first_frame_latents.shape[2]
            latent_sequence.append(first_frame_latents)

        if mosaic_latent is not None:
            mosaic_latent = mosaic_latent.to(device=latents.device, dtype=latents.dtype)
            if mosaic_latent.shape[:2] != latents.shape[:2] or mosaic_latent.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "mosaic_latent and latents must share batch/channel/spatial shape, got "
                    f"{tuple(mosaic_latent.shape)} vs {tuple(latents.shape)}."
                )
            if mosaic_latent.shape[2] > latents.shape[2]:
                raise ValueError(
                    "mosaic_latent temporal length must be <= latents temporal length, got "
                    f"{mosaic_latent.shape[2]} vs {latents.shape[2]}."
                )
            mosaic_frame_count = mosaic_latent.shape[2]
            mosaic_frame_indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(latents.shape[2]),
                mosaic_frame_count=int(mosaic_frame_count),
                device=latents.device,
            )
            latent_sequence.append(mosaic_latent)
        else:
            mosaic_frame_indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(latents.shape[2]),
                mosaic_frame_count=0,
                device=latents.device,
            )

        latent_sequence.append(latents)
        if len(latent_sequence) > 1:
            latents = torch.cat(latent_sequence, dim=2)
        return {
            "latents": latents,
            "first_frame_count": first_frame_count,
            "mosaic_frame_count": mosaic_frame_count,
            "condition_frame_count": first_frame_count + mosaic_frame_count,
            "mosaic_frame_indices": mosaic_frame_indices,
        }


_MATRIX_LATENT_SEQUENCE = MatrixLatentSequence()


class MatrixPropeCameraBuilder:
    VAE_HW_SCALING = 16
    VAE_T_SCALING = 4

    def process(
        self,
        pipe,
        use_prope,
        h,
        w,
        dtype,
        device,
        first_frame_count,
        mosaic_frame_count,
        clean_latent_indices_prope_intrinsic=None,
        clean_latent_indices_prope_extrinsic=None,
        noisy_latent_indices_prope_intrinsic=None,
        noisy_latent_indices_prope_extrinsic=None,
        mosaic_frame_indices=None,
        mosaic_view_change=None,
        use_mosaic_view_change_prope=False,
        trans_scale=50.0,
    ):
        if not use_prope:
            return {"camera_info": None}
        if noisy_latent_indices_prope_intrinsic is None or noisy_latent_indices_prope_extrinsic is None:
            return {"camera_info": None}

        # Camera-matrix composition runs in fp32 regardless of the model
        # dtype: K normalization, SE(3) inversion and the P/P_T/P_inv
        # einsums suffer catastrophic cancellation in bf16 (relative
        # translations between nearby frames lose 13-32% accuracy). Only
        # the final products are cast back to ``dtype`` for the attention.
        noisy_intrinsic = self._camera_tensor(noisy_latent_indices_prope_intrinsic, device, torch.float32)
        noisy_extrinsic = self._camera_tensor(noisy_latent_indices_prope_extrinsic, device, torch.float32)
        intrinsic_parts = []
        extrinsic_parts = []

        if first_frame_count > 0:
            if clean_latent_indices_prope_intrinsic is None or clean_latent_indices_prope_extrinsic is None:
                return {"camera_info": None}
            clean_intrinsic = self._camera_tensor(clean_latent_indices_prope_intrinsic, device, torch.float32)
            clean_extrinsic = self._camera_tensor(clean_latent_indices_prope_extrinsic, device, torch.float32)
            intrinsic_parts.append(clean_intrinsic[:, : first_frame_count * self.VAE_T_SCALING])
            extrinsic_parts.append(clean_extrinsic[:, : first_frame_count * self.VAE_T_SCALING])

        if mosaic_frame_count > 0:
            noisy_latent_count = noisy_intrinsic.shape[1] // self.VAE_T_SCALING
            indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(noisy_latent_count),
                mosaic_frame_count=int(mosaic_frame_count),
                device=noisy_intrinsic.device,
            )
            frame_offsets = (
                indices[:, None] * self.VAE_T_SCALING
                + torch.arange(self.VAE_T_SCALING, dtype=torch.long, device=noisy_intrinsic.device)[None, :]
            )
            flat_offsets = frame_offsets.reshape(-1)
            intrinsic_parts.append(noisy_intrinsic.index_select(1, flat_offsets))
            extrinsic_parts.append(noisy_extrinsic.index_select(1, flat_offsets))

        intrinsic_parts.append(noisy_intrinsic)
        extrinsic_parts.append(noisy_extrinsic)
        prope_intrinsic = torch.cat(intrinsic_parts, dim=1).reshape(
            noisy_intrinsic.shape[0], -1, self.VAE_T_SCALING, 3, 3
        )
        prope_extrinsic = torch.cat(extrinsic_parts, dim=1).reshape(
            noisy_extrinsic.shape[0], -1, self.VAE_T_SCALING, 4, 4
        )

        w2c = prope_extrinsic.clone()
        # Recenter the world on the FIRST NOISY camera. PRoPE consumes
        # only pairwise products P_i P_j^{-1}, which are exactly invariant
        # to a common world shift (W_i T (W_j T)^{-1} = W_i W_j^{-1}), so
        # this changes nothing semantically -- but it makes the bf16
        # quantization of the final matrices act on window-local
        # translations instead of large absolute world coordinates, which
        # otherwise wipe out the small inter-frame motion by cancellation.
        # The reference must be the noisy window (always present, always
        # window-local), NOT slot 0: with pool history enabled slot 0 is
        # the OLDEST context camera, which can sit far from the generating
        # window and would leave the noisy translations large again.
        # NOTE: the shift-invariance argument above holds for the legacy
        # linear trans_scale (a global world rescale). For the nonlinear
        # "log"/"tanh" modes per-frame compression does NOT commute with
        # world shifts, so the recenter becomes part of the encoding
        # definition for every inference call through this unit.
        noisy_first = int(first_frame_count) + int(mosaic_frame_count)
        c_ref = invert_se3(w2c[:, noisy_first, 0])[..., :3, 3]
        w2c[..., :3, 3] = w2c[..., :3, 3] + torch.einsum("bstij,bj->bsti", w2c[..., :3, :3], c_ref)
        w2c[..., :3, 3] = self._apply_trans_scale(w2c[..., :3, 3], trans_scale)
        ks_norm = torch.zeros_like(prope_intrinsic)
        image_width = w * self.VAE_HW_SCALING * 2
        image_height = h * self.VAE_HW_SCALING * 2
        ks_norm[..., 0, 0] = prope_intrinsic[..., 0, 0] / image_width
        ks_norm[..., 1, 1] = prope_intrinsic[..., 1, 1] / image_height
        ks_norm[..., 0, 2] = prope_intrinsic[..., 0, 2] / image_width - 0.5
        ks_norm[..., 1, 2] = prope_intrinsic[..., 1, 2] / image_height - 0.5
        ks_norm[..., 2, 2] = 1.0
        p = torch.einsum("...ij,...jk->...ik", lift_k(ks_norm), w2c)
        p_t = p.transpose(-1, -2)
        p_inv = torch.einsum("...ij,...jk->...ik", invert_se3(w2c), lift_k(invert_k(ks_norm)))
        # fp32 composition done -- hand the attention the model dtype.
        w2c = w2c.to(dtype)
        p = p.to(dtype)
        p_t = p_t.to(dtype)
        p_inv = p_inv.to(dtype)
        view_change_positions = self._build_view_change_positions(
            use_mosaic_view_change_prope=use_mosaic_view_change_prope,
            mosaic_view_change=mosaic_view_change,
            batch_size=noisy_intrinsic.shape[0],
            first_frame_count=first_frame_count,
            mosaic_frame_count=mosaic_frame_count,
            noisy_frame_count=noisy_intrinsic.shape[1] // self.VAE_T_SCALING,
            h=h,
            w=w,
            device=device,
            dtype=dtype,
        )
        if view_change_positions is not None:
            return {"camera_info": (w2c, (p, p_t, p_inv), view_change_positions)}
        return {"camera_info": (w2c, (p, p_t, p_inv))}

    @staticmethod
    def _apply_trans_scale(t, trans_scale):
        """Compress the (recentered) w2c translation vectors ``t`` (..., 3).

        Numeric ``trans_scale`` keeps the legacy behaviour ``t / trans_scale``
        (a global world rescale; must match the dataset's extrinsic units).
        ``"log"``/``"logd4"``/``"tanh"`` apply a direction-preserving radial
        compression of the magnitude (``||t|| -> log1p(||t||)`` resp.
        ``log1p(||t||) / 4`` resp. ``tanh(||t||)``): small window-local motions
        pass through near-identity instead of being crushed by a fixed divisor,
        while far history baselines are log-compressed (unbounded but
        slow-growing) resp. saturated to 1. ``"logd4"`` is ``"log"`` with the
        whole compressed magnitude divided by 4 (equivalently log base
        ``e**4``): same dynamic range as ``"log"`` but the far baseline is
        re-anchored near ~1 instead of ~4, so large pose gaps no longer inflate
        the attention logits.
        """
        if isinstance(trans_scale, str):
            mode = trans_scale.strip().lower()
            norm = t.norm(dim=-1, keepdim=True)
            if mode == "log":
                factor = torch.log1p(norm) / norm.clamp_min(1e-8)
            elif mode == "logd4":
                factor = torch.log1p(norm) / norm.clamp_min(1e-8) / 4.0
            elif mode == "tanh":
                factor = torch.tanh(norm) / norm.clamp_min(1e-8)
            else:
                try:
                    return t / float(mode)
                except ValueError:
                    raise ValueError(
                        f"trans_scale must be a number, 'log', 'logd4' or 'tanh', got {trans_scale!r}."
                    ) from None
            return t * factor
        return t / float(trans_scale)

    @staticmethod
    def _canonical_view_change(batch_size, frame_count, h, w, device, dtype):
        out = torch.zeros(
            batch_size,
            int(frame_count),
            int(h),
            int(w),
            3,
            device=device,
            dtype=dtype,
        )
        out[..., 0] = 1.0
        return out

    def _build_view_change_positions(
        self,
        *,
        use_mosaic_view_change_prope,
        mosaic_view_change,
        batch_size,
        first_frame_count,
        mosaic_frame_count,
        noisy_frame_count,
        h,
        w,
        device,
        dtype,
    ):
        if not use_mosaic_view_change_prope:
            return None

        parts = []
        if first_frame_count > 0:
            parts.append(self._canonical_view_change(batch_size, first_frame_count, h, w, device, dtype))

        if mosaic_frame_count > 0:
            if mosaic_view_change is None:
                raise ValueError(
                    "mosaic_view_change is required when mosaic_view_change_prope "
                    "is enabled and mosaic tokens are present."
                )
            vc = torch.as_tensor(mosaic_view_change, device=device, dtype=dtype)
            if vc.ndim == 4:
                vc = vc.unsqueeze(0)
            if vc.ndim != 5 or vc.shape[-1] != 3:
                raise ValueError(f"mosaic_view_change must have shape (T,H,W,3) or (B,T,H,W,3); got {tuple(vc.shape)}.")
            if int(vc.shape[1]) != int(mosaic_frame_count):
                raise ValueError(
                    "mosaic_view_change temporal length must match mosaic_frame_count, "
                    f"got {int(vc.shape[1])} vs {int(mosaic_frame_count)}."
                )
            if int(vc.shape[2]) == int(h) * 2 and int(vc.shape[3]) == int(w) * 2:
                vc = vc[:, :, ::2, ::2]
            if int(vc.shape[2]) != int(h) or int(vc.shape[3]) != int(w):
                raise ValueError(
                    "mosaic_view_change spatial shape must match patch grid "
                    f"({int(h)},{int(w)}) or doubled latent grid "
                    f"({int(h) * 2},{int(w) * 2}); got "
                    f"({int(vc.shape[2])},{int(vc.shape[3])})."
                )
            if int(vc.shape[0]) == 1 and int(batch_size) > 1:
                vc = vc.expand(int(batch_size), -1, -1, -1, -1)
            elif int(vc.shape[0]) != int(batch_size):
                raise ValueError(
                    "mosaic_view_change batch size must be 1 or match model batch, "
                    f"got {int(vc.shape[0])} vs {int(batch_size)}."
                )
            valid = torch.isfinite(vc).all(dim=-1, keepdim=True) & (vc[..., 0:1] > 0)
            neutral = torch.zeros_like(vc)
            neutral[..., 0] = 1.0
            parts.append(torch.where(valid, vc, neutral))

        parts.append(self._canonical_view_change(batch_size, noisy_frame_count, h, w, device, dtype))
        return torch.cat(parts, dim=1).reshape(int(batch_size), -1, 3)

    @staticmethod
    def _camera_tensor(value, device, dtype):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        return value.to(device=device, dtype=dtype)


_MATRIX_PROPE_CAMERA = MatrixPropeCameraBuilder()


def matrix_game_35_forward(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fuse_vae_embedding_in_latents: bool = False,
    first_frame_latents: Optional[torch.Tensor] = None,
    mosaic_latent: Optional[torch.Tensor] = None,
    mosaic_timestep_zero: bool = True,
    mosaic_revgrid: Optional[np.ndarray] = None,
    mosaic_use_revgrid_rope: bool = False,
    mosaic_view_change: Optional[torch.Tensor] = None,
    mosaic_view_change_prope: bool = False,
    mosaic_mask_holes: bool = True,
    mosaic_drop_holes: bool = False,
    mosaic_frame_indices: Optional[torch.Tensor] = None,
    latent_rope_time_indices: Optional[torch.Tensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    subject_ref_latents: Optional[torch.Tensor] = None,
    subject_ref_slot_ratio: float = 0.5,
    subject_ref_time_gap: int = 1,
    subject_ref_prope_mode: str = "identity",
    clean_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
    clean_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
    noisy_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
    noisy_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Matrix-Game 3.5 denoiser with Mosaic, PRoPE, and subject memory."""
    mosaic_hole_mask = None
    latent_sequence = _MATRIX_LATENT_SEQUENCE.process(
        pipe=None,
        latents=latents,
        first_frame_latents=first_frame_latents,
        mosaic_latent=mosaic_latent,
        mosaic_frame_indices=mosaic_frame_indices,
    )
    latents = latent_sequence["latents"]
    first_frame_count = latent_sequence["first_frame_count"]
    mosaic_frame_count = latent_sequence["mosaic_frame_count"]
    condition_frame_count = latent_sequence["condition_frame_count"]
    mosaic_frame_indices = latent_sequence["mosaic_frame_indices"]
    noisy_frame_count = int(latents.shape[2] - condition_frame_count)
    if mosaic_latent is not None and mosaic_mask_holes:
        all_zero = (mosaic_latent == 0).all(dim=(0, 1))
        T_lat, H_lat_full, W_lat_full = all_zero.shape
        hole_patch = all_zero.reshape(T_lat, H_lat_full // 2, 2, W_lat_full // 2, 2).all(dim=(2, 4)).flatten()
        prefix = (
            torch.zeros(
                first_frame_count * hole_patch.shape[0] // mosaic_frame_count,
                dtype=torch.bool,
                device=hole_patch.device,
            )
            if first_frame_count > 0
            else None
        )
        suffix = torch.zeros(
            (latents.shape[2] - condition_frame_count) * hole_patch.shape[0] // mosaic_frame_count,
            dtype=torch.bool,
            device=hole_patch.device,
        )
        masks = []
        if prefix is not None:
            masks.append(prefix)
        masks.extend([hole_patch, suffix])
        mosaic_hole_mask = torch.cat(masks, dim=0)
    if dit.seperated_timestep and (
        fuse_vae_embedding_in_latents or first_frame_count > 0 or (mosaic_timestep_zero and mosaic_frame_count > 0)
    ):
        patch_count_per_frame = latents.shape[3] * latents.shape[4] // 4
        timestep_scalar = timestep.reshape(-1)[0]
        if condition_frame_count > 0 and mosaic_timestep_zero:
            noisy_steps = torch.ones((noisy_frame_count,), dtype=latents.dtype, device=latents.device) * timestep_scalar
            frame_steps = torch.cat(
                [torch.zeros((condition_frame_count,), dtype=latents.dtype, device=latents.device), noisy_steps], dim=0
            )
            timestep = frame_steps.repeat_interleave(patch_count_per_frame)
        else:
            timestep = torch.concat(
                [
                    torch.zeros((1, patch_count_per_frame), dtype=latents.dtype, device=latents.device),
                    torch.ones(
                        (latents.shape[2] - 1, patch_count_per_frame), dtype=latents.dtype, device=latents.device
                    )
                    * timestep_scalar,
                ]
            ).flatten()
        if mosaic_hole_mask is not None:
            timestep = timestep.clone()
            timestep[mosaic_hole_mask] = 1000
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    subject_ref_memory = None
    subject_ref_prefix_token_count = 0
    if subject_ref_latents is not None:
        subject_ref_memory = _build_subject_ref_memory_tokens(
            dit,
            subject_ref_latents,
            batch_size=int(x.shape[0]),
            video_h=int(height) if height is not None else int(latents.shape[3] * 16),
            video_w=int(width) if width is not None else int(latents.shape[4] * 16),
            subject_ref_slot_ratio=subject_ref_slot_ratio,
            subject_ref_time_gap=subject_ref_time_gap,
            device=x.device,
            dtype=x.dtype,
        )
        if subject_ref_memory is not None:
            subject_ref_prefix_token_count = int(subject_ref_memory["token_count"])
    prope_camera = _MATRIX_PROPE_CAMERA.process(
        pipe=None,
        use_prope=getattr(dit, "use_prope", False),
        h=h,
        w=w,
        dtype=x.dtype,
        device=x.device,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        clean_latent_indices_prope_intrinsic=clean_latent_indices_prope_intrinsic,
        clean_latent_indices_prope_extrinsic=clean_latent_indices_prope_extrinsic,
        noisy_latent_indices_prope_intrinsic=noisy_latent_indices_prope_intrinsic,
        noisy_latent_indices_prope_extrinsic=noisy_latent_indices_prope_extrinsic,
        mosaic_frame_indices=mosaic_frame_indices,
        mosaic_view_change=mosaic_view_change,
        use_mosaic_view_change_prope=mosaic_view_change_prope,
        trans_scale=getattr(dit, "trans_scale", 50.0),
    )
    camera_info = prope_camera["camera_info"]
    full_frame_count = first_frame_count + mosaic_frame_count + noisy_frame_count
    if subject_ref_prefix_token_count > 0:
        ref_zero_timestep = torch.zeros(subject_ref_prefix_token_count, dtype=latents.dtype, device=x.device)
        ref_t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, ref_zero_timestep).unsqueeze(0))
        ref_t_mod = dit.time_projection(ref_t).unflatten(2, (6, dit.dim))
        if t_mod.ndim != 4:
            raise ValueError(f"subject_ref prefix memory requires per-token t_mod; got shape {tuple(t_mod.shape)}.")
        t = torch.cat([ref_t.to(dtype=t.dtype), t], dim=1)
        t_mod = torch.cat([ref_t_mod.to(dtype=t_mod.dtype), t_mod], dim=1)
        x = torch.cat([subject_ref_memory["x"], x], dim=1)
        camera_info = _prepend_subject_ref_prope_camera_info(
            camera_info,
            prefix_token_count=subject_ref_prefix_token_count,
            tokens_per_frame=h * w,
            frame_count=full_frame_count,
            mode=subject_ref_prope_mode,
            clean_anchor_token_index=max(0, int(first_frame_count) - 1) * int(h * w),
        )
    freq_dev = dit.freqs[0].device
    rope_time_indices = _resolve_latent_rope_time_indices(
        latent_rope_time_indices,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        noisy_frame_count=noisy_frame_count,
        mosaic_frame_indices=mosaic_frame_indices,
        device=freq_dev,
    )
    if rope_time_indices.numel() and int(rope_time_indices.max().item()) >= int(dit.freqs[0].shape[0]):
        raise ValueError(
            f"latent_rope_time_indices contains a temporal index outside dit.freqs[0], max={int(rope_time_indices.max().item())}, available={int(dit.freqs[0].shape[0])}."
        )
    first_rope_times = rope_time_indices[:first_frame_count]
    mosaic_rope_times = rope_time_indices[first_frame_count : first_frame_count + mosaic_frame_count]
    noisy_rope_times = rope_time_indices[first_frame_count + mosaic_frame_count :]
    if mosaic_use_revgrid_rope and mosaic_frame_count > 0 and (mosaic_revgrid is not None) and (len(dit.freqs) >= 5):
        freq_chunks = []
        if first_frame_count > 0:
            first_freqs = torch.cat(
                [
                    dit.freqs[0][first_rope_times]
                    .view(first_frame_count, 1, 1, -1)
                    .expand(first_frame_count, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(first_frame_count, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(first_frame_count, h, w, -1),
                ],
                dim=-1,
            )
            freq_chunks.append(first_freqs)
        noisy_freqs = torch.cat(
            [
                dit.freqs[0][noisy_rope_times].view(noisy_frame_count, 1, 1, -1).expand(noisy_frame_count, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(noisy_frame_count, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(noisy_frame_count, h, w, -1),
            ],
            dim=-1,
        )
        rg_t = torch.from_numpy(np.asarray(mosaic_revgrid, dtype=np.float32)).to(freq_dev)
        if int(rg_t.shape[0]) != int(mosaic_frame_count):
            raise ValueError(
                f"mosaic_revgrid temporal length must match mosaic_frame_count, got {int(rg_t.shape[0])} vs {int(mosaic_frame_count)}."
            )
        mosaic_rope_frames = []
        for idx in range(mosaic_frame_count):
            rope_t = dit.freqs[0][mosaic_rope_times[idx]].view(1, 1, 1, -1).expand(1, h, w, -1)
            rg = rg_t[idx, ::2, ::2]
            ql_h = (rg[..., 1] * 8).long().clamp(0, dit.freqs[3].shape[0] - 1)
            ql_w = (rg[..., 0] * 8).long().clamp(0, dit.freqs[4].shape[0] - 1)
            rope_h = dit.freqs[3][ql_h].reshape(1, h, w, -1)
            rope_w = dit.freqs[4][ql_w].reshape(1, h, w, -1)
            invalid = (rg[..., 0] < 0) | (rg[..., 1] < 0)
            if invalid.any():
                inv = invalid.view(1, h, w, 1)
                rope_h = torch.where(inv, dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1), rope_h)
                rope_w = torch.where(inv, dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1), rope_w)
            mosaic_rope_frames.append(torch.cat([rope_t, rope_h, rope_w], dim=-1))
        mosaic_freqs = torch.cat(mosaic_rope_frames, dim=0)
        freq_chunks.extend([mosaic_freqs, noisy_freqs])
        freqs = torch.cat(freq_chunks, dim=0).reshape(f * h * w, 1, -1).to(x.device)
    else:
        freq_chunks = []
        if first_frame_count > 0:
            freq_chunks.append(
                torch.cat(
                    [
                        dit.freqs[0][first_rope_times]
                        .view(first_frame_count, 1, 1, -1)
                        .expand(first_frame_count, h, w, -1),
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(first_frame_count, h, w, -1),
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(first_frame_count, h, w, -1),
                    ],
                    dim=-1,
                )
            )
        if mosaic_frame_count > 0:
            freq_chunks.append(
                torch.cat(
                    [
                        dit.freqs[0][mosaic_rope_times]
                        .view(mosaic_frame_count, 1, 1, -1)
                        .expand(mosaic_frame_count, h, w, -1),
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(mosaic_frame_count, h, w, -1),
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(mosaic_frame_count, h, w, -1),
                    ],
                    dim=-1,
                )
            )
        freq_chunks.append(
            torch.cat(
                [
                    dit.freqs[0][noisy_rope_times]
                    .view(noisy_frame_count, 1, 1, -1)
                    .expand(noisy_frame_count, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(noisy_frame_count, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(noisy_frame_count, h, w, -1),
                ],
                dim=-1,
            )
        )
        freqs = torch.cat(freq_chunks, dim=0).reshape(f * h * w, 1, -1).to(x.device)
    if subject_ref_prefix_token_count > 0:
        freqs = torch.cat([subject_ref_memory["freqs"].to(device=x.device), freqs], dim=0)
    reference_token_count = 0
    cross_attn_keep_mask = None
    if mosaic_frame_count > 0 or subject_ref_prefix_token_count > 0:
        cross_attn_keep_mask = _build_mosaic_cross_attn_keep_mask(
            prefix_memory_token_count=subject_ref_prefix_token_count,
            reference_token_count=reference_token_count,
            first_frame_count=first_frame_count,
            mosaic_frame_count=mosaic_frame_count,
            noisy_frame_count=noisy_frame_count,
            tokens_per_frame=h * w,
            device=x.device,
        )
    mosaic_attn_mask = None
    drop_holes_keep_idx_full: Optional[torch.Tensor] = None
    drop_holes_pre_seq_len: Optional[int] = None
    if mosaic_hole_mask is not None:
        _hm = mosaic_hole_mask.to(x.device)
        hm_parts = []
        if subject_ref_prefix_token_count > 0:
            hm_parts.append(torch.zeros(subject_ref_prefix_token_count, dtype=torch.bool, device=x.device))
        if reference_token_count > 0:
            hm_parts.append(torch.zeros(reference_token_count, dtype=torch.bool, device=x.device))
        hm_parts.append(_hm)
        _hm_full = torch.cat(hm_parts, dim=0)
        if bool(mosaic_drop_holes):
            keep_full = ~_hm_full
            keep_idx_full = torch.nonzero(keep_full, as_tuple=False).squeeze(-1)
            keep_idx_latent = torch.nonzero(~_hm, as_tuple=False).squeeze(-1)
            drop_holes_keep_idx_full = keep_idx_full
            drop_holes_pre_seq_len = int(x.shape[1])
            x = x.index_select(1, keep_idx_full)
            freqs = freqs.index_select(0, keep_idx_full)
            if cross_attn_keep_mask is not None:
                cross_attn_keep_mask = cross_attn_keep_mask.index_select(0, keep_idx_full)
            time_keep_idx = keep_idx_full if subject_ref_prefix_token_count > 0 else keep_idx_latent
            if t_mod.ndim == 4:
                t_mod = t_mod.index_select(1, time_keep_idx)
            if t.ndim == 3:
                t = t.index_select(1, time_keep_idx)
            tokens_per_frame_int = h * w
            if subject_ref_prefix_token_count > 0:
                camera_info = _reindex_token_prope_camera_info(camera_info, keep_idx_full)
            else:
                camera_info = _drop_holes_reindex_prope_camera_info(
                    camera_info,
                    full_frame_count=full_frame_count,
                    tokens_per_frame=tokens_per_frame_int,
                    keep_idx_latent=keep_idx_latent,
                )
        else:
            x[:, _hm_full] = 0
            freqs[_hm_full] = 0
            mosaic_attn_mask = (~_hm_full).view(1, 1, 1, -1)
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs, mosaic_attn_mask, camera_info, cross_attn_keep_mask)
    x = dit.head(x, t)
    if drop_holes_keep_idx_full is not None and drop_holes_pre_seq_len is not None:
        full_out = x.new_zeros(x.shape[0], drop_holes_pre_seq_len, x.shape[2])
        full_out.index_copy_(1, drop_holes_keep_idx_full, x)
        x = full_out
    if subject_ref_prefix_token_count > 0:
        x = x[:, subject_ref_prefix_token_count:]
    x = dit.unpatchify(x, (f, h, w))
    if condition_frame_count > 0:
        x = x[:, :, condition_frame_count:, :, :]
    return x


class MatrixGame35Denoiser:
    """Expose Matrix Game 3.5 through the shared denoiser contract."""

    def __init__(self, model: WanModel) -> None:
        self.model = model

    @contextlib.contextmanager
    def _temporary_no_prope(self, disabled: bool):
        if not disabled:
            yield
            return
        states = []
        try:
            for module in (
                self.model,
                *(block for block in self.model.blocks),
                *(block.self_attn for block in self.model.blocks),
            ):
                states.append((module, getattr(module, "use_prope", False)))
                module.use_prope = False
            yield
        finally:
            for module, value in states:
                module.use_prope = value

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = dict(model_input.conditioning)
        context = conditioning.pop("context", None)
        if not isinstance(context, torch.Tensor):
            raise TypeError("Matrix Game 3.5 requires a tensor 'context' condition")
        negative_no_context = bool(conditioning.pop("negative_no_context", False))
        negative_no_prope = bool(conditioning.pop("negative_no_prope", False))
        if model_input.branch == "negative" and negative_no_context:
            conditioning = _negative_no_context_inputs_shared(conditioning)
        compute_dtype = next(self.model.parameters()).dtype
        autocast_enabled = compute_dtype in {torch.float16, torch.bfloat16}
        with self._temporary_no_prope(model_input.branch == "negative" and negative_no_prope):
            with torch.autocast(
                device_type=model_input.latents.device.type,
                dtype=compute_dtype,
                enabled=autocast_enabled,
            ):
                sample = matrix_game_35_forward(
                    dit=self.model,
                    latents=model_input.latents,
                    timestep=model_input.timestep.to(
                        device=model_input.latents.device,
                        dtype=compute_dtype,
                    ),
                    context=context,
                    **conditioning,
                )
        return DenoiserOutput(sample=sample)


def _configure_matrix_prope(model: WanModel, options) -> None:
    use_prope = bool(options.get("use_prope", False))
    interval = max(1, int(options.get("prope_attention_interval", 1)))
    disable_native = bool(options.get("prope_disable_native_rope", False))
    disable_temporal = bool(options.get("prope_disable_t_rope", False))
    camera_layout = str(options.get("prope_camera_layout", "full"))
    if disable_native and disable_temporal:
        raise ValueError("prope_disable_native_rope and prope_disable_t_rope are mutually exclusive")
    if disable_temporal and interval <= 1:
        raise ValueError("prope_disable_t_rope requires prope_attention_interval > 1")
    if camera_layout != "full" and not disable_temporal:
        raise ValueError(f"prope_camera_layout={camera_layout!r} requires prope_disable_t_rope")

    model.use_prope = use_prope
    model.prope_disable_native_rope = disable_native
    model.prope_disable_t_rope = disable_temporal
    model.prope_camera_layout = camera_layout
    model.trans_scale = options.get("trans_scale", 50.0)
    for block_id, block in enumerate(model.blocks):
        enabled = use_prope and block_id % interval == 0
        for module in (block, block.self_attn):
            module.use_prope = enabled
            module.prope_disable_native_rope = disable_native
            module.prope_disable_t_rope = disable_temporal
            module.prope_camera_layout = camera_layout


def build_matrix_game_35_denoiser(context: ComponentBuildContext) -> MatrixGame35Denoiser:
    """Load one released Matrix checkpoint using the shared native loader."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model_config = {
        **MATRIX_GAME_35_DIT_CONFIG,
        "subject_ref_memory_max_refs": int(
            context.component_options.get(
                "subject_ref_memory_max_refs",
                MATRIX_GAME_35_DIT_CONFIG["subject_ref_memory_max_refs"],
            )
        ),
    }
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanModel,
            config=model_config,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            layer_container="blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, WanModel):
        raise TypeError(f"expected Matrix WanModel, got {type(model).__name__}")
    _configure_matrix_prope(model, context.component_options)
    return MatrixGame35Denoiser(model)


__all__ = [
    "MATRIX_GAME_35_DIT_CONFIG",
    "MatrixGame35Denoiser",
    "MatrixLatentSequence",
    "MatrixPropeCameraBuilder",
    "WAN_VIDEO_PROPE_CAMERA_KEYS",
    "build_matrix_game_35_denoiser",
    "matrix_game_35_forward",
]
