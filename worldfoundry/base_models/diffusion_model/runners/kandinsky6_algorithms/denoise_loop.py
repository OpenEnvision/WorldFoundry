from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from worldfoundry.base_models.diffusion_model.kandinsky6_types import LatentBundle, TextEmbeds
from worldfoundry.base_models.diffusion_model.schedulers.kandinsky6.flow_matching import flow_match_timesteps
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.guidance import apply_cfg
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.prepare_ropes import compute_rope1d, compute_visual_rope


def _build_video_input(
    video: Tensor,
    has_visual_cond: bool,
    first_frames: Tensor | None,
    video_cu_seqlens: Tensor | None,
    visual_cond_scheme: str,
) -> Tensor:
    """Appends visual-conditioning channels [cond, mask] to the video latent.

    Schemes (K5 parity):
      - ``pretrain``: inject first_frames into cond channels at sequence starts
      - ``i2v``: overwrite latents at sequence starts; cond unused
      - ``tail_cond_first_frame``: overwrite / mask the last (reference) frame
    """
    if not has_visual_cond:
        return video

    cond = torch.zeros_like(video)
    mask = torch.zeros([*video.shape[:-1], 1], dtype=video.dtype, device=video.device)

    if first_frames is not None:
        ff = first_frames.to(device=video.device, dtype=video.dtype)
        if video_cu_seqlens is None:
            raise ValueError(f"{visual_cond_scheme} requires video_cu_seqlens")
        batched = video.ndim == 5
        starts = torch.arange(video.shape[0], device=video.device) if batched else video_cu_seqlens[:-1]
        tail_cond = visual_cond_scheme == "tail_cond_first_frame"
        inject = (torch.full_like(starts, video.shape[1] - 1) if batched and tail_cond else starts)
        if not batched and tail_cond:
            inject = video_cu_seqlens[1:] - 1

        if batched:
            if visual_cond_scheme == "i2v":
                video[:, 0] = ff
            elif tail_cond:
                video[:, -1] = ff
            elif visual_cond_scheme == "pretrain":
                cond[:, 0] = ff
        elif visual_cond_scheme == "i2v":
            video[starts] = ff
        elif tail_cond:
            video[inject] = ff
        elif visual_cond_scheme == "pretrain":
            cond[starts] = ff
        else:
            raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
        if batched:
            mask[:, -1 if tail_cond else 0] = 1
        else:
            mask[inject] = 1

    return torch.cat([video, cond, mask], dim=-1)


def _resolve_null_embeds(
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    null_text_rope: Tensor | list[Tensor | None],
) -> tuple[Tensor, Tensor, Tensor | list[Tensor]]:
    """Returns (text_embed, pooled_embed, text_rope) for the uncond DiT pass.

    When null_text_embeds is a list of two items (T2VA per-modality nulls),
    returns list arguments so the DiT can apply different null text per modality.
    """
    if isinstance(null_text_embeds, list):
        # null_text_embeds[0] = video null, null_text_embeds[1] = audio null (may be None)
        null_v = null_text_embeds[0]
        null_a = null_text_embeds[1]
        null_rope_v = null_text_rope[0] if isinstance(null_text_rope, list) else null_text_rope
        null_rope_a = null_text_rope[1] if isinstance(null_text_rope, list) else None

        if null_a is not None and null_rope_a is not None:
            # Different null text per modality — pass as list to DiT
            text_embed = [null_v["text_embeds"], null_a["text_embeds"]]
            pooled_embed = [null_v["pooled_embed"], null_a["pooled_embed"]]
            rope = [null_rope_v, null_rope_a]
        else:
            text_embed = null_v["text_embeds"]
            pooled_embed = null_v["pooled_embed"]
            rope = null_rope_v
    else:
        text_embed = null_text_embeds["text_embeds"]
        pooled_embed = null_text_embeds["pooled_embed"]
        rope = null_text_rope

    return text_embed, pooled_embed, rope


def _raw_dit(dit: nn.Module) -> nn.Module:
    inner = getattr(dit, "module", None)
    return inner if inner is not None and hasattr(dit, "set_cache") else dit


def _rebuild_text_rope(dit: nn.Module, template: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
    """Recompute 1-D text RoPE(s) matching template length(s) — A/B vs cached tensors."""
    raw = _raw_dit(dit)
    if isinstance(template, list):
        if getattr(raw, "is_multimodal", False):
            mods = [raw.video_text_rope, raw.audio_text_rope]
        else:
            mods = [raw.text_rope] * len(template)
        return [compute_rope1d(mod, int(t.shape[0])) for mod, t in zip(mods, template, strict=True)]
    if getattr(raw, "is_multimodal", False):
        return compute_rope1d(raw.video_text_rope, int(template.shape[0]))
    return compute_rope1d(raw.text_rope, int(template.shape[0]))


@torch.no_grad()
def denoise_loop(
    bundle: LatentBundle,
    dit: nn.Module,
    text_embeds: TextEmbeds,
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    visual_rope: Tensor | None,
    audio_rope: Tensor | None,
    text_rope: Tensor | list[Tensor],
    null_text_rope: Tensor | list[Tensor | None],
    num_steps: int,
    guidance_weight: float,
    scheduler_scale: float,
    first_frames: Tensor | None = None,
    visual_cond_scheme: str = "pretrain",
    sample_video: bool = True,
    sample_audio: bool = True,
    *,
    attention_mask: Tensor | None = None,
    null_attention_mask: Tensor | None = None,
    visual_token_type_ids: Tensor | None = None,
    recompute_ropes_each_step: bool = False,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback=None,
) -> LatentBundle:
    """Single-device Euler denoising loop for T2V and T2VA.

    RoPE tensors are normally precomputed by the caller (pipeline). Set
    ``recompute_ropes_each_step=True`` to rebuild them every DiT forward (A/B
    vs pipeline cache; not for production).
    """
    video = bundle.video  # (sum_T, H, W, C) or (sum_T, H, W, C+extra) for instruct
    audio = bundle.audio  # (sum_A, audio_dim) or None
    is_multimodal = video is not None and audio is not None

    device = video.device if video is not None else audio.device
    timesteps = flow_match_timesteps(num_steps, scheduler_scale, device)

    video_cu_seqlens = (
        bundle.video_cu_seqlens.to(device=device)
        if bundle.video_cu_seqlens is not None
        else None
    )
    audio_cu_seqlens = (
        bundle.audio_cu_seqlens.to(device=device)
        if bundle.audio_cu_seqlens is not None
        else None
    )
    bs = (
        video_cu_seqlens.shape[0] - 1
        if video_cu_seqlens is not None
        else audio_cu_seqlens.shape[0] - 1
    )

    null_te, null_pe, null_rope = _resolve_null_embeds(null_text_embeds, null_text_rope)

    out_c: int | None = None  # determined after first step, used to strip instruct channels
    raw = _raw_dit(dit)
    vis_shape = (
        (int(visual_rope.shape[0]), int(visual_rope.shape[1]), int(visual_rope.shape[2]))
        if visual_rope is not None
        else None
    )
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    tail_cond = visual_cond_scheme == "tail_cond_first_frame"
    ref_positions = (
        video_cu_seqlens[1:] - 1
        if tail_cond and video_cu_seqlens is not None
        else None
    )
    batched_video = video is not None and video.ndim == 5

    for t, dt in zip(timesteps[:-1], torch.diff(timesteps)):
        t_step: Tensor | list[Tensor] = t.unsqueeze(0).expand(bs) * 1000  # (bs,)

        model_input_v = (
            _build_video_input(
                video, dit.visual_cond, first_frames,
                video_cu_seqlens, visual_cond_scheme,
            )
            if video is not None else None
        )

        # Freeze one modality at t=0 for partial sampling (T2VA only)
        if is_multimodal:
            t_frozen = timesteps[-1].unsqueeze(0).expand(bs) * 1000
            if not sample_audio:
                t_step = [t_step, t_frozen]
            elif not sample_video:
                t_step = [t_frozen, t_step]

        def _forward(te, pe, rope, attn_mask):
            vr, ar, tr = visual_rope, audio_rope, rope
            if recompute_ropes_each_step:
                if vis_shape is not None:
                    vr = compute_visual_rope(raw.visual_rope, vis_shape, scale)
                if audio_rope is not None:
                    ar = compute_rope1d(raw.audio_rope, int(audio_rope.shape[0]))
                tr = _rebuild_text_rope(raw, rope)
            return dit(
                x_video=model_input_v,
                x_audio=audio,
                text_embed=te,
                pooled_text_embed=pe,
                time=t_step,
                visual_rope=vr,
                audio_rope=ar,
                text_rope=tr,
                attention_mask=attn_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        vel_cond = _forward(
            text_embeds["text_embeds"], text_embeds["pooled_embed"], text_rope, attention_mask,
        )
        vel_uncond = (
            _forward(null_te, null_pe, null_rope, null_attention_mask)
            if abs(guidance_weight - 1.0) > 1e-6
            else vel_cond
        )

        # Euler update per modality
        if isinstance(vel_cond, tuple):
            vel_v, vel_a = vel_cond
            uvel_v, uvel_a = vel_uncond if vel_uncond is not vel_cond else (vel_v, vel_a)
            if out_c is None:
                out_c = vel_v.shape[-1]
            if video is not None and sample_video:
                video[..., :out_c] += dt * apply_cfg(vel_v, uvel_v, guidance_weight)
                if tail_cond and first_frames is not None:
                    if batched_video:
                        video[:, -1] = first_frames.to(device=device, dtype=video.dtype)
                    elif ref_positions is not None:
                        video[ref_positions] = first_frames.to(device=device, dtype=video.dtype)
            if audio is not None and sample_audio:
                audio += dt * apply_cfg(vel_a, uvel_a, guidance_weight)
        else:
            if out_c is None:
                out_c = vel_cond.shape[-1]
            vel = apply_cfg(vel_cond, vel_uncond, guidance_weight)
            if video is not None and sample_video:
                video[..., :out_c] += dt * vel
                if tail_cond and first_frames is not None:
                    if batched_video:
                        video[:, -1] = first_frames.to(device=device, dtype=video.dtype)
                    elif ref_positions is not None:
                        video[ref_positions] = first_frames.to(device=device, dtype=video.dtype)
            elif audio is not None and sample_audio:
                audio += dt * vel

        if progress_callback is not None:
            progress_callback()

    # I2V / I2VA: keep injected reference frames unchanged
    if first_frames is not None and video is not None and video_cu_seqlens is not None:
        ff = first_frames.to(device=device, dtype=video.dtype)
        if visual_cond_scheme == "i2v":
            if batched_video:
                video[:, 0] = ff
            else:
                video[video_cu_seqlens[:-1]] = ff
        elif tail_cond:
            if batched_video:
                video[:, -1] = ff
            elif ref_positions is not None:
                video[ref_positions] = ff

    # Strip instruct extra-channel padding from the video latent
    if out_c is not None and video is not None:
        video = video[..., :out_c]

    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=video_cu_seqlens,
        audio_cu_seqlens=audio_cu_seqlens,
    )
