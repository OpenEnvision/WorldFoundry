"""Stage 5 tests: packed sequence + coupled denoise loop wired to the real DiT."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
)
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_time_shift_sigmas,
)
from worldfoundry.pipelines.minimax._minimax_h3 import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
    minimax_h3_denoise_loop,
)


def test_patchify_unpatchify_roundtrip() -> None:
    latent = torch.randn(1, 24, 2, 8, 8)
    rows = minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2))
    # t=2, h=4, w=4 -> 32 rows; channel*pt*ph*pw = 24*1*2*2 = 96.
    assert rows.shape == (32, 96)
    back = minimax_h3_unpatchify_video_tokens(
        rows, latent_shape=(2, 4, 4, 24), patch_size=(1, 2, 2)
    )
    assert back.shape == latent.shape
    assert torch.allclose(back, latent, atol=1e-5)


def test_t2va_packed_layout_alignment_and_tags() -> None:
    packed = minimax_h3_packed_sequence(
        text_len=10,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=4,
        include_keyframe_cond=False,
    )
    frame_rows = (8 // 2) * (8 // 2)  # 16
    video_rows = 2 * frame_rows  # 32
    audio_rows = 4 * 2  # 8
    used = 10 + 0 + audio_rows + video_rows  # 50
    assert packed["seq_len"] == 64  # padded to 64
    assert int(packed["cu_seqlens"][1]) == used
    assert packed["img_pos"].shape[0] == video_rows
    assert packed["audio_pos"].shape[0] == audio_rows
    tags = packed["token_tags"]
    assert (tags[packed["text_pos"]] == 1).all()
    assert (tags[packed["audio_pos"]] == 2).all()
    assert (tags[packed["img_pos"]] == 0).all()
    # all target rows for t2va
    assert packed["update_mask"].all()


def test_fl2va_first_keyframe_has_condition_rows() -> None:
    packed = minimax_h3_packed_sequence(
        text_len=10,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=4,
        include_keyframe_cond=True,
        keyframe_frame_indices=(0,),
        frame_count=22,
    )
    frame_rows = 16
    # img_pos = cond frame rows + target video rows; first frame_rows are cond.
    assert packed["img_pos"].shape[0] == frame_rows + 2 * frame_rows
    assert (~packed["update_mask"][:frame_rows]).all()  # cond rows pinned
    assert packed["update_mask"][frame_rows:].all()  # rest are targets


def test_fl2va_first_last_cond_blocks_use_exact_rope_span() -> None:
    # Locks the delicate fp64 temporal-span numerics against the SGLang
    # reference (first keyframe at text_len; last at text_len+span-frame_rescale).
    text_len, latent_t = 11, 37
    built = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=48,
        latent_w=76,
        audio_t=203,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0, -1],
        frame_count=124,
    )
    frame_rows = 24 * 38
    cond_rows = 2 * frame_rows
    assert int((~built["update_mask"]).sum()) == cond_rows
    assert int(built["img_pos"].shape[0]) == (2 + latent_t) * frame_rows
    cond_pos = built["img_pos"][:cond_rows].reshape(2, frame_rows)
    cond_t = [float(built["img_position_ids"][positions, 0].unique().item()) for positions in cond_pos]
    frame_rescale = 5.0 / 3.0
    temporal_span = sum(frame_rescale * (1, 4, 4, 4, 4)[i % 5] for i in range(latent_t))
    assert cond_t[0] == float(text_len)
    assert cond_t[1] == pytest.approx(float(text_len) + temporal_span - frame_rescale, abs=1e-12)


def test_ref2va_blocks_layout_has_audio_update_mask() -> None:
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=8,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=4,
        ref_blocks=[{"kind": "image", "latent_h": 8, "latent_w": 8}],
    )
    assert "audio_update_mask" in packed
    # image ref contributes frame_rows video cond rows, no audio refs.
    frame_rows = 16
    assert (~packed["update_mask"][:frame_rows]).all()
    assert packed["audio_update_mask"].all()  # no audio refs -> all targets


def _tiny_config() -> MiniMaxH3DiTArchConfig:
    return MiniMaxH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=64,
        timestep_input_dim=256,
        time_embed_hidden_size=256,
        time_embed_dim=128,
        adaln_out_features=18 * 256,
        final_adaln_out_features=2 * 256,
        rope_inv_freq_len=16,
    )


def test_t2va_denoise_loop_end_to_end() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniMaxH3DiTModel(cfg).to(device)
    with torch.no_grad():
        for block in model.blocks:
            block.adaln_proj.linear.weight.zero_()
            block.adaln_proj.linear.bias.zero_()
        model.final_layer.adaln_proj.linear.weight.zero_()
        model.final_layer.adaln_proj.linear.bias.zero_()
        # Shrink random projection weights so the untrained smoke model stays
        # within bf16 range across CUDA accumulation orderings (real
        # checkpoints have small trained weights).
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.mul_(0.1)
    model = model.eval()

    text_len, latent_t, latent_h, latent_w, audio_t = 6, 2, 8, 8, 4
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    text_embeddings = torch.randn(text_len, cfg.text_dim, dtype=torch.bfloat16)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=packed["token_tags"],
        device=device,
    )

    n_video_rows = int(packed["img_pos"].shape[0])
    n_audio_rows = int(packed["audio_pos"].shape[0])
    gen = torch.Generator().manual_seed(1234)
    initial_video = torch.randn(n_video_rows, 96, generator=gen)
    initial_audio = torch.randn(n_audio_rows, 32, generator=gen)

    sigmas = minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=12.0)
    sigmas_audio = minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=3.0)
    # equal length required; pad the shorter with its terminal value structure
    n = min(len(sigmas), len(sigmas_audio))
    sigmas, sigmas_audio = sigmas[:n], sigmas_audio[:n]

    video_out, audio_out = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=initial_video,
        initial_audio_rows=initial_audio,
        keyframe_cond_rows=None,
        sigmas_video=sigmas,
        sigmas_audio=sigmas_audio,
        device=device,
    )
    assert video_out.shape == (n_video_rows, 96)
    assert audio_out.shape == (n_audio_rows, 32)
    assert torch.isfinite(video_out).all()
    assert torch.isfinite(audio_out).all()
