"""Stage 4 tests for the MiniMax H3 DiT (shape/contract, single GPU or CPU)."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
    reorder_grouped_qkv_to_qkv,
)


def _tiny_config() -> MiniMaxH3DiTArchConfig:
    # Small but structurally faithful: head_dim must be 128 so RoPE rot_dim=96
    # leaves a 32-dim passthrough tail, matching the real model.
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


def test_reorder_grouped_qkv_roundtrip() -> None:
    num_groups, head_dim, hidden = 4, 8, 5
    # grouped layout: per group [q(1 head), k, v] -> 3*head_dim rows/group.
    grouped = torch.arange(num_groups * 3 * head_dim * hidden, dtype=torch.float32).reshape(
        num_groups * 3 * head_dim, hidden
    )
    out = reorder_grouped_qkv_to_qkv(grouped, num_query_groups=num_groups, heads_per_group=1, head_dim=head_dim)
    assert out.shape == grouped.shape
    # First block is all q rows (group 0's q, group 1's q, ...).
    q_block = out[: num_groups * head_dim]
    expected_first_q = grouped[:head_dim]  # group 0's q rows
    assert torch.allclose(q_block[:head_dim], expected_first_q)


def test_fp32_param_names_present_and_dtype_split() -> None:
    model = MiniMaxH3DiTModel(_tiny_config())
    names = {name for name, _ in model.named_parameters()}
    for fp32_name in MINIMAX_H3_FP32_PARAM_NAMES:
        assert fp32_name in names, fp32_name
        assert model.get_parameter(fp32_name).dtype == torch.float32
    # A block linear is bf16.
    assert model.blocks[0].mlp.fc1.weight.dtype == torch.bfloat16
    model.post_load_weights()  # must not raise


def _build_packed_inputs(cfg: MiniMaxH3DiTArchConfig, device: torch.device):
    # Minimal valid packed layout: [text | audio | video], padded to a doc.
    text_len, audio_len, video_len = 3, 4, 5
    seq_len = text_len + audio_len + video_len
    video_row = cfg.video_patch_output_dim
    text_pos = torch.arange(0, text_len, device=device)
    audio_pos = torch.arange(text_len, text_len + audio_len, device=device)
    img_pos = torch.arange(text_len + audio_len, seq_len, device=device)

    x = torch.randn(1, seq_len, video_row, device=device, dtype=torch.float32)
    audio_x = torch.randn(1, seq_len, cfg.audio_latents_dim, device=device, dtype=torch.float32)
    img_position_ids = torch.zeros(1, seq_len, 3, device=device, dtype=torch.float32)
    img_position_ids[0, :, 0] = torch.arange(seq_len, device=device)

    token_tags = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    token_tags[text_pos] = 1
    token_tags[audio_pos] = 2
    token_tags[img_pos] = 0
    inverse_indices = torch.zeros(seq_len, dtype=torch.long, device=device)

    prompt_embeds = torch.randn(text_len, cfg.text_dim, device=device, dtype=torch.bfloat16)
    unique_timesteps = torch.tensor([0.5], device=device, dtype=torch.float32)
    update_mask = torch.ones(video_len, device=device, dtype=torch.float32)

    packed = {"cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=device), "max_seqlen_q": seq_len}
    refiner = {"cu_seqlens_q": torch.tensor([0, text_len], dtype=torch.int32, device=device), "max_seqlen_q": text_len}

    return dict(
        x=x,
        audio_x=audio_x,
        img_position_ids=img_position_ids,
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        update_mask=update_mask,
        token_tags=token_tags,
        prompt_embeds=prompt_embeds,
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params=packed,
        refiner_packed_seq_params=refiner,
    ), video_len, audio_len, video_row


def test_forward_rejects_unexpected_kwargs() -> None:
    model = MiniMaxH3DiTModel(_tiny_config())
    with pytest.raises(TypeError):
        model.forward(bogus=1)


def test_forward_shapes() -> None:
    cfg = _tiny_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = MiniMaxH3DiTModel(cfg).to(device)
    with torch.no_grad():
        for block in model.blocks:
            block.adaln_proj.linear.weight.zero_()
            block.adaln_proj.linear.bias.zero_()
        model.final_layer.adaln_proj.linear.weight.zero_()
        model.final_layer.adaln_proj.linear.bias.zero_()
    model.eval()
    inputs, video_len, audio_len, video_row = _build_packed_inputs(cfg, device)
    with torch.no_grad():
        video_logits, audio_logits = model.forward(**inputs)
    assert video_logits.shape == (video_len, video_row)
    assert audio_logits.shape == (audio_len, cfg.audio_latents_dim)
    assert video_logits.dtype == torch.float32
    assert audio_logits.dtype == torch.float32
    assert torch.isfinite(video_logits).all()
    assert torch.isfinite(audio_logits).all()
