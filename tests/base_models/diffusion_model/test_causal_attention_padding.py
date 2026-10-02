"""Numerical regressions for FlexAttention's padded, uncached Wan paths.

Use CPU SDPA in place of the Flex provider while retaining the real modules,
projections, QK normalization and RoPE. Compare against unpadded attention so
aligned sequences and teacher-forcing pairs cannot silently lose their tokens.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F


VARIANTS = "worldfoundry.base_models.diffusion_model.models.networks.wan.variants"


def _cpu_flex(monkeypatch, module, tokens, block_size):
    calls = []

    def attention(*, query, key, value, block_mask):
        assert block_mask == "causal-test-mask"
        padded = ((tokens + block_size - 1) // block_size) * block_size
        assert query.shape[2] == key.shape[2] == value.shape[2] == padded
        assert not torch.count_nonzero(query[:, :, tokens:])
        assert not torch.count_nonzero(key[:, :, tokens:])
        assert not torch.count_nonzero(value[:, :, tokens:])
        calls.append((query, key, value))
        query_positions = torch.arange(padded).unsqueeze(1)
        key_positions = torch.arange(padded).unsqueeze(0)
        # Mask out padding keys as the real causal BlockMask does.
        allowed = (key_positions <= query_positions) & (key_positions < tokens)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)

    monkeypatch.setattr(module, "flex_attention", attention)
    return calls


def _unpadded_attention(query, key, value):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True
    ).transpose(1, 2)


@pytest.mark.parametrize("variant", ["causal_action_21", "forcing.self_forcing", "forcing.causal_forcing"])
@pytest.mark.parametrize("tokens", [127, 128, 129, 256])
def test_uncached_causal_attention_matches_unpadded_sdpa(monkeypatch, variant, tokens):
    module = importlib.import_module(f"{VARIANTS}.{variant}")
    calls = _cpu_flex(monkeypatch, module, tokens, 128)
    torch.manual_seed(31)
    attention = module.CausalWanSelfAttention(dim=24, num_heads=2).eval()
    hidden = torch.randn(1, tokens, 24)
    grid = torch.tensor([1, 1, tokens])
    if variant != "causal_action_21":
        grid = grid.unsqueeze(0)
    freqs = module.rope_params(tokens, attention.head_dim)

    with torch.no_grad():
        query = attention.norm_q(attention.q(hidden)).reshape(1, tokens, 2, 12)
        key = attention.norm_k(attention.k(hidden)).reshape(1, tokens, 2, 12)
        value = attention.v(hidden).reshape(1, tokens, 2, 12)
        query = module.rope_apply(query, grid, freqs).type_as(value)
        key = module.rope_apply(key, grid, freqs).type_as(value)
        expected = attention.o(_unpadded_attention(query, key, value).flatten(2))
        actual = attention(
            hidden, seq_lens=torch.tensor([tokens]), grid_sizes=grid,
            freqs=freqs, block_mask="causal-test-mask", kv_cache=None,
        )

    assert len(calls) == 1
    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("variant", ["forcing.self_forcing", "forcing.causal_forcing"])
@pytest.mark.parametrize("clean_tokens", [63, 64, 65, 128])
def test_teacher_forcing_retains_both_rope_segments(monkeypatch, variant, clean_tokens):
    module = importlib.import_module(f"{VARIANTS}.{variant}")
    tokens = clean_tokens * 2
    calls = _cpu_flex(monkeypatch, module, tokens, 128)
    torch.manual_seed(43)
    attention = module.CausalWanSelfAttention(dim=24, num_heads=2).eval()
    hidden = torch.randn(1, tokens, 24)
    grid = torch.tensor([[1, 1, clean_tokens]])
    freqs = module.rope_params(clean_tokens, attention.head_dim)

    with torch.no_grad():
        query = attention.norm_q(attention.q(hidden)).reshape(1, tokens, 2, 12)
        key = attention.norm_k(attention.k(hidden)).reshape(1, tokens, 2, 12)
        value = attention.v(hidden).reshape(1, tokens, 2, 12)
        # Clean and noisy segments use the same positions, not consecutive RoPE.
        query = torch.cat([module.rope_apply(part, grid, freqs) for part in query.chunk(2, dim=1)], dim=1)
        key = torch.cat([module.rope_apply(part, grid, freqs) for part in key.chunk(2, dim=1)], dim=1)
        expected = attention.o(_unpadded_attention(query.type_as(value), key.type_as(value), value).flatten(2))
        actual = attention(
            hidden, seq_lens=torch.tensor([clean_tokens]), grid_sizes=grid,
            freqs=freqs, block_mask="causal-test-mask", kv_cache=None,
        )

    assert len(calls) == 1
    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("channel", ["mouse", "keyboard"])
@pytest.mark.parametrize("frames", [31, 32, 33, 64])
def test_uncached_action_attention_retains_aligned_frames(monkeypatch, channel, frames):
    module = importlib.import_module(f"{VARIANTS}.action_conditioning_21")
    calls = _cpu_flex(monkeypatch, module, frames, 32)
    torch.manual_seed(59)
    action = module.ActionModule(
        hidden_size=4, img_hidden_size=8, mouse_hidden_dim=64, keyboard_hidden_dim=64,
        heads_num=1, windows_size=1, enable_mouse=channel == "mouse",
        enable_keyboard=channel == "keyboard",
    ).eval()
    spatial_tokens = 880  # This action graph requires its real spatial geometry.
    hidden = torch.randn(1, frames * spatial_tokens, 8)
    input_frames = (frames - 1) * action.vae_time_compression_ratio + 1
    keyboard = torch.randn(1, input_frames, 6)
    mouse = torch.randn(1, input_frames, 2) if channel == "mouse" else None

    with torch.no_grad():
        actual = action(
            hidden, frames, 1, spatial_tokens, mouse_condition=mouse,
            keyboard_condition=keyboard, block_mask_mouse="causal-test-mask",
            block_mask_keyboard="causal-test-mask", is_causal=True,
            num_frame_per_block=frames,
        )
        assert len(calls) == 1
        query, key, value = (tensor[:, :, :frames] for tensor in calls[0])
        reference = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        # Restore the spatial-major query batches to the graph's frame-major tokens.
        reference = reference.transpose(1, 2).reshape(1, spatial_tokens, frames, 64)
        reference = reference.transpose(1, 2).reshape(1, frames * spatial_tokens, 64)
        projection = action.proj_mouse if channel == "mouse" else action.proj_keyboard
        expected = hidden + projection(reference)

    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
