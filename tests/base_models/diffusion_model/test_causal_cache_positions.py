"""Managed MG2 cache positions must progress without reading device scalars."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import action_conditioning_21 as action_wan
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import causal_action_21 as wan
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
)


class _NoScalarRead(torch.Tensor):
    def item(self):
        pytest.fail("managed cache positions must not call Tensor.item()")


def _cache(batch, capacity, heads, width, *, device="cpu", managed=False):
    cache = {
        "k": torch.zeros(batch, capacity, heads, width, device=device),
        "v": torch.zeros(batch, capacity, heads, width, device=device),
        "global_end_index": torch.tensor(0, device=device),
        "local_end_index": torch.tensor(0, device=device),
    }
    if managed:
        cache.update(_host_global_end_index=0, _host_local_end_index=0)
        for name in ("global_end_index", "local_end_index"):
            cache[name] = cache[name].as_subclass(_NoScalarRead)
    return cache


def _sdpa(query, key, value):
    return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(
        1, 2
    )


def _assert_cache_equal(actual, expected, *, global_end, local_end):
    for key in ("k", "v", "global_end_index", "local_end_index"):
        torch.testing.assert_close(actual[key].as_subclass(torch.Tensor), expected[key])
    assert actual["_host_global_end_index"] == global_end
    assert actual["_host_local_end_index"] == local_end


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_self_attention_host_positions_match_legacy_cache_through_rewrites_and_rollover(monkeypatch, device):
    monkeypatch.setattr(wan, "attention", _sdpa)
    torch.manual_seed(107)
    attention = wan.CausalWanSelfAttention(24, 2, local_attn_size=4, sink_size=1).to(device).eval()
    expected_cache = _cache(1, 24, 2, 12, device=device)
    actual_cache = _cache(1, 24, 2, 12, device=device, managed=True)
    grid = torch.tensor([2, 1, 4])
    freqs = wan.rope_params(16, 12).to(device)
    with torch.no_grad():
        for start in [0, 0, 8, 8, 16, 24, 24, 32]:
            hidden = torch.randn(1, 8, 24, device=device)
            kwargs = dict(
                seq_lens=torch.tensor([8]), grid_sizes=grid, freqs=freqs, block_mask=None, current_start=start
            )
            expected = attention(hidden, kv_cache=expected_cache, **kwargs)
            actual = attention(hidden, kv_cache=actual_cache, **kwargs)
            torch.testing.assert_close(actual, expected)
            _assert_cache_equal(actual_cache, expected_cache, global_end=start + 8, local_end=min(start + 8, 24))


@pytest.mark.parametrize("channel", ["mouse", "keyboard"])
def test_action_host_positions_match_legacy_cache_through_context_refresh_and_rollover(monkeypatch, channel):
    monkeypatch.setattr(action_wan, "_flash_attention", _sdpa)
    torch.manual_seed(109)
    action = action_wan.ActionModule(
        hidden_size=4,
        img_hidden_size=8,
        mouse_hidden_dim=64,
        keyboard_hidden_dim=64,
        heads_num=1,
        windows_size=1,
        enable_mouse=channel == "mouse",
        enable_keyboard=channel == "keyboard",
        local_attn_size=6,
    ).eval()
    expected_cache = _cache(880 if channel == "mouse" else 1, 6, 1, 64)
    actual_cache = _cache(880 if channel == "mouse" else 1, 6, 1, 64, managed=True)
    keyboard = torch.randn(1, 45, 6)
    mouse = torch.randn(1, 45, 2)
    with torch.no_grad():
        for start in [0, 0, 2, 2, 4, 6, 6, 8]:
            hidden = torch.randn(1, 2 * 880, 8)
            final_frame = 1 + 4 * (start + 1)
            kwargs = dict(
                tt=2,
                th=1,
                tw=880,
                mouse_condition=mouse[:, :final_frame] if channel == "mouse" else None,
                keyboard_condition=keyboard[:, :final_frame],
                is_causal=True,
                start_frame=start,
                num_frame_per_block=2,
            )
            key = "kv_cache_mouse" if channel == "mouse" else "kv_cache_keyboard"
            expected = action(hidden, **kwargs, **{key: expected_cache})
            actual = action(hidden, **kwargs, **{key: actual_cache})
            torch.testing.assert_close(actual, expected)
            _assert_cache_equal(actual_cache, expected_cache, global_end=start + 2, local_end=min(start + 2, 6))


def test_pipeline_reinitializes_host_and_device_positions_together():
    # Exercise the real owner/factories at a small cache geometry, avoiding a
    # production model or multi-GiB cache allocation.
    pipeline = CausalInferencePipeline.__new__(CausalInferencePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.local_attn_size = 2
    pipeline.frame_seq_length = 4
    pipeline.num_transformer_blocks = 1
    for _ in range(2):
        pipeline._initialize_kv_cache(1, torch.float32, "cpu")
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.float32, "cpu")
        for cache in [pipeline.kv_cache1[0], pipeline.kv_cache_mouse[0], pipeline.kv_cache_keyboard[0]]:
            assert cache["_host_global_end_index"] == cache["_host_local_end_index"] == 0
            assert int(cache["global_end_index"]) == int(cache["local_end_index"]) == 0
            cache["_host_global_end_index"] = 32
            cache["_host_local_end_index"] = 8
            cache["global_end_index"].fill_(32)
            cache["local_end_index"].fill_(8)
