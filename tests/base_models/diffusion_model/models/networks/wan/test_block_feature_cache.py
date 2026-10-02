from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    WanModel,
)
from worldfoundry.core.acceleration.cache import (
    BlockTaylorSeerCache,
    DualBlockFeatureCache,
    DynamicBlockFeatureCache,
    FirstBlockFeatureCache,
)


class _CountingResidualBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(value))
        self.calls = 0
        self.inplace_requests: list[object] = []

    def forward(self, hidden, context, t_mod, freqs, **kwargs):
        del context, t_mod, freqs
        self.calls += 1
        self.inplace_requests.append(
            kwargs.get("_worldfoundry_inplace_residual", False)
        )
        return hidden + self.delta


class _SliceHead(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return hidden[..., :2]


def _tiny_counting_wan(block_count: int) -> WanModel:
    model = WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=8,
        freq_dim=4,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=0,
        eps=1e-6,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    model.blocks = torch.nn.ModuleList(
        [_CountingResidualBlock(float(index + 1)) for index in range(block_count)]
    )
    model.head = _SliceHead()
    return model


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(1, 2, 1, 1, 2),
        "timestep": torch.tensor([1.0]),
        "context": torch.randn(1, 2, 8),
    }


@pytest.mark.parametrize(
    ("cache", "block_count", "expected_calls", "expected_skips"),
    (
        (
            FirstBlockFeatureCache(0.1, downsample_factor=2),
            4,
            [2, 1, 1, 1],
            (1, 2, 3),
        ),
        (
            DualBlockFeatureCache(0.1, downsample_factor=2),
            11,
            [2] * 5 + [1] + [2] * 5,
            (5,),
        ),
        (
            DynamicBlockFeatureCache(0.1, downsample_factor=2),
            4,
            [1, 1, 1, 1],
            (0, 1, 2, 3),
        ),
    ),
)
def test_wan_model_really_skips_selected_transformer_blocks(
    cache,
    block_count: int,
    expected_calls: list[int],
    expected_skips: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    model = _tiny_counting_wan(block_count)
    inputs = _inputs()

    with torch.no_grad():
        expected = model(
            **inputs,
            feature_cache=cache,
            feature_cache_step=0,
            feature_cache_total_steps=4,
        )
        actual = model(
            **inputs,
            feature_cache=cache,
            feature_cache_step=1,
            feature_cache_total_steps=4,
        )

    torch.testing.assert_close(actual, expected)
    assert [block.calls for block in model.blocks] == expected_calls
    assert cache.events[-1].hit is True
    assert cache.events[-1].skipped_blocks == expected_skips
    assert cache.skipped_block_calls == len(expected_skips)


@pytest.mark.parametrize(
    ("cache", "block_count"),
    (
        (FirstBlockFeatureCache(0.1), 4),
        (DualBlockFeatureCache(0.1), 11),
        (DynamicBlockFeatureCache(0.1), 4),
    ),
)
def test_wan_model_autograd_executes_every_block_and_does_not_seed_reuse_state(
    cache,
    block_count: int,
) -> None:
    torch.manual_seed(0)
    model = _tiny_counting_wan(block_count)
    inputs = _inputs()
    inputs["x"].requires_grad_()

    first = model(
        **inputs,
        feature_cache=cache,
        feature_cache_step=0,
        feature_cache_total_steps=4,
    )
    second = model(
        **inputs,
        feature_cache=cache,
        feature_cache_step=1,
        feature_cache_total_steps=4,
    )
    (first.sum() + second.sum()).backward()

    assert [block.calls for block in model.blocks] == [2] * block_count
    assert [event.reason for event in cache.events] == ["autograd", "autograd"]
    assert all(event.skipped_blocks == () for event in cache.events)
    assert inputs["x"].grad is not None
    for block in model.blocks:
        assert block.delta.grad is not None
        assert torch.count_nonzero(block.delta.grad).item() == 1


def test_wan_model_does_not_route_inplace_residual_into_feature_cache() -> None:
    model = _tiny_counting_wan(2)
    cache = FirstBlockFeatureCache(0.1)

    with torch.no_grad():
        model(
            **_inputs(),
            feature_cache=cache,
            feature_cache_step=0,
            feature_cache_total_steps=4,
            _worldfoundry_inplace_residual=True,
        )

    for block in model.blocks:
        assert block.inplace_requests == [False]


def test_wan_model_does_not_route_inplace_residual_under_autograd() -> None:
    model = _tiny_counting_wan(2)
    inputs = _inputs()
    inputs["x"].requires_grad_()

    output = model(**inputs, _worldfoundry_inplace_residual=True)
    output.sum().backward()

    for block in model.blocks:
        assert block.inplace_requests == [False]
    assert inputs["x"].grad is not None


def test_dit_block_phase_dense_path_matches_normal_forward() -> None:
    torch.manual_seed(7)
    model = _tiny_counting_wan(0)
    block = DiTBlock(False, dim=12, num_heads=1, ffn_dim=24).eval()
    hidden = torch.randn(1, 2, 12)
    context = torch.randn(1, 3, 12)
    t_mod = torch.randn(1, 6, 12)
    freqs = model.rotary_frequencies((1, 1, 2), device=hidden.device)

    with torch.no_grad():
        expected = block(hidden, context, t_mod, freqs)
        actual, phases = block.forward_with_phase_cache(
            hidden,
            context,
            t_mod,
            freqs,
        )

    torch.testing.assert_close(actual, expected)
    assert set(phases) == {"self_attn_out", "cross_attn_out", "ffn_out"}
    assert all(value.shape == hidden.shape for value in phases.values())


def test_dit_block_phase_hit_applies_current_timestep_gates() -> None:
    block = DiTBlock(False, dim=4, num_heads=1, ffn_dim=8).eval()
    with torch.no_grad():
        block.modulation.zero_()
    hidden = torch.zeros(1, 2, 4)
    t_mod = torch.zeros(1, 6, 4)
    t_mod[:, 2] = 2.0  # current self-attention gate
    t_mod[:, 5] = 3.0  # current FFN gate
    cached_phases = {
        "self_attn_out": torch.ones_like(hidden),
        "cross_attn_out": torch.full_like(hidden, 4.0),
        "ffn_out": torch.full_like(hidden, 5.0),
    }

    with torch.no_grad():
        actual, returned = block.forward_with_phase_cache(
            hidden,
            torch.empty(1, 0, 4),
            t_mod,
            torch.empty(0),
            cached_phases=cached_phases,
        )

    # 0 + gate_msa(2)*self(1) + cross(4) + gate_mlp(3)*ffn(5)
    torch.testing.assert_close(actual, torch.full_like(hidden, 21.0))
    assert returned == cached_phases


def test_wan_model_block_taylor_really_skips_all_heavy_block_phases() -> None:
    torch.manual_seed(11)
    model = WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=8,
        freq_dim=4,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=2,
        eps=1e-6,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    cache = BlockTaylorSeerCache(total_steps=4)
    phase_calls = {"self": 0, "cross": 0, "ffn": 0}
    hooks = []
    for block in model.blocks:
        hooks.extend(
            (
                block.self_attn.register_forward_hook(
                    lambda *_args: phase_calls.__setitem__(
                        "self", phase_calls["self"] + 1
                    )
                ),
                block.cross_attn.register_forward_hook(
                    lambda *_args: phase_calls.__setitem__(
                        "cross", phase_calls["cross"] + 1
                    )
                ),
                block.ffn.register_forward_hook(
                    lambda *_args: phase_calls.__setitem__(
                        "ffn", phase_calls["ffn"] + 1
                    )
                ),
            )
        )
    inputs = _inputs()
    try:
        with torch.no_grad():
            dense = model(
                **inputs,
                feature_cache=cache,
                feature_cache_step=0,
                feature_cache_total_steps=4,
            )
            hit = model(
                **inputs,
                feature_cache=cache,
                feature_cache_step=1,
                feature_cache_total_steps=4,
            )
    finally:
        for hook in hooks:
            hook.remove()

    torch.testing.assert_close(hit, dense)
    assert phase_calls == {"self": 2, "cross": 2, "ffn": 2}
    assert cache.events[-1].skipped_blocks == (0, 1)
    assert cache.receipt()["skipped_block_calls"] == 2
