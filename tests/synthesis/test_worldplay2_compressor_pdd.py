from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.synthesis.visual_generation.worldplay2.modeling.compressor import (
    CausalMemCompressModel,
)
from worldfoundry.synthesis.visual_generation.worldplay2.scheduler import FixedPDD4Scheduler


def _small_compressor():
    model = CausalMemCompressModel(
        input_dim=36, output_dim=16, dims=(8, 8, 16, 32),
        spatial_down=(1, 1, 1), temporal_down=(1, 0, 0),
    ).eval()
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator) * 0.2 - 0.1)
    return model


def _history():
    return torch.randn(1, 36, 4, 8, 8, generator=torch.Generator().manual_seed(321))


def test_memory_compressor_matches_official_cpu_fixture():
    # WorldPlay2 c5d83e32099116ff3a1437a8a05c764d579f704b, FP32.
    expected = torch.tensor([
        -0.083155513, -0.015315615, 0.102643222, 0.071084976,
        0.083292909, 0.032873850, -0.086716771, 0.074784778,
        0.007257611, 0.019582987, 0.076510668, -0.065199517,
        -0.032419272, -0.023564085, 0.097194172, -0.134412587,
        -0.074237719, -0.008368328, 0.092378639, 0.050076704,
        0.110255465, -0.003969010, -0.085606590, 0.018940452,
        -0.007356245, 0.024954583, 0.106124252, -0.066255584,
        -0.014349535, -0.035785142, 0.066028319, -0.117542684,
    ]).reshape(1, 2, 16)
    with torch.no_grad():
        actual = _small_compressor()(_history())
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_memory_compressor_preserves_causal_history():
    model, history = _small_compressor(), _history()
    changed = history.clone()
    changed[:, :, 2:] += 2
    with torch.no_grad():
        original, altered = model(history), model(changed)
    torch.testing.assert_close(original[:, :1], altered[:, :1], atol=0, rtol=0)
    assert (original[:, 1:] - altered[:, 1:]).abs().max() > 0.1


def test_fixed_pdd_schedule_preserves_expert_boundary_and_four_blocks():
    scheduler = FixedPDD4Scheduler()
    assert [(block.expert, block.start, block.end) for block in scheduler.ordered_blocks] == [
        ("high", 0, 28), ("high", 28, 56), ("low", 56, 92), ("low", 92, 128),
    ]
    torch.testing.assert_close(scheduler.sigmas[[0, 56, 128]], torch.tensor([1.0, 0.9, 0.0]))
    assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])


@pytest.mark.parametrize("expert", ["high", "low"])
def test_compact_pdd_head_matches_interval_displacement_and_endpoint(expert):
    scheduler = FixedPDD4Scheduler()
    start, end = scheduler.expert_range(expert)
    generator = torch.Generator().manual_seed(123)
    weights = torch.randn(end - start, 4, 8, generator=generator)
    biases = torch.randn(end - start, 4, generator=generator)
    inputs = torch.randn(3, 8, generator=generator)
    compact_weights, compact_biases = scheduler.compact_head(expert, weights, biases)
    for block in scheduler.blocks(expert):
        expected_displacement = torch.zeros(3, 4)
        for index in range(block.start, block.end - 1):
            velocity = F.linear(inputs, weights[index - start], biases[index - start])
            expected_displacement += (scheduler.sigmas[index + 1] - scheduler.sigmas[index]) * velocity
        expected_endpoint = F.linear(inputs, weights[block.end - 1 - start], biases[block.end - 1 - start])
        actual = F.linear(inputs, compact_weights[block.local_index], compact_biases[block.local_index])
        torch.testing.assert_close(actual[:, :4], expected_displacement, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual[:, 4:], expected_endpoint)


class _RecordingState(Mapping):
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def __contains__(self, key):
        return key in self.tensors

    def __getitem__(self, key):
        self.reads.append(key)
        return self.tensors[key]

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)


def test_compact_checkpoint_conversion_keeps_body_and_heads_lazy():
    state = _RecordingState({
        "head.block_weight": torch.randn(2, 8, 16),
        "head.block_bias": torch.randn(2, 8),
        "blocks.0.self_attn.q.weight": torch.randn(16, 16),
    })
    converted = FixedPDD4Scheduler().convert_state_dict(state, expert="high")
    assert converted is state
    assert state.reads == []


def test_interval_checkpoint_conversion_reads_only_head_banks():
    scheduler = FixedPDD4Scheduler()
    count = scheduler.boundary_index
    body = torch.randn(16, 16)
    state = _RecordingState({
        "head.weight": torch.randn(count, 4, 8),
        "head.bias": torch.randn(count, 4),
        "blocks.0.self_attn.q.weight": body,
    })
    converted = scheduler.convert_state_dict(state, expert="high")
    assert state.reads == ["head.weight", "head.bias"]
    assert set(converted) == {"head.block_weight", "head.block_bias", "blocks.0.self_attn.q.weight"}
    assert converted["blocks.0.self_attn.q.weight"] is body
    assert converted["head.block_weight"].shape == (2, 8, 8)
