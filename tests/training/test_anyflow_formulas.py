from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.post_training.distillation.anyflow.math import (
    allocate_flowmap_intervals,
    anyflow_distribution_gradient,
    anyflow_dmd_proxy_loss,
    anyflow_real_guidance,
    balance_flowmap_losses,
    beta08_train_weight,
    flowmap_central_difference,
    flowmap_inference_schedule,
    flowmap_interpolate,
    flowmap_step,
    flowmap_target,
    fused_guidance_prediction,
    shift_flowmap_time,
)

_FIXTURE = Path(__file__).parent / "fixtures/source_formulas/anyflow.json"


def _source() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_released_shift_and_inference_grid_match_source_values() -> None:
    source = _source()
    actual = shift_flowmap_time(torch.tensor(source["inputs"]["shift_five"]), 5.0)
    torch.testing.assert_close(
        actual,
        torch.tensor(source["expected"]["shift_five"]),
        rtol=source["rtol"],
        atol=source["atol"],
    )
    schedule = flowmap_inference_schedule(2, shift=5.0, device="cpu")
    torch.testing.assert_close(
        schedule,
        torch.tensor(
            source["expected"]["inference_grid_two_steps"],
            dtype=torch.float64,
        ),
        rtol=source["rtol"],
        atol=source["atol"],
    )


def test_flowmap_interpolation_and_destination_step_are_exact() -> None:
    clean = torch.ones(2, 1, 3, 1, 1)
    noise = torch.full_like(clean, 5.0)
    time = torch.tensor([0.25, 0.75])
    noisy = flowmap_interpolate(clean, noise, time)
    torch.testing.assert_close(noisy[0], torch.full_like(noisy[0], 2.0))
    velocity = noise - clean
    recovered = flowmap_step(noisy, velocity, time, torch.zeros_like(time))
    torch.testing.assert_close(recovered, clean)


def test_global_interval_allocation_uses_global_indices() -> None:
    first = torch.tensor([0.2, 0.9, 0.4])
    second = torch.tensor([0.8, 0.1, 0.6])
    time, destination, diffusion = allocate_flowmap_intervals(
        first,
        second,
        global_start_index=2,
        global_batch_size=7,
        diffusion_ratio=0.5,
        consistency_ratio=0.25,
    )
    torch.testing.assert_close(time, torch.tensor([0.8, 0.9, 0.6]))
    assert diffusion.tolist() == [True, True, False]
    torch.testing.assert_close(destination, torch.tensor([0.8, 0.9, 0.0]))


def test_central_target_and_both_guidance_conventions() -> None:
    plus = torch.tensor([[[[[7.0]]]]])
    minus = torch.tensor([[[[[1.0]]]]])
    derivative = flowmap_central_difference(
        plus,
        minus,
        epsilon=2.0,
        guidance_scale=3.0,
    )
    torch.testing.assert_close(derivative, torch.full_like(derivative, 0.5))
    clean = torch.ones_like(derivative)
    noise = torch.full_like(derivative, 5.0)
    target = flowmap_target(
        clean,
        noise,
        derivative,
        torch.tensor([8.0]),
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(target, torch.full_like(target, 1.0))
    conditional = torch.tensor([3.0])
    unconditional = torch.tensor([1.0])
    torch.testing.assert_close(
        fused_guidance_prediction(conditional, unconditional, 2.0),
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(
        anyflow_real_guidance(conditional, unconditional, 2.0),
        torch.tensor([7.0]),
    )


def test_beta08_nearest_grid_and_balancing_match_source_formula() -> None:
    source = _source()
    values = source["inputs"]["beta_weight"]
    expected = torch.tensor(source["expected"]["beta_weight"])
    actual = beta08_train_weight(
        torch.tensor(values["timesteps"]),
        num_train_timesteps=values["num_train_timesteps"],
        shift=values["shift"],
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=source["rtol"],
        atol=source["atol"],
    )
    losses = torch.tensor([2.0, 4.0, 8.0], requires_grad=True)
    balanced = balance_flowmap_losses(
        losses,
        torch.tensor([True, False, False]),
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        balanced.detach(),
        torch.tensor([2.0, 1.999995, 1.9999975]),
    )
    balanced.sum().backward()
    assert losses.grad is not None


def test_dmd_gradient_cleanup_and_fp64_proxy_inject_expected_gradient() -> None:
    generated = torch.tensor([[[[[1.0]]]], [[[[2.0]]]]], requires_grad=True)
    fake = torch.tensor([[[[[1.0]]]], [[[[4.0]]]]])
    real = torch.tensor([[[[[1.0]]]], [[[[0.0]]]]])
    gradient, normalizer = anyflow_distribution_gradient(generated, fake, real)
    assert gradient[0].item() == 0.0
    assert normalizer[0].item() == 0.0
    assert gradient[1].item() == pytest.approx(2.0)
    loss = anyflow_dmd_proxy_loss(generated, gradient)
    assert loss.dtype == torch.float64
    loss.backward()
    torch.testing.assert_close(
        generated.grad,
        torch.tensor([[[[[0.0]]]], [[[[2.0]]]]]),
    )


def test_far_partition_geometry_matches_released_cache_capacities() -> None:
    fixture = _source()
    source = fixture["inputs"]["far"]
    expected = fixture["expected"]["far"]
    partition = TemporalChunkPartition(
        chunks=tuple(source["chunk_partition"]),
        full_chunk_limit=source["full_chunk_limit"],
    )
    assert partition.frame_count == expected["frame_count"]
    assert partition.context_target_frames(expected["target_frames"]) == (
        expected["context_frames"],
        expected["target_frames"],
    )
    geometry = partition.token_geometry(
        latent_height=source["latent_height"],
        latent_width=source["latent_width"],
    )
    assert geometry.full_tokens_per_frame == expected["full_tokens_per_frame"]
    assert geometry.compressed_tokens_per_frame == expected[
        "compressed_tokens_per_frame"
    ]
    assert geometry.full_capacity == expected["full_capacity"]
    assert geometry.compressed_capacity == expected["compressed_capacity"]
