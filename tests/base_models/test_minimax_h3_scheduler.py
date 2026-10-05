"""Stage 1 tests for the MiniMax H3 rectified-flow Euler eta=0 scheduler."""

from __future__ import annotations

import math

import pytest
import torch

from worldfoundry.base_models.diffusion_model.schedulers import minimax_h3 as sched


def test_rf_v_to_x0_matches_flow_definition() -> None:
    xt = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    t = torch.tensor(0.3)
    x0 = sched.minimax_h3_rf_v_to_x0(xt, v, t)
    assert torch.allclose(x0, xt + (1.0 - 0.3) * v, atol=1e-6)


def test_euler_eta0_step_ratio_rule() -> None:
    state = torch.randn(4, 4)
    denoised = torch.randn(4, 4)
    sigma_curr, sigma_next = 0.8, 0.5
    ratio = sigma_next / sigma_curr
    out = sched.minimax_h3_euler_eta0_step(
        state, denoised, sigma_curr=sigma_curr, sigma_next=sigma_next
    )
    assert torch.allclose(out, ratio * state + (1.0 - ratio) * denoised, atol=1e-6)


def test_euler_eta0_step_zero_sigma_is_identity() -> None:
    state = torch.randn(4, 4)
    denoised = torch.randn(4, 4)
    out = sched.minimax_h3_euler_eta0_step(state, denoised, sigma_curr=0.0, sigma_next=0.0)
    assert torch.allclose(out, state)


def test_time_shift_sigmas_monotone_with_terminal_zero() -> None:
    # The [1, 0] linspace already ends at 0, so no terminal 0 is appended and
    # the schedule length equals num_steps (matches the SGLang reference).
    sigmas = sched.minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12.0)
    assert len(sigmas) == 50
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == 0.0
    assert all(a >= b for a, b in zip(sigmas, sigmas[1:]))  # non-increasing


def test_time_shift_matches_closed_form() -> None:
    shift = 3.0
    sigmas = sched.minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=shift)
    base = torch.linspace(1.0, 0.0, 5)
    expected = [float(shift * s / (1 + (shift - 1) * s)) for s in base.tolist()]
    # First five entries equal the closed form (terminal 0 already present).
    for got, want in zip(sigmas, expected):
        assert got == pytest.approx(want, abs=1e-6)


def test_single_step_preserves_cardinality() -> None:
    assert len(sched.minimax_h3_time_shift_sigmas(num_steps=1, shift_scale=12.0)) == 1


@pytest.mark.parametrize(
    "frames,expected_aligned,expected_latent_t",
    [(1, 5, 2), (5, 5, 2), (6, 22, 7), (22, 22, 7), (39, 39, 12)],
)
def test_frame_and_latent_alignment(frames, expected_aligned, expected_latent_t) -> None:
    aligned = sched.minimax_h3_align_frame_count(frames)
    assert aligned == expected_aligned
    assert (aligned - 5) % 17 == 0 or aligned == 5
    assert sched.minimax_h3_video_latent_t(aligned) == expected_latent_t
    assert sched.minimax_h3_frame_count_from_video_latent_t(expected_latent_t) == aligned


def test_audio_latent_t_40hz() -> None:
    assert sched.minimax_h3_audio_latent_t(5.0) == 200
    assert sched.minimax_h3_audio_latent_t(4.375) == 175


def test_step_denoising_couples_modalities_independently() -> None:
    scheduler = sched.build_minimax_h3_euler_ancestral_scheduler()
    video = torch.randn(6, 96)
    audio = torch.randn(6, 32)
    v_t, a_t = torch.tensor(0.4), torch.tensor(0.6)
    out = scheduler.step_denoising(
        input_visual_latent=video,
        input_audio_latent=audio,
        timestep=v_t,
        noise_pred_visual=torch.randn(6, 96),
        noise_pred_audio=torch.randn(6, 32),
        sigma_curr=0.6,
        sigma_next=0.4,
        video_timestep=v_t,
        audio_timestep=a_t,
        video_sigma_curr=1.0 - 0.4,
        video_sigma_next=0.4,
        audio_sigma_curr=1.0 - 0.6,
        audio_sigma_next=0.3,
    )
    assert out["output_visual_latent"].shape == video.shape
    assert out["output_audio_latent"].shape == audio.shape


def test_step_denoising_rejects_inconsistent_sigma() -> None:
    scheduler = sched.build_minimax_h3_euler_ancestral_scheduler()
    with pytest.raises(ValueError):
        scheduler.step_denoising(
            input_visual_latent=torch.randn(4, 96),
            input_audio_latent=torch.randn(4, 32),
            timestep=torch.tensor(0.4),
            noise_pred_visual=torch.randn(4, 96),
            noise_pred_audio=torch.randn(4, 32),
            sigma_curr=0.9,  # != 1 - 0.4
            sigma_next=0.3,
        )
