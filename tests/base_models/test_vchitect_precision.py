"""Precision contracts needed for the released Vchitect-2 sampler."""

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput, DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.denoisers.vchitect import VchitectDenoiser
from worldfoundry.base_models.diffusion_model.models.initializers.vchitect import VchitectLatentInitializer
from worldfoundry.base_models.diffusion_model.schedulers.vchitect import VchitectFlowMatchEulerScheduler


def test_vchitect_initial_noise_uses_official_float32_rng() -> None:
    request = DiffusionRequest(prompt="room", height=64, width=64, num_frames=2)
    actual = VchitectLatentInitializer().initialize(
        request,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    expected = torch.randn((1, 2, 16, 8, 8), generator=torch.Generator().manual_seed(42))
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vchitect_euler_rounding_matches_diffusers() -> None:
    sampling = SamplingConfig(num_inference_steps=8, seed=42)
    native = VchitectFlowMatchEulerScheduler(shift=3.0)
    released = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    released.set_timesteps(sampling.num_inference_steps, device="cpu")
    schedule = native.schedule(sampling, device=torch.device("cpu"), dtype=torch.bfloat16)
    torch.testing.assert_close(
        torch.stack([step.timestep for step in schedule]), released.timesteps, rtol=0, atol=0
    )
    actual = torch.randn((1, 2, 16, 8, 8), generator=torch.Generator().manual_seed(42))
    expected = actual.clone()
    for index, step in enumerate(schedule):
        flow = torch.full_like(actual, 0.125 + index / 32, dtype=torch.bfloat16)
        actual = native.step(flow, step, actual, generator=torch.Generator())
        expected = released.step(flow, released.timesteps[index], expected, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype is torch.bfloat16


def test_vchitect_denoiser_keeps_official_mixed_input_dtypes() -> None:
    class CaptureModel:
        def __call__(self, latents, *, encoder_hidden_states, pooled_projections, timestep):
            assert latents.dtype is torch.float32
            assert encoder_hidden_states.dtype is torch.float32
            assert pooled_projections.dtype is torch.bfloat16
            assert timestep.shape == (1,)
            return torch.zeros_like(latents, dtype=torch.bfloat16)

    latents = torch.zeros((1, 2, 16, 8, 8))
    result = VchitectDenoiser(CaptureModel())(
        DenoiserInput(
            latents=latents,
            timestep=torch.tensor(1000.0),
            next_timestep=torch.tensor(900.0),
            conditioning={
                "prompt_embeds": torch.zeros((1, 333, 4096)),
                "pooled_prompt_embeds": torch.zeros((1, 2048), dtype=torch.bfloat16),
            },
            step_index=0,
            total_steps=8,
        )
    )
    assert result.sample.dtype is torch.bfloat16
