# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Real small SD components and native schedule/reference comparisons."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.extensions.base import DiffusionExtension, DiffusionRunContext
from worldfoundry.base_models.diffusion_model.models.autoencoders.stable_diffusion import SDImageCodec
from worldfoundry.base_models.diffusion_model.models.denoisers.stable_diffusion import SDUNetDenoiser
from worldfoundry.base_models.diffusion_model.models.initializers.stable_diffusion import SDImageInitializer
from worldfoundry.base_models.diffusion_model.recipes.stable_diffusion import (
    marigold_normals_recipe,
    sd2_inpaint_recipe,
    wonderworld_depth_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.stable_diffusion import SDDiscreteScheduler

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("algorithm", ["ddim", "euler-unscaled"])
@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("steps", [1, 4, 30])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stateless_schedule_matches_reference(algorithm: str, prediction: str, steps: int, dtype: torch.dtype) -> None:
    config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": prediction,
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
    }
    native = SDDiscreteScheduler(config, algorithm=algorithm)
    reference = cast(
        DDIMScheduler | EulerDiscreteScheduler,
        (DDIMScheduler if algorithm == "ddim" else EulerDiscreteScheduler).from_config(config),
    )
    reference.set_timesteps(steps)
    if isinstance(reference, EulerDiscreteScheduler):
        reference.is_scale_input_called = True
    schedule = native.schedule(SamplingConfig(num_inference_steps=steps), device=torch.device("cpu"), dtype=dtype)
    torch.testing.assert_close(torch.stack([step.timestep for step in schedule]), reference.timesteps, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(17)
    actual = torch.randn((1, 4, 4, 4), generator=generator, dtype=dtype)
    expected = actual.clone()
    for step in schedule:
        prediction_tensor = torch.randn(actual.shape, generator=generator, dtype=dtype)
        before = actual.clone()
        actual = native.step(prediction_tensor, step, actual, generator=generator)
        if isinstance(reference, DDIMScheduler):
            expected = reference.step(prediction_tensor, int(step.timestep), expected, return_dict=False)[0]
        else:
            expected = reference.step(prediction_tensor, step.timestep, expected, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A guidance probe cannot change the next real scheduler update.
        repeated = native.step(prediction_tensor, step, before, generator=generator)
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
        assert native.scale_model_input(before, step) is before


def test_ddim_rejects_offset_overflow_and_accepts_valid_training_boundary() -> None:
    config = {"steps_offset": 1, "clip_sample": False, "set_alpha_to_one": False}
    scheduler = SDDiscreteScheduler(config, algorithm="ddim")
    with pytest.raises(ValueError, match="DDIM schedule exceeds training timesteps"):
        scheduler.schedule(SamplingConfig(num_inference_steps=1000), device=torch.device("cpu"), dtype=torch.float32)
    for offset, count in ((1, 999), (0, 1000)):
        configured = {**config, "steps_offset": offset}
        scheduler = SDDiscreteScheduler(configured, algorithm="ddim")
        reference = cast(DDIMScheduler, DDIMScheduler.from_config(configured))
        reference.set_timesteps(count)
        schedule = scheduler.schedule(
            SamplingConfig(num_inference_steps=count), device=torch.device("cpu"), dtype=torch.float32
        )
        assert len(schedule) == count and int(schedule[0].timestep) == 999
        torch.testing.assert_close(
            torch.stack([step.timestep for step in schedule]), reference.timesteps, rtol=0, atol=0
        )
        latent = torch.ones(1, 4, 2, 2)
        prediction = latent * 0.1
        actual = scheduler.step(prediction, schedule[0], latent, generator=torch.Generator())
        expected = reference.step(prediction, int(schedule[0].timestep), latent, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_euler_accepts_all_training_steps() -> None:
    scheduler = SDDiscreteScheduler(
        {"prediction_type": "v_prediction", "clip_sample": False}, algorithm="euler-unscaled"
    )
    schedule = scheduler.schedule(
        SamplingConfig(num_inference_steps=1000), device=torch.device("cpu"), dtype=torch.float32
    )
    assert len(schedule) == 1000 and float(schedule[0].timestep) == 999 and float(schedule[-1].timestep) == 0
    latent = torch.ones(1, 4, 2, 2)
    assert torch.isfinite(scheduler.step(latent * 0.1, schedule[0], latent, generator=torch.Generator())).all()


@pytest.mark.parametrize("mode,channels", [("inpaint", 9), ("prediction", 8)])
def test_tiny_real_unet_channel_order(mode: str, channels: int) -> None:
    torch.manual_seed(3)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=channels,
        out_channels=4,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
        layers_per_block=1,
    ).eval()
    noisy, rgb = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    mask, text = torch.ones(1, 1, 8, 8), torch.randn(1, 2, 8)
    conditions = {"context": text, "rgb_latents": rgb, "mask": mask, "masked_image_latents": rgb}
    model_input = DenoiserInput(noisy, torch.tensor(500), torch.tensor(250), conditions, 0, 4)
    actual = SDUNetDenoiser(unet, mode=mode, dtype=torch.float32)(model_input).sample
    sample = torch.cat((noisy, mask, rgb), dim=1) if mode == "inpaint" else torch.cat((rgb, noisy), dim=1)
    expected = unet(sample, 500, encoder_hidden_states=text).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sd_cfg_uses_one_ordered_batch_with_extension_lifecycle() -> None:
    calls: list[int] = []
    events: list[tuple[str, int]] = []

    class UNet(torch.nn.Module):
        def forward(
            self,
            pixels: torch.Tensor,
            timestep: torch.Tensor,
            *,
            encoder_hidden_states: torch.Tensor,
            return_dict: bool,
        ) -> tuple[torch.Tensor]:
            calls.append(len(pixels))
            offset = encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1)
            return (pixels[:, :4] + offset + len(pixels),)

    class Observe(DiffusionExtension):
        extension_id = "observe-cfg"

        def before_denoiser(self, context: DiffusionRunContext, model_input: DenoiserInput) -> DenoiserInput:
            events.append((model_input.branch, len(model_input.latents)))
            return model_input

        def after_denoiser(self, context: DiffusionRunContext, model_output: DenoiserOutput) -> DenoiserOutput:
            events.append(("after", len(model_output.sample)))
            return model_output

    denoiser = SDUNetDenoiser(UNet(), mode="inpaint", dtype=torch.float32)
    components = cast(RunnerComponents, SimpleNamespace(denoiser=denoiser))
    runner = NativeDiffusionRunner(model_id="sd-cfg", components=components, extensions=(Observe(),))
    request = DiffusionRequest(prompt="room", sampling=SamplingConfig(guidance_scale=7.5))
    context = DiffusionRunContext(
        request=request,
        components=components,
        generator=torch.Generator(),
        conditioning=Conditioning(
            positive={"context": torch.ones(1, 2, 8)},
            negative={"context": torch.zeros(1, 2, 8)},
            shared={"mask": torch.ones(1, 1, 2, 2), "masked_image_latents": torch.zeros(1, 4, 2, 2)},
        ),
        step=SchedulerStep(0, torch.tensor(500), torch.tensor(250)),
    )
    output = runner.predict(context, torch.zeros(1, 4, 2, 2))
    torch.testing.assert_close(output.sample, torch.full((1, 4, 2, 2), 9.5), rtol=0, atol=0)
    assert calls == [2] and events == [("cfg-pair", 2), ("after", 2)]


def test_shared_vae_mean_sampling_mask_and_noise_order() -> None:
    torch.manual_seed(5)
    vae = AutoencoderKL(
        block_out_channels=(8, 16, 16, 16),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=4,
        layers_per_block=1,
    ).eval()
    codec = SDImageCodec(vae, mode="rgb", dtype=torch.float32)
    pixels = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 16:] = 1
    request = DiffusionRequest(prompt="room", height=32, width=32, inputs={"image": pixels, "mask": mask})
    actual = SDImageInitializer(mode="inpaint").initialize_with_encoder(
        request,
        latent_encoder=codec,
        generator=torch.Generator().manual_seed(19),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(19)
    noise = torch.randn(1, 4, 4, 4, generator=generator)
    masked = (pixels * (mask < 0.5)).contiguous(memory_format=torch.channels_last)
    expected = vae.encode(masked).latent_dist.sample(generator=generator) * vae.config["scaling_factor"]
    torch.testing.assert_close(actual.latents, noise, rtol=0, atol=0)
    torch.testing.assert_close(actual.conditioning["masked_image_latents"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        codec.encode(pixels), vae.encode(pixels).latent_dist.mean * vae.config["scaling_factor"], rtol=0, atol=0
    )


def test_depth_decoder_averages_before_clipping() -> None:
    class VAE(torch.nn.Module):
        config = type("Config", (), {"scaling_factor": 0.18215})()

        def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
            return (latents[:, :3],)

    codec = SDImageCodec(VAE(), mode="depth", dtype=torch.float32)
    values = torch.tensor([3.0, -2.0, 0.5]).reshape(1, 3, 1, 1)
    actual = codec.decode(values * codec.scale, DiffusionRequest(prompt=""))
    expected = (values.mean(1, keepdim=True).clamp(-1, 1) + 1) / 2 / 200 + 0.001
    torch.testing.assert_close(actual, expected)


def test_depth_output_and_guidance_preserve_their_distinct_precisions() -> None:
    class VAE(torch.nn.Module):
        config = type("Config", (), {"scaling_factor": 1.0})()

        def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
            return (latents[:, :3],)

    codec = SDImageCodec(VAE(), mode="depth", dtype=torch.bfloat16)
    request = DiffusionRequest(prompt="")
    latent = torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16).reshape(1, 3, 1, 1)
    normalized = (latent.mean(1, keepdim=True).clamp(-1, 1) + 1) / 2
    actual = codec.decode(latent, request)
    guided = codec.decode_for_guidance(latent, request)
    assert actual.dtype == torch.float32 and guided.dtype == torch.bfloat16
    torch.testing.assert_close(actual, normalized.float() / 200 + 0.001, rtol=0, atol=0)
    torch.testing.assert_close(guided, normalized / 200 + 0.001, rtol=0, atol=0)


def test_normals_codec_matches_the_released_observer_facing_z_convention() -> None:
    class VAE(torch.nn.Module):
        config = type("Config", (), {"scaling_factor": 1.0})()

        def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
            return (latents,)

    codec = SDImageCodec(VAE(), mode="normals", dtype=torch.float32)
    normal = torch.tensor([0.3, 0.4, -0.5]).reshape(1, 3, 1, 1)
    actual = codec.decode(normal, DiffusionRequest(prompt="", height=1, width=1))
    expected = torch.tensor([0.6, 0.8, 0]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("provider", [sd2_inpaint_recipe, wonderworld_depth_recipe, marigold_normals_recipe])
def test_recipe_shares_one_codec_and_pins_all_weights(provider) -> None:
    recipe = provider()
    assert recipe.execution.strategy == "standard"
    assert recipe.execution.bindings["latent_encoder"] == recipe.execution.bindings["decoder"]
    assert all(len(checkpoint.revision) == 40 for checkpoint in recipe.checkpoints.values())
    assert all(checkpoint.files for checkpoint in recipe.checkpoints.values())
