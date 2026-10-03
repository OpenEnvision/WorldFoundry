from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents


def _runner(events, *, hook=None, final=False):
    class Conditioner:
        def encode(self, request, **kwargs):
            return Conditioning(positive={})

    class Initializer:
        def initialize(self, request, *, device, dtype, generator):
            return torch.randn(2, device=device, dtype=dtype, generator=generator)

    class Denoiser:
        def __call__(self, value):
            events.append(f"denoise:{value.step_index}")
            return DenoiserOutput(sample=value.latents + 1)

        def end_request(self, request_id, *, error):
            events.append("cleanup:error" if error else "cleanup:success")

    class Scheduler:
        def schedule(self, sampling, *, device, dtype):
            return tuple(
                SchedulerStep(
                    index=index, timestep=torch.tensor(float(index)), next_timestep=torch.tensor(float(index + 1))
                )
                for index in range(sampling.num_inference_steps)
            )

        def scale_model_input(self, latents, step):
            return latents

        def step(self, prediction, step, latents, *, generator):
            events.append(f"update:{step.index}")
            return prediction

        def final_denoise_step(self):
            return (
                SchedulerStep(index=2, timestep=torch.tensor(2.0), next_timestep=torch.tensor(3.0)) if final else None
            )

    class Decoder:
        def decode(self, latents, request):
            events.append("decode")
            return latents * 2

    class Extension(DiffusionExtension):
        def on_diffusion_complete(self, context):
            events.append("complete")
            if hook is not None:
                hook(context)

        def on_run_error(self, context, error):
            events.append("extension:error")

    return NativeDiffusionRunner(
        model_id="completion-test",
        components=RunnerComponents(
            denoiser=Denoiser(),
            conditioner=Conditioner(),
            latent_initializer=Initializer(),
            scheduler=Scheduler(),
            decoder=Decoder(),
        ),
        extensions=(Extension(),),
    )


@pytest.mark.parametrize("final", [False, True])
def test_completion_hook_is_after_final_update_and_preserves_seeded_output(final):
    request = DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1, seed=43))
    events = []
    observed = []

    def observe(context):
        assert context.final_latents is not None
        observed.append((context, context.final_latents.clone()))

    runner = _runner(events, final=final, hook=observe)
    result = runner.run(request)
    expected_events = ["denoise:0", "update:0", "denoise:1", "update:1"]
    if final:
        expected_events.append("denoise:2")
    assert events == expected_events + ["complete", "decode", "cleanup:success"]
    reference = _runner([], final=final)
    reference.extensions = ()
    expected = reference.run(request)
    torch.testing.assert_close(result.latents, expected.latents, rtol=0, atol=0)
    torch.testing.assert_close(result.sample, expected.sample, rtol=0, atol=0)
    assert len(observed) == 1
    context, completed = observed[0]
    torch.testing.assert_close(completed, result.latents, rtol=0, atol=0)
    assert context.final_latents is None


def test_completion_hook_failure_skips_decoder_and_cleans_request_state():
    contexts = []

    def fail(context):
        contexts.append(context)
        assert context.final_latents is not None
        raise RuntimeError("completion failed")

    events = []
    runner = _runner(events, hook=fail)
    request = DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1))
    with pytest.raises(RuntimeError, match="completion failed"):
        runner.run(request)
    assert "decode" not in events
    assert events[-3:] == ["complete", "extension:error", "cleanup:error"]
    assert contexts[0].final_latents is None


def test_legacy_extension_without_completion_method_still_runs():
    runner = _runner([])
    extension = runner.extensions[0]
    methods = (
        "extension_id",
        "on_run_start",
        "prepare_conditioning",
        "before_denoiser",
        "after_denoiser",
        "after_step",
        "after_decode",
        "on_run_end",
        "on_run_error",
    )
    runner.extensions = (SimpleNamespace(**{name: getattr(extension, name) for name in methods}),)
    runner.run(DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1)))
