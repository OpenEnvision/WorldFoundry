from types import SimpleNamespace

import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents


def test_nvtx_covers_runner_stages_and_balances_ranges(monkeypatch):
    events = []
    monkeypatch.setenv("WORLDFOUNDRY_NVTX", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(name))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))
    schedule = [
        SchedulerStep(index=i, timestep=torch.tensor(float(i)), next_timestep=torch.tensor(float(i + 1)))
        for i in range(4)
    ]
    runner = NativeDiffusionRunner(
        model_id="nvtx-test",
        components=RunnerComponents(
            denoiser=lambda model_input: DenoiserOutput(sample=torch.zeros_like(model_input.latents)),
            conditioner=SimpleNamespace(
                encode=lambda *a, **kw: Conditioning(positive={"prompt": "p"}, negative={"prompt": "n"})
            ),
            latent_initializer=SimpleNamespace(initialize=lambda *a, **kw: torch.zeros(1)),
            scheduler=SimpleNamespace(
                schedule=lambda *a, **kw: schedule,
                scale_model_input=lambda latents, step: latents,
                step=lambda prediction, step, latents, **kw: latents,
            ),
            decoder=SimpleNamespace(decode=lambda latents, request: latents),
        ),
    )
    output = runner.run(
        DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=4, guidance_scale=3.0))
    )
    torch.testing.assert_close(output.sample, torch.zeros(1))
    assert events[0] == "worldfoundry.run"
    assert events.count("worldfoundry.encode") == 1
    assert events.count("worldfoundry.decode") == 1
    assert events.count("worldfoundry.denoise_step") == 4
    assert events.count("worldfoundry.denoiser.positive") == 4
    assert events.count("worldfoundry.denoiser.negative") == 4
    assert events.count("pop") * 2 == len(events)
