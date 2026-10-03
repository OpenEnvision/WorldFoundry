from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    LatentInitialization,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.runners.chunked import (
    ChunkedKVCacheRunner,
)


class _ChunkedDenoiser:
    def __init__(self) -> None:
        self.request_ids: list[str | None] = []
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def streaming_cache_layout(self) -> tuple[bool, ...]:
        return (False,)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self.request_ids.append(model_input.request_id)
        cache = model_input.conditioning["kv_cache"]
        assert isinstance(cache, list)
        if bool(model_input.conditioning["save_kv_cache"]):
            frames = int(model_input.latents.shape[2])
            for block in cache:
                block[0] = torch.zeros(1, 1, frames)
                block[1] = torch.zeros(1, 1, frames)
                block[2] = torch.zeros(1, 1, frames)
                block[-1] = torch.zeros(1, 1, frames, 1)
        return DenoiserOutput(
            sample=torch.zeros_like(model_input.latents),
            extras={"kv_cache": cache},
        )

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.end_calls.append((request_id, error))


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        del request
        return Conditioning(positive={})


class _Initializer:
    def initialize(self, *_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("encoded initialization is required")

    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        **_: object,
    ) -> LatentInitialization:
        del request
        return LatentInitialization(torch.zeros(1, 1, 5, 1, 1))


class _Scheduler:
    def schedule(self, sampling, *, device, dtype) -> tuple[SchedulerStep, ...]:
        return tuple(
            SchedulerStep(
                index=index,
                timestep=torch.tensor(float(index), device=device, dtype=dtype),
                next_timestep=torch.tensor(float(index + 1), device=device, dtype=dtype),
            )
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents, step) -> torch.Tensor:
        del step
        return latents

    def step(self, model_output, step, latents, *, generator) -> torch.Tensor:
        del model_output, step, generator
        return latents


class _Codec:
    def __init__(self, *, fail_decode: bool = False) -> None:
        self.fail_decode = fail_decode

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        if self.fail_decode:
            raise RuntimeError("decode failed")
        return latents


def _request() -> DiffusionRequest:
    return DiffusionRequest(
        prompt="test",
        height=16,
        width=16,
        num_frames=33,
        sampling=SamplingConfig(
            num_inference_steps=2,
            guidance_scale=1.0,
            seed=7,
        ),
    )


def _runner(*, fail_decode: bool = False) -> tuple[ChunkedKVCacheRunner, _ChunkedDenoiser]:
    denoiser = _ChunkedDenoiser()
    codec = _Codec(fail_decode=fail_decode)
    runner = ChunkedKVCacheRunner(
        model_id="chunked-cleanup-test",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_Initializer(),
            scheduler=_Scheduler(),
            decoder=codec,
            latent_encoder=codec,
        ),
        base_chunk_frames=2,
        num_cached_chunks=1,
        device="cpu",
        dtype=torch.float32,
    )
    return runner, denoiser


def test_chunked_runner_finalizes_request_after_success() -> None:
    runner, denoiser = _runner()

    runner.run(_request())

    assert len(denoiser.end_calls) == 1
    request_id, error = denoiser.end_calls[0]
    assert request_id
    assert error is None
    assert set(denoiser.request_ids) == {request_id}


@pytest.mark.parametrize("fail_hook", [False, True])
def test_chunked_completion_runs_once_after_commits_and_cleans_on_failure(fail_hook) -> None:
    runner, denoiser = _runner()
    observed = []
    failure = RuntimeError("completion failed")

    class Observer(DiffusionExtension):
        def on_diffusion_complete(self, context):
            observed.append((context, context.final_latents.clone(), len(denoiser.request_ids)))
            if fail_hook:
                raise failure

    runner.extensions = (Observer(),)
    if fail_hook:
        with pytest.raises(RuntimeError, match="completion failed") as caught:
            runner.run(_request())
        assert caught.value is failure
        assert denoiser.end_calls[0][1] is failure
    else:
        output = runner.run(_request())
        torch.testing.assert_close(observed[0][1], output.latents, rtol=0, atol=0)
        reference, _ = _runner()
        torch.testing.assert_close(output.sample, reference.run(_request()).sample, rtol=0, atol=0)
    assert len(observed) == 1
    assert observed[0][0].final_latents is None
    assert observed[0][2] == len(denoiser.request_ids)


def test_chunked_runner_finalizes_request_after_failure() -> None:
    runner, denoiser = _runner(fail_decode=True)

    with pytest.raises(RuntimeError, match="decode failed") as caught:
        runner.run(_request())

    assert len(denoiser.end_calls) == 1
    request_id, error = denoiser.end_calls[0]
    assert request_id
    assert error is caught.value
    assert set(denoiser.request_ids) == {request_id}
