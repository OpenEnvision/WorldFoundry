"""CPU contracts for the ``autoregressive-window`` runner and its resident stream."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from einops import rearrange

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    LatentInitialization,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.runners.autoregressive import (
    AutoregressiveStream,
    AutoregressiveWindowRunner,
)
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.wan import (
    FastVideoCausalWanSelfForcingScheduler,
)


class _KVCache:
    def __init__(self) -> None:
        self.committed: list[torch.Tensor] = []


class _WindowDenoiser:
    """Deterministic window denoiser whose output depends on cache contents and window bounds."""

    block_size = 2
    variant = "fake-window"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.trace: list[str] = []
        self.cache_refs: list[weakref.ref[_KVCache]] = []
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def create_kv_cache(self, **_: object) -> _KVCache:
        cache = _KVCache()
        self.cache_refs.append(weakref.ref(cache))
        return cache

    def live_caches(self) -> int:
        gc.collect()
        return sum(ref() is not None for ref in self.cache_refs)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        if self.fail_on_call is not None and len(self.trace) == self.fail_on_call:
            raise RuntimeError("denoiser failed")
        values = model_input.conditioning
        cache = values["kv_cache"]
        assert isinstance(cache, _KVCache)
        committed = cache.committed
        start = values["current_start"]
        end = values["current_end"]
        rope = values["start_frame_for_rope"]
        assert isinstance(start, int) and isinstance(end, int) and isinstance(rope, int)
        self.trace.append(f"{model_input.branch}:{model_input.step_index}:{start}-{end}:{rope}")
        if model_input.branch.endswith("cache-commit"):
            committed.append(model_input.latents.clone())
            return DenoiserOutput(torch.zeros_like(model_input.latents))
        memory = committed[-1].mean() + 0.01 * len(committed) if committed else torch.tensor(0.0)
        context = values["context"]
        assert isinstance(context, torch.Tensor)
        sample = (
            0.7 * model_input.latents
            + 0.001 * model_input.timestep.float()
            + 0.1 * memory
            + 0.2 * context.mean()
            + 0.003 * start
            + 0.002 * rope
        )
        return DenoiserOutput(sample.to(model_input.latents.dtype))

    def end_request(self, request_id: str, *, error: BaseException | None = None) -> None:
        self.end_calls.append((request_id, error))


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        scale = len(prompt) / 10.0
        return Conditioning(
            positive={"context": torch.full((1, 1, 2), scale)},
            negative={"context": torch.full((1, 1, 2), -scale)},
        )


@dataclass(frozen=True)
class _Case:
    mode: str
    frames: int
    guidance: float
    n_views: int = 1
    kv_prefix_frames: int = 0
    first_frame: bool = False
    context_timestep: int = 0
    renoise_dtype: torch.dtype | None = None


class _Initializer:
    """Draw full-length noise, then optional KV-only prefix or first-frame latents."""

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        del device, dtype
        case = request.inputs["case"]
        assert isinstance(case, _Case)
        noise = torch.randn(1, 1, case.frames * case.n_views, 2, 2, generator=generator)
        conditioning: dict[str, object] = {"n_views": case.n_views, "latent_frames_per_view": case.frames}
        if case.kv_prefix_frames:
            conditioning["kv_prefix_latents"] = torch.randn(1, 1, case.kv_prefix_frames, 2, 2, generator=generator)
        if case.first_frame:
            conditioning["first_frame_latent"] = torch.full((1, 1, 1, 2, 2), 0.25)
        return LatentInitialization(noise, conditioning, artifacts={"frames": case.frames})


class _Codec:
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents * 2.0


class _HookRecorder(DiffusionExtension):
    extension_id = "hook-recorder"

    def __init__(self, events: list[str], final_latents: list[torch.Tensor] | None = None) -> None:
        self.events = events
        self.final_latents = final_latents if final_latents is not None else []

    def on_run_start(self, context):
        self.events.append("on_run_start")

    def prepare_conditioning(self, context, conditioning):
        self.events.append("prepare_conditioning")
        return conditioning

    def after_step(self, context, latents):
        self.events.append("after_step")
        return latents

    def on_diffusion_complete(self, context):
        self.events.append("on_diffusion_complete")
        self.final_latents.append(context.final_latents.clone())

    def after_decode(self, context, sample):
        self.events.append("after_decode")
        return sample + 1.0

    def on_run_end(self, context):
        self.events.append("on_run_end")

    def on_run_error(self, context, error):
        self.events.append(f"on_run_error:{error}")


_CASES: dict[str, _Case] = {
    "flow-cfg": _Case(mode="flow", frames=6, guidance=3.0),
    "flow-cfg-first-frame": _Case(mode="flow", frames=7, guidance=3.0, first_frame=True),
    "flow-cfg-kv-prefix": _Case(mode="flow", frames=4, guidance=3.0, kv_prefix_frames=2),
    "distilled-x0-views": _Case(
        mode="distilled-x0",
        frames=4,
        guidance=1.0,
        n_views=2,
        context_timestep=128,
        renoise_dtype=torch.float64,
    ),
    "distilled-x0-kv-prefix": _Case(mode="distilled-x0", frames=4, guidance=1.0, kv_prefix_frames=1),
}


def _runner(
    case: str,
    *,
    denoiser: _WindowDenoiser | None = None,
    events: list[str] | None = None,
    final_latents: list[torch.Tensor] | None = None,
) -> tuple[AutoregressiveWindowRunner, _WindowDenoiser]:
    spec = _CASES[case]
    denoiser = denoiser or _WindowDenoiser()
    distilled = spec.mode == "distilled-x0"
    runner = AutoregressiveWindowRunner(
        model_id=f"ar-window-{case}",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_Initializer(),
            scheduler=FastVideoCausalWanSelfForcingScheduler(raw_timesteps=(1000, 600, 300)),
            decoder=_Codec(),
        ),
        prediction_mode=spec.mode,
        fixed_timesteps=(1000, 750, 500, 250) if distilled else (),
        context_timestep=spec.context_timestep,
        renoise_dtype=spec.renoise_dtype,
        extensions=(_HookRecorder(events if events is not None else [], final_latents),),
    )
    return runner, denoiser


def _request(case: str, *, seed: int = 11, prompt: str = "a window") -> DiffusionRequest:
    spec = _CASES[case]
    return DiffusionRequest(
        prompt=prompt,
        negative_prompt="blur",
        sampling=SamplingConfig(
            num_inference_steps=4 if spec.mode == "distilled-x0" else 3,
            guidance_scale=spec.guidance,
            seed=seed,
        ),
        inputs={"case": spec},
    )


def _assemble(
    case: str,
    stream: AutoregressiveStream,
    blocks: Sequence[torch.Tensor],
    frames: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Place the stream's first frame and step() blocks into a BCTHW buffer in the runner's (V T) output layout."""

    spec = _CASES[case]
    output = torch.zeros(1, 1, spec.n_views, spec.frames, 2, 2)
    if stream.first_frame_latent is not None:
        output[:, :, :, :1] = rearrange(stream.first_frame_latent, "B C (V T) H W -> B C V T H W", V=spec.n_views)
    for block, (start, end) in zip(blocks, frames):
        output[:, :, :, start:end] = rearrange(block, "B C (V T) H W -> B C V T H W", V=spec.n_views)
    return rearrange(output, "B C V T H W -> B C (V T) H W")


# Recorded from the block loop of AutoregressiveWindowRunner.run with the fakes above.
_EXPECTED_LATENTS: dict[str, list[float]] = {
    "distilled-x0-kv-prefix": [
        1.4919034242630005,
        1.3486051559448242,
        0.7670676708221436,
        0.31693151593208313,
        1.0275397300720215,
        0.8648508787155151,
        0.8207715749740601,
        1.5649592876434326,
        1.3009936809539795,
        0.8946972489356995,
        1.6790004968643188,
        0.6394933462142944,
        0.9811571836471558,
        1.4962952136993408,
        0.7196954488754272,
        2.3534011840820312,
    ],
    "distilled-x0-views": [
        0.6293357014656067,
        1.6033533811569214,
        0.7626348733901978,
        1.713666319847107,
        0.6688594818115234,
        0.5304795503616333,
        1.7108970880508423,
        1.7324473857879639,
        0.9685859680175781,
        1.6189299821853638,
        1.6974507570266724,
        1.6675474643707275,
        2.1662425994873047,
        1.8037664890289307,
        1.7944371700286865,
        1.656100869178772,
        1.1614927053451538,
        1.3385897874832153,
        1.6753807067871094,
        1.1297667026519775,
        0.852503776550293,
        0.8456239700317383,
        0.5541252493858337,
        1.493637204170227,
        1.2132136821746826,
        0.9481362104415894,
        1.4952770471572876,
        0.8628222346305847,
        1.1585100889205933,
        1.1832374334335327,
        1.0549511909484863,
        0.49547597765922546,
    ],
    "flow-cfg": [
        -2.033787727355957,
        -0.8958042860031128,
        -1.3262321949005127,
        -2.1537094116210938,
        -1.8202208280563354,
        -2.418610095977783,
        -1.7968231439590454,
        -2.12512469291687,
        -1.3799412250518799,
        -1.6346807479858398,
        -1.2016974687576294,
        -1.0066194534301758,
        -1.6952240467071533,
        -0.9978312253952026,
        -1.6889379024505615,
        -1.3219337463378906,
        -1.098036527633667,
        -1.6715879440307617,
        -1.183634638786316,
        -1.824531078338623,
        -1.5040929317474365,
        -1.1245577335357666,
        -1.5354061126708984,
        -0.3866420090198517,
    ],
    "flow-cfg-first-frame": [
        0.25,
        0.25,
        0.25,
        0.25,
        -1.327897310256958,
        -2.4254097938537598,
        -1.7989696264266968,
        -2.299508571624756,
        -1.8832505941390991,
        -2.152794122695923,
        -0.575356662273407,
        -1.5182706117630005,
        -1.2162375450134277,
        -1.0903205871582031,
        -2.100032091140747,
        -1.1655242443084717,
        -1.4591363668441772,
        -1.172059416770935,
        -1.3946154117584229,
        -1.0006752014160156,
        -1.093847393989563,
        -1.7088916301727295,
        -1.3664556741714478,
        -1.2992476224899292,
        -1.5762113332748413,
        -0.5416985750198364,
        -1.6962016820907593,
        -1.578962802886963,
    ],
    "flow-cfg-kv-prefix": [
        -1.4118010997772217,
        -1.3537237644195557,
        -1.2132651805877686,
        -1.6155883073806763,
        -1.9901189804077148,
        -1.2823591232299805,
        -1.2575443983078003,
        -2.257206439971924,
        -1.3347737789154053,
        -1.3550186157226562,
        -1.0229334831237793,
        -0.8628696203231812,
        -1.2275199890136719,
        -1.5878913402557373,
        -1.5539191961288452,
        -1.1864451169967651,
    ],
}
_EXPECTED_TRACE: dict[str, list[str]] = {
    "distilled-x0-kv-prefix": [
        "positive-cache-commit:0:0-1:0",
        "positive:0:1-3:1",
        "positive:1:1-3:1",
        "positive:2:1-3:1",
        "positive:3:1-3:1",
        "positive-cache-commit:3:1-3:1",
        "positive:0:3-5:3",
        "positive:1:3-5:3",
        "positive:2:3-5:3",
        "positive:3:3-5:3",
        "positive-cache-commit:3:3-5:3",
    ],
    "distilled-x0-views": [
        "positive:0:0-4:0",
        "positive:1:0-4:0",
        "positive:2:0-4:0",
        "positive:3:0-4:0",
        "positive-cache-commit:3:0-4:0",
        "positive:0:4-8:2",
        "positive:1:4-8:2",
        "positive:2:4-8:2",
        "positive:3:4-8:2",
        "positive-cache-commit:3:4-8:2",
    ],
    "flow-cfg": [
        "positive:0:0-2:0",
        "negative:0:0-2:0",
        "positive:1:0-2:0",
        "negative:1:0-2:0",
        "positive:2:0-2:0",
        "negative:2:0-2:0",
        "positive-cache-commit:2:0-2:0",
        "negative-cache-commit:2:0-2:0",
        "positive:0:2-4:2",
        "negative:0:2-4:2",
        "positive:1:2-4:2",
        "negative:1:2-4:2",
        "positive:2:2-4:2",
        "negative:2:2-4:2",
        "positive-cache-commit:2:2-4:2",
        "negative-cache-commit:2:2-4:2",
        "positive:0:4-6:4",
        "negative:0:4-6:4",
        "positive:1:4-6:4",
        "negative:1:4-6:4",
        "positive:2:4-6:4",
        "negative:2:4-6:4",
        "positive-cache-commit:2:4-6:4",
        "negative-cache-commit:2:4-6:4",
    ],
    "flow-cfg-first-frame": [
        "positive-cache-commit:2:0-1:0",
        "negative-cache-commit:2:0-1:0",
        "positive:0:1-3:1",
        "negative:0:1-3:1",
        "positive:1:1-3:1",
        "negative:1:1-3:1",
        "positive:2:1-3:1",
        "negative:2:1-3:1",
        "positive-cache-commit:2:1-3:1",
        "negative-cache-commit:2:1-3:1",
        "positive:0:3-5:3",
        "negative:0:3-5:3",
        "positive:1:3-5:3",
        "negative:1:3-5:3",
        "positive:2:3-5:3",
        "negative:2:3-5:3",
        "positive-cache-commit:2:3-5:3",
        "negative-cache-commit:2:3-5:3",
        "positive:0:5-7:5",
        "negative:0:5-7:5",
        "positive:1:5-7:5",
        "negative:1:5-7:5",
        "positive:2:5-7:5",
        "negative:2:5-7:5",
        "positive-cache-commit:2:5-7:5",
        "negative-cache-commit:2:5-7:5",
    ],
    "flow-cfg-kv-prefix": [
        "positive-cache-commit:0:0-2:0",
        "negative-cache-commit:0:0-2:0",
        "positive:0:2-4:2",
        "negative:0:2-4:2",
        "positive:1:2-4:2",
        "negative:1:2-4:2",
        "positive:2:2-4:2",
        "negative:2:2-4:2",
        "positive-cache-commit:2:2-4:2",
        "negative-cache-commit:2:2-4:2",
        "positive:0:4-6:4",
        "negative:0:4-6:4",
        "positive:1:4-6:4",
        "negative:1:4-6:4",
        "positive:2:4-6:4",
        "negative:2:4-6:4",
        "positive-cache-commit:2:4-6:4",
        "negative-cache-commit:2:4-6:4",
    ],
}
_EXPECTED_EVENTS: dict[str, list[str]] = {
    "distilled-x0-kv-prefix": [
        "on_run_start",
        "prepare_conditioning",
        "on_diffusion_complete",
        "after_decode",
        "on_run_end",
    ],
    "distilled-x0-views": [
        "on_run_start",
        "prepare_conditioning",
        "on_diffusion_complete",
        "after_decode",
        "on_run_end",
    ],
    "flow-cfg": [
        "on_run_start",
        "prepare_conditioning",
        *["after_step"] * 9,
        "on_diffusion_complete",
        "after_decode",
        "on_run_end",
    ],
    "flow-cfg-first-frame": [
        "on_run_start",
        "prepare_conditioning",
        *["after_step"] * 9,
        "on_diffusion_complete",
        "after_decode",
        "on_run_end",
    ],
    "flow-cfg-kv-prefix": [
        "on_run_start",
        "prepare_conditioning",
        *["after_step"] * 6,
        "on_diffusion_complete",
        "after_decode",
        "on_run_end",
    ],
}
_EXPECTED_METADATA: dict[str, dict[str, object]] = {
    "distilled-x0-kv-prefix": {
        "model_id": "ar-window-distilled-x0-kv-prefix",
        "seed": 11,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "execution_strategy": "autoregressive-window",
        "prediction_mode": "distilled-x0",
        "block_size": 2,
        "n_views": 1,
        "first_frame_conditioned": False,
        "kv_prefix_frames": 1,
    },
    "distilled-x0-views": {
        "model_id": "ar-window-distilled-x0-views",
        "seed": 11,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "execution_strategy": "autoregressive-window",
        "prediction_mode": "distilled-x0",
        "block_size": 2,
        "n_views": 2,
        "first_frame_conditioned": False,
        "kv_prefix_frames": 0,
    },
    "flow-cfg": {
        "model_id": "ar-window-flow-cfg",
        "seed": 11,
        "num_inference_steps": 3,
        "guidance_scale": 3.0,
        "execution_strategy": "autoregressive-window",
        "prediction_mode": "flow",
        "block_size": 2,
        "n_views": 1,
        "first_frame_conditioned": False,
        "kv_prefix_frames": 0,
    },
    "flow-cfg-first-frame": {
        "model_id": "ar-window-flow-cfg-first-frame",
        "seed": 11,
        "num_inference_steps": 3,
        "guidance_scale": 3.0,
        "execution_strategy": "autoregressive-window",
        "prediction_mode": "flow",
        "block_size": 2,
        "n_views": 1,
        "first_frame_conditioned": True,
        "kv_prefix_frames": 0,
    },
    "flow-cfg-kv-prefix": {
        "model_id": "ar-window-flow-cfg-kv-prefix",
        "seed": 11,
        "num_inference_steps": 3,
        "guidance_scale": 3.0,
        "execution_strategy": "autoregressive-window",
        "prediction_mode": "flow",
        "block_size": 2,
        "n_views": 1,
        "first_frame_conditioned": False,
        "kv_prefix_frames": 2,
    },
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_run_matches_recorded_rollout(case: str) -> None:
    events: list[str] = []
    runner, denoiser = _runner(case, events=events)
    output = runner.run(_request(case))
    expected = torch.tensor(_EXPECTED_LATENTS[case], dtype=torch.float32).reshape(output.latents.shape)
    assert torch.equal(output.latents, expected)
    assert torch.equal(output.sample, output.latents * 2.0 + 1.0)
    assert denoiser.trace == _EXPECTED_TRACE[case]
    assert events == _EXPECTED_EVENTS[case]
    assert output.metadata == _EXPECTED_METADATA[case]
    assert output.artifacts == {"frames": _CASES[case].frames}
    assert [error for _, error in denoiser.end_calls] == [None]


def _step_all(stream: AutoregressiveStream) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    blocks: list[torch.Tensor] = []
    frames: list[tuple[int, int]] = []
    already_completed = stream.completed_blocks
    while not stream.done:
        blocks.append(stream.step().clone())
        assert stream.last_block_frames is not None
        frames.append(stream.last_block_frames)
        assert stream.completed_blocks == already_completed + len(blocks)
    return blocks, frames


@pytest.mark.parametrize("case", sorted(_CASES))
def test_stream_blocks_assemble_to_run_latents(case: str) -> None:
    events: list[str] = []
    final_latents: list[torch.Tensor] = []
    runner, denoiser = _runner(case, events=events, final_latents=final_latents)
    stream = runner.open_stream(_request(case))
    assert stream.completed_blocks == 0
    assert stream.last_block_frames is None
    assert (stream.first_frame_latent is not None) == _CASES[case].first_frame
    blocks, frames = _step_all(stream)
    assert stream.total_blocks == len(blocks)
    stream.close()

    expected_steps = _EXPECTED_EVENTS[case].count("after_step")
    assert events == [
        "on_run_start",
        "prepare_conditioning",
        *["after_step"] * expected_steps,
        "on_diffusion_complete",
        "on_run_end",
    ]
    assert denoiser.trace == _EXPECTED_TRACE[case]
    assert [error for _, error in denoiser.end_calls] == [None]
    assert denoiser.live_caches() == 0
    assembled = _assemble(case, stream, blocks, frames)
    assert len(final_latents) == 1
    assert torch.equal(final_latents[0], assembled)
    assert torch.equal(assembled, runner.run(_request(case)).latents)


@pytest.mark.parametrize("case", ["flow-cfg", "distilled-x0-views"])
def test_runner_isolates_a_b_a_requests_across_cancelled_stream(case: str) -> None:
    runner, denoiser = _runner(case)
    first = runner.run(_request(case))
    cancelled = runner.open_stream(_request(case, seed=12, prompt="another window"))
    cancelled.step()
    cancelled.close()
    other = runner.run(_request(case, seed=12, prompt="another window"))
    again = runner.run(_request(case))

    assert not torch.equal(first.latents, other.latents)
    assert torch.equal(first.latents, again.latents)
    assert torch.equal(first.sample, again.sample)
    assert denoiser.live_caches() == 0


def test_close_mid_rollout_releases_caches_and_ends_run_once() -> None:
    events: list[str] = []
    runner, denoiser = _runner("flow-cfg", events=events)
    stream = runner.open_stream(_request("flow-cfg"))
    stream.step()
    assert denoiser.live_caches() == 2

    stream.close()
    stream.close()

    assert events.count("on_run_end") == 1
    assert "on_diffusion_complete" not in events
    assert not any(event.startswith("on_run_error") for event in events)
    assert [error for _, error in denoiser.end_calls] == [None]
    assert denoiser.live_caches() == 0
    assert stream.completed_blocks == 1
    assert not stream.done
    with pytest.raises(RuntimeError, match="closed"):
        stream.step()


def test_step_failure_closes_stream_and_next_stream_matches_fresh_run() -> None:
    events: list[str] = []
    failing_call = _EXPECTED_TRACE["flow-cfg"].index("positive:0:2-4:2")
    runner, denoiser = _runner("flow-cfg", denoiser=_WindowDenoiser(fail_on_call=failing_call), events=events)
    stream = runner.open_stream(_request("flow-cfg"))
    stream.step()
    with pytest.raises(RuntimeError, match="denoiser failed") as caught:
        stream.step()

    assert events.count("on_run_error:denoiser failed") == 1
    assert "on_run_end" not in events
    assert denoiser.end_calls[-1][1] is caught.value
    assert stream.completed_blocks == 1
    with pytest.raises(RuntimeError, match="closed"):
        stream.step()
    stream.close()
    assert len(denoiser.end_calls) == 1
    del caught
    denoiser.end_calls.clear()
    assert denoiser.live_caches() == 0

    denoiser.fail_on_call = None
    recovered = runner.open_stream(_request("flow-cfg"))
    blocks, frames = _step_all(recovered)
    recovered.close()
    reference, _ = _runner("flow-cfg")
    assert torch.equal(_assemble("flow-cfg", recovered, blocks, frames), reference.run(_request("flow-cfg")).latents)


def test_runner_allows_one_active_stream() -> None:
    events: list[str] = []
    runner, _ = _runner("flow-cfg", events=events)
    stream = runner.open_stream(_request("flow-cfg"))
    first = stream.step().clone()
    first_frames = stream.last_block_frames
    with pytest.raises(RuntimeError, match="already open"):
        runner.open_stream(_request("flow-cfg"))
    with pytest.raises(RuntimeError, match="already open"):
        runner.run(_request("flow-cfg"))
    assert events.count("on_run_start") == 1

    assert first_frames == (0, 2)
    blocks, frames = _step_all(stream)
    stream.close()
    assembled = _assemble("flow-cfg", stream, [first, *blocks], [first_frames, *frames])
    assert torch.equal(assembled, runner.run(_request("flow-cfg")).latents)


def test_first_frame_only_request_completes_at_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_CASES, "first-frame-only", _Case(mode="flow", frames=1, guidance=3.0, first_frame=True))
    events: list[str] = []
    final_latents: list[torch.Tensor] = []
    runner, _ = _runner("first-frame-only", events=events, final_latents=final_latents)
    stream = runner.open_stream(_request("first-frame-only"))
    assert stream.done and stream.total_blocks == 0
    assert stream.first_frame_latent is not None
    assert len(final_latents) == 1
    assert torch.equal(final_latents[0], stream.first_frame_latent)
    stream.close()

    output = runner.run(_request("first-frame-only"))
    assert torch.equal(output.latents, stream.first_frame_latent)
    assert events[-3:] == ["on_diffusion_complete", "after_decode", "on_run_end"]
    assert len(final_latents) == 2


def _first_block_runner(
    *,
    fixed_timesteps: tuple[int, ...],
    first_block_timesteps: tuple[int, ...],
    denoiser: _WindowDenoiser,
) -> AutoregressiveWindowRunner:
    return AutoregressiveWindowRunner(
        model_id="ar-window-first-block",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_Initializer(),
            scheduler=FastVideoCausalWanSelfForcingScheduler(raw_timesteps=(1000, 600, 300)),
            decoder=_Codec(),
        ),
        prediction_mode="distilled-x0",
        fixed_timesteps=fixed_timesteps,
        first_block_timesteps=first_block_timesteps,
    )


@pytest.mark.parametrize("first_frame", [False, True])
def test_first_block_timesteps_schedule_only_the_first_noisy_block(
    first_frame: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _Case(mode="distilled-x0", frames=5 if first_frame else 4, guidance=1.0, first_frame=first_frame)
    monkeypatch.setitem(_CASES, "first-block", spec)
    denoiser = _WindowDenoiser()
    runner = _first_block_runner(
        fixed_timesteps=(1000, 500),
        first_block_timesteps=(1000, 750, 500, 250),
        denoiser=denoiser,
    )
    request = _request("first-block")
    request = DiffusionRequest(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1.0, seed=11),
        inputs=request.inputs,
    )
    runner.run(request)

    # Commits report the request's last step index (num_inference_steps - 1).
    first, second = (1, 3) if first_frame else (0, 2)
    expected = [f"positive:{index}:{first}-{first + 2}:{first}" for index in range(4)]
    expected.append(f"positive-cache-commit:1:{first}-{first + 2}:{first}")
    expected.extend(f"positive:{index}:{second}-{second + 2}:{second}" for index in range(2))
    expected.append(f"positive-cache-commit:1:{second}-{second + 2}:{second}")
    if first_frame:
        expected.insert(0, "positive-cache-commit:1:0-1:0")
    assert denoiser.trace == expected

    with pytest.raises(ValueError, match="requires 2 inference steps"):
        runner.run(_request("first-block"))


def test_first_block_timesteps_equal_to_fixed_steps_keep_the_rollout_bitwise() -> None:
    plain, _ = _runner("distilled-x0-kv-prefix")
    explicit = _first_block_runner(
        fixed_timesteps=(1000, 750, 500, 250),
        first_block_timesteps=(1000, 750, 500, 250),
        denoiser=_WindowDenoiser(),
    )
    assert plain.first_block_timesteps == ()
    request = _request("distilled-x0-kv-prefix")
    assert torch.equal(explicit.run(request).latents, plain.run(request).latents)


def test_first_block_timesteps_require_distilled_x0() -> None:
    with pytest.raises(ValueError, match="first_block_timesteps requires distilled-x0"):
        AutoregressiveWindowRunner(
            model_id="ar-window-flow",
            components=RunnerComponents(
                denoiser=_WindowDenoiser(),
                conditioner=_Conditioner(),
                latent_initializer=_Initializer(),
                scheduler=FastVideoCausalWanSelfForcingScheduler(raw_timesteps=(1000, 600, 300)),
                decoder=_Codec(),
            ),
            prediction_mode="flow",
            first_block_timesteps=(1000, 750),
        )


@pytest.mark.parametrize(("batch", "frames"), [(1, 1), (1, 3), (2, 1)])
def test_bf16_renoise_follows_official_add_noise_promotion(batch: int, frames: int) -> None:
    from worldfoundry.core.nn import FlowMatchScheduler

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    timestep, sigma = scheduler.timesteps[250], scheduler.sigmas[250]
    generator = torch.Generator().manual_seed(4)
    clean = torch.randn(batch, 16, frames, 8, 8, generator=generator).to(torch.bfloat16)
    noise = torch.randn(batch, 16, frames, 8, 8, generator=generator).to(torch.bfloat16)

    renoised = AutoregressiveWindowRunner._renoise_in_dtype(clean, noise, sigma)

    # The official runtime re-noises [B*F, C, H, W] frames with one timestep per frame.
    official = (
        scheduler.add_noise(
            clean.transpose(1, 2).flatten(0, 1),
            noise.transpose(1, 2).flatten(0, 1),
            timestep * torch.ones(batch * frames),
        )
        .unflatten(0, (batch, frames))
        .transpose(1, 2)
    )
    in_float32 = ((1 - sigma) * clean.float() + sigma * noise.float()).to(torch.bfloat16)
    assert renoised.dtype == torch.bfloat16
    assert torch.equal(renoised, official)
    # One frame per block keeps the official arithmetic in bf16, which differs from float32.
    assert torch.equal(official, in_float32) is (batch * frames > 1)
