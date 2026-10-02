from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.objectives.flow_matching import flow_shift_sigmas  # noqa: E402
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    DMDTrainingBatch,
    dmd_distribution_gradient,
)
from worldfoundry.training.post_training.distillation.self_forcing import (  # noqa: E402
    SelfForcingConfig,
    SelfForcingRolloutSampler,
    shifted_few_step_schedule,
)

FIXTURE = Path(__file__).parent / "fixtures" / "source_formulas" / "self-forcing.json"


def _source() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _CausalModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.4))


class _RecordingCausalAdapter:
    def __init__(self) -> None:
        self.module = _CausalModule()
        self.events: list[str] = []
        self.selected: list[torch.Tensor] = []

    def initialize_cache(self, reference, *, sample_ids, conditioning):
        del sample_ids, conditioning
        return {
            "active_block": -1,
            "committed_blocks": 0,
            "embedding": torch.zeros(
                reference.shape[0],
                1,
                1,
                1,
                1,
                device=reference.device,
                dtype=reference.dtype,
            ),
        }

    def predict_clean_chunk(
        self,
        noisy_chunk,
        timesteps,
        sigmas,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
        training,
    ):
        del sigmas, start_frame, sample_ids, conditioning
        assert cache["committed_blocks"] == block_index
        mode = "append" if cache["active_block"] != block_index else "overwrite"
        cache["active_block"] = block_index
        step = {1000: 0, 938: 1, 833: 2, 625: 3}[round(float(timesteps[0].item()))]
        grad = "grad" if torch.is_grad_enabled() else "no-grad"
        self.events.append(f"predict:{block_index}:{step}:{mode}:{grad}")
        result = noisy_chunk * self.module.gain + cache["embedding"]
        if training:
            result.retain_grad()
            self.selected.append(result)
        return result

    def commit_clean_chunk(
        self,
        clean_chunk,
        *,
        block_index,
        start_frame,
        sample_ids,
        conditioning,
        cache,
    ):
        del start_frame, sample_ids, conditioning
        assert not clean_chunk.requires_grad
        assert cache["active_block"] == block_index
        self.events.append(f"commit:{block_index}:overwrite:clean-t0")
        cache["embedding"] = clean_chunk.mean(dim=(1, 2, 3, 4), keepdim=True)
        cache["committed_blocks"] = block_index + 1
        return cache


def _sampler(*, exit_step_mode: str = "sequence"):
    source = _source()["inputs"]["schedule"]
    schedule = shifted_few_step_schedule(
        tuple(source["raw_timesteps"]),
        num_train_timesteps=source["num_train_timesteps"],
        flow_shift=source["flow_shift"],
    )
    adapter = _RecordingCausalAdapter()
    return (
        SelfForcingRolloutSampler(
            adapter,
            SelfForcingConfig(
                schedule=schedule,
                frames_per_block=2,
                frame_dim=2,
                exit_step_mode=exit_step_mode,
            ),
        ),
        adapter,
    )


def _batch() -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=("video",),
        clean_latents=torch.zeros(1, 1, 4, 1, 1),
        conditioning={},
        unconditional_conditioning={},
    )


def test_self_forcing_source_schedule_shift_clamp_and_per_sample_dmd_math() -> None:
    source = _source()
    inputs = source["inputs"]
    expected = source["expected"]
    schedule_source = inputs["schedule"]
    schedule = shifted_few_step_schedule(
        tuple(schedule_source["raw_timesteps"]),
        num_train_timesteps=schedule_source["num_train_timesteps"],
        flow_shift=schedule_source["flow_shift"],
    )
    assert schedule.timesteps == pytest.approx(
        expected["schedule"]["effective_timesteps"],
        rel=source["rtol"],
        abs=source["atol"],
    )
    assert schedule.sigmas == pytest.approx(
        expected["schedule"]["effective_sigmas"],
        rel=source["rtol"],
        abs=source["atol"],
    )

    score = inputs["score_shift_and_clamp"]
    actual_sigmas = flow_shift_sigmas(
        torch.tensor(score["base_sigmas"]),
        score["flow_shift"],
    ).clamp(min=score["minimum"], max=score["maximum"])
    torch.testing.assert_close(
        actual_sigmas,
        torch.tensor(expected["score_shift_and_clamp"]["sigmas"]),
        rtol=source["rtol"],
        atol=source["atol"],
    )

    normalizer = inputs["per_sample_normalizer"]
    gradient, denominator = dmd_distribution_gradient(
        torch.tensor(normalizer["generated"]),
        torch.tensor(normalizer["fake_score"]),
        torch.tensor(normalizer["real_score"]),
        per_sample_normalization=True,
    )
    torch.testing.assert_close(
        denominator,
        torch.tensor(expected["per_sample_normalizer"]["denominator"]),
        rtol=source["rtol"],
        atol=source["atol"],
    )
    torch.testing.assert_close(
        gradient,
        torch.tensor(expected["per_sample_normalizer"]["gradient"]),
        rtol=source["rtol"],
        atol=source["atol"],
    )


def test_self_forcing_rollout_executes_source_overwrite_commit_trace_and_truncates_cache_gradient() -> None:
    sampler, adapter = _sampler()
    batch = _batch()
    source = _source()
    trace_inputs = source["inputs"]["rollout_trace"]
    result = sampler.rollout(
        batch,
        torch.ones_like(batch.clean_latents),
        exit_indices=tuple(trace_inputs["exit_indices"]),
        generator=torch.Generator().manual_seed(7),
        training=True,
    )

    assert adapter.events == source["expected"]["rollout_trace"]["events"]
    assert [block.block_idx for block in result.cache_state.blocks] == [0, 1]
    assert [block.frame_start for block in result.cache_state.blocks] == [0, 2]
    assert len(adapter.selected) == 2
    adapter.selected[0].retain_grad()
    adapter.selected[1].sum().backward()
    assert adapter.selected[0].grad is None
    assert adapter.selected[1].grad is not None


def test_self_forcing_training_and_inference_use_one_rollout_path() -> None:
    sampler, adapter = _sampler()
    batch = _batch()
    noise = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1, 1)
    generator = torch.Generator().manual_seed(19)
    direct = sampler.rollout(
        batch,
        noise,
        exit_indices=(3, 3),
        generator=generator,
        training=False,
    )
    direct_events = tuple(adapter.events)

    adapter.events.clear()
    inferred = sampler.inference(
        batch,
        noise,
        generator=torch.Generator().manual_seed(19),
    )

    torch.testing.assert_close(inferred.clean_latents, direct.clean_latents, rtol=0, atol=0)
    assert tuple(adapter.events) == direct_events


def test_self_forcing_exit_rng_is_sequence_wide_and_exactly_resumable() -> None:
    sampler, _ = _sampler()
    reference = _batch().clean_latents
    generator = torch.Generator().manual_seed(101)
    first = sampler.sample_exit_indices(reference, generator=generator)
    state = generator.get_state()
    expected = sampler.sample_exit_indices(reference, generator=generator)

    restored = torch.Generator()
    restored.set_state(state)
    actual = sampler.sample_exit_indices(reference, generator=restored)

    assert len(set(first)) == 1
    assert len(set(expected)) == 1
    assert actual == expected
