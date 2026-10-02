from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.objectives.flow_matching import (  # noqa: E402
    flow_shift_sigmas,
)
from worldfoundry.training.post_training.distillation.dmd import (  # noqa: E402
    dmd_distribution_gradient,
    dmd_teacher_guidance,
    sample_dmd_score_sigmas,
)
from worldfoundry.training.post_training.distillation.reward_forcing import (  # noqa: E402
    NativeRewardForcingLossAdapter,
    RewardForcingConfig,
    RewardForcingTrainingBatch,
    VideoAlignMotionQualityReward,
    reward_forcing_multiplier,
    rewarded_dmd_proxy_loss,
)
from worldfoundry.training.post_training.distillation.self_forcing import (  # noqa: E402
    SelfForcingRolloutSampler,
)
from worldfoundry.training.post_training.rewards import (  # noqa: E402
    RewardResult,
)

FIXTURE = Path(__file__).parent / "fixtures" / "source_formulas" / "reward-forcing.json"


def _source() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _batch(*, count: int = 2, frames: int = 2) -> RewardForcingTrainingBatch:
    return RewardForcingTrainingBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(count)),
        clean_latents=torch.zeros(count, 1, frames, 1, 1),
        conditioning={"text_embeddings": torch.ones(count, 1)},
        unconditional_conditioning={"text_embeddings": torch.zeros(count, 1)},
        prompts=tuple(f"prompt {index}" for index in range(count)),
    )


def test_released_schedule_score_sampling_and_cfg_coefficient() -> None:
    config = RewardForcingConfig()
    assert config.schedule.timesteps == pytest.approx((1000.0, 937.5, 833.3333333333334, 625.0))
    assert config.schedule.sigmas == pytest.approx((1.0, 0.9375, 0.8333333333333334, 0.625))
    assert config.rollout_config.exit_step_mode == "sequence"
    assert config.dmd_config.shared_score_timestep is False
    assert config.dmd_config.per_sample_normalization is True
    assert config.student_scheduler_cadence == "generator-update"
    assert (
        config.local_attention_frames,
        config.ema_sink_frames,
        config.ema_sink_decay,
    ) == (9, 3, 0.999)
    with pytest.raises(ValueError, match="ema_sink_decay"):
        RewardForcingConfig(ema_sink_decay=0.0)
    with pytest.raises(ValueError, match="ema_decay"):
        RewardForcingConfig(ema_decay=0.0)

    reference = torch.zeros(4, 1, 1)
    actual_generator = torch.Generator().manual_seed(91)
    actual = sample_dmd_score_sigmas(
        reference,
        config.dmd_config,
        generator=actual_generator,
    )
    expected_generator = torch.Generator().manual_seed(91)
    indices = torch.randint(0, 1000, (4,), generator=expected_generator)
    base = indices.float() / 1000.0
    expected = (5.0 * base / (1.0 + 4.0 * base)).clamp(0.02, 0.98)
    torch.testing.assert_close(actual, expected)

    guided = dmd_teacher_guidance(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        config.teacher_guidance_scale,
    )
    # Released code uses cond + 3 * (cond - uncond), not standard CFG scale 3.
    torch.testing.assert_close(guided, torch.tensor([5.0]))


def test_rewarded_proxy_matches_released_double_precision_weighting_and_gradient() -> None:
    generated = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gradient = torch.tensor([[2.0, 4.0], [1.0, 1.0]])
    rewards = torch.tensor([0.0, 0.5])
    multipliers = reward_forcing_multiplier(rewards, beta=2.0)
    reduced = rewarded_dmd_proxy_loss(
        generated,
        gradient,
        multipliers,
        sample_weights=torch.tensor([1.0, 2.0]),
    )
    expected_numerator = 0.5 * ((2.0**2 + 4.0**2) + 2.0 * math.e * (1.0**2 + 1.0**2))
    expected_denominator = 2.0 + 4.0
    assert reduced.loss.dtype is torch.float64
    assert reduced.numerator.item() == pytest.approx(expected_numerator)
    assert reduced.denominator.item() == pytest.approx(expected_denominator)
    reduced.loss.backward()
    expected_gradient = torch.tensor(
        [
            [2.0 / expected_denominator, 4.0 / expected_denominator],
            [2.0 * math.e / expected_denominator, 2.0 * math.e / expected_denominator],
        ]
    )
    torch.testing.assert_close(generated.grad, expected_gradient)


class _Evaluator:
    def __init__(self) -> None:
        self.requests = ()

    def evaluate(self, requests):
        self.requests = requests
        return tuple(
            RewardResult(
                request_id=request.request_id,
                rollout_id=request.rollout_id,
                values={
                    "video_quality": 4.0,
                    "motion_quality": 3.0 + index * 2.0,
                    "text_alignment": 2.0,
                },
                valid={
                    "video_quality": True,
                    "motion_quality": True,
                    "text_alignment": True,
                },
                diagnostics={},
                latency_ms=0.0,
            )
            for index, request in enumerate(requests)
        )


def test_videoalign_bridge_requests_all_heads_and_returns_normalized_mq() -> None:
    evaluator = _Evaluator()
    adapter = VideoAlignMotionQualityReward(
        evaluator,
        checkpoint_identity="videoalign-checkpoint",
        calibration_mean=1.0,
        calibration_std=2.0,
    )
    batch = _batch()
    videos = torch.zeros(2, 3, 5, 2, 2)
    reward = adapter.score_motion_quality(videos, batch)
    torch.testing.assert_close(reward, torch.tensor([1.0, 2.0]))
    assert tuple(request.prompt for request in evaluator.requests) == batch.prompts
    assert all(
        request.reward_ids == ("video_quality", "motion_quality", "text_alignment") for request in evaluator.requests
    )
    for index, request in enumerate(evaluator.requests):
        torch.testing.assert_close(request.artifacts["video"], videos[index])


class _StudentModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.4))


class _CausalStudent:
    def __init__(self) -> None:
        self.module = _StudentModule()

    def initialize_cache(self, reference, *, sample_ids, conditioning):
        del reference, sample_ids, conditioning
        return {}

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
        del timesteps, sigmas, block_index, start_frame, sample_ids, conditioning, cache, training
        return noisy_chunk * self.module.gain

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
        del clean_chunk, block_index, start_frame, sample_ids, conditioning
        return cache


class _ScoreModule(torch.nn.Module):
    def __init__(self, value: float, *, frozen: bool) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))
        if frozen:
            self.requires_grad_(False)


class _Score:
    def __init__(self, value: float, *, frozen: bool = False) -> None:
        self.module = _ScoreModule(value, frozen=frozen)
        self.branches: list[str] = []
        self.training_modes: list[tuple[str, bool]] = []

    def predict_clean(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning
        self.branches.append(branch)
        self.training_modes.append(("clean", training))
        offset = -0.2 if branch == "negative" else 0.2
        return torch.zeros_like(noisy_latents) + self.module.value + offset

    def predict_velocity(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning, branch
        self.training_modes.append(("velocity", training))
        return noisy_latents * self.module.value


class _Decoder:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.checkpoint_identity = "reward-decoder-checkpoint"
        self.saw_grad_latents: list[bool] = []

    def decode_reward_videos(self, clean_latents, *, sample_ids, conditioning):
        del sample_ids, conditioning
        self.saw_grad_latents.append(clean_latents.requires_grad)
        return clean_latents.expand(-1, 3, -1, -1, -1)


class _MotionReward:
    def __init__(self) -> None:
        self.reward = torch.tensor([0.0, 0.5], requires_grad=True)
        self.checkpoint_identity = "motion-reward-checkpoint"
        self.owned_module = None
        self.calibration_mean = 1.1646
        self.calibration_std = 1.3811
        self.normalization_epsilon = 0.0

    def score_motion_quality(self, videos, batch):
        assert not videos.requires_grad
        assert videos.shape[0] == batch.batch_size
        return self.reward


def _objective():
    config = RewardForcingConfig(
        denoising_timesteps=(1000.0,),
        frames_per_block=1,
        training_frames=2,
        local_attention_frames=2,
        ema_sink_frames=1,
        ema_start_step=0,
    )
    student = _CausalStudent()
    real = _Score(0.7, frozen=True)
    fake = _Score(0.5)
    decoder = _Decoder()
    reward = _MotionReward()
    sampler = SelfForcingRolloutSampler(student, config.rollout_config)
    losses = NativeRewardForcingLossAdapter(
        real,
        fake,
        sampler,
        decoder,
        reward,
        config,
    )
    return losses, student, real, fake, decoder, reward


def test_fixed_reward_forcing_source_formulas_match_released_redmd_math() -> None:
    source = _source()
    assert set(source) == {"inputs", "expected", "atol", "rtol"}
    inputs = source["inputs"]
    expected = source["expected"]
    atol = source["atol"]
    rtol = source["rtol"]

    generated = torch.tensor(inputs["generated_clean"], dtype=torch.float32)
    fake = torch.tensor(inputs["fake_score_clean"], dtype=torch.float32)
    conditional = torch.tensor(
        inputs["real_conditional_clean"],
        dtype=torch.float32,
    )
    unconditional = torch.tensor(
        inputs["real_unconditional_clean"],
        dtype=torch.float32,
    )
    guided = dmd_teacher_guidance(
        conditional,
        unconditional,
        inputs["guidance_scale"],
    )
    gradient, normalizer = dmd_distribution_gradient(
        generated,
        fake,
        guided,
        per_sample_normalization=True,
    )
    multiplier = reward_forcing_multiplier(
        torch.tensor(inputs["normalized_motion_quality"], dtype=torch.float32),
        inputs["reward_beta"],
    )
    proxy = rewarded_dmd_proxy_loss(generated, gradient, multiplier)

    score_timesteps = torch.tensor(
        inputs["raw_score_timesteps"],
        dtype=torch.float64,
    )
    shifted_score = (
        flow_shift_sigmas(
            score_timesteps / inputs["num_train_timesteps"],
            inputs["score_flow_shift"],
        )
        * inputs["num_train_timesteps"]
    ).clamp(
        min=inputs["score_min_timestep"],
        max=inputs["score_max_timestep"],
    )
    denoising_timesteps = torch.tensor(
        inputs["denoising_timesteps"],
        dtype=torch.float64,
    )
    shifted_denoising = (
        flow_shift_sigmas(
            denoising_timesteps / inputs["num_train_timesteps"],
            inputs["denoising_flow_shift"],
        )
        * inputs["num_train_timesteps"]
    )
    ema_sink = torch.tensor(inputs["ema_sink_current"], dtype=torch.float64) * inputs["ema_sink_decay"] + torch.tensor(
        inputs["ema_sink_evicted"], dtype=torch.float64
    ) * (1.0 - inputs["ema_sink_decay"])

    torch.testing.assert_close(
        guided,
        torch.tensor(expected["guided_real_clean"]),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        normalizer,
        torch.tensor(expected["dmd_normalizer"]),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        gradient,
        torch.tensor(expected["distribution_gradient"]),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        multiplier,
        torch.tensor(expected["reward_multiplier"]),
        atol=atol,
        rtol=rtol,
    )
    assert proxy.loss.item() == pytest.approx(
        expected["rewarded_proxy_loss"],
        abs=atol,
        rel=rtol,
    )
    torch.testing.assert_close(
        shifted_score,
        torch.tensor(expected["shifted_score_timesteps"], dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        shifted_denoising,
        torch.tensor(
            expected["shifted_denoising_timesteps"],
            dtype=torch.float64,
        ),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        ema_sink,
        torch.tensor(expected["ema_sink_updated"], dtype=torch.float64),
        atol=atol,
        rtol=rtol,
    )


def test_generator_reward_is_detached_and_only_student_receives_gradient() -> None:
    losses, student, real, fake, decoder, reward = _objective()
    result = losses.generator_loss(
        _batch(),
        generator=torch.Generator().manual_seed(17),
    )
    assert decoder.saw_grad_latents == [False]
    torch.testing.assert_close(
        result.metrics["reward_multiplier"],
        torch.tensor([1.0, math.e]),
    )
    assert real.branches == ["positive", "negative"]
    assert fake.branches == ["positive"]
    result.loss.backward()
    assert student.module.gain.grad is not None
    assert fake.module.value.grad is None
    assert real.module.value.grad is None
    assert next(decoder.module.parameters()).grad is None
    assert reward.reward.grad is None


def test_fake_score_loss_uses_fresh_no_grad_rollout_and_updates_only_fake_score() -> None:
    losses, student, real, fake, decoder, reward = _objective()
    result = losses.fake_score_loss(
        _batch(),
        generator=torch.Generator().manual_seed(23),
    )
    result.loss.backward()
    assert fake.module.value.grad is not None
    assert fake.training_modes == [("velocity", False)]
    assert student.module.gain.grad is None
    assert real.module.value.grad is None
    assert decoder.saw_grad_latents == []
    assert reward.reward.grad is None
