from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training import (  # noqa: E402
    DMDConfig,
    DMDTrainingBatch,
    FewStepSchedule,
    FlowDMDLossAdapter,
    FlowSDEIndexSchedule,
    FlowTrajectorySampler,
    NativeFlowTrajectoryReplay,
    PostTrainingParallelContext,
    clipped_policy_loss,
    dmd_distribution_gradient,
    dmd_proxy_loss,
    dmd_teacher_guidance,
    flow_match_sigma_schedule,
    flow_sde_transition,
    normalize_grouped_advantages,
    shared_variance_gaussian_kl,
    simulate_few_step_student,
)
from worldfoundry.training.post_training.rl.rollout_strategies.transition import (  # noqa: E402
    ConstantDiffusionFlowTransition,
)
from worldfoundry.training.post_training.rl.transitions.constant_diffusion import (  # noqa: E402
    constant_diffusion_flow_transition,
)
from worldfoundry.training.recipes.post_training.common import (  # noqa: E402
    scheduled_clip_range,
    validate_clip_schedule,
)


def test_clip_schedules_match_the_pinned_unirl_optimizer_step_formulas() -> None:
    assert validate_clip_schedule("linear_decay", 4) == ("linear-decay", 4)
    assert scheduled_clip_range(
        0.2,
        schedule="constant",
        schedule_steps=None,
        optimizer_step=99,
    ) == 0.2
    assert scheduled_clip_range(
        0.2,
        schedule="linear-decay",
        schedule_steps=4,
        optimizer_step=2,
    ) == pytest.approx(0.15)
    assert scheduled_clip_range(
        0.2,
        schedule="linear-decay",
        schedule_steps=4,
        optimizer_step=10,
    ) == pytest.approx(0.1)
    assert scheduled_clip_range(
        0.2,
        schedule="cosine-decay",
        schedule_steps=4,
        optimizer_step=2,
    ) == pytest.approx(0.1)
    assert scheduled_clip_range(
        0.2,
        schedule="cosine-decay",
        schedule_steps=4,
        optimizer_step=4,
    ) == pytest.approx(0.0, abs=1.0e-15)

    with pytest.raises(ValueError, match="unused by a constant"):
        validate_clip_schedule("constant", 4)
    with pytest.raises(TypeError, match="requires integer"):
        validate_clip_schedule("cosine-decay", None)


def test_flow_match_schedule_and_dynamic_sde_indices_are_deterministic() -> None:
    expected = (
        1.0,
        0.97826087474823,
        0.9545454382896423,
        0.9285714030265808,
        0.8999999761581421,
        0.8684210777282715,
        0.8333333134651184,
        0.7941176295280457,
        0.75,
        0.699999988079071,
        0.6428571343421936,
        0.5769230723381042,
        0.5,
        0.40909090638160706,
        0.30000001192092896,
        0.1666666716337204,
        0.0,
    )
    schedule = FlowSDEIndexSchedule(
        transition_count=16,
        timestep_fraction=(0.0, 0.6),
        num_sde_steps=8,
    )

    assert flow_match_sigma_schedule(16, shift=3.0) == expected
    assert schedule.resolve(0) == (0, 1, 2, 3, 4, 5, 7, 8)
    assert schedule.resolve(1) == (0, 1, 3, 4, 5, 6, 7, 8)
    assert schedule.resolve(2) == schedule.resolve(0)
    assert schedule.resolve(3) == schedule.resolve(1)
    assert schedule.identity["transition_count"] == 16


def test_flow_sde_sigma_one_uses_the_second_schedule_sigma_as_sigma_max() -> None:
    sample = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    velocity = torch.tensor([[0.75, 0.125]], dtype=torch.float32)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    sigma_next = torch.tensor([0.75], dtype=torch.float32)
    eta = 0.25
    sigma_max = 0.8

    result = flow_sde_transition(
        velocity,
        sample,
        sigma,
        sigma_next,
        eta=eta,
        sigma_max=sigma_max,
        next_sample=torch.zeros_like(sample),
        trajectory_dtype=torch.float32,
    )

    diffusion = torch.sqrt(sigma / (1.0 - torch.tensor([sigma_max]))) * eta
    dt = sigma_next - sigma
    expected_mean = sample * (1.0 + diffusion.square()[:, None] / (2.0 * sigma[:, None]) * dt[:, None])
    expected_mean += (
        velocity * (1.0 + diffusion.square()[:, None] * (1.0 - sigma[:, None]) / (2.0 * sigma[:, None])) * dt[:, None]
    )
    expected_scale = diffusion[:, None] * torch.sqrt(-dt)[:, None]

    torch.testing.assert_close(result.mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(result.scale, expected_scale, rtol=0, atol=0)


def test_constant_diffusion_transition_matches_closed_form_mean_and_scale() -> None:
    sample = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
    velocity = torch.tensor([[0.75, 0.125], [-0.25, 0.5]], dtype=torch.float32)
    sigma = torch.tensor([1.0, 0.8], dtype=torch.float32)
    sigma_next = torch.tensor([0.75, 0.4], dtype=torch.float32)
    eta = 0.25
    observed = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)

    result = constant_diffusion_flow_transition(
        velocity,
        sample,
        sigma,
        sigma_next,
        eta=eta,
        next_sample=observed,
        trajectory_dtype=torch.float32,
    )

    sigma_broadcast = sigma[:, None]
    dt = (sigma_next - sigma)[:, None]
    correction = eta**2 / (2.0 * sigma_broadcast)
    expected_mean = sample * (1.0 + correction * dt)
    expected_mean += velocity * (1.0 + correction * (1.0 - sigma_broadcast)) * dt
    expected_scale = torch.full_like(sigma_broadcast, eta) * torch.sqrt(-dt)
    torch.testing.assert_close(result.mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(result.scale, expected_scale, rtol=0, atol=0)


def test_constant_diffusion_trajectory_replay_uses_frozen_transition_identity() -> None:
    policy = _ToyFlowPredictor(0.15)
    strategy = ConstantDiffusionFlowTransition(eta=0.7)
    sampler = FlowTrajectorySampler(
        policy,
        transition_strategy=strategy,
        trajectory_dtype=torch.bfloat16,
    )
    trajectory = sampler.sample(
        torch.randn(4, 2, generator=torch.Generator().manual_seed(37)),
        torch.tensor([1.0, 0.7, 0.3, 0.0]),
        sample_ids=("a", "b", "c", "d"),
        group_ids=("first", "first", "second", "second"),
        conditioning={"bias": 0.0},
        policy_revision="constant-diffusion-policy",
        sde_step_indices=(0, 2),
        generator=torch.Generator().manual_seed(41),
    )

    replay = NativeFlowTrajectoryReplay(policy).replay(trajectory, training=True)

    assert dict(trajectory.transition_identity) == {
        "kind": "constant-diffusion",
        "eta": 0.7,
    }
    torch.testing.assert_close(replay.log_probs, trajectory.old_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(
        replay.transition_means,
        trajectory.transition_means,
        rtol=0,
        atol=0,
    )


class _ToyFlowPredictor:
    def __init__(self, gain: float, *, trainable: bool = True) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.module.weight.fill_(gain)
        self.module.requires_grad_(trainable)
        self.calls: list[tuple[bool, str]] = []

    @staticmethod
    def _sigma(sigmas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return sigmas.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del sample_ids
        self.calls.append((training, branch))
        self.module.train(training)
        bias = float(conditioning.get("bias", 0.0))
        gain = self.module.weight.reshape(())
        return noisy_latents * gain + bias

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        return noisy_latents - self._sigma(sigmas, noisy_latents) * velocity


def _dmd_batch(batch_size: int = 2) -> DMDTrainingBatch:
    return DMDTrainingBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        clean_latents=torch.linspace(-1.0, 1.0, batch_size * 4).reshape(batch_size, 1, 2, 2),
        conditioning={"bias": 0.1},
        unconditional_conditioning={"bias": -0.1},
    )


def test_few_step_schedule_is_explicit_and_strict() -> None:
    schedule = FewStepSchedule.from_effective_timesteps(
        (1000, 757, 522),
        num_train_timesteps=1000,
    )

    assert schedule.sigmas == (1.0, 0.757, 0.522)
    with pytest.raises(ValueError, match="strictly descending"):
        FewStepSchedule((1000, 757, 757), (1.0, 0.757, 0.5))
    with pytest.raises(ValueError, match=r"in \(0,1\]"):
        FewStepSchedule((1000, 500), (1.0, 0.0))


def test_dmd_guidance_gradient_and_proxy_autograd_match_golden_formula() -> None:
    conditional = torch.tensor([2.0, 4.0])
    unconditional = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(
        dmd_teacher_guidance(conditional, unconditional, 3.5),
        torch.tensor([5.5, 7.5]),
    )

    generated = torch.tensor([1.0, 3.0], requires_grad=True)
    fake = torch.tensor([2.0, 1.0])
    real = torch.tensor([0.0, 1.0])
    gradient, denominator = dmd_distribution_gradient(generated, fake, real)
    expected = (fake - real) / torch.tensor(1.5)
    torch.testing.assert_close(denominator, torch.tensor(1.5))
    torch.testing.assert_close(gradient, expected)

    loss = dmd_proxy_loss(generated, gradient)
    loss.backward()
    torch.testing.assert_close(generated.grad, expected / generated.numel())


def test_few_step_simulation_keeps_only_selected_student_step_differentiable() -> None:
    predictor = _ToyFlowPredictor(0.25)
    schedule = FewStepSchedule((1000, 750, 500), (1.0, 0.75, 0.5))
    output = simulate_few_step_student(
        predictor,
        _dmd_batch(),
        schedule,
        generator=torch.Generator().manual_seed(13),
        target_index=2,
        training=True,
    )
    output.clean_latents.square().mean().backward()

    assert predictor.calls == [(False, "positive"), (False, "positive"), (True, "positive")]
    assert output.target_index == 2
    assert predictor.module.weight.grad is not None
    assert bool(torch.isfinite(predictor.module.weight.grad).all())


def test_flow_dmd_roles_route_gradients_to_student_then_fake_score_only() -> None:
    student = _ToyFlowPredictor(0.2)
    teacher = _ToyFlowPredictor(0.6, trainable=False)
    fake_score = _ToyFlowPredictor(0.4)
    adapter = FlowDMDLossAdapter(
        student,
        teacher,
        fake_score,
        DMDConfig(
            FewStepSchedule((1000, 600), (1.0, 0.6)),
            score_min_sigma=0.2,
            score_max_sigma=0.8,
            normalization_epsilon=1.0e-8,
        ),
    )
    batch = _dmd_batch()

    generator_result = adapter.generator_loss(batch, generator=torch.Generator().manual_seed(17))
    generator_result.loss.backward()
    assert student.module.weight.grad is not None
    assert fake_score.module.weight.grad is None
    assert teacher.module.weight.grad is None

    student.module.zero_grad(set_to_none=True)
    fake_result = adapter.fake_score_loss(batch, generator=torch.Generator().manual_seed(19))
    fake_result.loss.backward()
    assert student.module.weight.grad is None
    assert fake_score.module.weight.grad is not None
    assert teacher.module.weight.grad is None
    assert torch.isfinite(generator_result.loss)
    assert torch.isfinite(fake_result.loss)


def test_flow_sde_sampling_and_storage_precision_replay_have_identical_log_prob() -> None:
    sample = torch.tensor([[0.3, -0.2], [0.7, 0.1]], dtype=torch.float32)
    velocity = torch.tensor([[0.4, -0.5], [0.2, 0.8]], dtype=torch.float32)
    sigma = torch.tensor([1.0, 0.8])
    sigma_next = torch.tensor([0.7, 0.4])
    sampled = flow_sde_transition(
        velocity,
        sample,
        sigma,
        sigma_next,
        eta=0.7,
        generator=torch.Generator().manual_seed(23),
        trajectory_dtype=torch.bfloat16,
    )
    replayed = flow_sde_transition(
        velocity,
        sample,
        sigma,
        sigma_next,
        eta=0.7,
        next_sample=sampled.next_sample,
        trajectory_dtype=torch.bfloat16,
    )

    assert sampled.next_sample.dtype is torch.bfloat16
    torch.testing.assert_close(replayed.mean, sampled.mean, rtol=0, atol=0)
    torch.testing.assert_close(replayed.scale, sampled.scale, rtol=0, atol=0)
    torch.testing.assert_close(replayed.log_prob, sampled.log_prob, rtol=0, atol=0)


def test_zero_eta_is_deterministic_and_has_no_policy_log_prob() -> None:
    sample = torch.tensor([[0.2, -0.4]])
    velocity = torch.tensor([[0.6, 0.1]])
    result = flow_sde_transition(velocity, sample, 0.8, 0.5, eta=0.0)

    torch.testing.assert_close(result.next_sample, sample + (0.5 - 0.8) * velocity)
    assert result.log_prob is None


def test_native_trajectory_rollout_and_replay_are_exact_before_update() -> None:
    policy = _ToyFlowPredictor(0.15)
    sampler = FlowTrajectorySampler(policy, eta=0.7, trajectory_dtype=torch.bfloat16)
    initial = torch.randn(4, 2, generator=torch.Generator().manual_seed(29))
    trajectory = sampler.sample(
        initial,
        torch.tensor([1.0, 0.7, 0.3, 0.0]),
        sample_ids=("a", "b", "c", "d"),
        group_ids=("group-a", "group-a", "group-b", "group-b"),
        conditioning={"bias": 0.0},
        policy_revision="policy-root",
        sde_step_indices=(0, 2),
        generator=torch.Generator().manual_seed(31),
    )
    replay = NativeFlowTrajectoryReplay(policy).replay(trajectory, training=True)

    assert trajectory.latents.shape == (4, 4, 2)
    assert trajectory.old_log_probs.shape == (4, 2)
    torch.testing.assert_close(replay.log_probs, trajectory.old_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(replay.transition_means, trajectory.transition_means, rtol=0, atol=0)


def test_group_advantages_handle_constant_groups_without_nan() -> None:
    rewards = torch.tensor([1.0, 3.0, 5.0, 5.0])
    result = normalize_grouped_advantages(
        rewards,
        ("first", "first", "constant", "constant"),
    )

    torch.testing.assert_close(
        result.advantages,
        torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )
    assert bool(torch.isfinite(result.advantages).all())
    with pytest.raises(ValueError, match="at least two"):
        normalize_grouped_advantages(torch.tensor([1.0, 2.0]), ("alone", "other"))


@pytest.mark.parametrize(
    ("normalization", "expected"),
    (
        (
            "group-population-variance",
            torch.tensor([-1.0, 1.0, -1.0, 1.0]) / (1.0 + 1.0e-4) ** 0.5,
        ),
        (
            "group-population-std",
            torch.tensor([-1.0, 1.0, -1.0, 1.0]) / (1.0 + 1.0e-4),
        ),
        (
            "group-sample-std",
            torch.tensor([-1.0, 1.0, -1.0, 1.0]) / (2.0**0.5 + 1.0e-4),
        ),
    ),
)
def test_group_advantage_modes_match_their_source_denominators(
    normalization: str,
    expected: torch.Tensor,
) -> None:
    result = normalize_grouped_advantages(
        torch.tensor([1.0, 3.0, 5.0, 7.0]),
        ("first", "first", "second", "second"),
        epsilon=1.0e-4,
        normalization=normalization,
    )

    torch.testing.assert_close(result.advantages, expected)


@pytest.mark.parametrize("correction", (0, 1))
def test_group_mean_global_std_keeps_group_centering(correction: int) -> None:
    rewards = torch.tensor([1.0, 3.0, 5.0, 9.0])
    mode = "group-mean-global-population-std" if correction == 0 else "group-mean-global-sample-std"
    result = normalize_grouped_advantages(
        rewards,
        ("first", "first", "second", "second"),
        epsilon=1.0e-4,
        normalization=mode,
    )
    denominator = rewards.std(correction=correction) + 1.0e-4

    torch.testing.assert_close(
        result.advantages,
        torch.tensor([-1.0, 1.0, -2.0, 2.0]) / denominator,
    )


def test_group_advantages_do_not_erase_small_but_finite_reward_differences() -> None:
    result = normalize_grouped_advantages(
        torch.tensor([0.0, 1.0e-10]),
        ("group", "group"),
        epsilon=1.0e-8,
        normalization="group-population-std",
    )

    assert result.advantages[0] < 0
    assert result.advantages[1] > 0
    assert bool(torch.isfinite(result.advantages).all())


def test_data_parallel_standard_deviation_reduces_global_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.training.post_training.shared.distributed as distributed

    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)

    def add_peer_statistics(statistics, *, op, group) -> None:
        del op, group
        statistics.add_(torch.tensor([2.0, 14.0, 106.0], dtype=statistics.dtype))

    monkeypatch.setattr(distributed.dist, "all_reduce", add_peer_statistics)
    context = PostTrainingParallelContext(rank=0, world_size=2, process_group="dp")

    population = context.global_standard_deviation(torch.tensor([1.0, 3.0]), correction=0)
    sample = context.global_standard_deviation(torch.tensor([1.0, 3.0]), correction=1)

    torch.testing.assert_close(population, torch.tensor(8.75).sqrt())
    torch.testing.assert_close(sample, torch.tensor(35.0 / 3.0).sqrt())


def test_clipped_policy_loss_is_identity_anchored_and_gaussian_kl_has_declared_reduction() -> None:
    old = torch.tensor([[-1.0, -2.0], [-0.5, -0.25]])
    advantages = torch.tensor([1.0, -1.0])
    identity = clipped_policy_loss(old.clone(), old, advantages, clip_range=1.0e-4)

    torch.testing.assert_close(identity.ratio, torch.ones_like(old))
    torch.testing.assert_close(identity.approx_kl, torch.tensor(0.0))
    torch.testing.assert_close(identity.clip_fraction, torch.tensor(0.0))
    torch.testing.assert_close(identity.loss, torch.tensor(0.0))

    new_means = torch.ones(2, 3, 4)
    reference_means = torch.zeros_like(new_means)
    scale = torch.full((2, 3, 1), 2.0)
    per_transition = shared_variance_gaussian_kl(
        new_means,
        reference_means,
        scale,
        reduction="none",
    )
    torch.testing.assert_close(per_transition, torch.full((2, 3), 0.125))
