from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.dfd import (  # noqa: E402
    DFDConfig,
    DFDTrainingBatch,
    NativeDFDLossAdapter,
    data_forcing_teacher_data,
    dfd_distribution_gradient,
    dfd_proxy_loss_per_sample,
    prepare_dfd_student_prediction,
    shifted_uniform_timesteps,
)


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


def _expand(levels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return levels.to(reference).reshape((levels.shape[0],) + (1,) * (reference.ndim - 1))


class _Prediction:
    noise_process_kind = "flow-matching"

    def __init__(self, value: float, identity: str, *, frozen: bool = False) -> None:
        self.module = _Scale(value)
        if frozen:
            self.module.requires_grad_(False)
        self.checkpoint_identity = identity
        self.add_noise_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.prediction_calls: list[dict[str, object]] = []

    def add_noise(self, clean_latents, noise, timesteps):
        self.add_noise_calls.append(
            (
                clean_latents.detach().clone(),
                noise.detach().clone(),
                timesteps.detach().clone(),
            )
        )
        levels = _expand(timesteps, clean_latents)
        return clean_latents + levels * (noise - clean_latents)

    def predict_clean(
        self,
        noisy_latents,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del timesteps, sample_ids, conditioning
        self.prediction_calls.append(
            {
                "input": noisy_latents.detach().clone(),
                "training": training,
                "branch": branch,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return noisy_latents * self.module.weight


class _FakeScore(_Prediction):
    def denoising_loss_per_sample(
        self,
        clean_latents,
        noisy_latents,
        noise,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del noise, timesteps, sample_ids, conditioning, training
        prediction = noisy_latents * self.module.weight
        return (prediction - clean_latents).float().square().flatten(1).mean(1)


class _Discriminator:
    def __init__(self) -> None:
        self.module = _Scale(0.3)
        self.checkpoint_identity = "disc"
        self.inputs: list[torch.Tensor] = []

    def discriminator_logits(
        self,
        noisy_latents,
        timesteps,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del timesteps, sample_ids, conditioning, training
        self.inputs.append(noisy_latents.detach().clone())
        return noisy_latents.float().flatten(1).mean(1) * self.module.weight


def _batch() -> DFDTrainingBatch:
    return DFDTrainingBatch(
        sample_ids=("first", "second"),
        real_latents=torch.tensor([[1.0, -1.0], [0.5, 2.0]]),
        conditioning={"prompt": "paired"},
        unconditional_conditioning={"prompt": ""},
    )


def _config(**changes: object) -> DFDConfig:
    values: dict[str, object] = {
        "student_timesteps": (0.9, 0.5, 0.0),
        "generator_adversarial_weight": 0.03,
        "discriminator_weight": 1.0,
    }
    values.update(changes)
    return DFDConfig(**values)


def test_data_forcing_uses_real_values_with_identity_generated_gradient() -> None:
    generated = torch.tensor([[2.0, 3.0]], requires_grad=True)
    real = torch.tensor([[5.0, 7.0]])
    teacher_data = data_forcing_teacher_data(generated, real, enabled=True)
    torch.testing.assert_close(teacher_data, real)
    teacher_data.sum().backward()
    torch.testing.assert_close(generated.grad, torch.ones_like(generated))
    detached = generated.detach()
    assert data_forcing_teacher_data(detached, real, enabled=False) is detached


def test_distribution_gradient_uses_clean_generation_normalizer_and_proxy_field() -> None:
    generated = torch.tensor([[2.0, 4.0], [1.0, 5.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    fake = torch.tensor([[3.0, 5.0], [2.0, 7.0]])
    gradient, normalizer = dfd_distribution_gradient(
        generated,
        fake,
        teacher,
        epsilon=1.0,
    )
    torch.testing.assert_close(normalizer, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(gradient, (fake - teacher) / 3.0)
    loss = dfd_proxy_loss_per_sample(generated, gradient).sum()
    loss.backward()
    torch.testing.assert_close(generated.grad, gradient / generated.shape[1])


def test_shifted_sampler_matches_released_rational_transform_and_clamp() -> None:
    uniform = torch.tensor([0.0, 0.5, 0.999999], dtype=torch.float64)
    actual = shifted_uniform_timesteps(
        uniform,
        minimum=0.001,
        maximum=0.999,
        shift=5.0,
    )
    raw = uniform * 0.998 + 0.001
    expected = (raw * 5.0 / (raw * 4.0 + 1.0)).clamp(0.001, 0.999)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype is torch.float64


def test_multistep_training_selects_one_real_noised_state_and_one_prediction() -> None:
    student = _Prediction(0.8, "student")
    batch = _batch()
    result = prepare_dfd_student_prediction(
        student,
        batch,
        _config(),
        generator=torch.Generator().manual_seed(13),
        training=True,
    )
    assert len(student.prediction_calls) == 1
    assert len(student.add_noise_calls) == 1
    assert set(result.timesteps.tolist()) <= {0.9, 0.5}
    clean, noise, timesteps = student.add_noise_calls[0]
    torch.testing.assert_close(clean, batch.real_latents)
    expected_input = clean + _expand(timesteps, clean) * (noise - clean)
    torch.testing.assert_close(result.input_latents, expected_input)


@pytest.mark.parametrize("forcing", [False, True])
def test_student_objective_matches_teacher_to_selected_data_and_isolates_roles(
    forcing: bool,
) -> None:
    student = _Prediction(0.8, "student")
    teacher = _Prediction(0.6, "teacher", frozen=True)
    fake_score = _FakeScore(0.7, "fake")
    discriminator = _Discriminator()
    losses = NativeDFDLossAdapter(
        student,
        teacher,
        fake_score,
        _config(),
        discriminator=discriminator,
    )
    result = losses.student_loss(
        _batch(),
        data_forcing=forcing,
        generator=torch.Generator().manual_seed(19),
    )
    # Student corruption calls are: training input, generated score input,
    # and condition-matched teacher score input. The latter two share t/noise.
    assert len(student.add_noise_calls) == 3
    generated_clean, generated_noise, generated_t = student.add_noise_calls[1]
    teacher_clean, teacher_noise, teacher_t = student.add_noise_calls[2]
    torch.testing.assert_close(generated_noise, teacher_noise)
    torch.testing.assert_close(generated_t, teacher_t)
    if forcing:
        torch.testing.assert_close(teacher_clean, _batch().real_latents)
        assert not torch.equal(teacher.prediction_calls[0]["input"], fake_score.prediction_calls[0]["input"])
    else:
        torch.testing.assert_close(teacher_clean, generated_clean)
        torch.testing.assert_close(teacher.prediction_calls[0]["input"], fake_score.prediction_calls[0]["input"])
    assert [call["branch"] for call in teacher.prediction_calls] == ["positive", "negative"]
    result.loss.backward()
    assert student.module.weight.grad is not None
    assert fake_score.module.weight.grad is None
    assert teacher.module.weight.grad is None
    assert discriminator.module.weight.grad is None


def test_guidance_objective_updates_fake_score_and_discriminator_with_same_real_noise() -> None:
    student = _Prediction(0.8, "student")
    teacher = _Prediction(0.6, "teacher", frozen=True)
    fake_score = _FakeScore(0.7, "fake")
    discriminator = _Discriminator()
    losses = NativeDFDLossAdapter(
        student,
        teacher,
        fake_score,
        _config(),
        discriminator=discriminator,
    )
    result = losses.guidance_loss(
        _batch(),
        generator=torch.Generator().manual_seed(23),
    )
    assert len(fake_score.add_noise_calls) == 2
    _, fake_noise, fake_t = fake_score.add_noise_calls[0]
    real_clean, real_noise, real_t = fake_score.add_noise_calls[1]
    torch.testing.assert_close(fake_noise, real_noise)
    torch.testing.assert_close(fake_t, real_t)
    torch.testing.assert_close(real_clean, _batch().real_latents)
    result.loss.backward()
    assert student.module.weight.grad is None
    assert teacher.module.weight.grad is None
    assert fake_score.module.weight.grad is not None
    assert discriminator.module.weight.grad is not None


def test_paper_no_gan_profile_does_not_require_discriminator() -> None:
    config = _config(generator_adversarial_weight=0.0, discriminator_weight=0.0)
    losses = NativeDFDLossAdapter(
        _Prediction(0.8, "student"),
        _Prediction(0.6, "teacher", frozen=True),
        _FakeScore(0.7, "fake"),
        config,
    )
    result = losses.student_loss(
        _batch(),
        data_forcing=True,
        generator=torch.Generator().manual_seed(29),
    )
    assert result.metrics["generator_adversarial"] == 0
