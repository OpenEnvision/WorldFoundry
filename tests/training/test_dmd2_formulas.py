from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.dmd2.math import (  # noqa: E402
    dmd2_distribution_gradient,
    dmd2_generator_adversarial_loss,
    dmd2_guidance_adversarial_loss,
    dmd2_proxy_loss_per_sample,
    dmd2_weighted_total,
)
from worldfoundry.training.post_training.distillation.dmd2.objective import (  # noqa: E402
    dmd2_teacher_guidance,
)
from worldfoundry.training.recipes import DMD2AlgorithmSpec  # noqa: E402


def _fixture() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "source_formulas" / "dmd2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_dmd2_distribution_matching_matches_formula_fixture() -> None:
    fixture = _fixture()
    case = fixture["inputs"]["distribution_matching"]
    expected = fixture["expected"]["distribution_matching"]
    sample = torch.tensor(case["score_sample"], dtype=torch.float32)
    fake = torch.tensor(case["fake_score_clean"], dtype=torch.float32)
    real = torch.tensor(case["real_score_clean"], dtype=torch.float32)
    expected_gradient = torch.tensor(expected["gradient"], dtype=torch.float32)

    gradient, normalizer = dmd2_distribution_gradient(
        sample,
        fake,
        real,
        normalization_axes=tuple(case["normalization_axes"]),
    )

    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}
    torch.testing.assert_close(gradient, expected_gradient, **tolerance)
    torch.testing.assert_close(normalizer, torch.tensor(expected["normalizer"]), **tolerance)
    proxy = dmd2_proxy_loss_per_sample(sample.requires_grad_(), gradient)
    torch.testing.assert_close(proxy, torch.tensor(expected["proxy_per_sample"]), **tolerance)


def test_dmd2_normalizer_is_per_sample_and_requires_explicit_video_axes() -> None:
    sample = torch.tensor([[[[[2.0, 4.0]]]], [[[[10.0, 14.0]]]]])
    real = torch.zeros_like(sample)
    fake = torch.ones_like(sample)
    gradient, normalizer = dmd2_distribution_gradient(
        sample,
        fake,
        real,
        normalization_axes=(1, 2, 3, 4),
    )
    torch.testing.assert_close(normalizer, torch.tensor([3.0, 12.0]))
    assert not torch.equal(gradient[0], gradient[1])
    with pytest.raises(ValueError, match="every non-batch latent axis"):
        dmd2_distribution_gradient(
            sample,
            fake,
            real,
            normalization_axes=(1, 3, 4),
        )


def test_dmd2_batch_duplication_preserves_parameter_gradient() -> None:
    def parameter_gradient(repeats: int) -> torch.Tensor:
        parameter = torch.nn.Parameter(torch.tensor(0.4))
        features = torch.tensor([[1.0], [2.0]]).repeat_interleave(repeats, dim=0)
        score_sample = torch.tensor([[2.0], [4.0]]).repeat_interleave(repeats, dim=0)
        fake = torch.tensor([[2.0], [3.0]]).repeat_interleave(repeats, dim=0)
        real = torch.tensor([[1.0], [1.0]]).repeat_interleave(repeats, dim=0)
        generated = parameter * features
        gradient, _ = dmd2_distribution_gradient(
            score_sample,
            fake,
            real,
            normalization_axes=(1,),
        )
        dmd2_proxy_loss_per_sample(generated, gradient).mean().backward()
        assert parameter.grad is not None
        return parameter.grad.detach().clone()

    torch.testing.assert_close(parameter_gradient(1), parameter_gradient(3))


def test_dmd2_softplus_gan_losses_match_formula_fixture() -> None:
    fixture = _fixture()
    case = fixture["inputs"]["softplus_gan"]
    expected = fixture["expected"]["softplus_gan"]
    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}
    real = torch.tensor(case["real_logits"])
    fake = torch.tensor(case["fake_logits"])
    torch.testing.assert_close(
        dmd2_generator_adversarial_loss(fake),
        torch.tensor(expected["generator"]),
        **tolerance,
    )
    torch.testing.assert_close(
        dmd2_guidance_adversarial_loss(real, fake),
        torch.tensor(expected["guidance"]),
        **tolerance,
    )


def test_dmd2_teacher_guidance_has_cfg_zero_and_one_anchors() -> None:
    fixture = _fixture()
    case = fixture["inputs"]["teacher_guidance"]
    expected = fixture["expected"]["teacher_guidance"]
    conditional = torch.tensor(case["conditional"])
    unconditional = torch.tensor(case["unconditional"])
    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}
    torch.testing.assert_close(
        dmd2_teacher_guidance(conditional, unconditional, case["scales"][0]),
        torch.tensor(expected["scale_zero"]),
        **tolerance,
    )
    torch.testing.assert_close(
        dmd2_teacher_guidance(conditional, unconditional, case["scales"][1]),
        torch.tensor(expected["scale_one"]),
        **tolerance,
    )


def test_dmd2_generic_recipe_uses_official_sd_guidance_default() -> None:
    spec = DMD2AlgorithmSpec(
        student_timesteps=(999.0,),
        student_sigmas=(0.999,),
        real_score_checkpoint="teacher",
        guidance_checkpoint="guidance",
        normalization_axes=(1,),
    )
    assert spec.teacher_guidance_scale == 6.0


@pytest.mark.parametrize("role", ["generator", "guidance"])
def test_dmd2_total_loss_matches_formula_fixture(role: str) -> None:
    fixture = _fixture()
    case = fixture["inputs"]["weighted_totals"]
    components = {
        name: torch.tensor(values)
        for name, values in case[f"{role}_components"].items()
    }
    actual = dmd2_weighted_total(components, case[f"{role}_weights"])
    torch.testing.assert_close(
        actual,
        torch.tensor(fixture["expected"]["weighted_totals"][role]),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def test_dmd2_formula_fixture_records_scheduler_cadence() -> None:
    fixture = _fixture()
    cadence = fixture["inputs"]["cadence"]
    expected = fixture["expected"]["cadence"]
    interval = cadence["generator_update_interval"]
    due = [step % interval == 0 for step in cadence["global_steps"]]
    assert due == expected["generator_due"]
    iteration_steps = list(range(1, len(due) + 1))
    assert iteration_steps == expected["iteration_scheduler_steps"]
    committed = 0
    generator_scheduler_steps: list[int] = []
    for is_due in due:
        committed += int(is_due)
        generator_scheduler_steps.append(committed)
    assert generator_scheduler_steps == expected["generator_update_scheduler_steps"]
