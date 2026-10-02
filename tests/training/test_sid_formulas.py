from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.sid.math import (  # noqa: E402
    sid_classifier_free_guidance,
    sid_fake_score_adversarial_loss_per_sample,
    sid_fake_score_flow_loss_per_sample,
    sid_generator_adversarial_loss_per_sample,
    sid_generator_loss_per_sample,
    sid_score_weight,
)


def _fixture() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "source_formulas" / "sid.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_sid_cfg_matches_zero_one_and_official_default_anchors() -> None:
    fixture = _fixture()
    case = fixture["inputs"]["cfg"]
    expected = fixture["expected"]["cfg"]
    unconditional = torch.tensor(case["unconditional"])
    conditional = torch.tensor(case["conditional"])
    keys = ("scale_zero", "scale_one", "scale_four_point_five")
    for scale, key in zip(case["scales"], keys, strict=True):
        torch.testing.assert_close(
            sid_classifier_free_guidance(unconditional, conditional, scale),
            torch.tensor(expected[key]),
            atol=fixture["atol"],
            rtol=fixture["rtol"],
        )


@pytest.mark.parametrize(
    ("alpha", "expected_key"),
    [(1.0, "generator_alpha_one"), (0.25, "generator_alpha_quarter")],
)
def test_sid_general_alpha_score_identity_matches_formula_fixture(alpha: float, expected_key: str) -> None:
    fixture = _fixture()
    case = fixture["inputs"]["generator"]
    actual = sid_generator_loss_per_sample(
        torch.tensor(case["generated"]),
        torch.tensor(case["teacher_clean"]),
        torch.tensor(case["fake_clean"]),
        torch.tensor(case["score_weight"]),
        alpha=alpha,
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(fixture["expected"][expected_key]),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def test_sid_fake_score_flow_sum_matches_formula_fixture() -> None:
    fixture = _fixture()
    case = fixture["inputs"]["fake_score_flow"]
    actual = sid_fake_score_flow_loss_per_sample(
        torch.tensor(case["prediction"]),
        torch.tensor(case["target"]),
    )
    torch.testing.assert_close(actual, torch.tensor(fixture["expected"]["fake_score_flow"]))


@pytest.mark.parametrize(
    "scheme",
    [
        "1-minus-sigma",
        "1-minus-sigma-squared",
        "1-over-sigma",
        "1-over-sigma2",
        "snr-sqrt",
        "snr",
        "sid-legacy",
    ],
)
def test_sid_official_weighting_schemes_match_formula_fixture(scheme: str) -> None:
    fixture = _fixture()
    case = fixture["inputs"]["generator"]
    actual = sid_score_weight(
        torch.tensor(fixture["inputs"]["score_sigmas"]),
        scheme=scheme,
        epsilon=1.0e-5,
        generated=torch.tensor(case["generated"]),
        teacher_clean=torch.tensor(case["teacher_clean"]),
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(fixture["expected"]["score_weights"][scheme]),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def test_sid_diffusion_gan_bce_clamp_and_latent_scaling() -> None:
    fake = torch.tensor([[-20.0, 0.0], [2.0, 20.0]])
    real = torch.tensor([[20.0, 0.0], [2.0, -20.0]])
    score_weight = torch.tensor([0.5, 2.0])
    expected_g = torch.nn.functional.binary_cross_entropy_with_logits(
        fake.clamp(-10, 10),
        torch.ones_like(fake),
        reduction="none",
    ).mean(1) * score_weight * 8
    expected_d = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(
            real.clamp(-10, 10), torch.ones_like(real), reduction="none"
        ).mean(1)
        + torch.nn.functional.binary_cross_entropy_with_logits(
            fake.clamp(-10, 10), torch.zeros_like(fake), reduction="none"
        ).mean(1)
    ) * 8
    torch.testing.assert_close(
        sid_generator_adversarial_loss_per_sample(fake, score_weight, latent_elements=8),
        expected_g,
    )
    torch.testing.assert_close(
        sid_fake_score_adversarial_loss_per_sample(real, fake, latent_elements=8),
        expected_d,
    )
