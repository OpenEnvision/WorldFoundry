from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from worldfoundry.training.post_training.distillation.consistency.math import (
    batch_coefficients,
)
from worldfoundry.training.post_training.distillation.rcm import (
    NativeRFRCMPredictionAdapter,
)
from worldfoundry.training.post_training.distillation.rcm.math import (
    bidirectional_scm_loss,
    causal_scm_loss,
    exact_dmd_proxy_loss,
)

_FIXTURE = Path(__file__).parent / "fixtures/source_formulas/rcm.json"


class _KnownRFVelocity:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.module = nn.Linear(1, 1, bias=False)
        self.velocity = velocity

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del sigmas, kwargs
        return self.velocity.to(noisy_latents) + self.module.weight.sum() * 0

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        velocity = self.predict_velocity(noisy_latents, sigmas, **kwargs)
        return noisy_latents - batch_coefficients(sigmas, noisy_latents) * velocity


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_native_rf_adapter_matches_official_unit_scale_trigflow_wrapper() -> None:
    clean = torch.tensor([[0.4, -0.2], [0.1, 0.7], [-0.6, 0.3]])
    noise = torch.tensor([[-0.5, 0.8], [0.6, -0.1], [0.2, -0.9]])
    time = torch.tensor([0.0, 0.7, torch.pi / 2])
    coefficient = batch_coefficients(time, clean)
    noisy = torch.cos(coefficient) * clean + torch.sin(coefficient) * noise
    adapter = NativeRFRCMPredictionAdapter(_KnownRFVelocity(noise - clean))
    prediction = adapter.predict(
        noisy,
        time,
        sample_ids=("zero", "middle", "terminal"),
        conditioning={},
        training=True,
    )
    expected_velocity = torch.cos(coefficient) * noise - torch.sin(coefficient) * clean
    torch.testing.assert_close(prediction.clean_latents, clean, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(prediction.velocity, expected_velocity, atol=2e-6, rtol=2e-6)


def test_bidirectional_and_causal_continuous_targets_match_fixed_source_values() -> None:
    fixture = _fixture()
    values = fixture["inputs"]["continuous"]
    expected = fixture["expected"]["continuous"]
    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}
    tensors = {
        name: torch.tensor(values[name], dtype=torch.float64)
        for name in (
            "current",
            "stopped",
            "teacher",
            "directional_derivative",
            "noisy",
        )
    }
    bidirectional, bidirectional_norm, invalid = bidirectional_scm_loss(
        tensors["current"],
        tensors["stopped"],
        tensors["teacher"],
        tensors["directional_derivative"],
        tensors["noisy"],
        torch.tensor(values["trig_time"], dtype=torch.float64),
        warmup_ratio=values["warmup_ratio"],
    )
    torch.testing.assert_close(
        bidirectional,
        torch.tensor(expected["bidirectional_loss"], dtype=torch.float64),
        **tolerance,
    )
    torch.testing.assert_close(
        bidirectional_norm,
        torch.tensor(expected["bidirectional_norm"], dtype=torch.float64),
        **tolerance,
    )
    assert not bool(invalid.any())

    causal, causal_norm, invalid = causal_scm_loss(
        tensors["current"],
        tensors["stopped"],
        tensors["teacher"],
        tensors["directional_derivative"],
        torch.tensor(values["rf_time"], dtype=torch.float64),
        warmup_ratio=values["warmup_ratio"],
    )
    torch.testing.assert_close(
        causal,
        torch.tensor(expected["causal_loss"], dtype=torch.float64),
        **tolerance,
    )
    torch.testing.assert_close(
        causal_norm,
        torch.tensor(expected["causal_norm"], dtype=torch.float64),
        **tolerance,
    )
    assert not bool(invalid.any())


def test_rcm_dmd_uses_per_sample_fp64_normalizer_and_no_half_proxy() -> None:
    fixture = _fixture()
    values = fixture["inputs"]["dmd"]
    expected = fixture["expected"]["dmd"]
    tolerance = {"atol": fixture["atol"], "rtol": fixture["rtol"]}
    generated = torch.tensor(values["generated"], requires_grad=True)
    fake = torch.tensor(values["fake"])
    teacher = torch.tensor(values["teacher"])
    loss, normalizer, invalid = exact_dmd_proxy_loss(generated, fake, teacher)
    torch.testing.assert_close(
        normalizer.flatten(),
        torch.tensor(expected["normalizer"], dtype=torch.float64),
        **tolerance,
    )
    torch.testing.assert_close(
        loss,
        torch.tensor(expected["loss"], dtype=torch.float64),
        **tolerance,
    )
    assert loss.dtype == torch.float64
    assert not bool(invalid.any())
    loss.backward()
    torch.testing.assert_close(
        generated.grad,
        torch.tensor(expected["generated_gradient"]),
        **tolerance,
    )
