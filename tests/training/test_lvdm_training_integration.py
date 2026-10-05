"""Formula gates against YingqingHe/LVDM commit d251dccfbf6352826f5c5681abd86e87ed7e6371.

Compared source paths:
  configs/lvdm_short/ucf.yaml
  lvdm/models/ddpm3d.py
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api.contracts import TrainingBatch  # noqa: E402
from worldfoundry.training.engine.single_device import SingleDeviceTrainEngine  # noqa: E402
from worldfoundry.training.models.lvdm import LVDMUnconditionalTrainAdapter  # noqa: E402
from worldfoundry.training.objectives.classic_diffusion import (  # noqa: E402
    lvdm_linear_beta_schedule,
    lvdm_short_objective,
)


class _UnconditionalDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.input_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    def forward(self, noisy, timesteps):
        return noisy * self.scale + timesteps.reshape(-1, 1, 1, 1, 1) * 0.0


def test_lvdm_short_schedule_and_epsilon_l1_match_released_formula() -> None:
    actual_betas = lvdm_linear_beta_schedule(1000, beta_start=0.0015, beta_end=0.0155).numpy()
    expected_betas = np.linspace(np.sqrt(0.0015), np.sqrt(0.0155), 1000, dtype=np.float64) ** 2
    np.testing.assert_allclose(actual_betas, expected_betas, rtol=0.0, atol=1.0e-15)

    denoiser = _UnconditionalDenoiser()
    adapter = LVDMUnconditionalTrainAdapter(denoiser, codec=None)
    clean = torch.linspace(-1.0, 1.0, 2 * 4 * 4 * 2 * 2).reshape(2, 4, 4, 2, 2)
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("clip-a", "clip-b"),
            prompts=("", ""),
            conditions={"clean_latents": clean},
        )
    )
    objective = lvdm_short_objective()
    objective_batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(41))
    timesteps = objective_batch.timesteps
    alpha_cumprod = torch.from_numpy(np.cumprod(1.0 - expected_betas)).float()
    alpha = alpha_cumprod[timesteps].sqrt().reshape(2, 1, 1, 1, 1)
    sigma = (1.0 - alpha_cumprod[timesteps]).sqrt().reshape(2, 1, 1, 1, 1)
    expected_noisy = alpha * clean + sigma * objective_batch.noise
    torch.testing.assert_close(objective_batch.model_input, expected_noisy)
    torch.testing.assert_close(objective_batch.target, objective_batch.noise)

    prediction = adapter.forward_train(objective_batch)
    result = objective.compute_loss(prediction, objective_batch)
    expected_l1 = (prediction.float() - objective_batch.noise.float()).abs().mean()
    torch.testing.assert_close(result.loss, expected_l1)


def test_lvdm_short_uses_generic_optimizer_engine() -> None:
    denoiser = _UnconditionalDenoiser()
    adapter = LVDMUnconditionalTrainAdapter(denoiser, codec=None)
    objective = lvdm_short_objective()
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=0.01, weight_decay=0.0)
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer, max_grad_norm=10.0)
    batch = TrainingBatch(
        sample_ids=("clip",),
        prompts=("",),
        conditions={"clean_latents": torch.randn(1, 4, 4, 2, 2)},
    )
    before = denoiser.scale.detach().clone()
    result = engine.train_step(batch, generator=torch.Generator().manual_seed(7))
    assert result.sample_count == 1
    assert engine.global_step == 1
    assert not torch.equal(before, denoiser.scale.detach())
