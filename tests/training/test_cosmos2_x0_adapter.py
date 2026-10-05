from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import Cosmos2TrainAdapter  # noqa: E402
from worldfoundry.training.objectives import X0DiffusionConfig, X0DiffusionObjective  # noqa: E402


class _TinyCosmos2Model(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, network_input, timestep, context, *, fps=16.0, condition_mask=None, padding_mask=None):
        del timestep, context, fps, condition_mask, padding_mask
        return self.proj(network_input)


class _TinyCosmos2Denoiser:
    """Rectified-flow preconditioning + condition blend, returns clean x0."""

    def __init__(self, model: _TinyCosmos2Model) -> None:
        self.model = model
        self.seen_timestep: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        for key in ("context", "condition_latents", "condition_mask", "condition_indicator"):
            assert isinstance(values[key], torch.Tensor), key
        latents = model_input.latents
        self.seen_timestep = model_input.timestep.detach().clone()
        sigma = model_input.timestep.to(latents).reshape(-1, 1, 1, 1, 1)
        c_in = 1.0 / (sigma + 1.0)
        c_out = -sigma / (sigma + 1.0)
        cond_mask = values["condition_mask"].to(latents)
        cond_lat = values["condition_latents"].to(latents)
        network_input = latents * c_in
        network_input = cond_mask * cond_lat + (1 - cond_mask) * network_input
        prediction = self.model(network_input, sigma, values["context"])
        clean = c_in * latents + c_out * prediction
        clean = cond_mask * cond_lat + (1 - cond_mask) * clean
        return DenoiserOutput(sample=clean)


def _adapter(denoiser: _TinyCosmos2Denoiser) -> Cosmos2TrainAdapter:
    return Cosmos2TrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=4, spatial_compression=2,
    )


def _batch(*, total_frames: int = 5, condition_frames: int = 1, batch: int = 1) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=tuple(f"c{i}" for i in range(batch)),
        prompts=tuple("cached" for _ in range(batch)),
        conditions={
            "clean_latents": torch.randn(batch, 4, total_frames, 2, 2),
            "context": torch.randn(batch, 8, 16),
            "num_latent_conditional_frames": condition_frames,
        },
    )


def test_cosmos2_uses_x0_prediction_type_and_raw_sigma_scale() -> None:
    adapter = _adapter(_TinyCosmos2Denoiser(_TinyCosmos2Model()))
    assert adapter.prediction_type == "x0"
    # Cosmos2 takes the raw Karras sigma, NOT sigma*1000.
    assert adapter.model_timestep_scale == 1.0


def test_cosmos2_x0_forward_loss_backward_and_condition_mask() -> None:
    torch.manual_seed(73)
    denoiser = _TinyCosmos2Denoiser(_TinyCosmos2Model(4))
    adapter = _adapter(denoiser)
    prepared = adapter.prepare_batch(_batch(total_frames=5, condition_frames=1))
    assert prepared.metadata["model_family"] == "cosmos-predict2-video"
    # Only generated frames scored; the given first frame is masked out.
    mask = prepared.loss_mask
    assert torch.equal(mask[:, :, :1], torch.zeros(1, 1, 1, 1, 1))
    assert torch.equal(mask[:, :, 1:], torch.ones(1, 1, 4, 1, 1))

    objective = X0DiffusionObjective(X0DiffusionConfig(weighting="rectified_flow"))
    assert objective.prediction_type == adapter.prediction_type
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(79))
    # x0 corruption is additive Karras noise: model_input = x0 + sigma*noise.
    sigma = corrupted.sigmas.reshape(-1, 1, 1, 1, 1)
    expected = prepared.clean_latents + sigma * corrupted.noise
    torch.testing.assert_close(corrupted.model_input, expected, rtol=1e-5, atol=1e-5)
    # Karras sigmas are positive and can exceed 1 (not the [0,1] flow sigma).
    assert bool((corrupted.sigmas > 0).all())

    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    # The denoiser received the raw sigma unchanged (scale 1.0).
    assert denoiser.seen_timestep is not None
    torch.testing.assert_close(denoiser.seen_timestep, corrupted.sigmas, rtol=0, atol=0)
    assert torch.isfinite(result.loss)
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_cosmos2_rejects_multi_media_batch() -> None:
    adapter = _adapter(_TinyCosmos2Denoiser(_TinyCosmos2Model()))
    with pytest.raises(ValueError, match="batch size one"):
        adapter.prepare_batch(_batch(batch=2))
