from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import Cosmos1TrainAdapter  # noqa: E402
from worldfoundry.training.objectives import X0DiffusionConfig, X0DiffusionObjective  # noqa: E402


class _TinyCosmos1Model(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, network_input, c_noise, context, *, attention_mask=None, fps=24.0, padding_mask=None):
        del c_noise, context, attention_mask, fps, padding_mask
        return self.proj(network_input)


class _TinyCosmos1Denoiser:
    """EDM preconditioning + pose concat, returns clean x0."""

    def __init__(self, model: _TinyCosmos1Model, *, sigma_data: float = 0.5) -> None:
        self.model = model
        self.sigma_data = sigma_data
        self.seen_timestep: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        v = model_input.conditioning
        for key in (
            "context", "condition_latents", "condition_indicator",
            "condition_video_input_mask", "condition_video_pose", "condition_noise",
        ):
            assert isinstance(v[key], torch.Tensor), key
        latents = model_input.latents
        self.seen_timestep = model_input.timestep.detach().clone()
        sigma = model_input.timestep.to(latents).reshape(-1, 1, 1, 1, 1)
        sd = self.sigma_data
        c_in = torch.rsqrt(sigma.square() + sd**2)
        network_input = torch.cat(
            (latents * c_in, v["condition_video_input_mask"].to(latents), v["condition_video_pose"].to(latents)),
            dim=1,
        )
        prediction = self.model(network_input, sigma.log() * 0.25, v["context"])
        c_skip = sd**2 / (sigma.square() + sd**2)
        c_out = sigma * sd / torch.sqrt(sigma.square() + sd**2)
        return DenoiserOutput(sample=c_skip * latents + c_out * prediction)


def _adapter() -> Cosmos1TrainAdapter:
    model = _TinyCosmos1Model(in_channels=6, out_channels=4)  # 4 latent + 1 mask + 1 pose
    return Cosmos1TrainAdapter(
        _TinyCosmos1Denoiser(model), codec=None, conditioner=None,
        expected_latent_channels=4, spatial_compression=2,
    )


def _batch(*, frames: int = 5) -> TrainingBatch:
    indicator = torch.zeros(1, 1, frames, 1, 1)
    indicator[:, :, :1] = 1.0
    return TrainingBatch(
        sample_ids=("g",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, frames, 2, 2),
            "context": torch.randn(1, 8, 16),
            "condition_latents": torch.randn(1, 4, frames, 2, 2),
            "condition_indicator": indicator,
            "condition_video_input_mask": torch.zeros(1, 1, frames, 2, 2),
            "condition_video_pose": torch.zeros(1, 1, frames, 2, 2),
            "condition_noise": torch.randn(1, 4, frames, 2, 2),
        },
    )


def test_cosmos1_x0_type_and_raw_sigma() -> None:
    adapter = _adapter()
    assert adapter.prediction_type == "x0"
    assert adapter.model_timestep_scale == 1.0


def test_cosmos1_edm_forward_loss_backward_and_mask() -> None:
    torch.manual_seed(83)
    adapter = _adapter()
    denoiser = adapter.denoiser
    prepared = adapter.prepare_batch(_batch(frames=5))
    assert prepared.metadata["model_family"] == "cosmos1-gen3c-video"
    mask = prepared.loss_mask
    assert tuple(mask.shape) == (1, 1, 5, 1, 1)
    assert torch.equal(mask[:, :, :1], torch.zeros(1, 1, 1, 1, 1))
    assert torch.equal(mask[:, :, 1:], torch.ones(1, 1, 4, 1, 1))

    objective = X0DiffusionObjective(X0DiffusionConfig(weighting="edm", sigma_data=0.5))
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(89))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert denoiser.seen_timestep is not None
    torch.testing.assert_close(denoiser.seen_timestep, corrupted.sigmas, rtol=0, atol=0)
    assert torch.isfinite(result.loss)
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_cosmos1_requires_pose_conditioning() -> None:
    adapter = _adapter()
    batch = TrainingBatch(
        sample_ids=("g",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, 5, 2, 2),
            "context": torch.randn(1, 8, 16),
            "condition_latents": torch.randn(1, 4, 5, 2, 2),
            "condition_indicator": torch.zeros(1, 1, 5, 1, 1),
            "condition_video_input_mask": torch.zeros(1, 1, 5, 2, 2),
            "condition_noise": torch.randn(1, 4, 5, 2, 2),
        },
    )
    with pytest.raises(ValueError, match="condition_video_pose"):
        adapter.prepare_batch(batch)
