from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import StepVideoTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyStepModel(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Latents arrive [B,F,C,H,W]; move channels to conv position and back.
        moved = latents.movedim(2, 1)
        out = self.proj(moved)
        return out.movedim(1, 2)


class _TinyStepDenoiser:
    def __init__(self, model: _TinyStepModel) -> None:
        self.model = model
        self.seen_timestep: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = model_input.conditioning
        for key in ("prompt_embeds", "clip_embeds", "attention_mask"):
            assert isinstance(conditioning[key], torch.Tensor)
        self.seen_timestep = model_input.timestep.detach().clone()
        return DenoiserOutput(sample=self.model(model_input.latents))


def _adapter() -> StepVideoTrainAdapter:
    model = _TinyStepModel(channels=8)
    return StepVideoTrainAdapter(
        _TinyStepDenoiser(model),
        codec=None,
        conditioner=None,
        expected_latent_channels=8,
        spatial_compression=2,
    )


def _cached_batch(*, channels: int = 8) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 3, channels, 4, 4),
            "prompt_embeds": torch.randn(1, 6, 16),
            "clip_embeds": torch.randn(1, 8),
            "attention_mask": torch.ones(1, 6),
        },
    )


def test_step_video_layout_forward_loss_backward_and_timestep() -> None:
    torch.manual_seed(5)
    model = _TinyStepModel(channels=8)
    denoiser = _TinyStepDenoiser(model)
    adapter = StepVideoTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=8,
        spatial_compression=2,
    )
    # StepVideo rescales the incoming timestep by 1000 inside the model, so the
    # adapter must feed the raw flow sigma (scale 1.0), not sigma * 1000.
    assert adapter.model_timestep_scale == 1.0

    prepared = adapter.prepare_batch(_cached_batch())
    assert prepared.metadata["model_family"] == "step-video"
    assert tuple(prepared.clean_latents.shape) == (1, 3, 8, 4, 4)

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(7))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    # The denoiser must receive the flow sigma unchanged (0..1), not sigma*1000.
    assert denoiser.seen_timestep is not None
    torch.testing.assert_close(denoiser.seen_timestep, corrupted.sigmas, rtol=0, atol=0)
    assert tuple(prediction.shape) == (1, 3, 8, 4, 4)
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_step_video_channel_axis_geometry_fails_closed() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="channels"):
        adapter.prepare_batch(_cached_batch(channels=4))


def test_step_video_missing_conditioning_fails_closed() -> None:
    adapter = _adapter()
    batch = TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 3, 8, 4, 4), "prompt_embeds": torch.randn(1, 6, 16)},
    )
    with pytest.raises(ValueError, match="clip_embeds"):
        adapter.prepare_batch(batch)
