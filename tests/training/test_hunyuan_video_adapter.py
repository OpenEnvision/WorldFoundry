from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import HunyuanVideoTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyHunyuanModel(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyHunyuanDenoiser:
    def __init__(self, model: _TinyHunyuanModel) -> None:
        self.model = model
        self.seen_timestep: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = model_input.conditioning
        for key in ("text_states", "text_mask", "text_states_2"):
            assert isinstance(conditioning[key], torch.Tensor)
        self.seen_timestep = model_input.timestep.detach().clone()
        return DenoiserOutput(sample=self.model(model_input.latents))


def _adapter(denoiser: _TinyHunyuanDenoiser) -> HunyuanVideoTrainAdapter:
    return HunyuanVideoTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
    )


def _cached_batch(*, channels: int = 4) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, channels, 2, 2, 2),
            "text_states": torch.randn(1, 6, 16),
            "text_mask": torch.ones(1, 6),
            "text_states_2": torch.randn(1, 8),
        },
    )


def test_hunyuan_video_forward_loss_backward_and_timestep_scale() -> None:
    torch.manual_seed(19)
    denoiser = _TinyHunyuanDenoiser(_TinyHunyuanModel(channels=4))
    adapter = _adapter(denoiser)
    # HunyuanVideo's DiT reads timesteps on the sigma*1000 domain (Wan-style
    # scheduler, no internal rescale), so the adapter keeps scale 1000.
    assert adapter.model_timestep_scale == 1000.0

    prepared = adapter.prepare_batch(_cached_batch())
    assert prepared.metadata["model_family"] == "hunyuan-video"
    assert tuple(prepared.clean_latents.shape) == (1, 4, 2, 2, 2)

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(23))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert denoiser.seen_timestep is not None
    torch.testing.assert_close(denoiser.seen_timestep, corrupted.sigmas * 1000.0, rtol=0, atol=0)
    assert tuple(prediction.shape) == (1, 4, 2, 2, 2)
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_hunyuan_video_channel_geometry_fails_closed() -> None:
    adapter = _adapter(_TinyHunyuanDenoiser(_TinyHunyuanModel(channels=4)))
    with pytest.raises(ValueError, match="channels"):
        adapter.prepare_batch(_cached_batch(channels=8))


def test_hunyuan_video_missing_conditioning_fails_closed() -> None:
    adapter = _adapter(_TinyHunyuanDenoiser(_TinyHunyuanModel(channels=4)))
    batch = TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 4, 2, 2, 2), "text_states": torch.randn(1, 6, 16)},
    )
    with pytest.raises(ValueError, match="text_mask"):
        adapter.prepare_batch(batch)
