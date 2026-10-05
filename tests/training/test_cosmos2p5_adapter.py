from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import Cosmos2p5TrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyCosmosModel(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyCosmosDenoiser:
    """Mirrors Cosmos25Denoiser's condition-freeze + GT-velocity mix contract."""

    def __init__(self, model: _TinyCosmosModel) -> None:
        self.model = model
        self.seen_condition_keys: set | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        self.seen_condition_keys = set(values)
        for key in ("context", "condition_latents", "condition_mask", "condition_indicator", "initial_noise"):
            assert isinstance(values[key], torch.Tensor), key
        latents = model_input.latents
        condition_mask = values["condition_mask"].to(latents)
        condition_latents = values["condition_latents"].to(latents)
        initial_noise = values["initial_noise"].to(latents)
        # Re-freeze condition frames to clean (inference blend), predict velocity.
        model_latents = condition_mask * condition_latents + (1 - condition_mask) * latents
        prediction = self.model(model_latents)
        ground_truth_velocity = initial_noise - condition_latents
        prediction = ground_truth_velocity * condition_mask + prediction * (1 - condition_mask)
        return DenoiserOutput(sample=prediction)


def _adapter(denoiser: _TinyCosmosDenoiser) -> Cosmos2p5TrainAdapter:
    return Cosmos2p5TrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
    )


def _cached_batch(*, total_frames: int = 5, condition_frames: int = 1, batch: int = 1) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=tuple(f"c{i}" for i in range(batch)),
        prompts=tuple("cached" for _ in range(batch)),
        conditions={
            "clean_latents": torch.randn(batch, 4, total_frames, 2, 2),
            "context": torch.randn(batch, 8, 16),
            "num_latent_conditional_frames": condition_frames,
        },
    )


def test_cosmos2p5_loss_mask_zeros_condition_frames() -> None:
    adapter = _adapter(_TinyCosmosDenoiser(_TinyCosmosModel()))
    prepared = adapter.prepare_batch(_cached_batch(total_frames=5, condition_frames=1))
    assert prepared.metadata["model_family"] == "cosmos-predict2.5-video"
    assert prepared.metadata["num_latent_conditional_frames"] == 1
    mask = prepared.loss_mask
    assert tuple(mask.shape) == (1, 1, 5, 1, 1)
    # First frame is the given condition (masked out); frames 1..4 are scored.
    assert torch.equal(mask[:, :, :1], torch.zeros(1, 1, 1, 1, 1))
    assert torch.equal(mask[:, :, 1:], torch.ones(1, 1, 4, 1, 1))
    ind = prepared.conditioning["condition_indicator"]
    cmask = prepared.conditioning["condition_mask"]
    assert tuple(ind.shape) == (1, 1, 5, 1, 1)
    assert tuple(cmask.shape) == (1, 1, 5, 2, 2)


def test_cosmos2p5_forward_loss_backward_and_scale() -> None:
    torch.manual_seed(37)
    denoiser = _TinyCosmosDenoiser(_TinyCosmosModel(channels=4))
    adapter = _adapter(denoiser)
    assert adapter.model_timestep_scale == 1000.0
    prepared = adapter.prepare_batch(_cached_batch(total_frames=5, condition_frames=1))

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(41))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert denoiser.seen_condition_keys is not None
    required = {"condition_latents", "condition_mask", "condition_indicator", "initial_noise"}
    assert required <= denoiser.seen_condition_keys
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_cosmos2p5_rejects_multi_media_batch() -> None:
    adapter = _adapter(_TinyCosmosDenoiser(_TinyCosmosModel()))
    with pytest.raises(ValueError, match="batch size one"):
        adapter.prepare_batch(_cached_batch(batch=2))


def test_cosmos2p5_rejects_condition_covering_whole_clip() -> None:
    adapter = _adapter(_TinyCosmosDenoiser(_TinyCosmosModel()))
    with pytest.raises(ValueError, match="num_latent_conditional_frames"):
        adapter.prepare_batch(_cached_batch(total_frames=3, condition_frames=3))
