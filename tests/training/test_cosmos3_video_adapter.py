from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    MultiModalDenoiserInput,
    MultiModalDenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import Cosmos3VideoTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import FlowMatchingConfig, FlowMatchingObjective  # noqa: E402


class _TinyCosmos3Model(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyCosmos3Denoiser:
    """Omni contract denoiser returning per-modality flow velocity for video."""

    def __init__(self, model: _TinyCosmos3Model) -> None:
        self.model = model
        self.seen_input_ids = False
        self.seen_positions = False

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        state = model_input.modalities["video"]
        self.seen_input_ids = isinstance(model_input.conditioning.get("input_ids"), torch.Tensor)
        self.seen_positions = state.positions is not None
        # Private threading keys must be stripped before the denoiser call.
        assert not any(str(k).startswith("_cosmos3") for k in model_input.conditioning)
        assert state.denoise_mask is not None
        return MultiModalDenoiserOutput(samples={"video": self.model(state.latent)})


def _adapter(denoiser: _TinyCosmos3Denoiser) -> Cosmos3VideoTrainAdapter:
    return Cosmos3VideoTrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=4, spatial_compression=2,
    )


def _batch() -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, 3, 2, 2),
            "input_ids": torch.arange(6).unsqueeze(0),
        },
    )


def test_cosmos3_video_only_flow_velocity_type() -> None:
    adapter = _adapter(_TinyCosmos3Denoiser(_TinyCosmos3Model()))
    assert adapter.prediction_type == "flow_velocity"
    prepared = adapter.prepare_batch(_batch())
    assert prepared.metadata["model_family"] == "cosmos3-omni-video"
    assert set(prepared.clean_latents) == {"video"}
    assert tuple(prepared.clean_latents["video"].shape) == (1, 4, 3, 2, 2)


def test_cosmos3_multimodal_forward_loss_backward() -> None:
    torch.manual_seed(103)
    denoiser = _TinyCosmos3Denoiser(_TinyCosmos3Model(4))
    adapter = _adapter(denoiser)
    prepared = adapter.prepare_batch(_batch())

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(107))
    prediction = adapter.forward_train(corrupted)
    assert set(prediction) == {"video"}
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert denoiser.seen_input_ids and denoiser.seen_positions
    assert torch.isfinite(result.loss)
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_cosmos3_requires_input_ids() -> None:
    adapter = _adapter(_TinyCosmos3Denoiser(_TinyCosmos3Model()))
    batch = TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 4, 3, 2, 2)},
    )
    with pytest.raises(ValueError, match="input_ids"):
        adapter.prepare_batch(batch)


def test_cosmos3_rejects_multi_media_batch() -> None:
    adapter = _adapter(_TinyCosmos3Denoiser(_TinyCosmos3Model()))
    batch = TrainingBatch(
        sample_ids=("a", "b"),
        prompts=("cached", "cached"),
        conditions={
            "clean_latents": torch.randn(2, 4, 3, 2, 2),
            "input_ids": torch.arange(6).unsqueeze(0).expand(2, -1),
        },
    )
    with pytest.raises(ValueError, match="batch size one"):
        adapter.prepare_batch(batch)
