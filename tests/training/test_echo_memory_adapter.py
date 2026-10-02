from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import EchoMemoryTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyEchoModel(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyEchoDenoiser:
    """Records the latents and conditioning the model actually receives."""

    def __init__(self, model: _TinyEchoModel) -> None:
        self.model = model
        self.seen_latents: torch.Tensor | None = None
        self.seen_conditioning: dict | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = model_input.conditioning
        assert isinstance(conditioning["context"], torch.Tensor)
        assert isinstance(conditioning["actions"], torch.Tensor)
        assert isinstance(conditioning["num_context_frames"], int)
        # Private threading keys must be stripped before reaching the model.
        assert not any(key.startswith("_echo") for key in conditioning)
        self.seen_latents = model_input.latents.detach().clone()
        self.seen_conditioning = dict(conditioning)
        return DenoiserOutput(sample=self.model(model_input.latents))


def _adapter(denoiser: _TinyEchoDenoiser) -> EchoMemoryTrainAdapter:
    return EchoMemoryTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
        action_dim=12,
    )


def _cached_batch(*, total_frames: int = 5, context_frames: int = 2) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("echo",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, total_frames, 2, 2),
            "context": torch.randn(1, 8, 16),
            "actions": torch.randn(1, total_frames, 12),
            "num_context_frames": context_frames,
        },
    )


def test_echo_loss_mask_zeros_context_suffix() -> None:
    adapter = _adapter(_TinyEchoDenoiser(_TinyEchoModel()))
    prepared = adapter.prepare_batch(_cached_batch(total_frames=5, context_frames=2))
    assert prepared.metadata["model_family"] == "echo-memory-video"
    assert prepared.metadata["num_context_frames"] == 2
    mask = prepared.loss_mask
    assert tuple(mask.shape) == (1, 1, 5, 1, 1)
    # Target frames (first 3) contribute to the loss; context suffix (last 2) does not.
    assert torch.equal(mask[:, :, :3], torch.ones(1, 1, 3, 1, 1))
    assert torch.equal(mask[:, :, 3:], torch.zeros(1, 1, 2, 1, 1))


def test_echo_forward_refreezes_clean_context_suffix() -> None:
    torch.manual_seed(29)
    denoiser = _TinyEchoDenoiser(_TinyEchoModel(channels=4))
    adapter = _adapter(denoiser)
    prepared = adapter.prepare_batch(_cached_batch(total_frames=5, context_frames=2))
    clean = prepared.clean_latents
    assert adapter.model_timestep_scale == 1000.0

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(31))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    # The denoiser must have received the clean context suffix (last 2 frames),
    # NOT the objective-corrupted values.
    assert denoiser.seen_latents is not None
    torch.testing.assert_close(denoiser.seen_latents[:, :, 3:], clean[:, :, 3:], rtol=0, atol=0)
    # Target frames should be the corrupted latents (differ from clean).
    assert not torch.allclose(denoiser.seen_latents[:, :, :3], clean[:, :, :3])
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_echo_rejects_context_frames_covering_whole_clip() -> None:
    adapter = _adapter(_TinyEchoDenoiser(_TinyEchoModel()))
    with pytest.raises(ValueError, match="num_context_frames"):
        adapter.prepare_batch(_cached_batch(total_frames=3, context_frames=3))


def test_echo_requires_actions() -> None:
    adapter = _adapter(_TinyEchoDenoiser(_TinyEchoModel()))
    batch = TrainingBatch(
        sample_ids=("echo",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, 5, 2, 2),
            "context": torch.randn(1, 8, 16),
            "num_context_frames": 2,
        },
    )
    with pytest.raises(ValueError, match="actions"):
        adapter.prepare_batch(batch)
