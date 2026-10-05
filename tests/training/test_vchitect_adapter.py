from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import VchitectTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyVchitectModel(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        moved = latents.movedim(2, 1)
        return self.proj(moved).movedim(1, 2)


class _TinyVchitectDenoiser:
    def __init__(self, model: _TinyVchitectModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = model_input.conditioning
        assert isinstance(conditioning["prompt_embeds"], torch.Tensor)
        assert isinstance(conditioning["pooled_prompt_embeds"], torch.Tensor)
        return DenoiserOutput(sample=self.model(model_input.latents))


def _adapter() -> VchitectTrainAdapter:
    return VchitectTrainAdapter(
        _TinyVchitectDenoiser(_TinyVchitectModel(channels=4)),
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        spatial_compression=2,
    )


def _cached_batch() -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 3, 4, 4, 4),
            "prompt_embeds": torch.randn(1, 6, 16),
            "pooled_prompt_embeds": torch.randn(1, 8),
        },
    )


def test_vchitect_forward_loss_backward_and_family_tag() -> None:
    torch.manual_seed(13)
    adapter = _adapter()
    prepared = adapter.prepare_batch(_cached_batch())
    assert prepared.metadata["model_family"] == "vchitect-video"
    assert tuple(prepared.clean_latents.shape) == (1, 3, 4, 4, 4)

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(17))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert tuple(prediction.shape) == (1, 3, 4, 4, 4)
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_vchitect_missing_pooled_projection_fails_closed() -> None:
    adapter = _adapter()
    batch = TrainingBatch(
        sample_ids=("clip",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 3, 4, 4, 4), "prompt_embeds": torch.randn(1, 6, 16)},
    )
    with pytest.raises(ValueError, match="pooled_prompt_embeds"):
        adapter.prepare_batch(batch)
