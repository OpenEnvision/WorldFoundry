from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import (  # noqa: E402
    HunyuanVideo15TrainAdapter,
    HunyuanVideoI2VTrainAdapter,
)
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyModel(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.double_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _I2VDenoiser:
    """Original I2V: zeroes the first-frame output (token replacement)."""

    def __init__(self, model: _TinyModel) -> None:
        self.model = model
        self.seen_latents: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        for key in ("text_states", "text_mask", "text_states_2"):
            assert isinstance(model_input.conditioning[key], torch.Tensor)
        assert not any(key.startswith("_hunyuan") for key in model_input.conditioning)
        self.seen_latents = model_input.latents.detach().clone()
        sample = self.model(model_input.latents).clone()
        sample[:, :, :1] = 0
        return DenoiserOutput(sample=sample)


class _H15Denoiser:
    """H15: concatenates condition on the channel axis, predicts full velocity."""

    def __init__(self, model: _TinyModel) -> None:
        self.model = model
        self.seen_channels: int | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        cond = model_input.conditioning
        for key in ("text_states", "text_mask", "condition_latents", "byt5_text_states", "byt5_text_mask"):
            assert isinstance(cond[key], torch.Tensor), key
        merged = torch.cat((model_input.latents, cond["condition_latents"].to(model_input.latents)), dim=1)
        self.seen_channels = int(merged.shape[1])
        return DenoiserOutput(sample=self.model(model_input.latents))


def test_i2v_freezes_first_frame_and_masks_it() -> None:
    torch.manual_seed(43)
    denoiser = _I2VDenoiser(_TinyModel(4))
    adapter = HunyuanVideoI2VTrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=4,
    )
    assert adapter.model_timestep_scale == 1000.0
    clean = torch.randn(1, 4, 5, 2, 2)
    batch = TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={
            "clean_latents": clean,
            "text_states": torch.randn(1, 6, 16),
            "text_mask": torch.ones(1, 6),
            "text_states_2": torch.randn(1, 8),
        },
    )
    prepared = adapter.prepare_batch(batch)
    assert prepared.metadata["model_family"] == "hunyuan-video-i2v"
    mask = prepared.loss_mask
    assert torch.equal(mask[:, :, :1], torch.zeros(1, 1, 1, 1, 1))
    assert torch.equal(mask[:, :, 1:], torch.ones(1, 1, 4, 1, 1))

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(47))
    prediction = adapter.forward_train(corrupted)
    objective.compute_loss(prediction, corrupted).loss.backward()

    # First frame reached the model clean; later frames corrupted.
    assert denoiser.seen_latents is not None
    seen = denoiser.seen_latents
    clean_dev = clean.to(seen)
    torch.testing.assert_close(seen[:, :, :1], clean_dev[:, :, :1], rtol=0, atol=0)
    assert not torch.allclose(seen[:, :, 1:], clean_dev[:, :, 1:])
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_h15_concat_condition_full_frame_loss() -> None:
    torch.manual_seed(51)
    denoiser = _H15Denoiser(_TinyModel(8))
    adapter = HunyuanVideo15TrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=8, spatial_compression=16,
    )
    clean = torch.randn(1, 8, 3, 2, 2)
    batch = TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={
            "clean_latents": clean,
            "text_states": torch.randn(1, 6, 16),
            "text_mask": torch.ones(1, 6),
            "condition_latents": torch.randn(1, 8, 3, 2, 2),
            "byt5_text_states": torch.randn(1, 4, 16),
            "byt5_text_mask": torch.ones(1, 4),
        },
    )
    prepared = adapter.prepare_batch(batch)
    assert prepared.metadata["model_family"] == "hunyuan-video-1.5"
    assert prepared.loss_mask is None  # every frame is generated and scored

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(53))
    prediction = adapter.forward_train(corrupted)
    objective.compute_loss(prediction, corrupted).loss.backward()

    assert denoiser.seen_channels == 16  # 8 latent + 8 condition, channel-concat
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_h15_requires_condition_latents() -> None:
    adapter = HunyuanVideo15TrainAdapter(
        _H15Denoiser(_TinyModel(8)), codec=None, conditioner=None, expected_latent_channels=8,
    )
    batch = TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 8, 3, 2, 2),
            "text_states": torch.randn(1, 6, 16),
            "text_mask": torch.ones(1, 6),
            "byt5_text_states": torch.randn(1, 4, 16),
            "byt5_text_mask": torch.ones(1, 4),
        },
    )
    with pytest.raises(ValueError, match="condition_latents"):
        adapter.prepare_batch(batch)
