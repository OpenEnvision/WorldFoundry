from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import MatrixGame35TrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyMatrixModel(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyMatrixDenoiser:
    """Mirrors the contract: concat clean anchor internally, output only noisy frames."""

    def __init__(self, model: _TinyMatrixModel) -> None:
        self.model = model
        self.seen_condition_keys: set | None = None
        self.seen_frames: int | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        cond = dict(model_input.conditioning)
        assert isinstance(cond.pop("context"), torch.Tensor)
        assert isinstance(cond["first_frame_latents"], torch.Tensor)
        self.seen_condition_keys = set(model_input.conditioning)
        self.seen_frames = int(model_input.latents.shape[2])
        # Output is pure velocity for the noisy window (already matches input).
        return DenoiserOutput(sample=self.model(model_input.latents))


def _adapter(denoiser: _TinyMatrixDenoiser) -> MatrixGame35TrainAdapter:
    return MatrixGame35TrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=8, spatial_compression=16,
    )


def _batch(*, window: int = 4, batch: int = 1) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=tuple(f"m{i}" for i in range(batch)),
        prompts=tuple("cached" for _ in range(batch)),
        conditions={
            "clean_latents": torch.randn(batch, 8, window, 2, 2),
            "context": torch.randn(batch, 6, 16),
            "first_frame_latents": torch.randn(batch, 8, 1, 2, 2),
            "action_inputs": torch.randn(batch, window, 4),
        },
    )


def test_matrix_game_window_only_no_mask_conditions_passthrough() -> None:
    torch.manual_seed(59)
    denoiser = _TinyMatrixDenoiser(_TinyMatrixModel(8))
    adapter = _adapter(denoiser)
    assert adapter.model_timestep_scale == 1000.0
    prepared = adapter.prepare_batch(_batch(window=4))
    assert prepared.metadata["model_family"] == "matrix-game-3.5-video"
    # No first-frame freeze / loss mask: the anchor never enters model_input.
    assert prepared.loss_mask is None
    assert tuple(prepared.clean_latents.shape) == (1, 8, 4, 2, 2)

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(61))
    prediction = adapter.forward_train(corrupted)
    objective.compute_loss(prediction, corrupted).loss.backward()

    # The denoiser saw only the generating window and the anchor/action conditions.
    assert denoiser.seen_frames == 4
    assert {"first_frame_latents", "action_inputs"} <= denoiser.seen_condition_keys
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_matrix_game_requires_first_frame_anchor() -> None:
    adapter = _adapter(_TinyMatrixDenoiser(_TinyMatrixModel(8)))
    batch = TrainingBatch(
        sample_ids=("m",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 8, 4, 2, 2), "context": torch.randn(1, 6, 16)},
    )
    with pytest.raises(ValueError, match="first_frame_latents"):
        adapter.prepare_batch(batch)


def test_matrix_game_rejects_multi_media_batch() -> None:
    adapter = _adapter(_TinyMatrixDenoiser(_TinyMatrixModel(8)))
    with pytest.raises(ValueError, match="batch size one"):
        adapter.prepare_batch(_batch(batch=2))
