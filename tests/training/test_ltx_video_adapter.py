from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    MultiModalDenoiserInput,
    MultiModalDenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import LTXVideoTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import FlowMatchingConfig, FlowMatchingObjective  # noqa: E402


class _TinyLTXModel(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class _TinyLTXDenoiser:
    """Multi-modal contract denoiser returning x0 on the video token sequence."""

    def __init__(self, model: _TinyLTXModel) -> None:
        self.model = model
        self.seen_tokens: int | None = None
        self.seen_positions: bool = False

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        state = model_input.modalities["video"]
        assert isinstance(model_input.conditioning["video_context"], torch.Tensor)
        # Private threading keys must be stripped before the denoiser call.
        assert not any(str(k).startswith("_ltx") for k in model_input.conditioning)
        assert state.positions is not None and state.denoise_mask is not None
        self.seen_tokens = int(state.latent.shape[1])
        # positions are [B, coords, tokens, bounds]; the token axis matches latent.
        self.seen_positions = int(state.positions.shape[2]) == int(state.latent.shape[1])
        # Predict a velocity, return x0 = latent - sigma*velocity (LTX surface).
        sigma = model_input.timestep.reshape(-1, 1, 1)
        velocity = self.model(state.latent)
        x0 = state.latent - sigma * velocity
        return MultiModalDenoiserOutput(samples={"video": x0})


def _adapter(denoiser: _TinyLTXDenoiser) -> LTXVideoTrainAdapter:
    return LTXVideoTrainAdapter(denoiser, codec=None, conditioner=None, expected_latent_channels=8)


def _batch() -> TrainingBatch:
    # [B, C=8, T=2, H=2, W=2] -> 8 tokens after patch size 1.
    return TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 8, 2, 2, 2),
            "video_context": torch.randn(1, 6, 16),
        },
    )


def test_ltx_prepare_builds_token_sequence_tree() -> None:
    adapter = _adapter(_TinyLTXDenoiser(_TinyLTXModel(8)))
    prepared = adapter.prepare_batch(_batch())
    assert prepared.metadata["model_family"] == "ltx-video"
    # clean_latents is a per-modality tree of patchified tokens [B, tokens, C].
    assert set(prepared.clean_latents) == {"video"}
    tokens = prepared.clean_latents["video"]
    assert tokens.ndim == 3 and int(tokens.shape[0]) == 1 and int(tokens.shape[2]) == 8
    assert int(tokens.shape[1]) == 2 * 2 * 2  # T*H*W tokens


def test_ltx_multimodal_forward_loss_backward_x0_to_velocity() -> None:
    torch.manual_seed(97)
    denoiser = _TinyLTXDenoiser(_TinyLTXModel(8))
    adapter = _adapter(denoiser)
    assert adapter.prediction_type == "flow_velocity"
    prepared = adapter.prepare_batch(_batch())

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(101))
    prediction = adapter.forward_train(corrupted)
    assert set(prediction) == {"video"}
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert denoiser.seen_tokens == 8
    assert denoiser.seen_positions  # positions cover the token axis
    assert torch.isfinite(result.loss)
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_ltx_requires_video_context() -> None:
    adapter = _adapter(_TinyLTXDenoiser(_TinyLTXModel(8)))
    batch = TrainingBatch(
        sample_ids=("v",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 8, 2, 2, 2)},
    )
    with pytest.raises(ValueError, match="video_context"):
        adapter.prepare_batch(batch)
