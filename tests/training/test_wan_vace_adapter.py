from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import WanVaceTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyVaceModel(nn.Module):
    """Minimal Wan-VACE-shaped denoiser network with the attention-mode hook."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Identity()])
        self.patch_size = (1, 2, 2)
        self.compatibility_mode = False

    def set_attention_compatibility_mode(self, enabled: bool) -> None:
        self.compatibility_mode = bool(enabled)


class _TinyVaceDenoiser:
    def __init__(self, model: _TinyVaceModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        conditioning = model_input.conditioning
        # Contract check: VACE control signal must be present on every call.
        assert isinstance(conditioning["context"], torch.Tensor)
        assert isinstance(conditioning["vace_context"], torch.Tensor)
        latents = model_input.latents
        scale = float(conditioning.get("vace_context_scale", 1.0))
        return DenoiserOutput(sample=self.model.proj(latents) * scale)


def _adapter() -> WanVaceTrainAdapter:
    model = _TinyVaceModel(channels=4)
    return WanVaceTrainAdapter(
        _TinyVaceDenoiser(model),
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
        expected_text_length=4,
        expected_context_features=16,
    )


def _cached_batch(*, with_vace: bool = True) -> TrainingBatch:
    conditions = {
        "clean_latents": torch.randn(1, 4, 2, 2, 2),
        "context": torch.randn(1, 4, 16),
    }
    if with_vace:
        conditions["vace_context"] = torch.randn(1, 96, 2, 2, 2)
    return TrainingBatch(
        sample_ids=("vace",),
        prompts=("cached",),
        conditions=conditions,
        metadata={"target_num_frames": 5, "target_height": 4, "target_width": 4},
    )


def test_wan_vace_forward_loss_backward_and_family_tag() -> None:
    torch.manual_seed(11)
    adapter = _adapter()
    prepared = adapter.prepare_batch(_cached_batch())
    assert prepared.metadata["model_family"] == "wan-vace-video"
    assert "vace_context" in prepared.conditioning

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(3))
    prediction = adapter.forward_train(corrupted)
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    assert tuple(prediction.shape) == (1, 4, 2, 2, 2)
    assert torch.isfinite(result.loss)
    gradients = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def test_wan_vace_missing_control_signal_fails_closed() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="vace_context"):
        adapter.prepare_batch(_cached_batch(with_vace=False))
