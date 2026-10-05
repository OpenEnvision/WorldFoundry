from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import GammaWorldBidirectionalTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyGammaNet(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.proj(latents)


class _TinyGammaDenoiser:
    """Bidirectional Gamma denoiser: blends clean condition frames, returns velocity."""

    is_bidirectional = True

    def __init__(self, net: _TinyGammaNet) -> None:
        self.net = net
        self.seen_condition_keys: set | None = None
        self.seen_input: torch.Tensor | None = None

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        cond = model_input.conditioning
        assert isinstance(cond["crossattn_emb"], torch.Tensor)
        assert cond.get("action_inputs") is not None
        assert "block_noise" not in cond and "initial_noise" not in cond
        self.seen_condition_keys = set(cond)
        latents = model_input.latents
        gt = cond.get("gt_frames")
        mask = cond.get("condition_video_input_mask_B_C_T_H_W")
        if isinstance(gt, torch.Tensor) and isinstance(mask, torch.Tensor):
            em = mask.expand(-1, latents.shape[1], -1, -1, -1).to(latents)
            latents = gt.to(latents) * em + latents * (1 - em)
        self.seen_input = latents.detach().clone()
        return DenoiserOutput(sample=self.net(latents))


def _adapter(denoiser: _TinyGammaDenoiser) -> GammaWorldBidirectionalTrainAdapter:
    return GammaWorldBidirectionalTrainAdapter(
        denoiser, codec=None, conditioner=None, expected_latent_channels=4,
    )


def _batch(*, frames: int = 4, with_condition: bool = True) -> TrainingBatch:
    conditions = {
        "clean_latents": torch.randn(1, 4, frames, 2, 2),
        "crossattn_emb": torch.randn(1, 6, 16),
        "action_inputs": {"actions": [{"keyboard": torch.randn(1, frames, 3)}]},
    }
    if with_condition:
        mask = torch.zeros(1, 1, frames, 2, 2)
        mask[:, :, :1] = 1.0  # first frame given
        conditions["condition_video_input_mask_B_C_T_H_W"] = mask
        conditions["gt_frames"] = torch.randn(1, 4, frames, 2, 2)
    return TrainingBatch(sample_ids=("g",), prompts=("cached",), conditions=conditions)


def test_gamma_world_uses_net_attribute_and_scale() -> None:
    net = _TinyGammaNet(4)
    adapter = _adapter(_TinyGammaDenoiser(net))
    assert adapter.trainable_module is net
    assert adapter.model_timestep_scale == 1000.0


def test_gamma_world_condition_mask_drives_loss_mask_and_forward() -> None:
    torch.manual_seed(67)
    denoiser = _TinyGammaDenoiser(_TinyGammaNet(4))
    adapter = _adapter(denoiser)
    prepared = adapter.prepare_batch(_batch(frames=4, with_condition=True))
    assert prepared.metadata["model_family"] == "gamma-world-bidirectional-video"
    mask = prepared.loss_mask
    assert tuple(mask.shape) == (1, 1, 4, 2, 2)
    assert torch.equal(mask[:, :, :1], torch.zeros(1, 1, 1, 2, 2))
    assert torch.equal(mask[:, :, 1:], torch.ones(1, 1, 3, 2, 2))

    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform", num_train_timesteps=1000, flow_shift=1.0)
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(71))
    prediction = adapter.forward_train(corrupted)
    objective.compute_loss(prediction, corrupted).loss.backward()

    assert denoiser.seen_condition_keys is not None
    assert {"gt_frames", "condition_video_input_mask_B_C_T_H_W", "action_inputs"} <= denoiser.seen_condition_keys
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_gamma_world_unconditional_has_no_loss_mask() -> None:
    adapter = _adapter(_TinyGammaDenoiser(_TinyGammaNet(4)))
    prepared = adapter.prepare_batch(_batch(with_condition=False))
    assert prepared.loss_mask is None


def test_gamma_world_requires_action_inputs() -> None:
    adapter = _adapter(_TinyGammaDenoiser(_TinyGammaNet(4)))
    batch = TrainingBatch(
        sample_ids=("g",),
        prompts=("cached",),
        conditions={"clean_latents": torch.randn(1, 4, 4, 2, 2), "crossattn_emb": torch.randn(1, 6, 16)},
    )
    with pytest.raises(ValueError, match="action_inputs"):
        adapter.prepare_batch(batch)
