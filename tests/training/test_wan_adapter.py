from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.contracts import Conditioning  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (  # noqa: E402
    WanDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (  # noqa: E402
    AttentionModule,
    WanModel,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import WanTrainAdapter, wan_pixel_mask_to_latent  # noqa: E402
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


def _tiny_model() -> WanModel:
    return WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1.0e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    )


def _adapter(*, gradient_checkpointing: bool = False) -> WanTrainAdapter:
    model = _tiny_model()
    denoiser = WanDenoiser(model, compute_dtype=torch.float32)
    return WanTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
        expected_text_length=4,
        expected_context_features=16,
        gradient_checkpointing=gradient_checkpointing,
    )


def _cached_batch() -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("video",),
        prompts=("cached",),
        conditions={
            "clean_latents": torch.randn(1, 4, 2, 2, 2),
            "context": torch.randn(1, 4, 16),
            "latent_loss_mask": torch.tensor([[[[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]]),
            "valid_latent_mask": torch.ones(1, 1, 2, 2, 2, dtype=torch.bool),
        },
        metadata={
            "target_num_frames": 5,
            "target_height": 4,
            "target_width": 4,
        },
    )


def test_wan_causal_pixel_mask_preserves_partial_temporal_and_spatial_weights() -> None:
    mask = torch.ones(1, 1, 5, 4, 4)
    mask[:, :, 3:] = 0
    mask[:, :, :, :, 2:] = 0

    latent = wan_pixel_mask_to_latent(
        mask,
        pixel_shape=(1, 3, 5, 4, 4),
        latent_shape=(2, 2, 2),
        temporal_compression=4,
    )

    assert tuple(latent.shape) == (1, 1, 2, 2, 2)
    torch.testing.assert_close(latent[:, :, 0], torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]))
    torch.testing.assert_close(latent[:, :, 1], torch.tensor([[[[0.5, 0.0], [0.5, 0.0]]]]))


def test_cached_wan_forward_loss_backward_and_timestep_contract() -> None:
    torch.manual_seed(23)
    adapter = _adapter()
    prepared = adapter.prepare_batch(_cached_batch())
    objective = FlowMatchingObjective(
        FlowMatchingConfig(
            timestep_sampler="uniform",
            num_train_timesteps=1000,
            flow_shift=1.0,
        )
    )
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(29))
    observed: dict[str, object] = {}

    def capture(_module, _args, kwargs):
        observed.update(kwargs)

    handle = adapter.trainable_module.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        prediction = adapter.forward_train(corrupted)
    finally:
        handle.remove()
    result = objective.compute_loss(prediction, corrupted)
    result.loss.backward()

    torch.testing.assert_close(
        observed["timestep"],
        corrupted.sigmas * 1000.0,
        rtol=0,
        atol=0,
    )
    assert observed["use_gradient_checkpointing"] is False
    assert tuple(prediction.shape) == (1, 4, 2, 2, 2)
    assert torch.isfinite(result.loss)
    assert result.latent_token_count == 8
    assert prepared.metadata["precomputed_latents"] is True
    assert prepared.metadata["attention_compatibility_mode"] is True
    assert torch.equal(prepared.loss_mask[:, :, 1], torch.full((1, 4, 2, 2), 0.5))
    assert all(
        module.compatibility_mode
        for module in adapter.trainable_module.modules()
        if isinstance(module, AttentionModule)
    )
    gradients = [parameter.grad for parameter in adapter.trainable_module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_wan_activation_checkpoint_flag_reaches_native_blocks() -> None:
    adapter = _adapter(gradient_checkpointing=True)
    prepared = adapter.prepare_batch(_cached_batch())
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    corrupted = objective.corrupt(prepared, generator=torch.Generator().manual_seed(31))
    observed: dict[str, object] = {}

    def capture(_module, _args, kwargs):
        observed.update(kwargs)

    handle = adapter.trainable_module.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        prediction = adapter.forward_train(corrupted)
        objective.compute_loss(prediction, corrupted).loss.backward()
    finally:
        handle.remove()

    assert observed["use_gradient_checkpointing"] is True


def test_wan_cached_geometry_and_context_drift_fail_closed() -> None:
    adapter = _adapter()
    bad_geometry = _cached_batch()
    bad_geometry = TrainingBatch(
        sample_ids=bad_geometry.sample_ids,
        prompts=bad_geometry.prompts,
        conditions=bad_geometry.conditions,
        metadata={**bad_geometry.metadata, "target_num_frames": 9},
    )
    with pytest.raises(ValueError, match="codec-implied geometry"):
        adapter.prepare_batch(bad_geometry)

    batch = _cached_batch()
    bad_context = TrainingBatch(
        sample_ids=batch.sample_ids,
        prompts=batch.prompts,
        conditions={**batch.conditions, "context": torch.zeros(1, 3, 16)},
        metadata=batch.metadata,
    )
    with pytest.raises(ValueError, match="context must have shape"):
        adapter.prepare_batch(bad_context)


class _TinyCodec:
    temporal_compression_factor = 4
    spatial_compression_factor = 2

    def __init__(self) -> None:
        self.vae = torch.nn.Conv3d(3, 4, kernel_size=1, bias=False)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        first = pixels[:, :, :1]
        later = (
            pixels[:, :, 1:]
            .reshape(
                pixels.shape[0],
                pixels.shape[1],
                1,
                4,
                pixels.shape[3],
                pixels.shape[4],
            )
            .mean(dim=3)
        )
        temporal = torch.cat((first, later), dim=2)
        spatial = torch.nn.functional.avg_pool3d(
            temporal,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        return self.vae(spatial)


class _TinyConditioner:
    def __init__(self) -> None:
        self.text_encoder = torch.nn.Linear(2, 16, bias=False)

    def encode(self, request, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        values = torch.ones(request.batch_size, 4, 2, device=device, dtype=dtype)
        return Conditioning(positive={"context": self.text_encoder(values)})


def test_raw_wan_adapter_freezes_encoders_and_projects_pixel_mask() -> None:
    codec = _TinyCodec()
    conditioner = _TinyConditioner()
    model = _tiny_model()
    adapter = WanTrainAdapter(
        WanDenoiser(model, compute_dtype=torch.float32),
        codec,
        conditioner,
        expected_latent_channels=4,
        temporal_compression=4,
        spatial_compression=2,
        expected_text_length=4,
        expected_context_features=16,
    )
    mask = torch.ones(1, 1, 5, 4, 4)
    mask[:, :, 3:] = 0
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("raw",),
            prompts=("a moving square",),
            pixel_values=torch.randn(1, 3, 5, 4, 4),
            valid_mask=mask,
        )
    )

    assert tuple(prepared.clean_latents.shape) == (1, 4, 2, 2, 2)
    assert tuple(prepared.conditioning["context"].shape) == (1, 4, 16)
    assert torch.equal(prepared.loss_mask[:, :, 1], torch.full((1, 1, 2, 2), 0.5))
    assert all(not parameter.requires_grad for parameter in codec.vae.parameters())
    assert all(not parameter.requires_grad for parameter in conditioner.text_encoder.parameters())
    assert codec.vae.training is False
    assert conditioner.text_encoder.training is False
