from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.contracts import Conditioning  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser  # noqa: E402
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import (  # noqa: E402
    CaptionEmbedder,
    CaptionEmbedderDoubleBr,
)
from worldfoundry.training.api import TrainingBatch  # noqa: E402
from worldfoundry.training.models import SanaTrainAdapter  # noqa: E402
from worldfoundry.training.objectives import FlowMatchingConfig, FlowMatchingObjective  # noqa: E402


class _TinyCodec:
    scaling_factor = 0.5

    def __init__(self) -> None:
        self.model = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images) * self.scaling_factor


class _TinyConditioner:
    def __init__(self) -> None:
        self.encoder = torch.nn.Linear(2, 4, bias=False)
        self.calls = 0

    def encode(self, request, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        self.calls += 1
        features = torch.ones(request.batch_size, 3, 2, device=device, dtype=dtype)
        context = self.encoder(features)[:, None]
        return Conditioning(
            positive={
                "context": context,
                "context_mask": torch.ones(request.batch_size, 3, device=device, dtype=torch.long),
            },
            shared={},
        )


class _TinyCaptionEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.uncond_prob = 0.1


class _TinySanaGraph(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.75))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.y_embedder = _TinyCaptionEmbedder()
        self.last_timestep: torch.Tensor | None = None
        self.last_data_info: dict[str, object] | None = None

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        del mask, kwargs
        self.last_timestep = timestep.detach().clone()
        self.last_data_info = data_info
        context_bias = y.float().mean(dim=(1, 2, 3), keepdim=True)
        return self.gain * x + 0.001 * context_bias


def _adapter() -> tuple[SanaTrainAdapter, _TinySanaGraph, _TinyCodec, _TinyConditioner]:
    graph = _TinySanaGraph()
    codec = _TinyCodec()
    conditioner = _TinyConditioner()
    adapter = SanaTrainAdapter(
        SanaDenoiser(graph),
        codec,
        conditioner,
        expected_latent_channels=4,
    )
    return adapter, graph, codec, conditioner


def test_sana_fixed_tensor_prepare_forward_loss_and_backward() -> None:
    adapter, graph, codec, conditioner = _adapter()
    pixels = torch.linspace(-1.0, 1.0, 2 * 3 * 1 * 8 * 8).reshape(2, 3, 1, 8, 8)
    valid_mask = torch.ones(2, 1, 1, 8, 8)
    valid_mask[1, :, :, :, 4:] = 0
    raw = TrainingBatch(
        sample_ids=("sample-a", "sample-b"),
        prompts=("red cube", "blue sphere"),
        pixel_values=pixels,
        valid_mask=valid_mask,
        sample_weights=torch.tensor([1.0, 0.5]),
    )

    prepared = adapter.prepare_batch(raw)
    objective = FlowMatchingObjective(
        FlowMatchingConfig(
            timestep_sampler="uniform",
            flow_shift=3.0,
            num_train_timesteps=1000,
        )
    )
    objective_batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(123))
    prediction = adapter.forward_train(objective_batch)
    result = objective.compute_loss(prediction, objective_batch)
    result.loss.backward()

    assert prepared.clean_latents.shape == (2, 4, 4, 4)
    assert prepared.loss_mask.shape == (2, 1, 4, 4)
    assert conditioner.calls == 1
    assert graph.last_timestep is not None
    torch.testing.assert_close(graph.last_timestep, objective_batch.sigmas * 1000.0)
    assert graph.last_data_info is not None
    torch.testing.assert_close(graph.last_data_info["img_hw"], torch.tensor([[8.0, 8.0], [8.0, 8.0]]))
    assert graph.gain.grad is not None and bool(torch.isfinite(graph.gain.grad))
    assert all(parameter.grad is None for parameter in codec.model.parameters())
    assert all(parameter.grad is None for parameter in conditioner.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in codec.model.parameters())
    assert not any(parameter.requires_grad for parameter in conditioner.encoder.parameters())
    assert codec.model.training is False
    assert conditioner.encoder.training is False
    assert adapter.conditioning_dropout_probability == pytest.approx(0.1)
    assert result.sample_count == 2


def test_sana_precomputed_conditioning_skips_text_encoder() -> None:
    adapter, _, _, conditioner = _adapter()
    raw = TrainingBatch(
        sample_ids=("cached",),
        prompts=("unused cached prompt",),
        pixel_values=torch.zeros(1, 3, 1, 8, 8),
        conditions={
            "context": torch.ones(1, 1, 3, 4),
            "context_mask": torch.ones(1, 3, dtype=torch.long),
        },
    )

    prepared = adapter.prepare_batch(raw)

    assert conditioner.calls == 0
    assert prepared.conditioning["context"].shape == (1, 1, 3, 4)


def test_sana_precomputed_latents_and_context_need_no_encoders() -> None:
    graph = _TinySanaGraph()
    adapter = SanaTrainAdapter(
        SanaDenoiser(graph),
        codec=None,
        conditioner=None,
        expected_latent_channels=4,
        spatial_compression=2,
    )
    clean_latents = torch.randn(1, 4, 4, 4)
    raw = TrainingBatch(
        sample_ids=("cached",),
        prompts=("already encoded",),
        conditions={
            "clean_latents": clean_latents,
            "latent_loss_mask": torch.ones(1, 1, 4, 4),
            "context": torch.ones(1, 1, 3, 4),
            "context_mask": torch.ones(1, 3, dtype=torch.long),
        },
        metadata={"image_height": 8, "image_width": 8, "latent_scaling_factor": 0.5},
    )

    prepared = adapter.prepare_batch(raw)
    objective = FlowMatchingObjective(FlowMatchingConfig(flow_shift=3.0, num_train_timesteps=1000))
    objective_batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(31))
    result = objective.compute_loss(adapter.forward_train(objective_batch), objective_batch)
    result.loss.backward()

    torch.testing.assert_close(prepared.clean_latents, clean_latents)
    assert prepared.loss_mask.shape == (1, 1, 4, 4)
    assert prepared.metadata["precomputed_latents"] is True
    assert prepared.metadata["pixel_shape"] is None
    assert graph.gain.grad is not None


def test_sana_image_adapter_rejects_video_batches() -> None:
    adapter, _, _, _ = _adapter()
    raw = TrainingBatch(
        sample_ids=("video",),
        prompts=("moving cube",),
        pixel_values=torch.zeros(1, 3, 2, 8, 8),
    )

    with pytest.raises(ValueError, match="SANA image pixels"):
        adapter.prepare_batch(raw)


def test_sana_caption_dropout_samples_on_the_input_device() -> None:
    caption = torch.randn(2, 1, 3, 2)
    image_embedder = CaptionEmbedder(
        2,
        4,
        uncond_prob=1.0,
        act_layer=lambda: torch.nn.GELU(approximate="tanh"),
        token_num=3,
    )
    image_output = image_embedder(caption, train=True)

    video_embedder = CaptionEmbedderDoubleBr(
        2,
        4,
        uncond_prob=1.0,
        act_layer=lambda: torch.nn.GELU(approximate="tanh"),
        token_num=3,
    )
    video_output, dropped_caption = video_embedder(caption, train=True)

    assert image_output.device == caption.device
    assert video_output.device == caption.device
    assert dropped_caption.device == caption.device
    (image_output.sum() + video_output.sum()).backward()
