"""CPU contracts for Wan VAE decode acceleration dispatch and telemetry."""

from __future__ import annotations

import contextlib
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
    component as wan_component,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    WanTAEPreviewDecoder,
    WanVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    WanVideoVAE,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.tae_22 import (
    TAEW22StreamingDecoder,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.z_dim = 4
        self.model = type("Backbone", (), {"clear_cache": lambda self: None})()
        self.decode_autocast_dtype = None
        self.calls: list[dict[str, object]] = []

    _spatial_tile_tasks = staticmethod(WanVideoVAE._spatial_tile_tasks)

    def _decode_autocast_is_redundant(self, input_dtype):
        return self.decode_autocast_dtype == input_dtype == self.anchor.dtype

    def _decode_autocast(self, input_dtype=None):
        del input_dtype
        return contextlib.nullcontext()

    def decode(self, hidden_states, device, **kwargs):
        self.calls.append({"kind": "local", "device": device, **kwargs})
        return hidden_states[:, :3]


class _FakeConvVAE(_FakeVAE):
    def __init__(self) -> None:
        super().__init__()
        self.conv3d = torch.nn.Conv3d(4, 4, kernel_size=3, padding=1)


class _FakeTAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.reset_calls = 0
        self.inputs: list[torch.Tensor] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.inputs.append(latents)
        batch = latents.shape[0]
        return torch.full((batch, 121, 3, 32, 48), 0.75, dtype=latents.dtype)

    def parallel_tiled_decode(self, hidden_states, device, *args, **kwargs):
        self.calls.append(
            {"kind": "parallel", "device": device, "args": args, **kwargs}
        )
        return hidden_states[:, :3]


def _latents(*, frames: int = 5, height: int = 16, width: int = 16) -> torch.Tensor:
    return torch.randn(1, 4, frames, height, width)


class _StreamingDecoder(torch.nn.Module):
    def forward(self, value, *, feat_cache, feat_idx):
        previous = feat_cache[0]
        output = value if previous is None else value + previous
        feat_cache[0] = value
        feat_idx[0] += 1
        return output, feat_cache, feat_idx


class _StreamingBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2 = torch.nn.Identity()
        self.decoder = _StreamingDecoder()
        self.clear_cache()

    def clear_cache(self) -> None:
        self._conv_idx = [0]
        self._feat_map = [None]


class _ParallelBackbone:
    @staticmethod
    def decode(value, scale):
        del scale
        return value


def _parallel_decode_worker(rank, world_size, init_path, result_queue) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        vae = WanVideoVAE.__new__(WanVideoVAE)
        torch.nn.Module.__init__(vae)
        vae.model = _ParallelBackbone()
        vae.scale = [torch.zeros(3), torch.ones(3)]
        vae.z_dim = 3
        vae.upsampling_factor = 1
        hidden = torch.arange(3 * 1 * 4 * 4, dtype=torch.float32).reshape(
            1,
            3,
            1,
            4,
            4,
        ) / 100
        output = vae.parallel_tiled_decode(
            hidden,
            torch.device("cpu"),
            (3, 3),
            (2, 2),
        )
        result_queue.put(bool(torch.equal(output, hidden)))
    finally:
        dist.destroy_process_group()


def test_spatial_tiling_is_consumed_and_reported() -> None:
    vae = _FakeVAE()
    decoder = WanVideoDecoder(
        vae,
        device=torch.device("cpu"),
        tiled=True,
        tile_size=(8, 8),
        tile_stride=(4, 4),
    )
    output = decoder.decode(_latents())
    assert output.shape[:3] == (1, 3, 5)
    assert vae.calls[0]["tiled"] is True
    report = decoder.runtime_optimization_report()
    assert report["effective"]["vae_decode"] == "spatial-tiled"
    assert report["effective"]["vae_spatial_tiles"] > 1
    assert report["quality_tier"] == "numerically-approximate"


def test_temporal_streaming_is_exact_and_reaches_main_decode() -> None:
    vae = _FakeVAE()
    decoder = WanVideoDecoder(
        vae,
        device=torch.device("cpu"),
        temporal_chunk_size=2,
    )
    decoder.decode(_latents(frames=5))
    assert vae.calls[0]["temporal_chunk_size"] == 2
    report = decoder.runtime_optimization_report()
    assert report["effective"]["vae_decode"] == "causal-temporal-streaming"
    assert report["runtime"]["temporal_chunked_decode_calls"] == 1
    assert report["quality_tier"] == "exact"


def test_taew22_preview_adapts_layout_range_and_reports_approximation() -> None:
    tae = _FakeTAE()
    decoder = WanTAEPreviewDecoder(
        tae,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        checkpoint_path="/models/taew2_2.pth",
    )
    latents = torch.randn(1, 48, 31, 2, 3)
    output = decoder.decode(latents)

    assert tae.inputs[0].shape == (1, 31, 48, 2, 3)
    assert tae.inputs[0].dtype == torch.bfloat16
    assert output.shape == (1, 3, 121, 32, 48)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == torch.tensor(0.5, dtype=torch.bfloat16))
    report = decoder.runtime_optimization_report()
    assert report["effective"]["vae_preview_decoder"] == "taew2.2"
    assert report["runtime"]["preview_decode_calls"] == 1
    assert report["quality_tier"] == "algorithmically-approximate-preview"


def test_taew22_loads_direct_safetensors_checkpoint(tmp_path) -> None:
    from safetensors.torch import save_file

    source = torch.nn.Sequential(torch.nn.Linear(4, 3))
    expected = {
        f"decoder.{name}": value.detach().clone()
        for name, value in source.state_dict().items()
    }
    checkpoint = tmp_path / "taew2_2.safetensors"
    save_file(expected, str(checkpoint))
    target = torch.nn.Sequential(torch.nn.Linear(4, 3))

    TAEW22StreamingDecoder._load_decoder(
        SimpleNamespace(decoder=target),
        checkpoint,
    )

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)


def test_temporal_chunking_keeps_causal_cache_across_chunk_boundaries() -> None:
    vae = WanVideoVAE.__new__(WanVideoVAE)
    torch.nn.Module.__init__(vae)
    vae.model = _StreamingBackbone()
    vae.scale = [torch.zeros(3), torch.ones(3)]
    vae.z_dim = 3
    latent = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1, 1) / 100
    output = vae.temporal_chunked_decode(
        latent,
        torch.device("cpu"),
        chunk_size=2,
    )
    expected = latent.clone()
    expected[:, :, 1:] += latent[:, :, :-1]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("temporal_chunk_size", [0, 2])
def test_resident_decode_does_not_stage_latents_through_cpu(
    temporal_chunk_size,
) -> None:
    """A meta tensor makes any accidental ``to('cpu')`` transfer fail."""

    vae = WanVideoVAE.__new__(WanVideoVAE)
    torch.nn.Module.__init__(vae)
    vae.decode_autocast_dtype = None
    seen: list[torch.device] = []

    def decode_on_resident_device(hidden_state, device, **kwargs):
        del device, kwargs
        seen.append(hidden_state.device)
        return hidden_state[:, :3]

    vae.single_decode = decode_on_resident_device
    vae.temporal_chunked_decode = decode_on_resident_device
    latents = torch.empty((1, 4, 5, 2, 2), device="meta")

    output = vae.decode(
        latents,
        torch.device("meta"),
        temporal_chunk_size=temporal_chunk_size,
    )

    assert output.device.type == "meta"
    assert seen == [torch.device("meta")]


def test_resident_encode_does_not_stage_video_through_cpu() -> None:
    """Dense encode preserves device residency just like dense decode."""

    vae = WanVideoVAE.__new__(WanVideoVAE)
    torch.nn.Module.__init__(vae)
    seen: list[torch.device] = []

    def encode_on_resident_device(video, device):
        del device
        seen.append(video.device)
        return video[:, :2]

    vae.single_encode = encode_on_resident_device
    videos = torch.empty((1, 3, 5, 2, 2), device="meta")

    output = vae.encode(videos, torch.device("meta"))

    assert output.device.type == "meta"
    assert seen == [torch.device("meta")]


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is unavailable",
)
def test_parallel_tiled_decode_uses_real_gloo_collectives(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_path = str(tmp_path / "vae-parallel-rendezvous")
    processes = [
        context.Process(
            target=_parallel_decode_worker,
            args=(rank, 2, init_path, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert [result_queue.get(timeout=2) for _ in processes] == [True, True]


def test_single_spatial_tile_is_an_explicit_fallback() -> None:
    decoder = WanVideoDecoder(
        _FakeVAE(),
        device=torch.device("cpu"),
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    )
    decoder.decode(_latents(height=8, width=8))
    report = decoder.runtime_optimization_report()
    assert report["effective"]["vae_decode"] == "spatial-tiled (single-tile)"
    assert report["runtime"]["single_spatial_tile_calls"] == 1
    assert any("one tile" in fallback for fallback in report["fallbacks"])


def test_decode_telemetry_is_scoped_to_the_last_request() -> None:
    decoder = WanVideoDecoder(
        _FakeVAE(),
        device=torch.device("cpu"),
        tiled=True,
        tile_size=(8, 8),
        tile_stride=(4, 4),
    )
    decoder.decode(_latents(height=16, width=16))
    first = decoder.runtime_optimization_report()
    assert first["runtime"]["window_id"] == 1
    assert first["runtime"]["last_spatial_tile_count"] > 1
    assert first["runtime"]["single_spatial_tile_calls"] == 0

    decoder.decode(_latents(height=4, width=4))
    second = decoder.runtime_optimization_report()
    assert second["runtime"]["window_id"] == 2
    assert second["runtime"]["decode_calls"] == 1
    assert second["runtime"]["spatial_tiled_decode_calls"] == 1
    assert second["runtime"]["last_spatial_tile_count"] == 1
    assert second["runtime"]["single_spatial_tile_calls"] == 1
    assert second["runtime"]["lifetime"]["decode_calls"] == 2
    assert second["runtime"]["lifetime"]["spatial_tiled_decode_calls"] == 2


def test_temporal_chunk_receipt_requires_multiple_chunks() -> None:
    decoder = WanVideoDecoder(
        _FakeVAE(),
        device=torch.device("cpu"),
        temporal_chunk_size=4,
    )
    decoder.decode(_latents(frames=4))
    single = decoder.runtime_optimization_report()
    assert single["runtime"]["temporal_chunked_decode_calls"] == 0
    assert single["runtime"]["single_temporal_chunk_calls"] == 1
    assert single["runtime"]["last_temporal_chunk_count"] == 1

    decoder.decode(_latents(frames=5))
    multiple = decoder.runtime_optimization_report()
    assert multiple["runtime"]["temporal_chunked_decode_calls"] == 1
    assert multiple["runtime"]["single_temporal_chunk_calls"] == 0
    assert multiple["runtime"]["last_temporal_chunk_count"] == 2
    assert multiple["runtime"]["lifetime"]["temporal_chunked_decode_calls"] == 1


def test_block_offload_keeps_wan_vae_resident_without_legacy_wrappers(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_load(self, spec, checkpoint, policy):
        del self, checkpoint
        seen["spec"] = spec
        seen["policy"] = policy
        return _FakeVAE()

    monkeypatch.setattr(wan_component.NativeModuleLoader, "load", fake_load)
    decoder = wan_component._load_wan_video_decoder(
        CheckpointSpec(source="unused.safetensors"),
        RuntimePolicy(
            device="cpu",
            offload=OffloadPolicy(
                mode=OffloadMode.BLOCK,
                target="cpu",
                pin_memory=True,
            ),
        ),
        module_class=_FakeVAE,
    )

    load_policy = seen["policy"]
    assert isinstance(load_policy, RuntimePolicy)
    assert load_policy.offload.mode is OffloadMode.NONE
    report = decoder.runtime_optimization_report()
    assert report["requested"]["vae_offload"] == "block"
    assert report["effective"]["vae_offload"] == "resident-no-legacy-wrapper"
    assert not any(
        module.__class__.__name__.startswith("AutoWrapped")
        for module in decoder.vae.modules()
    )


def test_resident_bf16_and_channels_last_3d_are_applied_and_reported(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_load(self, spec, checkpoint, policy):
        del self, spec, checkpoint
        seen["policy"] = policy
        return _FakeConvVAE().to(dtype=policy.dtype)

    monkeypatch.setattr(wan_component.NativeModuleLoader, "load", fake_load)
    decoder = wan_component._load_wan_video_decoder(
        CheckpointSpec(source="unused.safetensors"),
        RuntimePolicy(
            device="cpu",
            dtype=torch.float32,
            options={
                "vae_weight_dtype": "bf16",
                "vae_channels_last_3d": True,
                "vae_decode_autocast": "bf16",
            },
        ),
        module_class=_FakeConvVAE,
    )

    load_policy = seen["policy"]
    assert isinstance(load_policy, RuntimePolicy)
    assert load_policy.dtype is torch.bfloat16
    assert decoder.dtype is torch.bfloat16
    assert decoder.vae.conv3d.weight.is_contiguous(
        memory_format=torch.channels_last_3d
    )
    report = decoder.runtime_optimization_report()
    assert report["requested"]["vae_weight_dtype"] == "torch.bfloat16"
    assert report["requested"]["vae_channels_last_3d"] is True
    assert report["effective"]["vae_weight_dtype"] == "torch.bfloat16"
    assert report["effective"]["vae_channels_last_3d"] == "enabled"
    assert report["effective"]["vae_channels_last_3d_conv3d"] == 1
    assert report["effective"]["vae_conv3d_count"] == 1
    assert report["effective"]["vae_decode_autocast_context"] == (
        "elided-resident-dtype"
    )


def test_channels_last_3d_option_requires_a_bool() -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        wan_component._resolve_vae_channels_last_3d(
            RuntimePolicy(options={"vae_channels_last_3d": "yes"})
        )


def test_parallel_decode_rejects_an_uninitialized_process_group(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized torchrun process group"):
        WanVideoDecoder(
            _FakeVAE(),
            device=torch.device("cpu"),
            parallel_degree=2,
        )


@pytest.mark.parametrize(
    ("tile_size", "tile_stride", "message"),
    [
        ((0, 8), (4, 4), "positive"),
        ((8, 8), (9, 4), "cannot exceed"),
    ],
)
def test_invalid_tile_geometry_is_rejected(tile_size, tile_stride, message) -> None:
    with pytest.raises(ValueError, match=message):
        WanVideoDecoder(
            _FakeVAE(),
            device=torch.device("cpu"),
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
