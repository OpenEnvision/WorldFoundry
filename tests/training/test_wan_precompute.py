from __future__ import annotations

import json
from pathlib import Path

import pytest

av = pytest.importorskip("av")
numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.base_models.diffusion_model.contracts import Conditioning  # noqa: E402
from worldfoundry.training.data import (  # noqa: E402
    MediaReference,
    SharedConditioningStore,
    TrainingManifestDataset,
    TrainingSample,
    VideoCachedDataset,
    VideoCacheStore,
    VideoDecodeConfig,
    VideoDecodingDataset,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
    collate_video_cached_samples,
)
from worldfoundry.training.data.wan.contracts import (  # noqa: E402
    WAN_LATENT_MEAN,
    WAN_LATENT_STD,
    wan_cache_contract,
    wan_latent_normalization,
)
from worldfoundry.training.data.wan.encoding import (  # noqa: E402
    WanFeatureEncoder,
    WanVideoFeatureEncoder,
)
from worldfoundry.training.data.wan.training_cache import (  # noqa: E402
    prepare_wan_training_cache_from_audits,
)
from worldfoundry.training.safety import PromptSafetyAudit  # noqa: E402
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
)


def _write_video(
    path: Path,
    *,
    frames: int = 5,
    height: int = 16,
    width: int = 16,
    fps: int = 5,
) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = numpy.full((height, width, 3), index * 20, dtype=numpy.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _safe_audit(prompt: str) -> PromptSafetyAudit:
    return PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )


def _dataset(tmp_path: Path) -> tuple[VideoDecodingDataset, PromptSafetyAudit]:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    prompt = "a square becomes brighter"
    audit = _safe_audit(prompt)
    sample = TrainingSample(
        sample_id="wan-video",
        task="t2v",
        prompt=prompt,
        media=MediaReference(
            uri=video.name,
            mime_type="video/x-matroska",
            size_bytes=video.stat().st_size,
        ),
        width=16,
        height=16,
        num_frames=5,
        fps=5.0,
        conditions={},
        split="train",
        safety={"prompt_safe": True, "model_revision": audit.model_revision},
    )
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        verify_files=True,
    )
    assignments = assign_video_buckets(
        tuple(manifest),
        buckets=(VideoResolutionBucket(5, 16, 16, "umt5-sequence"),),
        geometry=VideoLatentGeometry(8, 8, 4, "first-frame"),
        conditioning_layout="umt5-sequence",
    )
    return (
        VideoDecodingDataset(
            manifest,
            assignments,
            config=VideoDecodeConfig(),
        ),
        audit,
    )


class _FakeVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("mean", torch.tensor(WAN_LATENT_MEAN))
        self.register_buffer("std", torch.tensor(WAN_LATENT_STD))


class _FakeCodec:
    temporal_compression_factor = 4
    spatial_compression_factor = 8

    def __init__(self) -> None:
        self.vae = _FakeVae()

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch = pixels.shape[0]
        first = pixels[:, :1, :1, ::8, ::8]
        later = pixels[:, :1, 1:].reshape(batch, 1, 1, 4, 16, 16).mean(dim=3)
        later = later[:, :, :, ::8, ::8]
        signal = torch.cat((first, later), dim=2)
        return signal.expand(batch, 16, 2, 2, 2).contiguous()


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class _FakeConditioner:
    def __init__(self) -> None:
        self.text_encoder = _FakeTextEncoder()

    def encode(self, request, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        context = torch.zeros(
            request.batch_size,
            512,
            4096,
            device=device,
            dtype=dtype,
        )
        context[:, 0, 0] = 2 if request.prompts[0] == "" else 1
        return Conditioning(positive={"context": context})


def test_wan_feature_cache_round_trip_binds_official_contract(tmp_path: Path) -> None:
    dataset, audit = _dataset(tmp_path)
    store = VideoCacheStore(tmp_path / "cache")
    result = prepare_wan_training_cache_from_audits(
        dataset=dataset,
        store=store,
        feature_encoder=WanFeatureEncoder(_FakeCodec(), _FakeConditioner()),
        safety_audits=(audit,),
        model_recipe="wan2.1-t2v-1.3b",
        codec={"repo_id": "vae", "revision": "main"},
        conditioner={"repo_id": "umt5", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
    )

    cached = VideoCachedDataset(
        tmp_path / "cache",
        expected_sample_ids=dataset.sample_ids,
    )
    batch = collate_video_cached_samples((cached[0],))
    entry = result.entries[0]

    assert result.index == cached.index
    assert entry.provenance.model_recipe == "wan2.1-t2v-1.3b"
    assert entry.provenance.latent_normalization == wan_latent_normalization()
    assert tuple(batch.conditions["clean_latents"].shape) == (1, 16, 2, 2, 2)
    assert tuple(batch.conditions["context"].shape) == (1, 512, 4096)
    assert tuple(batch.conditions["latent_loss_mask"].shape) == (1, 1, 2, 2, 2)
    assert bool(batch.conditions["valid_latent_mask"].all())
    assert entry.tensors["condition.context"].layout == "sequence-features"
    unconditional = SharedConditioningStore(tmp_path / "cache").read("unconditional")
    assert result.unconditional_conditioning == unconditional.artifact
    assert unconditional.artifact.identity.model_recipe == "wan2.1-t2v-1.3b"
    assert unconditional.artifact.identity.conditioner == {"repo_id": "umt5", "revision": "main"}
    assert unconditional.artifact.identity.tokenizer == {"repo_id": "tokenizer", "revision": "main"}
    assert unconditional.tensors["context"][0, 0].item() == 2


def test_wan_codec_normalization_drift_is_rejected() -> None:
    codec = _FakeCodec()
    codec.vae.std[0] += 0.01

    with pytest.raises(ValueError, match="channel std"):
        WanVideoFeatureEncoder(codec)


def test_wan_cache_contract_records_denoiser_geometry() -> None:
    baseline = wan_cache_contract("wan2.1-t2v-1.3b")

    assert baseline != wan_cache_contract(
        "wan2.1-t2v-1.3b",
        latent_patch_size=(1, 1, 2),
    )
    assert baseline != wan_cache_contract(
        "wan2.1-t2v-1.3b",
        context_features=2048,
    )
