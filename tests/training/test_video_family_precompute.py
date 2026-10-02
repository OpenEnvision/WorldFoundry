from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

av = pytest.importorskip("av")
numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.base_models.diffusion_model.contracts import Conditioning  # noqa: E402
from worldfoundry.training.data.cosmos.encoding import (  # noqa: E402
    CosmosTextFeatureEncoder,
    CosmosVideoFeatureEncoder,
)
from worldfoundry.training.data.cosmos.training_cache import (  # noqa: E402
    build_cosmos_video_decoding_dataset,
    prepare_cosmos_training_cache_from_audits,
)
from worldfoundry.training.data.dataset import TrainingManifestDataset  # noqa: E402
from worldfoundry.training.data.ltx.encoding import (  # noqa: E402
    LTXTextFeatureEncoder,
    LTXVideoFeatureEncoder,
)
from worldfoundry.training.data.ltx.training_cache import (  # noqa: E402
    build_ltx_video_decoding_dataset,
    prepare_ltx_training_cache_from_audits,
)
from worldfoundry.training.data.manifest import MediaReference, TrainingSample  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheStore,
    collate_video_cached_samples,
)
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
    PromptSafetyAudit,
)


def _write_video(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int = 5,
) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = numpy.full((height, width, 3), index * 10, dtype=numpy.uint8)
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


def _manifest(
    root: Path,
    *,
    frames: int,
    height: int,
    width: int,
    prompts: tuple[str, ...],
) -> tuple[TrainingManifestDataset, tuple[PromptSafetyAudit, ...]]:
    rows = []
    audits = tuple(_safe_audit(prompt) for prompt in prompts)
    for index, (prompt, audit) in enumerate(zip(prompts, audits, strict=True)):
        video = root / f"sample-{index}.mkv"
        _write_video(video, frames=frames, height=height, width=width)
        rows.append(
            TrainingSample(
                sample_id=f"sample-{index}",
                task="t2v",
                prompt=prompt,
                media=MediaReference(
                    uri=video.name,
                    mime_type="video/x-matroska",
                    size_bytes=video.stat().st_size,
                ),
                width=width,
                height=height,
                num_frames=frames,
                fps=5.0,
                conditions={},
                split="train",
                safety={"prompt_safe": True, "model_revision": audit.model_revision},
            )
        )
    path = root / "manifest.jsonl"
    path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows), encoding="utf-8")
    return TrainingManifestDataset.from_file(path, verify_files=True), audits


def _recipe(
    model_recipe: str,
    *,
    manifest: Path,
    cache: Path,
    frames: int,
    height: int,
    width: int,
) -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": f"cache-{model_recipe}", "output_dir": "unused"},
            "model": {"recipe": model_recipe},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": str(manifest),
                "cache": str(cache),
                "max_latent_tokens_per_microbatch": 1024,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {
                    "video_buckets": [{"num_frames": frames, "height": height, "width": width}],
                    "bucket_policy": {"allow_spatial_upscale": False},
                    "decode": {"frame_sampling": "head"},
                },
            },
            "objective": {
                "type": "flow-matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "uniform",
            },
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-4},
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
        }
    )


class _Anchor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class _FakeLTXConditioner:
    def __init__(self) -> None:
        self.model = _Anchor()

    def encode(self, request, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        context = torch.full((1, 4, 6), len(request.prompts[0]), device=device, dtype=dtype)
        return Conditioning(
            positive={
                "video_context": context,
                "audio_context": torch.zeros(1, 2, 3, device=device, dtype=dtype),
                "context_mask": torch.tensor([[1, 1, 1, 0]], device=device),
            }
        )


class _FakeLTXCodec:
    def __init__(self) -> None:
        self.encoder = _Anchor()

    @staticmethod
    def _latents(pixels: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full(
            (
                int(pixels.shape[0]),
                128,
                1 + (int(pixels.shape[2]) - 1) // 8,
                int(pixels.shape[3]) // 32,
                int(pixels.shape[4]) // 32,
            ),
            value,
            dtype=pixels.dtype,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._latents(pixels, 1.0)

    def encode_posterior(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._latents(pixels, 2.0)


@pytest.mark.parametrize(
    ("model_recipe", "expected_value"),
    [
        ("ltx-video-i2v", 2.0),
        ("ltx-2-i2v", 1.0),
        ("ltx-2.3-i2v", 1.0),
    ],
)
def test_ltx_fake_producer_writes_adapter_ready_cache(
    tmp_path: Path,
    model_recipe: str,
    expected_value: float,
) -> None:
    manifest, audits = _manifest(
        tmp_path,
        frames=9,
        height=32,
        width=32,
        prompts=("a short prompt",),
    )
    recipe = _recipe(
        model_recipe,
        manifest=manifest.manifest_path,
        cache=tmp_path / "cache",
        frames=9,
        height=32,
        width=32,
    )
    dataset = build_ltx_video_decoding_dataset(recipe, manifest)
    codec = _FakeLTXCodec()
    result = prepare_ltx_training_cache_from_audits(
        dataset=dataset,
        store=VideoCacheStore(tmp_path / "cache"),
        text_encoder=LTXTextFeatureEncoder(
            _FakeLTXConditioner(),
            device="cpu",
            dtype=torch.float32,
            include_audio=False,
        ),
        video_encoder=LTXVideoFeatureEncoder(
            codec,
            sample_posterior=model_recipe == "ltx-video-i2v",
        ),
        safety_audits=audits,
        model_recipe=model_recipe,
        codec={"name": "fake-ltx-vae"},
        conditioner={"name": "fake-ltx-text"},
        tokenizer={"name": "fake-ltx-tokenizer"},
    )

    cached = VideoCachedDataset(tmp_path / "cache", expected_sample_ids=manifest.sample_ids)
    batch = collate_video_cached_samples((cached[0],))

    assert result.index == cached.index
    assert set(batch.conditions) == {
        "clean_latents",
        "video_context",
        "context_mask",
        "latent_loss_mask",
        "valid_latent_mask",
    }
    assert tuple(batch.conditions["clean_latents"].shape) == (1, 128, 2, 1, 1)
    assert tuple(batch.conditions["video_context"].shape) == (1, 4, 6)
    assert tuple(batch.conditions["context_mask"].shape) == (1, 4)
    assert float(batch.conditions["clean_latents"].mean()) == expected_value
    assert "condition.audio_context" not in result.entries[0].tensors
    assert result.entries[0].tensors["condition.context_mask"].layout == "sequence"


class _FakeTokenizer:
    pad_token_id = 0


class _FakeCosmosConditioner:
    def __init__(self, model_recipe: str) -> None:
        self.model = _Anchor()
        self.tokenizer = _FakeTokenizer()
        self.model_recipe = model_recipe
        self.use_system_prompt = False
        self.requests = []

    def encode(self, request, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        self.requests.append(request)
        if self.model_recipe.startswith("cosmos3-"):
            length = 2 + len(request.prompts[0].split())
            return Conditioning(positive={"input_ids": torch.arange(1, length + 1, device=device)})
        context = torch.ones(1, 3, 5, device=device, dtype=dtype)
        negative = {"context": -context} if self.model_recipe.startswith("cosmos-predict2.5-") else {}
        return Conditioning(positive={"context": context}, negative=negative)


class _FakeCosmosVAE(torch.nn.Module):
    def __init__(self, *, channels: int, spatial_compression: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("mean", torch.linspace(-0.5, 0.5, channels))
        self.register_buffer("std", torch.linspace(0.5, 1.5, channels))
        self.channels = channels
        self.spatial_compression = spatial_compression

    def encode(self, videos, device, **_) -> torch.Tensor:
        video = videos[0]
        return torch.full(
            (
                1,
                self.channels,
                1 + (int(video.shape[1]) - 1) // 4,
                int(video.shape[2]) // self.spatial_compression,
                int(video.shape[3]) // self.spatial_compression,
            ),
            3.0,
            dtype=video.dtype,
            device=device,
        )


class _FakeCosmosComponent:
    def __init__(self, *, channels: int, spatial_compression: int) -> None:
        self.vae = _FakeCosmosVAE(
            channels=channels,
            spatial_compression=spatial_compression,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode([pixels[0]], pixels.device)


@pytest.mark.parametrize(
    ("model_recipe", "channels", "spatial", "expected_keys"),
    [
        (
            "cosmos-predict2-2b-video2world",
            16,
            8,
            {"context"},
        ),
        (
            "cosmos-predict2-14b-video2world",
            16,
            8,
            {"context"},
        ),
        (
            "cosmos-predict2.5-2b",
            16,
            8,
            {"context", "negative_context"},
        ),
        (
            "cosmos-predict2.5-14b",
            16,
            8,
            {"context", "negative_context"},
        ),
        ("cosmos3-nano", 48, 16, {"input_ids", "empty_input_ids"}),
        ("cosmos3-super", 48, 16, {"input_ids", "empty_input_ids"}),
    ],
)
def test_cosmos_fake_producer_writes_adapter_ready_cache(
    tmp_path: Path,
    model_recipe: str,
    channels: int,
    spatial: int,
    expected_keys: set[str],
) -> None:
    prompts = ("short", "a longer prompt") if model_recipe.startswith("cosmos3-") else ("prompt",)
    manifest, audits = _manifest(
        tmp_path,
        frames=5,
        height=spatial,
        width=spatial,
        prompts=prompts,
    )
    recipe = _recipe(
        model_recipe,
        manifest=manifest.manifest_path,
        cache=tmp_path / "cache",
        frames=5,
        height=spatial,
        width=spatial,
    )
    dataset = build_cosmos_video_decoding_dataset(recipe, manifest)
    conditioner = _FakeCosmosConditioner(model_recipe)
    text_encoder = CosmosTextFeatureEncoder(
        conditioner,
        model_recipe=model_recipe,
        device="cpu",
        dtype=torch.float32,
    )
    component = _FakeCosmosComponent(channels=channels, spatial_compression=spatial)
    result = prepare_cosmos_training_cache_from_audits(
        dataset=dataset,
        store=VideoCacheStore(tmp_path / "cache"),
        text_encoder=text_encoder,
        video_encoder=CosmosVideoFeatureEncoder(
            component,
            cosmos3=model_recipe.startswith("cosmos3-"),
            latent_channels=channels,
            temporal_compression=4,
            spatial_compression=spatial,
        ),
        safety_audits=audits,
        model_recipe=model_recipe,
        codec={"name": "fake-cosmos-vae"},
        conditioner={"name": "fake-cosmos-text"},
        tokenizer={"name": "fake-cosmos-tokenizer"},
    )

    cached = VideoCachedDataset(tmp_path / "cache", expected_sample_ids=manifest.sample_ids)
    batches = (
        tuple(collate_video_cached_samples((cached[index],)) for index in range(len(cached)))
        if model_recipe.startswith("cosmos3-")
        else (collate_video_cached_samples(tuple(cached)),)
    )
    batch = batches[0]

    assert result.index == cached.index
    assert set(batch.conditions) == {
        "clean_latents",
        "latent_loss_mask",
        "valid_latent_mask",
        *expected_keys,
    }
    assert tuple(batch.conditions["clean_latents"].shape) == (
        1,
        channels,
        2,
        1,
        1,
    )
    if model_recipe.startswith("cosmos3-"):
        assert [tuple(item.conditions["input_ids"].shape) for item in batches] == [(1, 3), (1, 5)]
        assert [tuple(item.conditions["empty_input_ids"].shape) for item in batches] == [(1, 2), (1, 2)]
        assert all(item.conditions["empty_input_ids"].tolist() == [[1, 2]] for item in batches)
        empty_requests = [request for request in conditioner.requests if request.prompts == ("",)]
        assert len(empty_requests) == len(prompts)
        assert all(request.inputs["add_duration_template"] is False for request in empty_requests)
        assert all(request.inputs["add_resolution_template"] is False for request in empty_requests)
    elif model_recipe.startswith("cosmos-predict2.5-"):
        assert conditioner.requests[0].negative_prompts == ("",)
    assert not {
        "condition_latents",
        "condition_mask",
        "condition_indicator",
        "denoise_masks",
    } & set(batch.conditions)
    assert len(result.entries[0].provenance.latent_normalization["channel_mean"]) == channels


def test_cosmos3_cache_rejects_system_prompt_tokenization() -> None:
    conditioner = _FakeCosmosConditioner("cosmos3-nano")
    conditioner.use_system_prompt = True
    with pytest.raises(ValueError, match="use_system_prompt=False"):
        CosmosTextFeatureEncoder(
            conditioner,
            model_recipe="cosmos3-nano",
            device="cpu",
            dtype=torch.float32,
        )


def test_cosmos3_config_builds_native_decode_dataset(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    recipe = TrainingRecipe.from_file(root / "configs/training/cosmos3_nano_vision_sft.yaml")
    manifest, _ = _manifest(
        tmp_path,
        frames=49,
        height=256,
        width=256,
        prompts=("a prompt",),
    )

    dataset = build_cosmos_video_decoding_dataset(recipe, manifest)

    assert dataset.assignments[0].bucket_key.conditioning_layout == "cosmos3-token-sequence"
    assert recipe.model.options["use_system_prompt"] is False


def test_train_cache_has_explicit_ltx_cosmos_dispatch_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    from worldfoundry.cli.training_commands.handlers import cache as handler
    from worldfoundry.training.data.cosmos import training_cache as cosmos_cache
    from worldfoundry.training.data.ltx import training_cache as ltx_cache

    calls: list[tuple[str, Path]] = []

    def materialize(recipe, **kwargs):
        calls.append((recipe.model.recipe, kwargs["base_dir"]))
        return SimpleNamespace(
            index=SimpleNamespace(to_dict=lambda: {"entries": []}),
            entries=(),
            safety_audits=(),
        )

    monkeypatch.setattr(ltx_cache, "materialize_ltx_training_cache", materialize)
    monkeypatch.setattr(cosmos_cache, "materialize_cosmos_training_cache", materialize)

    def args(recipe: TrainingRecipe) -> Namespace:
        recipe_path = tmp_path / f"{recipe.model.recipe}.yaml"
        recipe_path.touch()
        monkeypatch.setattr(handler, "load_cache_recipe", lambda _: recipe)
        return Namespace(
            recipe=recipe_path,
            base_dir=tmp_path,
            manifest=None,
            cache=None,
            device="cpu",
            checkpoint_override=None,
            skip_media_file_verification=False,
            safety_batch_size=1,
            prompt_audits=None,
        )

    for model_recipe in ("ltx-2-i2v", "cosmos3-nano"):
        recipe = _recipe(
            model_recipe,
            manifest=tmp_path / "unused.jsonl",
            cache=tmp_path / f"cache-{model_recipe}",
            frames=9 if model_recipe.startswith("ltx-") else 5,
            height=32 if model_recipe.startswith("ltx-") else 16,
            width=32 if model_recipe.startswith("ltx-") else 16,
        )
        assert handler._handle_train_cache(args(recipe)) == 0

    assert calls == [
        ("ltx-2-i2v", tmp_path),
        ("cosmos3-nano", tmp_path),
    ]
    recipe = _recipe(
        "ltx-2-i2v",
        manifest=tmp_path / "unused.jsonl",
        cache=tmp_path / "cache-unknown",
        frames=9,
        height=32,
        width=32,
    )
    monkeypatch.setattr(handler, "training_family", lambda _: "unknown-family")
    with pytest.raises(ValueError, match="no native materializer"):
        handler._handle_train_cache(args(recipe))
