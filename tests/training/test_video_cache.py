from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.training.data import (  # noqa: E402
    LatentTokenBatchSampler,
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
    VideoLatentGeometry,
    collate_video_cached_samples,
)


def _provenance(**overrides: object) -> VideoCacheProvenance:
    values: dict[str, object] = {
        "media_uri": "video.mp4",
        "prompt": "a video prompt",
        "model_recipe": "wan2.1-t2v-1.3b",
        "codec": {"repo_id": "wan", "revision": "main"},
        "conditioner": {"repo_id": "umt5", "revision": "main"},
        "tokenizer": {"repo_id": "tokenizer", "revision": "main"},
        "conditioning_inputs": {"max_length": 12},
        "safety_audit": {"safe": True, "model_revision": "main"},
        "frame_sampling": {"mode": "uniform", "frames": 17},
        "spatial_transform": {"mode": "center-crop", "height": 288, "width": 512},
        "latent_normalization": {"mean": 0.0, "std": 1.0},
        "task": "t2v",
        "conditioning_layout": "text",
        "aspect_bin": "16:9",
        "source_num_frames": 65,
        "source_height": 576,
        "source_width": 1024,
        "source_fps": 24.0,
        "target_num_frames": 17,
        "target_height": 288,
        "target_width": 512,
        "target_fps": 24.0,
        "latent_geometry": VideoLatentGeometry(8, 8, 4, "first-frame"),
    }
    values.update(overrides)
    return VideoCacheProvenance(**values)


def _write(store: VideoCacheStore, sample_id: str, provenance: VideoCacheProvenance, *, value: float = 0.0):
    return store.write_sample(
        sample_id=sample_id,
        provenance=provenance,
        clean_latents=torch.full((16, 5, 36, 64), value, dtype=torch.bfloat16),
        conditioning={
            "context": torch.full((12, 32), value, dtype=torch.bfloat16),
            "context_mask": torch.ones(12, dtype=torch.bool),
        },
        conditioning_layouts={
            "context": "sequence-features",
            "context_mask": "sequence",
        },
        latent_loss_mask=torch.ones(1, 5, 36, 64),
        valid_latent_mask=torch.ones(1, 5, 36, 64, dtype=torch.bool),
        sample_weight=torch.tensor(1.0),
    )


def test_video_cache_round_trip_collates_masks_conditions_and_token_metrics(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    provenance = _provenance()
    entries = [
        _write(
            store,
            f"sample-{index}",
            replace(provenance, media_uri=f"video-{index}.mp4"),
            value=float(index),
        )
        for index in range(2)
    ]
    index = store.write_index(entries=entries)

    dataset = VideoCachedDataset(tmp_path, expected_sample_ids=("sample-0", "sample-1"))
    batch = collate_video_cached_samples([dataset[0], dataset[1]])

    assert dataset.index == index
    assert dataset.bucket_keys == (provenance.bucket_key, provenance.bucket_key)
    assert tuple(batch.conditions["clean_latents"].shape) == (2, 16, 5, 36, 64)
    assert tuple(batch.conditions["latent_loss_mask"].shape) == (2, 1, 5, 36, 64)
    assert tuple(batch.conditions["valid_latent_mask"].shape) == (2, 1, 5, 36, 64)
    assert tuple(batch.conditions["context"].shape) == (2, 12, 32)
    assert batch.metadata["samples_per_microbatch"] == 2
    assert batch.metadata["latent_tokens_per_microbatch"] == 2 * 5 * 36 * 64


def test_video_cache_records_frame_and_normalization_configuration(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    base = _provenance()
    frame_changed = replace(base, frame_sampling={"mode": "head", "frames": 17})
    normalization_changed = replace(base, latent_normalization={"mean": 0.5, "std": 2.0})
    entries = (
        _write(store, "base", base),
        _write(store, "frame", frame_changed),
        _write(store, "normalization", normalization_changed),
    )
    assert entries[0].provenance != entries[1].provenance
    assert entries[0].provenance != entries[2].provenance


def test_video_cache_rejects_invalid_latent_and_temporal_masks(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    provenance = _provenance()
    with pytest.raises(ValueError, match="matching provenance"):
        store.write_sample(
            sample_id="wrong-shape",
            provenance=provenance,
            clean_latents=torch.zeros(16, 4, 36, 64),
        )
    with pytest.raises(ValueError, match="valid_latent_mask"):
        store.write_sample(
            sample_id="wrong-mask",
            provenance=provenance,
            clean_latents=torch.zeros(16, 5, 36, 64),
            valid_latent_mask=torch.ones(5, 36, 64, dtype=torch.bool),
        )


def test_video_cache_collator_rejects_bucket_and_contract_mixing(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    base = _provenance()
    other_contract = replace(base, codec={"repo_id": "other", "revision": "main"})
    first = store.audit_entry(_write(store, "first", base))
    second = store.audit_entry(_write(store, "second", other_contract))
    assert first is not None and second is not None
    with pytest.raises(ValueError, match="incompatible model or preprocessing"):
        collate_video_cached_samples([first, second])


def test_token_sampler_separates_same_shape_cache_contracts(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    base = _provenance()
    entries = (
        _write(store, "24-fps", base),
        _write(
            store,
            "30-fps",
            replace(base, media_uri="other.mp4", source_fps=30.0, target_fps=30.0),
        ),
    )
    store.write_index(entries=entries)
    dataset = VideoCachedDataset(tmp_path)
    sampler = LatentTokenBatchSampler(
        dataset,
        max_latent_tokens=2 * base.bucket_key.token_count,
        shuffle=False,
        tail_policy="pad",
    )

    batches = list(sampler)

    assert batches == [[0], [1]]
    assert dataset.batch_contracts[0] != dataset.batch_contracts[1]
    for indices in batches:
        collate_video_cached_samples([dataset[index] for index in indices])


def test_video_cache_detects_truncated_object(tmp_path) -> None:
    store = VideoCacheStore(tmp_path)
    entry = _write(store, "sample", _provenance())
    path = tmp_path / entry.object_path
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="size mismatch"):
        store.audit_entry(entry)
