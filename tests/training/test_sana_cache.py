from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.data import (  # noqa: E402
    SanaCachedDataset,
    SanaCacheEntry,
    SanaCacheProvenance,
    SanaCacheStore,
    collate_sana_cached_samples,
)


def _provenance(*, height: int = 64, width: int = 64) -> SanaCacheProvenance:
    return SanaCacheProvenance(
        media_uri="media.png",
        prompt="a blue cup",
        model_recipe="sana-600m-1024",
        codec={"repo_id": "dcae", "revision": "main"},
        conditioner={"repo_id": "gemma", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
        safety_audit={"safe": True, "model_revision": "main"},
        pixel_transform={"range": "[-1,1]"},
        prompt_enhancement={"enabled": True, "prefix": "describe"},
        image_height=height,
        image_width=width,
        spatial_compression=32,
        latent_scaling_factor=0.41407,
        max_text_length=3,
    )


def _write_sample(
    store: SanaCacheStore,
    *,
    sample_id: str = "sample",
    provenance: SanaCacheProvenance | None = None,
):
    resolved = provenance or _provenance()
    return store.write_sample(
        sample_id=sample_id,
        provenance=resolved,
        clean_latents=torch.arange(32 * 2 * 2, dtype=torch.float32).reshape(32, 2, 2),
        context=torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4),
        context_mask=torch.tensor([1, 1, 0]),
        latent_loss_mask=torch.ones(1, 2, 2),
        sample_weight=torch.tensor(0.75),
    )


def test_sana_cache_round_trip_uses_explicit_identity(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    first = _write_sample(store, sample_id="first")
    second = _write_sample(store, sample_id="second")
    index = store.write_index(entries=[first, second])

    assert first.object_path == "objects/first.safetensors"
    assert second.object_path == "objects/second.safetensors"
    assert len(tuple((tmp_path / "objects").rglob("*.safetensors"))) == 2
    raw_index = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert "a blue cup" in raw_index
    assert index.entries == (first, second)

    dataset = SanaCachedDataset(tmp_path, expected_sample_ids=("first", "second"))
    batch = collate_sana_cached_samples([dataset[0], dataset[1]])

    assert dataset.sample_ids == ("first", "second")
    assert batch.pixel_values is None
    assert batch.conditions["clean_latents"].shape == (2, 32, 2, 2)
    assert batch.conditions["context"].shape == (2, 1, 3, 4)
    assert batch.conditions["context_mask"].shape == (2, 3)
    assert batch.conditions["latent_loss_mask"].shape == (2, 1, 2, 2)
    torch.testing.assert_close(batch.sample_weights, torch.tensor([0.75, 0.75]))
    assert batch.prompts == ("a blue cup", "a blue cup")


def test_sana_cache_rejects_truncated_object(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    entry = _write_sample(store)
    store.write_index(entries=[entry])
    object_path = tmp_path / entry.object_path
    object_path.write_bytes(object_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="size mismatch"):
        SanaCachedDataset(tmp_path, audit_on_open=False)[0]


def test_sana_cache_rejects_object_path_mismatch(tmp_path: Path) -> None:
    entry = _write_sample(SanaCacheStore(tmp_path))
    payload = entry.to_dict()
    payload["object_path"] = "objects/other.safetensors"

    with pytest.raises(ValueError, match="object_path"):
        SanaCacheEntry.from_mapping(payload)


def test_sana_cache_rejects_index_identity_changes_and_wrong_sample_ids(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    entry = _write_sample(store)
    store.write_index(entries=[entry])
    index_path = tmp_path / "index.json"
    import json

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["entries"][0]["provenance"]["model_recipe"] = "different"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        SanaCachedDataset(tmp_path)

    store.write_index(entries=[entry])
    with pytest.raises(ValueError, match="sample IDs"):
        SanaCachedDataset(tmp_path, expected_sample_ids=("wrong",))


def test_sana_cache_collator_rejects_incompatible_preprocessing_buckets(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    small = _write_sample(store, sample_id="small")
    large_provenance = _provenance(height=96, width=64)
    large = store.write_sample(
        sample_id="large",
        provenance=large_provenance,
        clean_latents=torch.zeros(32, 3, 2),
        context=torch.zeros(1, 3, 4),
        context_mask=torch.ones(3, dtype=torch.long),
        latent_loss_mask=torch.ones(1, 3, 2),
        sample_weight=torch.tensor(1.0),
    )
    store.write_index(entries=[small, large])
    dataset = SanaCachedDataset(tmp_path)

    with pytest.raises(ValueError, match="tensor descriptors"):
        collate_sana_cached_samples([dataset[0], dataset[1]])


def test_sana_cache_validates_tensor_contract_before_write(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)

    with pytest.raises(ValueError, match="context must have shape"):
        store.write_sample(
            sample_id="bad",
            provenance=_provenance(),
            clean_latents=torch.zeros(32, 2, 2),
            context=torch.zeros(1, 2, 4),
            context_mask=torch.ones(2, dtype=torch.long),
        )
