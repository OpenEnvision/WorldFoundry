from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.training.data import SharedConditioningStore  # noqa: E402


def _write(store: SharedConditioningStore):
    return store.write(
        branch="unconditional",
        prompt="",
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repo_id": "encoder", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
        tensors={"context": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)},
        layouts={"context": "sequence-features"},
    )


def test_shared_conditioning_round_trip_is_idempotent(tmp_path) -> None:
    store = SharedConditioningStore(tmp_path)
    artifact = _write(store)
    same = _write(store)
    loaded = store.read("unconditional")

    assert same == artifact
    assert loaded.artifact == artifact
    assert artifact.identity.prompt == ""
    assert artifact.object_path == "shared-objects/unconditional.safetensors"
    torch.testing.assert_close(
        loaded.tensors["context"],
        torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
    )


def test_shared_conditioning_validates_object_size_and_manifest_branch(tmp_path) -> None:
    object_store = SharedConditioningStore(tmp_path / "object")
    artifact = _write(object_store)
    object_path = object_store.root / artifact.object_path
    object_path.write_bytes(object_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size mismatch"):
        object_store.read("unconditional")

    manifest_store = SharedConditioningStore(tmp_path / "manifest")
    _write(manifest_store)
    manifest = manifest_store.root / "unconditional-conditioning.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"branch":"unconditional"', '"branch":"changed"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="object_path"):
        manifest_store.read("unconditional")
