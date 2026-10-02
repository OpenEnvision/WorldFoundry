"""CPU-only regression tests for core-foundation cache atomicity fixes.

Covers the review findings CF-21 (checkpoint local cache), CF-30
(``io.cache._populate``), and CF-35 (``serialization._atomic_write_text``):

* an interrupted write must never publish a partial cache file;
* stale ``.tmp`` residue must not affect cache-hit checks or reads;
* a corrupt (pre-fix) cache file must be discarded and rebuilt from source;
* concurrent writers must not interleave bytes in the published file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest
import torch

from worldfoundry.core.checkpoint import load as checkpoint_load
from worldfoundry.core.io import cache as io_cache
from worldfoundry.core.io.serialization import write_json


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(4, dtype=torch.float32), "bias": torch.ones(2)}


def _assert_no_tmp_residue(directory: Path) -> None:
    leftovers = [p for p in directory.rglob("*.tmp")]
    assert leftovers == [], f"temp files leaked: {leftovers}"


class TestSaveToLocalCacheAtomicity:
    def test_publish_and_reload_pt(self, tmp_path: Path) -> None:
        target = tmp_path / "cache" / "model.pt"
        checkpoint_load._save_to_local_cache(_tiny_state_dict(), str(target), ".pt")
        assert target.exists()
        _assert_no_tmp_residue(tmp_path)
        loaded = torch.load(target, map_location="cpu", weights_only=True)
        assert torch.equal(loaded["weight"], _tiny_state_dict()["weight"])

    def test_publish_and_reload_safetensors(self, tmp_path: Path) -> None:
        target = tmp_path / "cache" / "model.safetensors"
        checkpoint_load._save_to_local_cache(_tiny_state_dict(), str(target), ".safetensors")
        assert target.exists()
        _assert_no_tmp_residue(tmp_path)
        loaded = checkpoint_load._load_checkpoint_from_local(str(target), ".safetensors")
        assert torch.equal(loaded["bias"], _tiny_state_dict()["bias"])

    def test_interrupted_write_publishes_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "cache" / "model.pt"

        def exploding_save(obj, handle, *args, **kwargs):
            handle.write(b"partial bytes")
            raise RuntimeError("simulated crash mid-serialization")

        monkeypatch.setattr(checkpoint_load.torch, "save", exploding_save)
        with pytest.raises(RuntimeError, match="simulated crash"):
            checkpoint_load._save_to_local_cache(_tiny_state_dict(), str(target), ".pt")
        # The final cache slot must stay empty so exists()-based hit checks miss.
        assert not target.exists()

    def test_stale_tmp_residue_does_not_affect_reads(self, tmp_path: Path) -> None:
        target = tmp_path / "cache" / "model.pt"
        target.parent.mkdir(parents=True)
        # Simulate a kill -9 mid-write from a previous process: only a temp
        # sibling remains.
        stale = target.with_name(f".{target.name}.deadbeef.tmp")
        stale.write_bytes(b"partial bytes from a crashed writer")
        assert not target.exists()  # cache-hit checks must miss

        checkpoint_load._save_to_local_cache(_tiny_state_dict(), str(target), ".pt")
        loaded = torch.load(target, map_location="cpu", weights_only=True)
        assert set(loaded) == {"weight", "bias"}

    def test_should_write_shared_cache_without_process_group(self) -> None:
        assert checkpoint_load._should_write_shared_cache() is True


class TestShardedIndexCacheSelfHeal:
    @pytest.fixture()
    def sharded_checkpoint(self, tmp_path: Path) -> tuple[str, Path, dict[str, torch.Tensor]]:
        from safetensors.torch import save_file

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        tensors = {"a.weight": torch.randn(3, 3), "b.weight": torch.randn(2)}
        save_file({"a.weight": tensors["a.weight"]}, source_dir / "shard-0.safetensors")
        save_file({"b.weight": tensors["b.weight"]}, source_dir / "shard-1.safetensors")
        index_path = source_dir / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"weight_map": {"a.weight": "shard-0.safetensors", "b.weight": "shard-1.safetensors"}})
        )
        cache_dir = tmp_path / "cache"
        return str(index_path), cache_dir, tensors

    def test_merge_then_cache_hit(self, sharded_checkpoint) -> None:
        index_path, cache_dir, tensors = sharded_checkpoint
        merged = checkpoint_load.load_single_checkpoint(index_path, local_cache_dir=str(cache_dir))
        assert torch.equal(merged["a.weight"], tensors["a.weight"])
        cache_file = checkpoint_load._sharded_safetensors_merge_cache_path(index_path, str(cache_dir))
        assert os.path.exists(cache_file)
        # Second load must come from the cache and produce identical tensors.
        again = checkpoint_load.load_single_checkpoint(index_path, local_cache_dir=str(cache_dir))
        assert torch.equal(again["b.weight"], tensors["b.weight"])

    def test_corrupt_cache_is_discarded_and_rebuilt(self, sharded_checkpoint) -> None:
        index_path, cache_dir, tensors = sharded_checkpoint
        checkpoint_load.load_single_checkpoint(index_path, local_cache_dir=str(cache_dir))
        cache_file = Path(checkpoint_load._sharded_safetensors_merge_cache_path(index_path, str(cache_dir)))
        # Simulate a truncated cache produced before the atomic-write fix.
        cache_file.write_bytes(b"garbage that is not a safetensors payload")

        merged = checkpoint_load.load_single_checkpoint(index_path, local_cache_dir=str(cache_dir))
        assert torch.equal(merged["a.weight"], tensors["a.weight"])
        # The cache slot must have been rebuilt into a loadable file.
        healed = checkpoint_load._load_checkpoint_from_local(str(cache_file), ".safetensors")
        assert torch.equal(healed["b.weight"], tensors["b.weight"])


class TestIoCachePopulate:
    def test_populate_copies_and_hits(self, tmp_path: Path) -> None:
        source = tmp_path / "asset.bin"
        source.write_bytes(b"payload-bytes")
        target = tmp_path / "cache" / "asset.bin"
        resolved = io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target))
        assert Path(resolved) == target
        assert target.read_bytes() == b"payload-bytes"
        _assert_no_tmp_residue(tmp_path / "cache")

    def test_interrupted_copy_leaves_cache_slot_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "asset.bin"
        source.write_bytes(b"payload-bytes")
        target = tmp_path / "cache" / "asset.bin"

        def exploding_copy(src, dst, **kwargs):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(b"payload")  # truncated: only half the bytes
            raise RuntimeError("simulated network drop")

        monkeypatch.setattr(io_cache, "copy_uri", exploding_copy)
        with pytest.raises(RuntimeError, match="simulated network drop"):
            io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target))
        assert not target.exists()
        _assert_no_tmp_residue(tmp_path / "cache")

        # Next attempt (network recovered) retries cleanly.
        monkeypatch.undo()
        io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target))
        assert target.read_bytes() == b"payload-bytes"

    def test_concurrent_populate_downloads_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # /tmp is guaranteed local so flock semantics hold regardless of the
        # (possibly network-mounted) pytest tmp_path location.
        workdir = Path(tempfile.mkdtemp(dir="/tmp"))
        try:
            source = workdir / "asset.bin"
            source.write_bytes(b"payload-bytes")
            target = workdir / "cache" / "asset.bin"

            calls: list[int] = []
            calls_lock = threading.Lock()
            real_copy = io_cache.copy_uri

            def slow_counting_copy(src, dst, **kwargs):
                with calls_lock:
                    calls.append(1)
                import time

                time.sleep(0.2)
                return real_copy(src, dst, **kwargs)

            monkeypatch.setattr(io_cache, "copy_uri", slow_counting_copy)

            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    io_cache.download_from_cache_or_uri(str(source), cache_fp=str(target))
                except BaseException as exc:  # pragma: no cover - failure reporting
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert errors == []
            assert target.read_bytes() == b"payload-bytes"
            assert len(calls) == 1, f"expected a single populate under the lock, got {len(calls)}"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestAtomicWriteText:
    def test_concurrent_writers_never_interleave(self, tmp_path: Path) -> None:
        target = tmp_path / "scorecard.json"
        payloads = [{"writer": index, "data": "x" * 2048} for index in range(4)]
        iterations = 25

        errors: list[BaseException] = []

        def worker(payload: dict) -> None:
            try:
                for _ in range(iterations):
                    write_json(target, payload)
            except BaseException as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(payload,)) for payload in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        # The published file must always be exactly one writer's payload.
        content = json.loads(target.read_text(encoding="utf-8"))
        assert content in [json.loads(json.dumps(p)) for p in payloads]
        _assert_no_tmp_residue(tmp_path)
