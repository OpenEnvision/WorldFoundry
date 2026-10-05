from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.orchestration.cache import (
    CacheKey,
    GenerationResultCache,
    canonical_json_bytes,
    canonical_json_dumps,
    file_sha256,
    generation_request_cacheable,
    json_sha256,
    make_cache_key,
    run_generation_with_cache,
    sha256_hex,
)
from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evaluation.api.artifacts import local_path_for_uri


@dataclass(frozen=True)
class _Payload:
    prompt: str
    controls: dict[str, object]


def test_canonical_json_is_stable_for_key_order_and_compact_utf8() -> None:
    left = {
        "outputs": ("video.mp4", 24),
        "inputs": {"prompt": "向左转", "seed": 7},
        "model": {"id": "fake", "params": {"steps": 4, "guidance": 1.5}},
    }
    right = {
        "model": {"params": {"guidance": 1.5, "steps": 4}, "id": "fake"},
        "inputs": {"seed": 7, "prompt": "向左转"},
        "outputs": ["video.mp4", 24],
    }

    expected = (
        '{"inputs":{"prompt":"向左转","seed":7},'
        '"model":{"id":"fake","params":{"guidance":1.5,"steps":4}},'
        '"outputs":["video.mp4",24]}'
    )
    assert canonical_json_dumps(left) == expected
    assert canonical_json_dumps(right) == expected
    assert json_sha256(left) == json_sha256(right)
    assert json_sha256(left) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_canonical_json_accepts_stdlib_dataclasses() -> None:
    payload = _Payload(prompt="turn right", controls={"seed": 42, "camera": ["pan", "slow"]})

    assert canonical_json_dumps(payload) == (
        '{"controls":{"camera":["pan","slow"],"seed":42},"prompt":"turn right"}'
    )


def test_canonical_json_rejects_unstable_or_non_json_values() -> None:
    with pytest.raises(TypeError, match="object keys must be strings"):
        canonical_json_dumps({1: "integer keys would collide with JSON string keys"})

    with pytest.raises(ValueError, match="NaN or Infinity"):
        canonical_json_dumps({"score": float("nan")})

    with pytest.raises(TypeError, match="Unsupported JSON cache payload type"):
        canonical_json_dumps({"labels": {"unordered", "set"}})


def test_sha256_helpers_hash_bytes_text_json_and_files(tmp_path: Path) -> None:
    data = b"worldfoundry-cache\n" * 3
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    assert sha256_hex(data) == expected
    assert sha256_hex(data.decode("utf-8")) == expected
    assert file_sha256(artifact, chunk_size=5) == expected
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_cache_key_hashes_payload_and_version_context_stably(tmp_path: Path) -> None:
    metric_impl = tmp_path / "metric.py"
    metric_impl.write_text("VERSION = '1'\n", encoding="utf-8")
    version_context = {
        "metric": "judge_score",
        "implementation_sha256": file_sha256(metric_impl),
    }

    key = make_cache_key(
        stage="judge",
        sample_id="sample-001",
        payload={
            "judge": {"temperature": 0, "model": "fake-judge"},
            "result": {"artifact": "memory://sample-001.mp4"},
        },
        version_context=version_context,
    )
    reordered_key = CacheKey.from_payload(
        stage="judge",
        sample_id="sample-001",
        payload={
            "result": {"artifact": "memory://sample-001.mp4"},
            "judge": {"model": "fake-judge", "temperature": 0},
        },
        version_context={
            "implementation_sha256": file_sha256(metric_impl),
            "metric": "judge_score",
        },
    )

    assert key == reordered_key
    assert key.to_dict() == {
        "stage": "judge",
        "sample_id": "sample-001",
        "payload_hash": json_sha256(
            {
                "judge": {"model": "fake-judge", "temperature": 0},
                "result": {"artifact": "memory://sample-001.mp4"},
            }
        ),
        "version_context_hash": json_sha256(version_context),
    }
    assert key.key_hash == json_sha256(key.to_dict())


def test_cache_key_validates_required_fields() -> None:
    digest = json_sha256({})

    with pytest.raises(ValueError, match="stage"):
        CacheKey(stage="", sample_id="sample", payload_hash=digest, version_context_hash=digest)

    with pytest.raises(ValueError, match="payload_hash"):
        CacheKey(stage="request", sample_id="sample", payload_hash="not-a-digest", version_context_hash=digest)


def test_generation_cache_enables_wal_and_bounded_busy_wait(tmp_path: Path) -> None:
    cache = GenerationResultCache(tmp_path / "cache", namespace="unit")

    connection = cache._connect()
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
    finally:
        connection.close()

    assert journal_mode == "wal"
    assert busy_timeout == 30_000
    assert synchronous == 1  # SQLITE_SYNC_NORMAL


def test_generation_cache_hit_survives_locked_counter_update(tmp_path: Path, monkeypatch) -> None:
    cache = GenerationResultCache(tmp_path / "cache", namespace="unit")
    key = make_cache_key(
        stage="generation",
        sample_id="sample-a",
        payload={"prompt": "turn left"},
        version_context={"model": "cache-model"},
    )
    cache.put(key, GenerationResult(sample_id="sample-a", model_id="cache-model"))
    original_connect = cache._connect

    class _LockedHitCounterConnection:
        def __init__(self) -> None:
            self._connection = original_connect()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            try:
                return self._connection.__exit__(exc_type, exc, traceback)
            finally:
                self._connection.close()

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def executemany(self, sql, parameters):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "_connect", _LockedHitCounterConnection)

    record = cache.get(key)

    assert record is not None
    assert record.result.sample_id == "sample-a"
    with sqlite3.connect(cache.cache_path) as connection:
        assert connection.execute("SELECT hits FROM generation_results").fetchone()[0] == 0


def test_generation_result_cache_reuses_successful_deterministic_outputs(tmp_path: Path) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"
    cache_dir = tmp_path / "cache"
    calls: list[list[str]] = []
    requests = [
        GenerationRequest(
            sample_id="sample-a",
            task_name="cacheable_generation",
            generation_kwargs={"temperature": 0, "do_sample": False},
            output_schema={"generated_video": {"kind": "video"}},
        )
    ]

    def generate(rows):
        rows = list(rows)
        calls.append([row.sample_id for row in rows])
        artifact = first_output_dir / "sample-a.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"video")
        return [
            GenerationResult(
                sample_id=row.sample_id,
                request_id=row.request_id,
                model_id="cache-model",
                artifacts={"generated_video": ArtifactRef(uri=artifact.name, kind="video")},
            )
            for row in rows
        ]

    first_results, first_stats = run_generation_with_cache(
        requests,
        generate,
        cache_dir=cache_dir,
        cache_mode="read-write",
        namespace="unit",
        version_context={"model": "cache-model", "runner": "unit"},
        artifact_base_dir=first_output_dir,
        run_id="first-run",
    )
    assert first_stats.writes == 1
    assert first_stats.hits == 0
    assert calls == [["sample-a"]]
    assert first_results[0].metadata.get("cache") is None

    second_results, second_stats = run_generation_with_cache(
        requests,
        lambda rows: pytest.fail(f"unexpected generation call: {list(rows)}"),
        cache_dir=cache_dir,
        cache_mode="read-write",
        namespace="unit",
        version_context={"runner": "unit", "model": "cache-model"},
        artifact_base_dir=second_output_dir,
        run_id="second-run",
    )

    assert second_stats.hits == 1
    assert second_stats.writes == 0
    assert second_results[0].metadata["cache"]["hit"] is True
    cached_artifact = local_path_for_uri(second_results[0].artifacts["generated_video"].uri)
    assert cached_artifact == (first_output_dir / "sample-a.mp4").resolve()
    assert cached_artifact.is_file()


def test_generation_result_cache_skips_non_deterministic_requests(tmp_path: Path) -> None:
    request = GenerationRequest(
        sample_id="sample-a",
        task_name="sampled_generation",
        generation_kwargs={"temperature": 0.7, "do_sample": True},
    )
    cacheable, reason = generation_request_cacheable(request)
    assert cacheable is False
    assert "temperature" in str(reason)

    calls: list[str] = []

    def generate(rows):
        calls.extend(row.sample_id for row in rows)
        return [GenerationResult(sample_id=row.sample_id, model_id="sampled-model") for row in rows]

    _, first_stats = run_generation_with_cache(
        [request],
        generate,
        cache_dir=tmp_path / "cache",
        cache_mode="read-write",
        namespace="unit",
        version_context={"model": "sampled-model"},
    )
    _, second_stats = run_generation_with_cache(
        [request],
        generate,
        cache_dir=tmp_path / "cache",
        cache_mode="read-write",
        namespace="unit",
        version_context={"model": "sampled-model"},
    )

    assert calls == ["sample-a", "sample-a"]
    assert first_stats.skipped == 1
    assert second_stats.skipped == 1
    assert second_stats.hits == 0
