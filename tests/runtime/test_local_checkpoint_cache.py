from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from worldfoundry.runtime import local_checkpoint_cache
from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime


def test_stage_checkpoint_is_disabled_without_explicit_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source" / "model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", raising=False)

    assert stage_checkpoint_for_realtime(source, required_paths=("config.json",)) == source


def test_stage_checkpoint_reuses_immutable_local_copy(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source" / "model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"weights")
    cache = tmp_path / "local"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    first = stage_checkpoint_for_realtime(
        source,
        required_paths=("config.json", "weights.bin"),
    )
    second = stage_checkpoint_for_realtime(
        source,
        required_paths=("config.json", "weights.bin"),
    )

    assert first == second
    assert first != source
    assert (first / "weights.bin").read_bytes() == b"weights"


def test_stage_checkpoint_can_copy_only_runtime_components(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source" / "model"
    (source / "tokenizer").mkdir(parents=True)
    (source / "tokenizer" / "config.json").write_text("{}", encoding="utf-8")
    (source / "vae.bin").write_bytes(b"vae")
    (source / "unused-transformer.bin").write_bytes(b"unused")
    cache = tmp_path / "local"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    staged = stage_checkpoint_for_realtime(
        source,
        required_paths=("vae.bin", "tokenizer/config.json"),
        include_paths=("vae.bin", "tokenizer"),
    )

    assert (staged / "vae.bin").read_bytes() == b"vae"
    assert (staged / "tokenizer" / "config.json").is_file()
    assert not (staged / "unused-transformer.bin").exists()


@pytest.mark.parametrize(
    ("parameter", "unsafe_path"),
    (
        ("required_paths", "../outside.bin"),
        ("required_paths", "/outside.bin"),
        ("include_paths", "nested/../../outside.bin"),
        ("include_paths", "/outside.bin"),
    ),
)
def test_stage_checkpoint_rejects_unsafe_relative_paths(
    tmp_path: Path,
    monkeypatch,
    parameter: str,
    unsafe_path: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    with pytest.raises(ValueError, match=parameter):
        stage_checkpoint_for_realtime(source, **{parameter: (unsafe_path,)})


def test_stage_checkpoint_rejects_source_symlink_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv(
        "WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE",
        str(tmp_path / "cache"),
    )

    with pytest.raises(ValueError, match="resolves outside"):
        stage_checkpoint_for_realtime(source, include_paths=("escape",))


def test_stage_checkpoint_rejects_unfiltered_symlink_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (source / "escape.bin").symlink_to(outside)
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv(
        "WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE",
        str(tmp_path / "cache"),
    )

    with pytest.raises(RuntimeError, match="checkpoint source symlink"):
        stage_checkpoint_for_realtime(source)


def test_stage_checkpoint_rejects_cache_target_symlink_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    cache = (tmp_path / "cache").resolve()
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = local_checkpoint_cache._cache_target(source, cache)
    target.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    with pytest.raises(RuntimeError, match="local checkpoint target"):
        stage_checkpoint_for_realtime(source)


def test_concurrent_staging_uses_unique_temporary_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source" / "model"
    source.mkdir(parents=True)
    (source / "weights.bin").write_bytes(b"weights")
    cache = tmp_path / "cache"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))
    original_copy = local_checkpoint_cache._copy_tree_parallel
    both_copies_started = Barrier(2)
    temporary_paths: list[Path] = []

    def synchronized_copy(source_path: Path, target: Path, **kwargs) -> None:
        temporary_paths.append(target)
        both_copies_started.wait(timeout=5)
        original_copy(source_path, target, **kwargs)

    monkeypatch.setattr(
        local_checkpoint_cache,
        "_copy_tree_parallel",
        synchronized_copy,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                stage_checkpoint_for_realtime,
                source,
                required_paths=("weights.bin",),
            )
            for _ in range(2)
        ]
        staged = [future.result(timeout=10) for future in futures]

    assert staged[0] == staged[1]
    assert len(temporary_paths) == 2
    assert temporary_paths[0] != temporary_paths[1]
    assert not any(path.exists() for path in temporary_paths)
