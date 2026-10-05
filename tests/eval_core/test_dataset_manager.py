from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.catalog.schema import BenchmarkDatasetRef, BenchmarkZooEntry
from worldfoundry.evaluation.tasks.datasets import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManager,
    DatasetRef,
    build_dataset_manifest,
    check_local_dataset,
    classify_dataset_access,
    dataset_location_env_var,
    load_dataset_manifest,
    hf_cache_dataset_dir,
    locate_local_dataset,
    normalize_hf_dataset_id,
    parse_dataset_refs,
    validate_dataset_manifest,
    write_dataset_manifest,
)


def test_parse_dataset_refs_accepts_benchmark_dataset_ref_and_manifest_shapes() -> None:
    benchmark_ref = BenchmarkDatasetRef(
        hf_dataset_id="https://huggingface.co/datasets/org/direct/tree/main",
        revision="abc1234",
        license="mit",
    )
    manifest = {
        "benchmark_id": "alpha",
        "dataset": {"not_applicable": True, "reason": "generated artifacts supplied by caller"},
        "official_sources": {
            "huggingface_datasets": [
                {"repo_id": "org/a", "sha": "abc1234", "license": "mit", "gated": False},
                {"url": "https://huggingface.co/datasets/org/b/tree/main", "license": "apache-2.0"},
            ]
        },
    }

    refs = parse_dataset_refs([benchmark_ref, manifest])

    assert [ref.hf_dataset_id for ref in refs if ref.hf_dataset_id] == ["org/direct", "org/a", "org/b"]
    assert any(ref.not_applicable and ref.reason == "generated artifacts supplied by caller" for ref in refs)
    assert refs[0].revision == "abc1234"
    assert next(ref for ref in refs if ref.hf_dataset_id == "org/a").revision == "abc1234"
    assert next(ref for ref in refs if ref.hf_dataset_id == "org/b").source == "official_sources.huggingface_datasets"


def test_parse_dataset_refs_preserves_benchmark_zoo_not_applicable_dataset() -> None:
    entry = BenchmarkZooEntry.from_dict(
        {
            "id": "dataset-free",
            "dataset": {"not_applicable": True, "reason": "uses caller-generated videos"},
        }
    )

    refs = parse_dataset_refs(entry)

    assert refs == (
        DatasetRef(
            not_applicable=True,
            reason="uses caller-generated videos",
            source="dataset",
            metadata=entry.dataset.to_dict(),
        ),
    )


def test_parse_dataset_refs_accepts_source_provenance_hf_datasets() -> None:
    refs = parse_dataset_refs(
        {
            "benchmark_id": "source-provenance-benchmark",
            "source_provenance": {
                "huggingface_datasets": [
                    {"repo_id": "org/source-data", "sha": "abc1234", "license": "mit"},
                ],
            },
        }
    )

    assert len(refs) == 1
    assert refs[0].hf_dataset_id == "org/source-data"
    assert refs[0].revision == "abc1234"
    assert refs[0].source == "source_provenance.huggingface_datasets"


def test_dataset_manager_builds_filtered_and_full_metadata_plans(tmp_path: Path) -> None:
    manager = DatasetManager(tmp_path / "cache" / "hfd")
    manifest = {
        "benchmark_id": "alpha",
        "official_sources": {
            "huggingface_datasets": [
                {"repo_id": "org/a", "sha": "abc1234", "license": "mit"},
                {"repo_id": "org/b", "revision": "main", "license": "cc-by-nc-4.0", "gated": "manual"},
            ]
        },
    }

    filtered = manager.build_download_plan(manifest, dataset_ids=["org/b"], metadata_mode="filtered")
    full = manager.build_download_plan(manifest, dataset_ids=["org/b"], metadata_mode="full")

    assert filtered.dataset_ids == ("org/b",)
    assert filtered.commands == (
        (
            "hf",
            "download",
            "org/b",
            "--repo-type",
            "dataset",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "main",
        ),
    )
    assert filtered.metadata["hf_dataset_ids"] == ["org/b"]
    assert "available_dataset_refs" not in filtered.metadata

    assert full.metadata["hf_dataset_ids"] == ["org/b"]
    assert full.metadata["available_hf_dataset_ids"] == ["org/a", "org/b"]
    assert full.access_reports[0].access_status == "restricted"
    assert full.access_reports[0].license_status == "review_required"
    assert {issue.code for issue in full.access_reports[0].issues} == {
        "gated_dataset",
        "license_review_required",
    }


def test_dataset_manager_reports_missing_requested_filter_in_full_metadata(tmp_path: Path) -> None:
    manager = DatasetManager(tmp_path / "cache" / "hfd")
    manifest = {"benchmark_id": "alpha", "hf_dataset_id": "org/a"}

    plan = manager.build_download_plan(manifest, dataset_ids=["org/missing"], metadata_mode="full")

    assert plan.refs == ()
    assert plan.commands == ()
    assert plan.metadata["missing_requested_dataset_ids"] == ["org/missing"]
    assert plan.metadata["available_hf_dataset_ids"] == ["org/a"]


def test_classify_dataset_access_distinguishes_open_missing_and_restricted_refs() -> None:
    open_report = classify_dataset_access(DatasetRef(hf_dataset_id="org/open", license="mit"))
    missing_report = classify_dataset_access(DatasetRef(hf_dataset_id="org/missing"))
    restricted_report = classify_dataset_access(
        DatasetRef(hf_dataset_id="org/restricted", license="other", private=True)
    )

    assert open_report.ok is True
    assert open_report.issues == ()
    assert missing_report.access_status == "public"
    assert missing_report.license_status == "missing"
    assert [issue.code for issue in missing_report.issues] == ["missing_license"]
    assert restricted_report.access_status == "restricted"
    assert restricted_report.requires_auth is True
    assert {issue.code for issue in restricted_report.issues} == {
        "private_dataset",
        "license_review_required",
    }


def test_local_dataset_check_detects_ready_branch_snapshot_and_incomplete_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = hf_cache_dataset_dir(cache_dir, "org/dataset")
    snapshot = dataset_dir / "snapshots" / "abc1234"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "test").write_text("abc1234", encoding="utf-8")
    (snapshot / "sample.json").write_text("{}", encoding="utf-8")

    ready = check_local_dataset("org/dataset", cache_dir, expected_revision="test")

    assert ready.ready is True
    assert ready.status == "ready"
    assert ready.referenced_snapshot == "abc1234"
    assert ready.snapshot_dirs == (snapshot,)

    (dataset_dir / "sample.incomplete").write_text("partial", encoding="utf-8")

    incomplete = check_local_dataset("org/dataset", cache_dir, expected_revision="test")

    assert incomplete.ready is False
    assert incomplete.status == "incomplete_files"
    assert incomplete.incomplete_files[0]["path"].endswith("sample.incomplete")


def test_local_dataset_check_detects_revision_mismatch_and_broken_links(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = hf_cache_dataset_dir(cache_dir, "org/dataset")
    snapshot = dataset_dir / "snapshots" / "abc1234"
    (dataset_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs" / "main").write_text("abc1234", encoding="utf-8")
    (snapshot / "broken.json").symlink_to(snapshot / "missing.json")

    mismatch = check_local_dataset("org/dataset", cache_dir, expected_revision="def5678")
    broken = check_local_dataset("org/dataset", cache_dir)

    assert mismatch.status == "revision_mismatch"
    assert mismatch.revision_matches is False
    assert broken.status == "broken_links"
    assert broken.broken_links[0].endswith("broken.json")


def test_local_dataset_check_accepts_direct_hfd_double_underscore_layout(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        '{"sha":"abc123def456"}',
        encoding="utf-8",
    )
    (dataset_dir / "data.json").write_text("{}", encoding="utf-8")

    ready = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert ready.ready is True
    assert ready.status == "ready"
    assert ready.local_layout == "direct_hfd"
    assert ready.direct_dataset_dir == dataset_dir
    assert ready.direct_file_count == 1
    assert ready.direct_revision == "abc123def456"


def test_local_dataset_check_accepts_nested_hfd_datasets_layout(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "hfd_datasets" / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        '{"sha":"abc123def456"}',
        encoding="utf-8",
    )
    (dataset_dir / "data.json").write_text("{}", encoding="utf-8")

    ready = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert ready.ready is True
    assert ready.status == "ready"
    assert ready.local_layout == "direct_hfd"
    assert ready.direct_dataset_dir == dataset_dir


def test_local_dataset_check_prefers_local_dir_metadata_revision_over_stale_hfd(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        '{"sha":"old123"}',
        encoding="utf-8",
    )
    metadata_dir = dataset_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "data.json.metadata").write_text(
        "abc123def456\netag\n123.0\n",
        encoding="utf-8",
    )
    (dataset_dir / "data.json").write_text("{}", encoding="utf-8")

    ready = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert ready.ready is True
    assert ready.status == "ready"
    assert ready.local_layout == "direct_hfd"
    assert ready.direct_revision == "abc123def456"


def test_local_dataset_check_rejects_direct_hfd_missing_metadata_siblings(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps(
            {
                "sha": "abc123def456",
                "siblings": [
                    {"rfilename": "present.json"},
                    {"rfilename": ".DS_Store"},
                    {"rfilename": "missing/video.mp4"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "present.json").write_text("{}", encoding="utf-8")

    status = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert status.ready is False
    assert status.status == "direct_hfd_incomplete_files"
    assert status.local_layout == "direct_hfd"
    assert status.direct_incomplete_files[0]["kind"] == "missing_expected_file"
    assert status.direct_incomplete_files[0]["relative_path"] == "missing/video.mp4"


def test_local_dataset_check_rejects_direct_hfd_aria2_partials(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = cache_dir / "hfd_datasets" / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        '{"sha":"abc123def456"}',
        encoding="utf-8",
    )
    (dataset_dir / "part-000").write_text("partial", encoding="utf-8")
    (dataset_dir / "part-000.aria2").write_text("metadata", encoding="utf-8")

    status = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert status.ready is False
    assert status.status == "direct_hfd_incomplete_files"
    assert status.direct_dataset_dir == dataset_dir
    assert status.direct_incomplete_files[0]["path"].endswith("part-000.aria2")


def test_local_dataset_check_uses_shallow_direct_hfd_metadata_free_scan(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    for index in range(300):
        _touch = dataset_dir / f"shard-{index:03d}" / "sample.json"
        _touch.parent.mkdir(parents=True, exist_ok=True)
        _touch.write_text("{}", encoding="utf-8")

    status = check_local_dataset("org/dataset", cache_dir)

    assert status.ready is True
    assert status.status == "ready"
    assert status.local_layout == "direct_hfd"
    assert status.direct_file_count == 300


def test_local_dataset_check_streams_huge_direct_hfd_sibling_lists(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"siblings": [{"rfilename": f"videos/sample-{index:04d}.mp4"} for index in range(600)]}),
        encoding="utf-8",
    )

    status = check_local_dataset("org/dataset", cache_dir)

    assert status.ready is False
    assert status.status == "direct_hfd_incomplete_files"
    assert status.direct_incomplete_files[0]["kind"] == "missing_expected_file"
    assert status.direct_incomplete_files[0]["relative_path"] == "videos/sample-0000.mp4"


def test_local_dataset_check_accepts_large_complete_direct_hfd_sibling_list(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    siblings = [{"rfilename": f"videos/sample-{index:04d}.mp4"} for index in range(150)]
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456", "siblings": siblings}),
        encoding="utf-8",
    )
    for sibling in siblings:
        path = dataset_dir / sibling["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("video", encoding="utf-8")

    status = check_local_dataset({"repo_id": "org/dataset", "revision": "abc123def456"}, cache_dir)

    assert status.ready is True
    assert status.status == "ready"
    assert status.direct_file_count == 150


def test_local_dataset_check_accepts_complete_direct_hfd_sibling_list_past_count_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.evaluation.tasks.datasets.manager as dataset_manager

    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / ".hfd").mkdir(parents=True)
    siblings = [{"rfilename": f"videos/sample-{index:04d}.mp4"} for index in range(12)]
    (dataset_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456", "siblings": siblings}),
        encoding="utf-8",
    )
    for sibling in siblings:
        path = dataset_dir / sibling["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("video", encoding="utf-8")
    monkeypatch.setattr(dataset_manager, "_DIRECT_DATASET_FILE_COUNT_LIMIT", 5)

    status = check_local_dataset({"repo_id": "org/dataset", "revision": "abc123def456"}, cache_dir)

    assert status.ready is True
    assert status.status == "ready"
    assert status.direct_file_count == 5
    assert status.direct_file_count_capped is True
    assert status.direct_incomplete_files == ()


def test_local_dataset_check_bounds_direct_hfd_download_metadata_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.evaluation.tasks.datasets.manager as dataset_manager

    cache_dir = tmp_path / "hfd_datasets"
    dataset_dir = cache_dir / "org__dataset"
    (dataset_dir / "sample.json").parent.mkdir(parents=True)
    (dataset_dir / "sample.json").write_text("{}", encoding="utf-8")
    metadata_root = dataset_dir / ".cache" / "huggingface" / "download"
    (metadata_root / "a").mkdir(parents=True)
    (metadata_root / "b").mkdir(parents=True)
    (metadata_root / "z").mkdir(parents=True)
    (metadata_root / "z" / "sample.metadata").write_text("abc123def456\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manager, "_DIRECT_DATASET_METADATA_DIR_LIMIT", 2)

    status = check_local_dataset("org/dataset", cache_dir, expected_revision="abc123def456")

    assert status.ready is False
    assert status.status == "direct_hfd_revision_mismatch"
    assert status.direct_revision is None


def test_download_plan_dedupes_top_level_dataset_when_revision_ref_exists(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.datasets import DatasetManager

    manager = DatasetManager(tmp_path / "hfd")
    plan = manager.build_download_plan(
        {
            "benchmark_id": "demo",
            "dataset": {"hf_dataset_id": "org/dataset"},
            "dataset_refs": [
                {
                    "hf_dataset_id": "org/dataset",
                    "revision": "abc123def456",
                    "license": "mit",
                }
            ],
        },
        check_local=True,
    )

    assert plan.dataset_ids == ("org/dataset",)
    assert len(plan.commands) == 1
    assert plan.commands[0] == (
        "hf",
        "download",
        "org/dataset",
        "--repo-type",
        "dataset",
        "--cache-dir",
        str(tmp_path / "hfd"),
        "--revision",
        "abc123def456",
    )
    assert len(plan.local_checks) == 1


def test_locate_local_dataset_prefers_explicit_env_path(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "vbench-human-annotation"
    dataset_dir.mkdir()
    env_var = dataset_location_env_var("Vchitect/VBench-2.0_human_annotation")

    location = locate_local_dataset(
        "Vchitect/VBench-2.0_human_annotation",
        env={env_var: str(dataset_dir)},
    )

    assert env_var == "WORLDFOUNDRY_DATASET_VCHITECT_VBENCH_2_0_HUMAN_ANNOTATION_PATH"
    assert location.ready is True
    assert location.source == "env"
    assert location.path == dataset_dir


def test_locate_local_dataset_uses_manifest_and_data_root(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    manifest_dataset = root / "Video-Bench" / "Video-Bench_human_annotation"
    manifest_dataset.mkdir(parents=True)
    manifest_path = tmp_path / "local_datasets.yaml"
    manifest_path.write_text(
        """
datasets:
  - hf_dataset_id: Video-Bench/Video-Bench_human_annotation
    path: benchmarks/Video-Bench/Video-Bench_human_annotation
""".strip(),
        encoding="utf-8",
    )
    data_root_dataset = root / "BestWishYsh--ChronoMagic-Bench"
    data_root_dataset.mkdir()

    manifest_location = locate_local_dataset(
        "Video-Bench/Video-Bench_human_annotation",
        manifest_path=manifest_path,
        env={},
    )
    data_root_location = locate_local_dataset(
        "BestWishYsh/ChronoMagic-Bench",
        data_root=root,
        env={},
    )

    assert manifest_location.ready is True
    assert manifest_location.source == "manifest"
    assert manifest_location.path == manifest_dataset
    assert data_root_location.ready is True
    assert data_root_location.source == "data_root"
    assert data_root_location.path == data_root_dataset


def test_locate_local_dataset_uses_local_assets_manifest(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "datasets" / "Howieeeee__WorldScore"
    dataset_dir.mkdir(parents=True)
    manifest_path = tmp_path / "local_assets_manifest.yaml"
    manifest_path.write_text(
        f"""
schema_version: worldfoundry-local-assets-v1
benchmarks:
  - id: worldscore
    assets:
      - id: dataset
        kind: dataset
        hf_dataset_id: Howieeeee/WorldScore
        path: {dataset_dir}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDFOUNDRY_LOCAL_ASSET_MANIFEST", str(manifest_path))
    monkeypatch.setenv("WORLDFOUNDRY_HOME", str(tmp_path / "home"))

    location = locate_local_dataset("Howieeeee/WorldScore", env=os.environ)

    assert location.ready is True
    assert location.source == "local_assets_manifest"
    assert location.path == dataset_dir.resolve()


def test_dataset_manager_locate_local_uses_ready_hf_cache_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "hfd"
    dataset_dir = hf_cache_dataset_dir(cache_dir, "Kaiyue/T2V-CompBench-Videos")
    snapshot = dataset_dir / "snapshots" / "92f9ef4642f244567e8aa2789827ec301500a2ff"
    snapshot.mkdir(parents=True)
    (dataset_dir / "refs").mkdir()
    (dataset_dir / "refs" / "main").write_text("92f9ef4642f244567e8aa2789827ec301500a2ff", encoding="utf-8")
    (snapshot / "sample.mp4").write_text("video", encoding="utf-8")
    manager = DatasetManager(cache_dir)

    location = manager.locate_local("Kaiyue/T2V-CompBench-Videos", env={})

    assert location.ready is True
    assert location.source == "hf_cache"
    assert location.path == snapshot
    assert location.cache_dataset_dir == dataset_dir


def test_locate_local_dataset_reports_missing_manifest_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing-local-datasets.json"

    location = locate_local_dataset(
        "TIGER-Lab/VideoFeedback",
        manifest_path=manifest_path,
        env={},
    )

    assert location.ready is False
    assert location.source == "manifest"
    assert location.status == "not_found"
    assert location.manifest_path == manifest_path


def test_normalize_hf_dataset_id_and_cache_dir() -> None:
    assert normalize_hf_dataset_id("https://huggingface.co/datasets/org/name/tree/main") == "org/name"
    assert normalize_hf_dataset_id("datasets/org/name") == "org/name"
    assert hf_cache_dataset_dir("/cache/hfd", "org/name") == Path("/cache/hfd/datasets--org--name")


def test_dataset_manifest_builds_stable_sample_contract(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        '{"sample_id": "sample-a", "prompt": "look left"}\n'
        '{"sample_id": "sample-b", "prompt": "look right"}\n',
        encoding="utf-8",
    )

    manifest = build_dataset_manifest(
        samples_path=samples_path,
        root=tmp_path,
        dataset_id="unit-dataset",
        split="smoke",
        license="mit",
        access={"status": "public"},
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)

    loaded = load_dataset_manifest(manifest_path)
    validation = validate_dataset_manifest(manifest_path)

    assert loaded.schema_version == DATASET_MANIFEST_SCHEMA_VERSION
    assert loaded.dataset_id == "unit-dataset"
    assert loaded.samples_path == "samples.jsonl"
    assert loaded.sample_count == 2
    assert loaded.sample_ids_sha256
    assert loaded.license == "mit"
    assert validation["ok"] is True
    assert validation["sample_count"] == 2
    assert validation["issues"] == []


def test_dataset_manifest_validation_reports_missing_samples_path(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text('{"sample_id": "sample-a", "prompt": "look left"}\n', encoding="utf-8")
    manifest = build_dataset_manifest(samples_path=samples_path, root=tmp_path, dataset_id="missing-samples")
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    samples_path.unlink()

    validation = validate_dataset_manifest(manifest_path)

    assert validation["ok"] is False
    assert validation["issues"][0].startswith("samples_path not found:")


def test_dataset_manifest_validation_warns_for_generated_sample_ids(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text('{"prompt": "look left"}\n', encoding="utf-8")
    manifest = build_dataset_manifest(samples_path=samples_path, root=tmp_path, dataset_id="generated-ids")
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)

    validation = validate_dataset_manifest(manifest_path)

    assert manifest.missing_sample_id_count == 1
    assert validation["ok"] is True
    assert validation["warnings"] == ["1 sample(s) rely on generated sample ids"]


def test_dataset_manifest_validation_reports_drift_and_duplicate_ids(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        '{"sample_id": "sample-a", "prompt": "first"}\n'
        '{"sample_id": "sample-b", "prompt": "second"}\n',
        encoding="utf-8",
    )
    manifest = build_dataset_manifest(samples_path=samples_path, root=tmp_path, dataset_id="drift")
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    samples_path.write_text(
        '{"sample_id": "sample-a", "prompt": "changed"}\n'
        '{"sample_id": "sample-a", "prompt": "duplicate"}\n',
        encoding="utf-8",
    )

    validation = validate_dataset_manifest(manifest_path)

    assert validation["ok"] is False
    assert "sha256 mismatch" in validation["issues"]
    assert "sample_ids_sha256 mismatch" in validation["issues"]
    assert "duplicate sample ids: sample-a" in validation["issues"]
