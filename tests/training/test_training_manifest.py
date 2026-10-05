from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.training.data import (
    TRAINING_SAMPLE_SCHEMA,
    TrainingManifestDataset,
    TrainingManifestError,
    inspect_training_manifest,
    load_training_manifest,
)


def _write_media(path: Path, payload: bytes) -> int:
    path.write_bytes(payload)
    return len(payload)


def _sample(
    *,
    sample_id: str,
    uri: str,
    size_bytes: int,
    split: str = "train",
    task: str = "text_to_image",
) -> dict[str, object]:
    return {
        "schema": TRAINING_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "task": task,
        "prompt": f"prompt for {sample_id}",
        "media": {"uri": uri, "size_bytes": size_bytes},
        "width": 32,
        "height": 24,
        "num_frames": 1,
        "fps": 1.0,
        "conditions": {"kind": "none"},
        "split": split,
        "safety": {"reviewed": True},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_manifest_dataset_validates_files_filters_split_and_freezes_rows(tmp_path: Path) -> None:
    first_size = _write_media(tmp_path / "first.bin", b"first-media")
    second_size = _write_media(tmp_path / "second.bin", b"second-media")
    manifest_path = tmp_path / "samples.jsonl"
    _write_jsonl(
        manifest_path,
        [
            _sample(sample_id="train-1", uri="first.bin", size_bytes=first_size),
            _sample(
                sample_id="valid-1",
                uri="second.bin",
                size_bytes=second_size,
                split="validation",
            ),
        ],
    )

    dataset = TrainingManifestDataset.from_file(
        manifest_path,
        split="train",
        verify_files=True,
    )

    assert len(dataset) == 1
    assert dataset.sample_ids == ("train-1",)
    assert dataset.index_for_sample_id("train-1") == 0
    assert dataset.manifest.report.split_counts == {"train": 1, "validation": 1}
    assert dataset[0].task == "text_to_image"
    with pytest.raises(TypeError):
        dataset[0].conditions["mutated"] = True  # type: ignore[index]


def test_manifest_inspector_collects_duplicate_and_row_schema_errors(tmp_path: Path) -> None:
    size = _write_media(tmp_path / "media.bin", b"media")
    duplicate = _sample(sample_id="same", uri="media.bin", size_bytes=size)
    invalid = _sample(sample_id="invalid", uri="media.bin", size_bytes=size)
    invalid["unexpected"] = True
    manifest_path = tmp_path / "bad.jsonl"
    _write_jsonl(manifest_path, [duplicate, duplicate, invalid])

    report = inspect_training_manifest(manifest_path, split="train", verify_files=True)

    assert report.ok is False
    assert report.error_count == 2
    assert {issue.code for issue in report.issues} == {"duplicate-sample-id", "invalid-sample"}
    assert report.to_dict()["selected_sample_count"] == 2
    with pytest.raises(TrainingManifestError) as captured:
        load_training_manifest(manifest_path, split="train")
    assert captured.value.report is report or captured.value.report.to_dict() == report.to_dict()


def test_manifest_size_mismatch_is_an_error(tmp_path: Path) -> None:
    _write_media(tmp_path / "media.bin", b"actual")
    manifest_path = tmp_path / "samples.jsonl"
    _write_jsonl(
        manifest_path,
        [_sample(sample_id="item", uri="media.bin", size_bytes=123)],
    )

    report = inspect_training_manifest(manifest_path, verify_files=True)

    assert report.ok is False
    assert [issue.code for issue in report.issues] == ["media-size-mismatch"]


def test_file_checks_can_be_disabled_for_remote_or_lazy_media(tmp_path: Path) -> None:
    manifest_path = tmp_path / "samples.jsonl"
    _write_jsonl(
        manifest_path,
        [_sample(sample_id="missing", uri="missing.bin", size_bytes=1)],
    )

    report = inspect_training_manifest(manifest_path, verify_files=False)

    assert report.ok


def test_json_container_and_parquet_manifests_share_the_contract(tmp_path: Path) -> None:
    size = _write_media(tmp_path / "media.bin", b"content")
    row = _sample(sample_id="item", uri="media.bin", size_bytes=size)

    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps({"samples": [row]}), encoding="utf-8")
    json_sample = load_training_manifest(json_path, split="train").samples[0]
    assert json_sample.sample_id == "item"

    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet_path = tmp_path / "samples.parquet"
    parquet.write_table(pyarrow.Table.from_pylist([row]), parquet_path)
    parquet_sample = load_training_manifest(parquet_path, split="train").samples[0]
    assert parquet_sample.to_dict() == json_sample.to_dict()
