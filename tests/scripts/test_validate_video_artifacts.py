from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inference" / "validate_video_artifacts.py"
SPEC = importlib.util.spec_from_file_location("worldfoundry_validate_video_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_video_artifacts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_video_artifacts
SPEC.loader.exec_module(validate_video_artifacts)


def test_loads_pretty_workspace_manifest_and_deduplicates_preview(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [str(video), "result_metadata.json"],
                "metadata": {"result": {"video_sha256": "a" * 64}},
                "preview_video": str(video),
                "status": "succeeded",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert validate_video_artifacts._load_manifest_rows(manifest) == [
        (
            1,
            {
                "kind": "generated_video",
                "mime_type": "video/mp4",
                "sha256": "a" * 64,
                "uri": str(video),
            },
        )
    ]


def test_workspace_artifact_sha_is_bound_only_to_matching_video(tmp_path: Path) -> None:
    video = tmp_path / "generated.mp4"
    control = tmp_path / "control.mp4"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [str(video), str(control)],
                "metadata": {
                    "result": {
                        "artifact_path": str(video),
                        "artifact_sha256": "b" * 64,
                    }
                },
                "preview_video": str(video),
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )

    assert validate_video_artifacts._load_manifest_rows(manifest) == [
        (
            1,
            {
                "kind": "generated_video",
                "mime_type": "video/mp4",
                "sha256": "b" * 64,
                "uri": str(video),
            },
        ),
        (
            1,
            {
                "kind": "generated_video",
                "mime_type": "video/mp4",
                "uri": str(control),
            },
        ),
    ]


def test_loads_json_array_and_jsonl_artifact_manifests(tmp_path: Path) -> None:
    rows = [
        {
            "kind": "generated_video",
            "mime_type": "video/mp4",
            "sha256": "a" * 64,
            "uri": "one.mp4",
        },
        {"kind": "image", "mime_type": "image/png", "sha256": "b" * 64, "uri": "one.png"},
    ]
    array_manifest = tmp_path / "artifacts.json"
    array_manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    assert validate_video_artifacts._load_manifest_rows(array_manifest) == [(1, rows[0]), (1, rows[1])]

    jsonl_manifest = tmp_path / "artifacts.jsonl"
    jsonl_manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert validate_video_artifacts._load_manifest_rows(jsonl_manifest) == [(1, rows[0]), (2, rows[1])]


def test_loads_completed_workspace_matrix_rows(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "ready-model",
                        "status": "completed",
                        "validation": {
                            "manifest_path": "/runs/ready/manifest.json",
                            "videos": [
                                {
                                    "path": "/runs/ready/output.mp4",
                                    "sha256": "c" * 64,
                                }
                            ],
                        },
                    },
                    {
                        "model_id": "blocked-model",
                        "status": "blocked_checkpoint",
                        "validation": {"videos": [{"path": "/runs/blocked.mp4"}]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert validate_video_artifacts._load_manifest_rows(matrix) == [
        (
            1,
            {
                "kind": "generated_video",
                "mime_type": "video/mp4",
                "model_id": "ready-model",
                "sha256": "c" * 64,
                "source_manifest": "/runs/ready/manifest.json",
                "uri": "/runs/ready/output.mp4",
            },
        )
    ]


def test_invalid_multiline_manifest_reports_source_line(tmp_path: Path) -> None:
    manifest = tmp_path / "broken.jsonl"
    manifest.write_text(
        '{"kind": "generated_video", "uri": "one.mp4"}\n{broken}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        validate_video_artifacts._load_manifest_rows(manifest)


def test_resolves_portable_and_root_relative_workspace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "checkout"
    video = repo_root / "artifacts" / "runs" / "demo.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    manifest = video.parent / "manifest.json"
    monkeypatch.setattr(validate_video_artifacts, "REPO_ROOT", repo_root)

    assert validate_video_artifacts._resolve_uri(
        manifest, "${WORLDFOUNDRY_REPO_ROOT}/artifacts/runs/demo.mp4"
    ) == video.resolve()
    assert validate_video_artifacts._resolve_uri(
        manifest, "/artifacts/runs/demo.mp4"
    ) == video.resolve()


def test_strict_video_validation_persists_stream_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrame:
        def __init__(self, value: int, time: float) -> None:
            self.value = value
            self.time = time

        def to_ndarray(self, *, format: str):
            assert format == "rgb24"
            return validate_video_artifacts.np.full((4, 6, 3), self.value, dtype="uint8")

    stream = SimpleNamespace(average_rate=2, duration=4, time_base=0.5)

    class FakeContainer:
        duration = None
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, selected_stream):
            assert selected_stream is stream
            return [FakeFrame(10, 0.0), FakeFrame(20, 0.5)]

    monkeypatch.setattr(validate_video_artifacts.av, "open", lambda *_args, **_kwargs: FakeContainer())

    result = validate_video_artifacts._validate_video(tmp_path / "video.mp4")

    assert result["duration_seconds"] == pytest.approx(2.0)
    assert result["decoded_frames"] == 2
    assert result["temporal_change"] is True
