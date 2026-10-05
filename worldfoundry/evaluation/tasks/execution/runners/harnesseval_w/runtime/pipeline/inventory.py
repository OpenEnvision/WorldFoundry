"""Read-only manifest and generation inventory for a HarnessEval evaluation batch."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ..io import atomic_write_json, atomic_write_text, file_fingerprint, read_json
from ..protocols import FAMILIES
from .planner import initial_observation_path


INVENTORY_SCHEMA = "harnesseval.inventory"
VideoProbe = Callable[[Path, bool], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, separator, denominator = value.partition("/")
    return float(numerator) / float(denominator) if separator else float(value)


def probe_video(video_path: Path, full_decode: bool = False) -> dict[str, Any]:
    """Read stream metadata and optionally decode the complete video."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")
    streams = (json.loads(completed.stdout).get("streams") or [])
    if not streams:
        raise RuntimeError("video has no readable video stream")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _rate(stream.get("avg_frame_rate"))
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("video has invalid dimensions or FPS")

    if full_decode:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if decoded.returncode:
            raise RuntimeError(decoded.stderr.strip() or "full video decode failed")

    return {
        "check": "full_decode" if full_decode else "ffprobe",
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count or None,
        "duration_seconds": round(duration, 6) if duration > 0 else None,
    }


def load_manifest_cases(
    manifests: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    issues: list[dict[str, Any]] = []
    for manifest_path in manifests:
        value = read_json(manifest_path)
        manifest_cases = value.get("cases") if isinstance(value, dict) else None
        if not isinstance(manifest_cases, list):
            raise ValueError(f"manifest has no cases array: {manifest_path}")
        for index, case in enumerate(manifest_cases):
            if not isinstance(case, dict) or not case.get("case_id"):
                issues.append(
                    {
                        "kind": "invalid_case",
                        "manifest": str(manifest_path),
                        "index": index,
                    }
                )
                continue
            case_id = str(case["case_id"])
            taxonomy = case.get("taxonomy") or {}
            axis = str(taxonomy.get("primary_axis") or "")
            family = str(taxonomy.get("probe_family") or "")
            if not axis or family not in FAMILIES:
                issues.append(
                    {
                        "kind": "invalid_taxonomy",
                        "case_id": case_id,
                        "primary_axis": axis,
                        "probe_family": family,
                    }
                )
                continue
            if case_id in cases:
                issues.append(
                    {
                        "kind": "duplicate_case_id",
                        "case_id": case_id,
                        "manifests": [str(sources[case_id]), str(manifest_path)],
                    }
                )
                continue
            cases[case_id] = case
            sources[case_id] = manifest_path
    return [cases[key] for key in sorted(cases)], issues


def _metadata_video(metadata: dict[str, Any], metadata_path: Path) -> Path | None:
    raw = metadata.get("output_video") or metadata.get("video_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else metadata_path.parent / path).resolve()


def _inspect_rollout(
    generation_root: Path,
    case: dict[str, Any],
    model_id: str,
    assets_root: Path | None,
    full_decode: bool,
    video_probe: VideoProbe,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    taxonomy = case["taxonomy"]
    axis = str(taxonomy["primary_axis"])
    family = str(taxonomy["probe_family"])
    rollout_dir = generation_root / "outputs" / axis / family / model_id / case_id
    metadata_path = rollout_dir / "metadata.json"
    video_path = rollout_dir / "output.mp4"
    initial = initial_observation_path(case, assets_root)
    issues = []
    metadata: dict[str, Any] = {}
    video: dict[str, Any] = {}

    if initial is None:
        issues.append("missing_initial_observation")
    if not rollout_dir.is_dir():
        issues.append("missing_rollout_directory")
    if not metadata_path.is_file():
        issues.append("missing_metadata")
    else:
        try:
            value = read_json(metadata_path)
            if not isinstance(value, dict):
                raise ValueError("metadata is not a JSON object")
            metadata = value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid_metadata: {exc}")

    if metadata:
        identities = {
            "case_id": (metadata.get("case_id"), case_id),
            "model_id": (metadata.get("model_id") or metadata.get("model_slug"), model_id),
            "primary_axis": (
                (metadata.get("taxonomy") or {}).get("primary_axis"),
                axis,
            ),
            "probe_family": (
                (metadata.get("taxonomy") or {}).get("probe_family"),
                family,
            ),
        }
        for name, (actual, expected) in identities.items():
            if str(actual or "") != expected:
                issues.append(f"metadata_{name}_mismatch")
        declared_video = _metadata_video(metadata, metadata_path)
        if declared_video is None:
            issues.append("metadata_missing_video_path")
        elif not declared_video.is_file():
            issues.append("metadata_video_missing")
        elif video_path.is_file() and declared_video != video_path.resolve():
            issues.append("metadata_video_path_mismatch")

    if not video_path.is_file():
        issues.append("missing_video")
    else:
        video = file_fingerprint(video_path)
        try:
            video["probe"] = video_probe(video_path.resolve(), full_decode)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            issues.append(f"invalid_video: {exc}")

    return {
        "case_id": case_id,
        "model_id": model_id,
        "primary_axis": axis,
        "probe_family": family,
        "status": "ok" if not issues else "invalid",
        "issues": issues,
        "paths": {
            "rollout_dir": str(rollout_dir.resolve(strict=False)),
            "metadata": str(metadata_path.resolve(strict=False)),
            "video": str(video_path.resolve(strict=False)),
            "initial_observation": str(initial) if initial else None,
        },
        "video": video,
    }


def build_inventory(
    manifests: Iterable[Path],
    generation_root: Path,
    model_ids: Iterable[str],
    *,
    assets_root: Path | None = None,
    workers: int = 16,
    full_decode: bool = False,
    video_probe: VideoProbe = probe_video,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = [path.resolve(strict=True) for path in manifests]
    generation_root = generation_root.resolve(strict=True)
    models = sorted(set(model_ids))
    if not models:
        raise ValueError("at least one expected model is required")
    if workers <= 0:
        raise ValueError("workers must be positive")

    cases, manifest_issues = load_manifest_cases(manifests)
    work = [(case, model) for case in cases for model in models]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(
            pool.map(
                lambda item: _inspect_rollout(
                    generation_root,
                    item[0],
                    item[1],
                    assets_root,
                    full_decode,
                    video_probe,
                ),
                work,
            )
        )

    expected = {
        (row["primary_axis"], row["probe_family"], row["model_id"], row["case_id"])
        for row in rows
    }
    actual = set()
    outputs_root = generation_root / "outputs"
    if outputs_root.is_dir():
        for path in outputs_root.glob("*/*/*/*"):
            if path.is_dir():
                parts = path.relative_to(outputs_root).parts
                if len(parts) == 4:
                    if parts[0].startswith(("_", ".")) or parts[2] not in models:
                        continue
                    actual.add(tuple(parts))
    unexpected = [
        {
            "primary_axis": axis,
            "probe_family": family,
            "model_id": model,
            "case_id": case_id,
        }
        for axis, family, model, case_id in sorted(actual - expected)
    ]

    issue_counts = Counter(issue.split(":", 1)[0] for row in rows for issue in row["issues"])
    family_counts = Counter(str(case["taxonomy"]["probe_family"]) for case in cases)
    status = "passed"
    if manifest_issues or unexpected or any(row["status"] != "ok" for row in rows):
        status = "failed"
    audit = {
        "schema_version": INVENTORY_SCHEMA,
        "created_at": utc_now(),
        "status": status,
        "inputs": {
            "manifests": [str(path) for path in manifests],
            "generation_root": str(generation_root),
            "assets_root": str(assets_root.resolve()) if assets_root else None,
            "models": models,
            "video_check": "full_decode" if full_decode else "ffprobe",
        },
        "case_count": len(cases),
        "model_count": len(models),
        "expected_rollout_count": len(rows),
        "valid_rollout_count": sum(row["status"] == "ok" for row in rows),
        "invalid_rollout_count": sum(row["status"] != "ok" for row in rows),
        "family_case_counts": dict(sorted(family_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "manifest_issues": manifest_issues,
        "unexpected_rollouts": unexpected,
        "invalid_rollouts": [row for row in rows if row["status"] != "ok"],
    }
    return rows, audit


def write_inventory(
    output_root: Path,
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, str]:
    output_root = output_root.resolve()
    matrix_path = output_root / "rollout_matrix.jsonl"
    audit_path = output_root / "INPUT_AUDIT.json"
    lines = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    atomic_write_text(matrix_path, lines)
    atomic_write_json(audit_path, audit)
    return {"rollout_matrix": str(matrix_path), "input_audit": str(audit_path)}
