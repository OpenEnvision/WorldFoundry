# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Load videos directly, or a JSONL manifest with additional sample fields."""

import json
import os
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ANNOTATION_PATH_FIELDS = ("condition_image", "target_object_mask_first_frame")
ARTIFACT_PATH_FIELDS = ("da3", "da3_scene", "target_poses", "vipe")


def _video_sample(path: Path, source: Path) -> dict:
    # Both public layouts carry unambiguous labels at a named root.
    if (
        len(path.parents) >= 5
        and path.parent.name.startswith("case_")
        and path.name == "video.mp4"
        and path.parents[4].name in {"samples", "inference_results", "WorldMemBench"}
    ):
        first, second, trajectory, case_id, _ = path.parts[-5:]
        if path.parents[4].name == "samples":
            model, category = first, second
        else:
            category, model = first, second
        return {
            "item_id": "/".join((model, category, trajectory, case_id)),
            "model_id": model,
            "category": category,
            "condition_id": trajectory,
            "sample_id": case_id,
            "video": str(path),
        }
    return {"item_id": path.relative_to(source).as_posix(), "video": str(path)}


def load_samples(
    source: str | Path,
    *,
    models: list[str] | None = None,
    categories: list[str] | None = None,
    trajectories: list[str] | None = None,
    cases: list[str] | None = None,
) -> list[dict]:
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        samples = [
            _video_sample(path, source)
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    elif source.is_file() and source.suffix.lower() == ".jsonl":
        samples = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                video = (source.parent / sample["video"]).resolve()
                if not video.is_file():
                    raise ValueError(f"video not found: {video}")
                sample["video"] = str(video)
                annotations = sample.get("annotations")
                if isinstance(annotations, dict):
                    for field in ANNOTATION_PATH_FIELDS:
                        if isinstance(annotations.get(field), str) and annotations[field]:
                            annotations[field] = str((source.parent / annotations[field]).resolve())
                artifacts = sample.get("artifacts")
                if isinstance(artifacts, dict):
                    for field in ARTIFACT_PATH_FIELDS:
                        if isinstance(artifacts.get(field), str) and artifacts[field]:
                            artifacts[field] = str((source.parent / artifacts[field]).resolve())
                sample.setdefault("item_id", str(number))
                samples.append(sample)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{source}:{number}: {error}") from error
    elif source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        samples = [_video_sample(source, source.parent)]
    else:
        raise ValueError(f"expected a video, video directory, or JSONL manifest: {source}")
    if not samples:
        raise ValueError(f"no samples found: {source}")
    ids = [sample["item_id"] for sample in samples]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("item_id must be a non-empty string")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id in manifest")
    for field, values in (
        ("model_id", models),
        ("category", categories),
        ("condition_id", trajectories),
        ("sample_id", cases),
    ):
        if values:
            if any(not sample.get(field) for sample in samples):
                raise ValueError(f"filter requires {field}; use the standard samples layout or a manifest")
            samples = [sample for sample in samples if sample[field] in values]
    if not samples:
        raise ValueError("no samples match the requested filters")
    return samples


def write_manifest(samples: list[dict], output: str | Path) -> Path:
    """Save relative media and artifact paths alongside sample annotations."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for sample in samples:
            row = {**sample, "video": os.path.relpath(sample["video"], output.parent)}
            if isinstance(sample.get("annotations"), dict):
                row["annotations"] = dict(sample["annotations"])
                for field in ANNOTATION_PATH_FIELDS:
                    value = row["annotations"].get(field)
                    if isinstance(value, str) and value:
                        row["annotations"][field] = os.path.relpath(value, output.parent)
            if isinstance(sample.get("artifacts"), dict):
                row["artifacts"] = dict(sample["artifacts"])
                for field in ARTIFACT_PATH_FIELDS:
                    value = row["artifacts"].get(field)
                    if isinstance(value, str) and value:
                        row["artifacts"][field] = os.path.relpath(value, output.parent)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return output
