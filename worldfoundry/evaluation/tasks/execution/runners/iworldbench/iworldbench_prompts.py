"""iWorld-Bench prompt materialization from released dataset metadata CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "iworld-bench"
IN_TREE_IWORLD_BENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "iworldbench"
METADATA_REL = Path("dataset/all_pack/metadata.csv")
CAMERA_FOLLOWING_METADATA_REL = Path("dataset/all_pack/camera_following_metadata.csv")
CANONICAL_PROMPT_COUNT = 4900

SPLIT_TASK_NAMES = {
    "diff": "Diff",
    "mem": "Mem",
    "camera_following": "CameraFollowing",
}

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def resolve_iworldbench_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_IWORLD_BENCH_ROOT"),
        IN_TREE_IWORLD_BENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_dataset_root(*, explicit: Path | None = None, repo_root: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT"),
        repo_root,
        resolve_iworldbench_root(),
    ):
        if candidate is not None and candidate.is_dir():
            metadata = candidate / METADATA_REL
            if metadata.is_file():
                return candidate.expanduser().resolve()
    bundled = bundled_benchmark_assets_root(BENCHMARK_ID)
    if (bundled / METADATA_REL).is_file():
        return bundled
    return None


def normalize_iworldbench_split(split: str) -> str:
    normalized = str(split).strip().lower().replace("-", "_")
    if normalized == "camerafollowing":
        normalized = "camera_following"
    if normalized not in SPLIT_TASK_NAMES:
        raise ValueError(f"unsupported iWorld-Bench split {split!r}; expected one of {tuple(SPLIT_TASK_NAMES)}")
    return normalized


def _metadata_relative_path(split: str) -> Path:
    return CAMERA_FOLLOWING_METADATA_REL if split == "camera_following" else METADATA_REL


def _metadata_under_root(root: Path, relative: Path, *, source: str) -> Path:
    candidate = root.expanduser().resolve() / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"iWorld-Bench metadata CSV not found under {source}: {candidate}")
    return candidate


def resolve_metadata_csv_path(
    *,
    explicit: Path | None = None,
    dataset_root: Path | None = None,
    repo_root: Path | None = None,
    split: str = "diff",
) -> Path:
    normalized_split = normalize_iworldbench_split(split)
    relative = _metadata_relative_path(normalized_split)
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"iWorld-Bench metadata CSV not found: {path}")
        return path
    if dataset_root is not None:
        # A caller-selected dataset must never be silently shadowed by the tiny
        # checked-in contract fixture.
        return _metadata_under_root(Path(dataset_root), relative, source="dataset_root")
    env_manifest = _env_path("WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"iWorld-Bench metadata CSV not found: {env_manifest}")
        return env_manifest
    env_dataset_root = _env_path("WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT")
    if env_dataset_root is not None:
        return _metadata_under_root(env_dataset_root, relative, source="WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT")
    if repo_root is not None:
        candidate = repo_root.expanduser().resolve() / relative
        if candidate.is_file():
            return candidate
    configured_root = resolve_iworldbench_root()
    if configured_root is not None:
        candidate = configured_root / relative
        if candidate.is_file():
            return candidate
    bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        "iWorld-Bench metadata CSV is missing. Set WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT, "
        "WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST, or WORLDFOUNDRY_IWORLD_BENCH_ROOT."
    )


def _generation_text(row: dict[str, Any]) -> str:
    for key in ("text_description", "prompt", "caption", "action_text", "description"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _prompt_id_from_row(row: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "video_id", "prompt_id", "index", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(index)


def _resolve_path(value: Any, *, search_roots: tuple[Path, ...]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    candidates = tuple(root / path for root in search_roots)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    # Preserve a deterministic absolute location so missing downloads fail at
    # model preflight instead of being interpreted relative to the process cwd.
    return str(candidates[0].resolve())


def _dataset_root_from_metadata(metadata_path: Path) -> Path | None:
    parent = metadata_path.parent
    if parent.name == "all_pack" and parent.parent.name == "dataset":
        return parent.parent.parent
    return None


def _control_metadata(
    row: dict[str, Any],
    *,
    metadata_path: Path,
    dataset_root: Path | None,
    split: str,
) -> dict[str, Any]:
    if split == "camera_following":
        columns = ("source_camera_txt_path", "source_camera_path")
        control_dir_name = "source_camera_txt"
    else:
        columns = ("control_txt_path", "camera_path")
        control_dir_name = "inference_txt"

    control_column = next((column for column in columns if str(row.get(column) or "").strip()), None)
    raw_value = str(row.get(control_column) or "").strip() if control_column else ""
    if not raw_value and split == "diff":
        values = tuple(str(row.get(key) or "").strip() for key in ("level", "translation", "rotation"))
        if all(values):
            raw_value = f"camera_{values[0]}_{values[1]}_{values[2]}.txt"
            control_column = "level/translation/rotation"
    if not raw_value and split == "mem":
        memory_id = str(row.get("memory_id") or "").strip()
        if memory_id:
            raw_value = f"memory_{memory_id}.txt"
            control_column = "memory_id"
    if not raw_value:
        return {}

    bundled_root = bundled_benchmark_assets_root(BENCHMARK_ID)
    roots: list[Path] = [metadata_path.parent]
    if dataset_root is not None:
        roots.extend(
            (
                dataset_root,
                dataset_root / "camera_trajectories" / control_dir_name,
            )
        )
    configured_root = resolve_iworldbench_root()
    if configured_root is not None:
        roots.extend((configured_root, configured_root / "camera_trajectories" / control_dir_name))
    roots.extend(
        (
            bundled_root,
            bundled_root / "camera_trajectories" / control_dir_name,
        )
    )
    resolved = _resolve_path(raw_value, search_roots=tuple(roots))
    return {
        "control_txt_path": resolved,
        "metadata_control_txt_path": raw_value,
        "control_column": control_column,
        "control_type": str(row.get("control_type") or ("source_camera" if split == "camera_following" else split)),
    }


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)
    return safe.strip("._") or "sample"


def _generation_id(prompt_id: str, control_txt_path: str | None) -> str:
    base = _safe_id(prompt_id)
    if not control_txt_path:
        return base
    control_stem = _safe_id(Path(control_txt_path).stem)
    if base == control_stem or base.endswith(f"_{control_stem}"):
        return base
    return f"{base}_{control_stem}"


def load_prompt_records(
    *,
    meta_csv_path: Path | None = None,
    dataset_root: Path | None = None,
    split: str = "diff",
) -> list[dict[str, Any]]:
    normalized_split = normalize_iworldbench_split(split)
    path = resolve_metadata_csv_path(
        explicit=meta_csv_path,
        dataset_root=dataset_root,
        split=normalized_split,
    )
    resolved_dataset_root = (
        Path(dataset_root).expanduser().resolve() if dataset_root is not None else _dataset_root_from_metadata(path)
    )
    expected_task = SPLIT_TASK_NAMES[normalized_split]
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if not row:
                continue
            row_task = str(row.get("task") or row.get("task_type") or "").strip()
            if row_task and row_task.casefold() != expected_task.casefold():
                continue
            prompt_id = _prompt_id_from_row(row, index)
            prompt = _generation_text(row)
            if not prompt_id:
                continue
            first_frame = _resolve_path(
                row.get("first_frame_path") or row.get("first_frame") or row.get("image_path") or row.get("asset_path"),
                search_roots=(path.parent,),
            )
            control = _control_metadata(
                row,
                metadata_path=path,
                dataset_root=resolved_dataset_root,
                split=normalized_split,
            )
            generation_id = _generation_id(prompt_id, control.get("control_txt_path"))
            records.append(
                {
                    "prompt_id": prompt_id,
                    "generation_id": generation_id,
                    "prompt": prompt,
                    "split": normalized_split,
                    "task": expected_task,
                    "first_frame": first_frame,
                    "control": control,
                    "official_video_name": f"{generation_id}.mp4",
                    "raw": dict(row),
                }
            )
    if not records:
        raise ValueError(f"iWorld-Bench prompt records are empty after validation: {path}")
    return records


def unique_prompt_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        generation_id = str(row.get("generation_id") or row["prompt_id"])
        if generation_id in seen:
            continue
        seen.add(generation_id)
        records.append(row)
    return records


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    explicit = str(record.get("official_video_name") or "").strip()
    if explicit:
        return Path(explicit).name
    return f"{_safe_id(record.get('generation_id') or record['prompt_id'])}.mp4"


def materialize_iworldbench_generation_requests(
    *,
    limit: int | None = None,
    meta_csv_path: Path | None = None,
    dataset_root: Path | None = None,
    split: str = "diff",
) -> tuple[GenerationRequest, ...]:
    normalized_split = normalize_iworldbench_split(split)
    records = unique_prompt_records(
        load_prompt_records(
            meta_csv_path=meta_csv_path,
            dataset_root=dataset_root,
            split=normalized_split,
        )
    )
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["generation_id"]
        inputs: dict[str, Any] = {
            "prompt_id": record["prompt_id"],
            "first_frame": record.get("first_frame"),
            "official_video_name": official_video_filename_for_record(record),
            "split": normalized_split,
        }
        if record.get("prompt"):
            inputs["prompt"] = record["prompt"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="iworld-bench",
                split=normalized_split,
                inputs=inputs,
                controls=record.get("control") or {},
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)
