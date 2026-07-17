"""T2V-CompBench prompt requests and generated-video layout bridge."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import (
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
    local_path_for_uri,
)
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "t2v-compbench"
PROMPTS_PER_CATEGORY = 200
CANONICAL_PROMPT_COUNT = 1400
GENERATION_MANIFEST_NAME = "t2v_compbench_generation_manifest.jsonl"
OFFICIAL_LAYOUT_MANIFEST_NAME = "t2v_compbench_official_layout_manifest.jsonl"

# Order follows the official prompt suite and leaderboard documentation.
CATEGORY_ORDER = (
    "consistent_attribute_binding",
    "dynamic_attribute_binding",
    "spatial_relationships",
    "motion_binding",
    "action_binding",
    "object_interactions",
    "generative_numeracy",
)
CATEGORY_PROTOCOL: dict[str, dict[str, str]] = {
    "consistent_attribute_binding": {
        "prompt_file": "meta_data/consistent_attribute_binding.json",
        "video_subdir": "consistent_attr",
    },
    "dynamic_attribute_binding": {
        "prompt_file": "meta_data/dynamic_attribute_binding.json",
        "video_subdir": "dynamic_attr",
    },
    "spatial_relationships": {
        "prompt_file": "meta_data/spatial_relationships.json",
        "video_subdir": "spatial_relationships",
    },
    "motion_binding": {
        "prompt_file": "meta_data/motion_binding.json",
        "video_subdir": "motion_binding",
    },
    "action_binding": {
        "prompt_file": "meta_data/action_binding.json",
        "video_subdir": "action_binding",
    },
    "object_interactions": {
        "prompt_file": "meta_data/object_interactions.json",
        "video_subdir": "interaction",
    },
    "generative_numeracy": {
        "prompt_file": "meta_data/generative_numeracy.json",
        "video_subdir": "generative_numeracy",
    },
}
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def resolve_t2v_compbench_assets(explicit: Path | None = None) -> Path:
    """Resolve the checked-in official prompt and metadata assets."""

    env_value = os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_ASSETS")
    path = explicit or (Path(env_value) if env_value else bundled_benchmark_assets_root(BENCHMARK_ID))
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"T2V-CompBench asset directory not found: {path}")
    return path


def prompt_id(category_id: str, prompt_number: int) -> str:
    """Return a stable, path-safe category-local prompt id."""

    return f"{category_id}-{prompt_number:04d}"


def sample_id(category_id: str, prompt_number: int) -> str:
    """Return the stable WorldFoundry sample id for an official prompt."""

    return f"{BENCHMARK_ID}-{prompt_id(category_id, prompt_number)}"


def load_t2v_compbench_prompt_records(
    *,
    assets_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load the exact 7 x 200 prompt rows indexed by the official evaluators."""

    root = resolve_t2v_compbench_assets(assets_root)
    records: list[dict[str, Any]] = []
    for category_id in CATEGORY_ORDER:
        protocol = CATEGORY_PROTOCOL[category_id]
        source = root / protocol["prompt_file"]
        if not source.is_file():
            raise FileNotFoundError(f"T2V-CompBench metadata file not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"T2V-CompBench metadata must be a JSON list: {source}")
        if len(payload) != PROMPTS_PER_CATEGORY:
            raise ValueError(
                f"T2V-CompBench category {category_id!r} has {len(payload)} metadata rows; "
                f"expected exactly {PROMPTS_PER_CATEGORY}: {source}"
            )
        for category_prompt_index, row in enumerate(payload):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"T2V-CompBench metadata row {category_prompt_index} must be an object: {source}"
                )
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"T2V-CompBench metadata row {category_prompt_index} has no prompt: {source}"
                )
            prompt_number = category_prompt_index + 1
            records.append(
                {
                    "sample_id": sample_id(category_id, prompt_number),
                    "category_id": category_id,
                    "prompt_id": prompt_id(category_id, prompt_number),
                    "category_prompt_index": category_prompt_index,
                    "official_prompt_number": prompt_number,
                    "prompt": prompt,
                    "official_video_name": f"{prompt_number:04d}.mp4",
                    "official_video_subdir": protocol["video_subdir"],
                }
            )
    if len(records) != CANONICAL_PROMPT_COUNT:
        raise ValueError(
            f"T2V-CompBench materialized {len(records)} prompts; expected {CANONICAL_PROMPT_COUNT}"
        )
    return records


def materialize_t2v_compbench_generation_requests(
    *,
    limit: int | None = None,
    assets_root: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    """Build model-independent generation requests from the official suite."""

    records = load_t2v_compbench_prompt_records(assets_root=assets_root)
    if limit is not None:
        if isinstance(limit, bool) or int(limit) <= 0:
            raise ValueError("T2V-CompBench generation limit must be a positive integer")
        records = records[: int(limit)]
    return tuple(
        GenerationRequest(
            sample_id=record["sample_id"],
            task_name=BENCHMARK_ID,
            split="standard",
            inputs={
                "prompt": record["prompt"],
                "prompt_id": record["prompt_id"],
                "category_id": record["category_id"],
                "category_prompt_index": record["category_prompt_index"],
                "official_prompt_number": record["official_prompt_number"],
                "official_video_name": record["official_video_name"],
                "official_video_subdir": record["official_video_subdir"],
            },
            output_schema={"generated_video": {"kind": "video"}},
        )
        for record in records
    )


def _canonical_records_by_sample_id(*, assets_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {
        str(record["sample_id"]): record
        for record in load_t2v_compbench_prompt_records(assets_root=assets_root)
    }


def _validated_request_record(
    request: GenerationRequest,
    *,
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = records_by_id.get(request.sample_id)
    if canonical is None:
        raise ValueError(f"unknown T2V-CompBench generation sample_id: {request.sample_id!r}")
    for field in (
        "prompt",
        "prompt_id",
        "category_id",
        "category_prompt_index",
        "official_prompt_number",
        "official_video_name",
        "official_video_subdir",
    ):
        if request.inputs.get(field) != canonical[field]:
            raise ValueError(
                f"T2V-CompBench request {request.sample_id!r} has invalid {field}: "
                f"expected {canonical[field]!r}, got {request.inputs.get(field)!r}"
            )
    return dict(canonical)


def _probe_decodable_mp4(path: Path) -> None:
    if path.suffix.lower() != ".mp4":
        raise ValueError(
            f"T2V-CompBench official staging requires MP4 artifacts; got {path.suffix or '<none>'}: {path}"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"T2V-CompBench generated video is missing or empty: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("T2V-CompBench video validation requires opencv-python") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        decoded, frame = capture.read() if opened else (False, None)
    finally:
        capture.release()
    if not opened or not decoded or frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError(f"T2V-CompBench generated video is not decodable: {path}")


def _safe_relative_video_path(value: Any, *, root: Path) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("T2V-CompBench generation manifest row is missing relative_path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe T2V-CompBench relative_path: {value!r}")
    source = (root / relative).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"T2V-CompBench relative_path escapes generated video root: {value!r}") from exc
    return relative, source


def _manifest_row_from_canonical(
    canonical: Mapping[str, Any],
    *,
    relative_path: str,
    source_uri: str | None,
    request_id: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "sample_id": canonical["sample_id"],
        "request_id": request_id,
        "category_id": canonical["category_id"],
        "prompt_id": canonical["prompt_id"],
        "category_prompt_index": canonical["category_prompt_index"],
        "official_prompt_number": canonical["official_prompt_number"],
        "prompt": canonical["prompt"],
        "official_video_name": canonical["official_video_name"],
        "official_video_subdir": canonical["official_video_subdir"],
        "relative_path": relative_path,
        "source_uri": source_uri,
        "status": status,
        "placeholder": False,
    }


def copy_t2v_compbench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    assets_root: Path | None = None,
) -> tuple[int, int]:
    """Strictly join model results to official category/index video names."""

    generation_output_dir = generation_output_dir.expanduser().resolve()
    requests_path = generation_output_dir / "requests.jsonl"
    results_path = generation_output_dir / "results.jsonl"
    if not requests_path.is_file():
        raise FileNotFoundError(f"T2V-CompBench generation request manifest is missing: {requests_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"T2V-CompBench generation result manifest is missing: {results_path}")

    canonical_by_id = _canonical_records_by_sample_id(assets_root=assets_root)
    requests_by_id: dict[str, GenerationRequest] = {}
    request_records: dict[str, dict[str, Any]] = {}
    official_keys: set[tuple[str, int]] = set()
    for row in read_jsonl_objects(requests_path):
        request = GenerationRequest.from_dict(row)
        if request.sample_id in requests_by_id:
            raise ValueError(f"duplicate T2V-CompBench request sample_id: {request.sample_id!r}")
        record = _validated_request_record(request, records_by_id=canonical_by_id)
        official_key = (str(record["category_id"]), int(record["official_prompt_number"]))
        if official_key in official_keys:
            raise ValueError(f"duplicate T2V-CompBench category/prompt mapping: {official_key!r}")
        official_keys.add(official_key)
        requests_by_id[request.sample_id] = request
        request_records[request.sample_id] = record
    if not requests_by_id:
        raise ValueError(f"T2V-CompBench generation request manifest is empty: {requests_path}")

    results_by_id: dict[str, GenerationResult] = {}
    for row in read_jsonl_objects(results_path):
        result = GenerationResult.from_dict(row)
        if result.sample_id in results_by_id:
            raise ValueError(f"duplicate T2V-CompBench result sample_id: {result.sample_id!r}")
        results_by_id[result.sample_id] = result
    missing = sorted(requests_by_id.keys() - results_by_id.keys())
    unexpected = sorted(results_by_id.keys() - requests_by_id.keys())
    if missing or unexpected:
        raise ValueError(
            "T2V-CompBench generation coverage mismatch: "
            f"missing results={missing[:8]}, unexpected results={unexpected[:8]}"
        )

    destination = generated_artifact_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to mix T2V-CompBench videos with an existing artifact directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".t2v-compbench-stage-", dir=destination.parent))
    manifest_rows: list[dict[str, Any]] = []
    try:
        ordered_ids = sorted(
            requests_by_id,
            key=lambda item: (
                CATEGORY_ORDER.index(str(request_records[item]["category_id"])),
                int(request_records[item]["official_prompt_number"]),
            ),
        )
        for current_sample_id in ordered_ids:
            request = requests_by_id[current_sample_id]
            result = results_by_id[current_sample_id]
            record = request_records[current_sample_id]
            if not is_generation_result_successful(result):
                raise ValueError(
                    f"T2V-CompBench generation failed for {current_sample_id!r}; "
                    f"complete requested coverage is required: {result.error}"
                )
            artifact = result.artifacts.get(output_artifact)
            if artifact is None:
                raise ValueError(
                    f"T2V-CompBench result {current_sample_id!r} has no {output_artifact!r} artifact"
                )
            source = local_path_for_uri(artifact.uri, base_dir=generation_output_dir)
            if source is None or not source.is_file():
                raise FileNotFoundError(
                    f"T2V-CompBench result {current_sample_id!r} does not reference a readable local video: "
                    f"{artifact.uri!r}"
                )
            _probe_decodable_mp4(source)
            relative_path = Path(str(record["official_video_subdir"])) / str(record["official_video_name"])
            target = stage / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            _probe_decodable_mp4(target)
            manifest_rows.append(
                _manifest_row_from_canonical(
                    record,
                    relative_path=relative_path.as_posix(),
                    source_uri=artifact.uri,
                    request_id=result.request_id or request.request_id,
                    status="copied",
                )
            )

        write_jsonl(stage / GENERATION_MANIFEST_NAME, manifest_rows)
        if destination.exists():
            destination.rmdir()
        stage.replace(destination)
        write_jsonl(artifact_manifest_path, manifest_rows)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return len(manifest_rows), 0


def _validate_generation_manifest_row(
    row: Mapping[str, Any],
    *,
    canonical_by_id: Mapping[str, Mapping[str, Any]],
    generated_video_dir: Path,
) -> tuple[dict[str, Any], Path]:
    current_sample_id = row.get("sample_id")
    if not isinstance(current_sample_id, str) or current_sample_id not in canonical_by_id:
        raise ValueError(f"unknown T2V-CompBench manifest sample_id: {current_sample_id!r}")
    canonical = canonical_by_id[current_sample_id]
    for field in (
        "category_id",
        "prompt_id",
        "category_prompt_index",
        "official_prompt_number",
        "prompt",
        "official_video_name",
        "official_video_subdir",
    ):
        if row.get(field) != canonical[field]:
            raise ValueError(
                f"T2V-CompBench manifest {current_sample_id!r} has invalid {field}: "
                f"expected {canonical[field]!r}, got {row.get(field)!r}"
            )
    if row.get("placeholder") is True:
        raise ValueError(f"T2V-CompBench manifest {current_sample_id!r} references a placeholder")
    status = str(row.get("status") or "").strip().lower()
    if status not in {"copied", "materialized", "generated", "succeeded"}:
        raise ValueError(
            f"T2V-CompBench manifest {current_sample_id!r} has non-success status: {status!r}"
        )
    relative_path, source = _safe_relative_video_path(row.get("relative_path"), root=generated_video_dir)
    _probe_decodable_mp4(source)
    normalized = _manifest_row_from_canonical(
        canonical,
        relative_path=relative_path.as_posix(),
        source_uri=None if row.get("source_uri") is None else str(row.get("source_uri")),
        request_id=None if row.get("request_id") is None else str(row.get("request_id")),
        status=status,
    )
    return normalized, source


def categories_from_generation_manifest(
    *,
    generated_video_dir: Path,
    generation_manifest_path: Path | None = None,
    assets_root: Path | None = None,
) -> tuple[str, ...]:
    """Return manifest categories in canonical order after strict validation."""

    report = validate_t2v_compbench_generation_manifest(
        generated_video_dir=generated_video_dir,
        generation_manifest_path=generation_manifest_path,
        assets_root=assets_root,
    )
    present = set(report["by_category"])
    return tuple(category for category in CATEGORY_ORDER if category in present)


def validate_t2v_compbench_generation_manifest(
    *,
    generated_video_dir: Path,
    generation_manifest_path: Path | None = None,
    assets_root: Path | None = None,
) -> dict[str, Any]:
    """Validate exact manifest coverage without guessing mappings from paths."""

    root = generated_video_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"T2V-CompBench generated video directory not found: {root}")
    manifest_path = (
        generation_manifest_path.expanduser().resolve()
        if generation_manifest_path is not None
        else root / GENERATION_MANIFEST_NAME
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "T2V-CompBench generated artifacts require an explicit mapping manifest; "
            f"not found: {manifest_path}"
        )
    rows = read_jsonl_objects(manifest_path)
    if not rows:
        raise ValueError(f"T2V-CompBench generation manifest is empty: {manifest_path}")

    canonical_by_id = _canonical_records_by_sample_id(assets_root=assets_root)
    validated_rows: list[dict[str, Any]] = []
    source_by_id: dict[str, Path] = {}
    seen_official_keys: set[tuple[str, int]] = set()
    seen_relative_paths: set[str] = set()
    for row in rows:
        normalized, source = _validate_generation_manifest_row(
            row,
            canonical_by_id=canonical_by_id,
            generated_video_dir=root,
        )
        current_sample_id = str(normalized["sample_id"])
        if current_sample_id in source_by_id:
            raise ValueError(f"duplicate T2V-CompBench manifest sample_id: {current_sample_id!r}")
        official_key = (
            str(normalized["category_id"]),
            int(normalized["official_prompt_number"]),
        )
        if official_key in seen_official_keys:
            raise ValueError(f"duplicate T2V-CompBench category/prompt mapping: {official_key!r}")
        relative_path = str(normalized["relative_path"])
        if relative_path in seen_relative_paths:
            raise ValueError(f"duplicate T2V-CompBench relative_path: {relative_path!r}")
        source_by_id[current_sample_id] = source
        seen_official_keys.add(official_key)
        seen_relative_paths.add(relative_path)
        validated_rows.append(normalized)

    referenced_paths = {path.resolve() for path in source_by_id.values()}
    actual_video_paths = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    unreferenced = sorted(str(path) for path in actual_video_paths - referenced_paths)
    if unreferenced:
        raise ValueError(
            "T2V-CompBench generated directory contains videos with no manifest mapping: "
            f"{unreferenced[:8]}"
        )

    by_category = Counter(str(row["category_id"]) for row in validated_rows)
    complete_categories = {
        category_id
        for category_id in CATEGORY_ORDER
        if by_category.get(category_id, 0) == PROMPTS_PER_CATEGORY
        and {
            int(row["official_prompt_number"])
            for row in validated_rows
            if row["category_id"] == category_id
        }
        == set(range(1, PROMPTS_PER_CATEGORY + 1))
    }
    full_suite = (
        len(validated_rows) == CANONICAL_PROMPT_COUNT
        and complete_categories == set(CATEGORY_ORDER)
    )
    return {
        "manifest_path": str(manifest_path),
        "generated_video_dir": str(root),
        "sample_count": len(validated_rows),
        "by_category": {category: by_category.get(category, 0) for category in CATEGORY_ORDER if by_category.get(category)},
        "categories": [category for category in CATEGORY_ORDER if by_category.get(category)],
        "complete_categories": [category for category in CATEGORY_ORDER if category in complete_categories],
        "canonical_prompt_count": CANONICAL_PROMPT_COUNT,
        "full_suite": full_suite,
        "bounded": not full_suite,
        "rows": validated_rows,
        "source_by_sample_id": source_by_id,
    }


def materialize_t2v_compbench_official_layout(
    *,
    generated_video_dir: Path,
    official_layout_dir: Path,
    generation_manifest_path: Path | None = None,
    selected_categories: Sequence[str] | None = None,
    assets_root: Path | None = None,
) -> dict[str, Any]:
    """Map generic generated files into the official category/NNNN.mp4 layout."""

    validation = validate_t2v_compbench_generation_manifest(
        generated_video_dir=generated_video_dir,
        generation_manifest_path=generation_manifest_path,
        assets_root=assets_root,
    )
    categories = tuple(selected_categories or validation["categories"])
    if not categories:
        raise ValueError("T2V-CompBench official staging selected zero categories")
    unknown = [category for category in categories if category not in CATEGORY_ORDER]
    if unknown:
        raise ValueError(f"unknown T2V-CompBench staging categories: {unknown}")
    rows = [row for row in validation["rows"] if row["category_id"] in categories]
    counts = Counter(str(row["category_id"]) for row in rows)
    missing_categories = [category for category in categories if counts.get(category, 0) == 0]
    if missing_categories:
        raise ValueError(
            "T2V-CompBench generated manifest has zero coverage for selected categories: "
            f"{missing_categories}"
        )

    destination = official_layout_dir.expanduser().resolve()
    managed_existing_layout = destination / OFFICIAL_LAYOUT_MANIFEST_NAME
    if destination.exists() and any(destination.iterdir()) and not managed_existing_layout.is_file():
        raise FileExistsError(
            f"refusing to mix T2V-CompBench official inputs with an existing directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".t2v-compbench-layout-", dir=destination.parent))
    staged_rows: list[dict[str, Any]] = []
    try:
        source_by_sample_id: Mapping[str, Path] = validation["source_by_sample_id"]
        for row in rows:
            source = source_by_sample_id[str(row["sample_id"])]
            relative_path = Path(str(row["official_video_subdir"])) / str(row["official_video_name"])
            target = stage / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
                materialization = "hardlink"
            except OSError:
                shutil.copy2(source, target)
                materialization = "copy"
            _probe_decodable_mp4(target)
            staged_row = dict(row)
            staged_row["source_relative_path"] = row["relative_path"]
            staged_row["relative_path"] = relative_path.as_posix()
            staged_row["materialization"] = materialization
            staged_rows.append(staged_row)
        write_jsonl(stage / OFFICIAL_LAYOUT_MANIFEST_NAME, staged_rows)
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    selected_full_suite = (
        set(categories) == set(CATEGORY_ORDER)
        and validation["full_suite"] is True
        and len(staged_rows) == CANONICAL_PROMPT_COUNT
    )
    return {
        "source_manifest": validation["manifest_path"],
        "generated_video_dir": validation["generated_video_dir"],
        "official_video_root": str(destination),
        "official_layout_manifest": str(destination / OFFICIAL_LAYOUT_MANIFEST_NAME),
        "sample_count": len(staged_rows),
        "by_category": {category: counts.get(category, 0) for category in categories},
        "selected_categories": list(categories),
        "canonical_prompt_count": CANONICAL_PROMPT_COUNT,
        "full_suite": selected_full_suite,
        "bounded": not selected_full_suite,
    }


__all__ = [
    "BENCHMARK_ID",
    "CANONICAL_PROMPT_COUNT",
    "CATEGORY_ORDER",
    "CATEGORY_PROTOCOL",
    "GENERATION_MANIFEST_NAME",
    "OFFICIAL_LAYOUT_MANIFEST_NAME",
    "PROMPTS_PER_CATEGORY",
    "categories_from_generation_manifest",
    "copy_t2v_compbench_generated_videos",
    "load_t2v_compbench_prompt_records",
    "materialize_t2v_compbench_generation_requests",
    "materialize_t2v_compbench_official_layout",
    "prompt_id",
    "resolve_t2v_compbench_assets",
    "sample_id",
    "validate_t2v_compbench_generation_manifest",
]
