"""In-tree Physics-IQ Original and Verified evaluation runtime."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .official.scoring import IQTable
from .physics_iq_prompts import VIEWS, load_description_rows, unique_generation_records
from .protocols import PhysicsIQProtocolSpec

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
UPSTREAM_REVISION = "b02cf26dc15d559d0ca4f63a6917070312dde185"


@dataclass(frozen=True)
class PhysicsIQDatasetLayout:
    """Resolved official dataset assets for one protocol and one frame rate."""

    root: Path
    fps: int
    reference_videos: Path
    reference_masks: Path


@dataclass(frozen=True)
class PhysicsIQRunConfig:
    protocol: PhysicsIQProtocolSpec
    dataset_root: Path | None
    descriptions_path: Path
    generated_video_dir: Path
    output_dir: Path
    generated_mask_dir: Path | None = None
    raw_metrics_path: Path | None = None
    n_processes: int = 0
    mask_threshold: int = 10
    validate_videos: bool = True
    lazy_integrity: bool = False
    limit: int | None = None


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def resolve_dataset_root(
    protocol: PhysicsIQProtocolSpec,
    explicit: Path | None = None,
) -> Path:
    """Resolve a dataset directory, accepting either it or its parent."""

    protocol_env = (
        "WORLDFOUNDRY_PHYSICS_IQ_VERIFIED_ROOT"
        if protocol.protocol == "verified"
        else "WORLDFOUNDRY_PHYSICS_IQ_ORIGINAL_ROOT"
    )
    framework_data_env = (
        "WORLDFOUNDRY_PHYSICS_IQ_VERIFIED_DATA_ROOT"
        if protocol.protocol == "verified"
        else "WORLDFOUNDRY_PHYSICS_IQ_DATA_ROOT"
    )
    candidate = explicit or _env_path(
        protocol_env,
        framework_data_env,
        "WORLDFOUNDRY_PHYSICS_IQ_DATASET_ROOT",
        "WORLDFOUNDRY_PHYSICS_IQ_ROOT",
        "WORLDFOUNDRY_BENCHMARK_DATA_ROOT",
    )
    if candidate is None:
        raise FileNotFoundError(
            f"{protocol.display_name} needs the official dataset assets. Pass --dataset-root or set "
            f"{protocol_env}. No upstream source checkout is required."
        )
    candidate = candidate.expanduser().resolve()
    roots = (candidate, candidate / protocol.dataset_dir_name)
    for root in roots:
        if (root / "split-videos" / "testing").is_dir():
            return root
    raise FileNotFoundError(
        f"Could not find split-videos/testing below {candidate} for {protocol.display_name}."
    )


def detect_video_fps(video_dir: Path) -> int:
    """Return the shared integral FPS used by generated videos."""

    import cv2

    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"No generated videos found in {video_dir}.")
    values: set[int] = set()
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        rounded = round(fps)
        if fps <= 0 or not math.isclose(fps, rounded, abs_tol=0.05):
            raise ValueError(f"Physics-IQ requires an integral FPS; {path.name} reports {fps}.")
        values.add(int(rounded))
    if len(values) != 1:
        raise ValueError(f"Generated videos use inconsistent FPS values: {sorted(values)}")
    return values.pop()


def _video_duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frames / fps if fps > 0 else 0.0


def _scenario_key_and_view(record: dict[str, str]) -> tuple[str, str]:
    scenario_name = Path(str(record.get("scenario") or "")).name
    scenario_parts = scenario_name.split("_", 3)
    if len(scenario_parts) != 4:
        raise ValueError(f"Unexpected Physics-IQ scenario filename: {scenario_name!r}")
    _file_id, view, take, scenario_key = scenario_parts
    if take != "take-1" or view not in VIEWS:
        raise ValueError(f"Unexpected Physics-IQ scenario filename: {scenario_name!r}")

    generated_name = Path(str(record.get("generated_video_name") or "")).name
    generated_parts = generated_name.split("_", 2)
    if (
        len(generated_parts) != 3
        or generated_parts[1] != view
        or generated_parts[2] != scenario_key
    ):
        raise ValueError(
            "Physics-IQ description row does not map its scenario to the generated video: "
            f"scenario={scenario_name!r}, generated_video_name={generated_name!r}."
        )
    return scenario_key, view


def select_complete_scenario_records(
    *,
    descriptions_path: Path,
    protocol: PhysicsIQProtocolSpec,
    limit: int | None,
) -> tuple[tuple[dict[str, str], ...], frozenset[str]]:
    """Select a bounded prefix containing every view of each chosen scenario."""

    records = unique_generation_records(
        load_description_rows(descriptions_path=descriptions_path, spec=protocol)
    )
    if limit is not None:
        if limit <= 0 or limit % len(VIEWS) != 0:
            raise ValueError(
                "Physics-IQ official scoring requires complete three-view scenario groups; "
                f"--limit must be a positive multiple of {len(VIEWS)}, got {limit}."
            )
        if limit > len(records):
            raise ValueError(
                f"Physics-IQ --limit {limit} exceeds the {len(records)} available videos."
            )
        records = records[:limit]

    grouped: dict[str, list[str]] = {}
    for record in records:
        scenario_key, view = _scenario_key_and_view(record)
        grouped.setdefault(scenario_key, []).append(view)
    expected_views = set(VIEWS)
    incomplete = {
        scenario_key: views
        for scenario_key, views in grouped.items()
        if len(views) != len(VIEWS) or set(views) != expected_views
    }
    if incomplete:
        examples = list(incomplete.items())[:3]
        raise ValueError(
            "Physics-IQ official scoring requires exactly one left, center, and right view "
            f"for every selected scenario; incomplete groups: {examples}."
        )
    return tuple(records), frozenset(grouped)


def _expected_video_names(records: Iterable[dict[str, str]]) -> list[str]:
    return [str(record["generated_video_name"]) for record in records]


def stage_generated_videos(
    *,
    source_dir: Path,
    staging_dir: Path,
    records: Iterable[dict[str, str]],
    validate: bool,
) -> dict[str, Any]:
    """Create official filenames without renaming or mutating model outputs."""

    expected = _expected_video_names(records)
    source_videos = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    exact = {path.name: path for path in source_videos}
    by_prefix: dict[str, list[Path]] = {}
    for path in source_videos:
        by_prefix.setdefault(path.name[:4], []).append(path)

    if staging_dir.exists():
        if staging_dir.is_symlink():
            staging_dir.unlink()
        else:
            shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    ambiguous: list[str] = []
    for expected_name in expected:
        source = exact.get(expected_name)
        if source is None:
            matches = by_prefix.get(expected_name[:4], [])
            if len(matches) == 1:
                source = matches[0]
            elif len(matches) > 1:
                ambiguous.append(expected_name)
                continue
        if source is None:
            missing.append(expected_name)
            continue
        target = staging_dir / expected_name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)

    if missing or ambiguous:
        raise ValueError(
            f"Generated-video coverage is incomplete: {len(missing)} missing, "
            f"{len(ambiguous)} ambiguous. First missing: {missing[:3]}; "
            f"first ambiguous: {ambiguous[:3]}."
        )
    if validate:
        invalid = [
            (name, _video_duration(staging_dir / name))
            for name in expected
            if not math.isclose(_video_duration(staging_dir / name), 5.0, abs_tol=0.05)
        ]
        if invalid:
            raise ValueError(f"Physics-IQ videos must be 5 seconds long; examples: {invalid[:3]}")
    return {
        "expected_count": len(expected),
        "source_count": len(source_videos),
        "staged_count": len(expected),
        "staging_dir": str(staging_dir.resolve()),
    }


def _ensure_reference_assets(
    *,
    dataset_root: Path,
    fps: int,
    output_dir: Path,
) -> PhysicsIQDatasetLayout:
    def checked_layout(reference: Path, masks: Path) -> PhysicsIQDatasetLayout:
        reference_count = len(list(reference.glob("*.mp4")))
        mask_count = len(list(masks.glob("*.mp4")))
        if reference_count < 396 or mask_count < 396:
            raise FileNotFoundError(
                "Physics-IQ reference assets are incomplete: expected 396 take/view videos and masks, "
                f"found {reference_count} videos and {mask_count} masks at {fps}FPS."
            )
        return PhysicsIQDatasetLayout(dataset_root, fps, reference, masks)

    reference = dataset_root / "split-videos" / "testing" / f"{fps}FPS"
    masks = dataset_root / "video-masks" / "real" / f"{fps}FPS"
    if reference.is_dir() and masks.is_dir():
        return checked_layout(reference, masks)

    source = dataset_root / "split-videos" / "testing" / "30FPS"
    if not source.is_dir():
        raise FileNotFoundError(f"Official 30FPS reference videos not found: {source}")
    reference = output_dir / "reference-cache" / f"{fps}FPS"
    if not reference.is_dir() or not any(reference.glob("*.mp4")):
        from .official.fps import change_video_fps

        change_video_fps(str(source), str(reference), fps)
    masks = output_dir / "reference-mask-cache" / f"{fps}FPS"
    if not masks.is_dir() or not any(masks.glob("*.mp4")):
        from .official.masks import generate_binary_masks

        generate_binary_masks(str(reference), str(masks), True)
    return checked_layout(reference, masks)


def score_raw_metrics_csv(
    path: Path,
    *,
    lazy_integrity: bool = False,
) -> dict[str, float]:
    """Apply both official score formulas to an official raw metric table."""

    table = IQTable.from_csv(str(path), lazy_integrity=lazy_integrity)
    return {key: float(value) for key, value in table.get_output_dict().items()}


def _ensure_generated_masks(
    *,
    generated_dir: Path,
    mask_dir: Path,
    threshold: int,
    owned_output: bool,
) -> Path:
    from .official.masks import generate_binary_masks

    source_videos = sorted(generated_dir.glob("*.mp4"))
    expected_count = len(source_videos)
    source_signature = {
        "threshold": threshold,
        "videos": [
            {
                "name": path.name,
                "target": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in source_videos
        ],
    }
    signature_path = mask_dir / ".worldfoundry-mask-sources.json"
    existing_count = len(list(mask_dir.glob("*.mp4"))) if mask_dir.is_dir() else 0
    cache_matches = False
    if owned_output and signature_path.is_file():
        try:
            cache_matches = json.loads(signature_path.read_text(encoding="utf-8")) == source_signature
        except (OSError, json.JSONDecodeError):
            cache_matches = False
    needs_generation = existing_count != expected_count or (owned_output and not cache_matches)
    if needs_generation:
        if mask_dir.exists():
            if not owned_output:
                raise ValueError(
                    f"Caller-provided generated mask directory has {existing_count} masks; "
                    f"expected {expected_count}: {mask_dir}"
                )
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        generate_binary_masks(str(generated_dir), str(mask_dir), False, threshold)
        signature_path.write_text(
            json.dumps(source_signature, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    actual_count = len(list(mask_dir.glob("*.mp4")))
    if actual_count != expected_count:
        raise RuntimeError(f"Generated {actual_count} masks for {expected_count} generated videos.")
    return mask_dir


def run_physics_iq_evaluation(config: PhysicsIQRunConfig) -> dict[str, Any]:
    """Run raw video metrics and official Original/Verified score aggregation."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records, selected_scenarios = select_complete_scenario_records(
        descriptions_path=config.descriptions_path,
        protocol=config.protocol,
        limit=config.limit,
    )
    metric_only = config.raw_metrics_path is not None
    if metric_only:
        dataset_root = config.dataset_root.resolve() if config.dataset_root is not None else None
        raw_metrics_path = config.raw_metrics_path.expanduser().resolve()
        if not raw_metrics_path.is_file():
            raise FileNotFoundError(f"Raw Physics-IQ metrics not found: {raw_metrics_path}")
        staging_summary: dict[str, Any] | None = None
        layout: PhysicsIQDatasetLayout | None = None
        fps: int | None = None
    else:
        dataset_root = resolve_dataset_root(config.protocol, config.dataset_root)
        staging_dir = output_dir / "staged-generated-videos"
        staging_summary = stage_generated_videos(
            source_dir=config.generated_video_dir,
            staging_dir=staging_dir,
            records=selected_records,
            validate=config.validate_videos,
        )
        fps = detect_video_fps(staging_dir)
        layout = _ensure_reference_assets(
            dataset_root=dataset_root,
            fps=fps,
            output_dir=output_dir,
        )
        generated_masks = _ensure_generated_masks(
            generated_dir=staging_dir,
            mask_dir=(config.generated_mask_dir or output_dir / "generated-masks").resolve(),
            threshold=config.mask_threshold,
            owned_output=config.generated_mask_dir is None,
        )
        raw_metrics_path = output_dir / "raw_metrics.csv"
        from .official.raw_metrics import process_videos

        process_videos(
            real_folder=str(layout.reference_videos),
            generated_folder=str(staging_dir),
            binary_real_folder=str(layout.reference_masks),
            binary_generated_folder=str(generated_masks),
            csv_file_path=str(raw_metrics_path),
            fps=fps,
            video_time_selection="first",
            n_processes=config.n_processes,
            selected_scenarios=(
                set(selected_scenarios) if config.limit is not None else None
            ),
        )
        if not raw_metrics_path.is_file():
            raise RuntimeError("Official raw metric engine did not produce raw_metrics.csv.")

    scores = score_raw_metrics_csv(raw_metrics_path, lazy_integrity=config.lazy_integrity)
    metrics_path = output_dir / "official_metrics.json"
    metrics_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "backend": "in_tree_official_score_protocol" if metric_only else "in_tree_official_raw_video_scorer",
        "protocol": config.protocol.protocol,
        "benchmark_id": config.protocol.benchmark_id,
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "fps": fps,
        "raw_metrics_path": str(raw_metrics_path.resolve()),
        "results_path": str(metrics_path.resolve()),
        "scores": scores,
        "primary_score_key": config.protocol.primary_score_key,
        "primary_score": scores[config.protocol.primary_score_key],
        "staging": staging_summary,
        "reference_videos": None if layout is None else str(layout.reference_videos),
        "reference_masks": None if layout is None else str(layout.reference_masks),
        "upstream_revision": UPSTREAM_REVISION,
    }


def discover_physics_iq_results(search_roots: Iterable[Path]) -> Path | None:
    """Find a metrics JSON/CSV or raw metric table for normalization-only runs."""

    names = (
        "official_metrics.json",
        "raw_metrics.csv",
        "physics_iq_results.csv",
        "physics_iq_results.json",
        "results_summary.csv",
    )
    for root in search_roots:
        if root.is_file() and root.suffix.lower() in {".csv", ".json"}:
            return root
        if not root.is_dir():
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None
