"""Generic video benchmark contract evaluation helpers.

This module provides metrics evaluation, metadata verification, video readability probes,
and adapters to ingest/normalize official scores for physics-grounded benchmarks.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.serialization import write_json, write_jsonl
from worldfoundry.core.time import utc_now_iso
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION

from worldfoundry.evaluation.tasks.execution.framework.io import optional_float
from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import OfficialMetricScore


JsonValue = Any
RECORD_FILE_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".csv"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
LOCAL_METRIC_IDS = (
    "artifact_manifest_coverage",
    "generated_video_exists",
    "generated_video_nonempty",
    "generated_video_readable",
    "generated_video_duration_check",
    "generated_video_fps_check",
    "generated_video_resolution_check",
    "metadata_consistency",
    "mcqa_accuracy",
)
OFFICIAL_BLOCKED_REASON = "Official benchmark scoring requires benchmark-specific official assets; WorldFoundry in-tree evaluator does not fabricate this score."
NO_MANIFEST_REASON = "No prompt/question manifest was supplied, so manifest-dependent checks are not available."
NO_PROBE_REASON = "Video probe metadata is unavailable; provide ffprobe on PATH or a sidecar JSON metadata file."




def _path_value(values: Mapping[str, JsonValue], *keys: str) -> Path | None:
    """Resolve the first non-empty path-like value from a keyword mapping.

    Args:
        values: Keyword argument mapping.
        keys: Candidate key names to inspect.
    """

    for key in keys:
        value = values.get(key)
        if value not in (None, ""):
            return Path(str(value))
    return None


def _prompt_manifest_path(
    benchmark_id: str,
    values: Mapping[str, JsonValue],
    extra_keys: tuple[str, ...] = (),
) -> Path | None:
    """Find the prompt or question manifest path from benchmark-specific aliases.

    Args:
        benchmark_id: Benchmark identifier used for benchmark-data-root defaults.
        values: Runner keyword arguments supplied by the caller.
    """

    explicit = _path_value(
        values,
        "prompt_manifest",
        "question_manifest",
        "physics_prompt_manifest",
        "videophy_prompt_manifest",
        "videophy2_prompt_manifest",
        "phyground_prompt_manifest",
        "prompt_suite_json",
        "generated_artifact_manifest",
        *extra_keys,
    )
    if explicit is not None:
        return explicit

    data_root = _benchmark_data_root(benchmark_id, values)
    if data_root is None:
        return None
    normalized = benchmark_id.replace("-", "_")
    candidates = (
        data_root / "prompts" / f"{benchmark_id}.json",
        data_root / "prompts" / f"{normalized}.json",
        data_root / benchmark_id / "prompts" / f"{benchmark_id}.json",
        data_root / normalized / "prompts" / f"{normalized}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _benchmark_data_root(benchmark_id: str, values: Mapping[str, JsonValue]) -> Path | None:
    """Resolve an optional user-supplied benchmark data root.

    Args:
        benchmark_id: Benchmark identifier used to derive environment aliases.
        values: Runner keyword arguments supplied by the caller.
    """

    root = _path_value(values, "benchmark_data_root", "data_root", "dataset_root")
    if root is not None:
        return root
    env_prefix = "".join(char if char.isalnum() else "_" for char in benchmark_id).upper()
    for env_name in (f"WORLDFOUNDRY_{env_prefix}_DATA_ROOT", "WORLDFOUNDRY_BENCHMARK_DATA_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    return None


def _text_value(values: Mapping[str, JsonValue], *keys: str) -> str | None:
    """Resolve the first non-empty text value from a keyword mapping."""

    for key in keys:
        value = values.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _load_records(path: Path) -> list[Mapping[str, JsonValue]]:
    """Load JSON, JSONL, or YAML manifest records.

    Args:
        path: Prompt, question, generated-artifact, or judge result manifest.
    """

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if isinstance(payload, Mapping):
        for key in ("samples", "records", "items", "data", "questions", "prompts", "results", "annotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _load_official_records(path: Path | None) -> list[Mapping[str, JsonValue]]:
    """Load optional official result records when a caller supplies them.

    Args:
        path: JSON, JSONL, YAML, or CSV result file from the official scorer.
    """

    if path is None or not path.exists():
        return []
    if path.is_dir():
        records: list[Mapping[str, JsonValue]] = []
        files = (
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in RECORD_FILE_SUFFIXES
        )
        for item in sorted(files):
            records.extend(_load_official_records(item))
        return records
    if path.suffix.lower() == ".csv":
        return [dict(row) for row in csv.DictReader(path.read_text(encoding="utf-8").splitlines())]
    return _load_records(path)


def _load_judge_rows(path: Path | None) -> dict[str, Mapping[str, JsonValue]]:
    """Load optional judge outputs keyed by sample id.

    Args:
        path: Optional JSON, JSONL, or YAML judge results path.
    """

    if path is None or not path.exists() or path.suffix.lower() == ".csv":
        return {}
    rows = _load_official_records(path)
    result: dict[str, Mapping[str, JsonValue]] = {}
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id:
            result[sample_id] = row
    return result


def _generated_video_files(root: Path | None) -> list[Path]:
    """List generated video files below an artifact directory.

    Args:
        root: Optional generated artifact directory.
    """

    if root is None or not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def _samples(
    prompt_records: list[Mapping[str, JsonValue]],
    judge_rows: Mapping[str, Mapping[str, JsonValue]],
    artifact_root: Path | None,
    generated_files: list[Path],
) -> list[dict[str, JsonValue]]:
    """Build canonical sample records from prompt manifests and generated artifacts.

    Args:
        prompt_records: Records loaded from the prompt/question manifest.
        judge_rows: Optional judge outputs keyed by sample id.
        artifact_root: Generated artifact directory.
        generated_files: Discovered generated video files.
    """

    if prompt_records:
        return [_sample_from_prompt(row, judge_rows, artifact_root, generated_files) for row in prompt_records]
    return [
        {
            "sample_id": path.stem,
            "prompt": None,
            "question": None,
            "reference": None,
            "judge_response": None,
            "video_path": path,
            "manifest_record": {},
        }
        for path in generated_files
    ]


def _sample_from_prompt(
    row: Mapping[str, JsonValue],
    judge_rows: Mapping[str, Mapping[str, JsonValue]],
    artifact_root: Path | None,
    generated_files: list[Path],
) -> dict[str, JsonValue]:
    """Convert one prompt/question row into the evaluator's canonical sample shape.

    Args:
        row: Prompt or question manifest row.
        judge_rows: Optional judge outputs keyed by sample id.
        artifact_root: Generated artifact directory.
        generated_files: Generated video files discovered from disk.
    """

    sample_id = _sample_id(row) or f"sample-{len(generated_files)}"
    judge_row = judge_rows.get(sample_id, {})
    video_path = _video_path(row, artifact_root, sample_id, generated_files)
    return {
        "sample_id": sample_id,
        "prompt": _first_value(row, "prompt", "caption", "text"),
        "question": _first_value(row, "question", "query"),
        "reference": _first_value(row, "label", "labels", "answer", "gold_answer", "reference", "target"),
        "judge_response": _first_value(judge_row, "response", "answer", "prediction", "judge_response")
        or _first_value(row, "judge_response", "prediction", "model_answer"),
        "video_path": video_path,
        "manifest_record": dict(row),
    }


def _sample_id(row: Mapping[str, JsonValue]) -> str | None:
    """Extract a stable sample identifier from common manifest fields.

    Args:
        row: Prompt, question, artifact, or judge record.
    """

    value = _first_value(row, "sample_id", "id_stem", "prompt_id", "id", "question_id", "video_id", "video", "uid")
    return None if value in (None, "") else str(value)


def _first_value(row: Mapping[str, JsonValue], *keys: str) -> JsonValue:
    """Return the first present value for a set of alias keys.

    Args:
        row: Source mapping.
        keys: Candidate field names.
    """

    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _video_path(
    row: Mapping[str, JsonValue],
    artifact_root: Path | None,
    sample_id: str,
    generated_files: list[Path],
) -> Path | None:
    """Resolve the generated video path for a sample.

    Args:
        row: Prompt or artifact manifest row.
        artifact_root: Generated artifact directory.
        sample_id: Canonical sample identifier.
        generated_files: Generated files used for stem-based lookup.
    """

    raw_path = _first_value(row, "generated_video_path", "video_path", "generated_video", "artifact_path", "path")
    if raw_path not in (None, ""):
        path = Path(str(raw_path))
        return path if path.is_absolute() or artifact_root is None else artifact_root / path
    for path in generated_files:
        if path.stem == sample_id:
            return path
    if artifact_root is not None:
        return artifact_root / f"{sample_id}.mp4"
    return None


def _local_metric_rows(
    samples: list[Mapping[str, JsonValue]],
    prompt_manifest_path: Path | None,
    generated_files: list[Path],
    kwargs: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    """Compute local manifest, artifact, probe, metadata, and MCQA metric rows.

    Args:
        samples: Canonical samples to score.
        prompt_manifest_path: Optional prompt manifest path.
        generated_files: Generated video files discovered from disk.
        kwargs: Optional caller-supplied video constraints.
    """

    probes = [_probe_sample(sample) for sample in samples]
    return [
        _manifest_coverage_row(samples, prompt_manifest_path, generated_files),
        _boolean_row("generated_video_exists", samples, [_sample_video_path(sample) is not None and _sample_video_path(sample).exists() for sample in samples]),
        _boolean_row("generated_video_nonempty", samples, [_is_nonempty(_sample_video_path(sample)) for sample in samples]),
        _readability_row(samples, probes),
        _probe_check_row("generated_video_duration_check", samples, probes, kwargs, "duration_seconds"),
        _probe_check_row("generated_video_fps_check", samples, probes, kwargs, "fps"),
        _resolution_check_row(samples, probes, kwargs),
        _metadata_consistency_row(samples, prompt_manifest_path),
        _mcqa_row(samples),
    ]


def _manifest_coverage_row(
    samples: list[Mapping[str, JsonValue]],
    prompt_manifest_path: Path | None,
    generated_files: list[Path],
) -> dict[str, JsonValue]:
    """Score whether manifest samples have corresponding generated videos.

    Args:
        samples: Canonical samples to inspect.
        prompt_manifest_path: Optional prompt manifest path.
        generated_files: Generated video files discovered from disk.
    """

    if prompt_manifest_path is None:
        return _not_available_row("artifact_manifest_coverage", NO_MANIFEST_REASON, {"generated_video_count": len(generated_files)})
    checks = [_sample_video_path(sample) is not None and _sample_video_path(sample).exists() for sample in samples]
    return _boolean_row(
        "artifact_manifest_coverage",
        samples,
        checks,
        evidence={"prompt_manifest": str(prompt_manifest_path), "generated_video_count": len(generated_files)},
    )


def _boolean_row(
    metric_id: str,
    samples: list[Mapping[str, JsonValue]],
    checks: list[bool],
    *,
    evidence: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Aggregate boolean per-sample checks into a metric row.

    Args:
        metric_id: Local metric id.
        samples: Canonical samples checked.
        checks: Boolean per-sample pass/fail values.
        evidence: Optional extra evidence fields.
    """

    total = len(checks)
    passed = sum(1 for item in checks if item)
    score = None if total == 0 else passed / total
    status = "not_available" if total == 0 else "passed" if passed == total else "failed"
    blocked_reason = "No generated artifacts or manifest samples were available." if total == 0 else None
    return {
        "metric_id": metric_id,
        "score": score,
        "value": {"passed": passed, "total": total},
        "status": status,
        "evidence": {
            "sample_ids": [str(sample["sample_id"]) for sample in samples],
            **dict(evidence or {}),
        },
        "blocked_reason": blocked_reason,
    }


def _sample_video_path(sample: Mapping[str, JsonValue]) -> Path | None:
    """Return a sample's video path as a Path object when present.

    Args:
        sample: Canonical sample mapping.
    """

    value = sample.get("video_path")
    return value if isinstance(value, Path) else None


def _is_nonempty(path: Path | None) -> bool:
    """Check whether a generated video path exists and has bytes.

    Args:
        path: Optional generated video path.
    """

    return path is not None and path.exists() and path.stat().st_size > 0


def _probe_sample(sample: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Probe one sample's video metadata using sidecar JSON or ffprobe.

    Args:
        sample: Canonical sample mapping containing a video path.
    """

    path = _sample_video_path(sample)
    if path is None or not path.exists():
        return {"available": False, "readable": False, "source": "missing_video", "reason": "generated video is missing"}
    sidecar = _sidecar_path(path)
    if sidecar is not None:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return {"available": True, "readable": bool(payload.get("readable", True)), "source": str(sidecar), **payload}
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return {"available": False, "readable": None, "source": None, "reason": NO_PROBE_REASON}
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"available": True, "readable": False, "source": "ffprobe", "reason": completed.stderr.strip()}
    return {"available": True, "readable": True, "source": "ffprobe", **_ffprobe_metadata(completed.stdout)}


def _sidecar_path(path: Path) -> Path | None:
    """Find a local JSON metadata sidecar for deterministic validation tests and offline probes.

    Args:
        path: Generated video path.
    """

    candidates = (
        path.with_name(f"{path.name}.json"),
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}.metadata.json"),
        path.with_name(f"{path.stem}.probe.json"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ffprobe_metadata(payload: str) -> dict[str, JsonValue]:
    """Normalize ffprobe JSON into stable video metadata fields.

    Args:
        payload: ffprobe JSON stdout.
    """

    data = json.loads(payload)
    streams = data.get("streams") if isinstance(data, Mapping) else None
    stream = streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], Mapping) else {}
    fps = _rate_value(stream.get("avg_frame_rate"))
    duration = optional_float(stream.get("duration"))
    frame_count = _int_value(stream.get("nb_frames"))
    if duration is None and frame_count is not None and fps not in (None, 0):
        duration = frame_count / fps
    return {
        "width": _int_value(stream.get("width")),
        "height": _int_value(stream.get("height")),
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
    }


def _rate_value(value: JsonValue) -> float | None:
    """Convert an ffprobe rational frame-rate value into float fps.

    Args:
        value: Raw ffprobe rate value.
    """

    if value in (None, "", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denom = float(denominator)
        return None if denom == 0 else float(numerator) / denom
    return float(text)


def _int_value(value: JsonValue) -> int | None:
    """Convert numeric metadata to int when present.

    Args:
        value: Raw numeric value.
    """

    numeric = optional_float(value)
    return None if numeric is None else int(numeric)


def _readability_row(samples: list[Mapping[str, JsonValue]], probes: list[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Score generated video readability based on local probe evidence.

    Args:
        samples: Canonical samples to inspect.
        probes: Probe results aligned with samples.
    """

    if not probes:
        return _not_available_row("generated_video_readable", "No generated video samples were available.")
    available = [probe for probe in probes if probe.get("available") is True]
    if not available:
        return _not_available_row("generated_video_readable", NO_PROBE_REASON, {"sample_count": len(samples)})
    checks = [probe.get("readable") is True for probe in available]
    return _boolean_row(
        "generated_video_readable",
        samples,
        checks,
        evidence={"probe_sources": [probe.get("source") for probe in available]},
    )


def _probe_check_row(
    metric_id: str,
    samples: list[Mapping[str, JsonValue]],
    probes: list[Mapping[str, JsonValue]],
    kwargs: Mapping[str, JsonValue],
    field: str,
) -> dict[str, JsonValue]:
    """Score duration or fps metadata availability and optional bounds.

    Args:
        metric_id: Metric id for the probe check.
        samples: Canonical samples to inspect.
        probes: Probe results aligned with samples.
        kwargs: Optional global constraints.
        field: Probe metadata field to evaluate.
    """

    usable = [probe for probe in probes if probe.get("available") is True and probe.get("readable") is True and probe.get(field) is not None]
    if not usable:
        return _not_available_row(metric_id, NO_PROBE_REASON, {"field": field, "sample_count": len(samples)})
    checks = [_value_in_bounds(float(probe[field]), kwargs, field) for probe in usable]
    return _boolean_row(metric_id, samples, checks, evidence={"values": [probe[field] for probe in usable], "field": field})


def _resolution_check_row(
    samples: list[Mapping[str, JsonValue]],
    probes: list[Mapping[str, JsonValue]],
    kwargs: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Score width and height metadata availability and optional bounds.

    Args:
        samples: Canonical samples to inspect.
        probes: Probe results aligned with samples.
        kwargs: Optional global resolution constraints.
    """

    usable = [
        probe
        for probe in probes
        if probe.get("available") is True
        and probe.get("readable") is True
        and probe.get("width") is not None
        and probe.get("height") is not None
    ]
    if not usable:
        return _not_available_row("generated_video_resolution_check", NO_PROBE_REASON, {"sample_count": len(samples)})
    checks = [
        _value_in_bounds(float(probe["width"]), kwargs, "width")
        and _value_in_bounds(float(probe["height"]), kwargs, "height")
        for probe in usable
    ]
    return _boolean_row(
        "generated_video_resolution_check",
        samples,
        checks,
        evidence={"values": [{"width": probe["width"], "height": probe["height"]} for probe in usable]},
    )


def _value_in_bounds(value: float, constraints: Mapping[str, JsonValue], field: str) -> bool:
    """Evaluate optional expected/min/max constraints for one numeric value.

    Args:
        value: Numeric probe value.
        constraints: Caller-supplied global constraints.
        field: Base field name such as fps or width.
    """

    expected = optional_float(constraints.get(f"expected_{field}", constraints.get(field)))
    tolerance = optional_float(
        constraints.get(f"{field}_tolerance", constraints.get(f"{field}_tolerance_seconds", 0.0))
    ) or 0.0
    minimum = optional_float(constraints.get(f"min_{field}"))
    maximum = optional_float(constraints.get(f"max_{field}"))
    expected_ok = True if expected is None else abs(value - expected) <= tolerance
    minimum_ok = True if minimum is None else value >= minimum
    maximum_ok = True if maximum is None else value <= maximum
    return expected_ok and minimum_ok and maximum_ok


def _metadata_consistency_row(
    samples: list[Mapping[str, JsonValue]],
    prompt_manifest_path: Path | None,
) -> dict[str, JsonValue]:
    """Score whether manifest ids and generated video stems agree.

    Args:
        samples: Canonical samples to inspect.
        prompt_manifest_path: Optional prompt manifest path.
    """

    if prompt_manifest_path is None:
        return _not_available_row("metadata_consistency", NO_MANIFEST_REASON)
    checks = []
    for sample in samples:
        path = _sample_video_path(sample)
        checks.append(path is not None and path.stem == str(sample["sample_id"]))
    return _boolean_row("metadata_consistency", samples, checks, evidence={"prompt_manifest": str(prompt_manifest_path)})


def _mcqa_row(samples: list[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Score multiple-choice answer accuracy when labels and judge responses are present.

    Args:
        samples: Canonical samples containing optional reference and judge response values.
    """

    scored = []
    for sample in samples:
        reference = _choice(sample.get("reference"))
        response = _choice(sample.get("judge_response"))
        if reference is not None and response is not None:
            scored.append(reference == response)
    if not scored:
        return _not_available_row(
            "mcqa_accuracy",
            "No sample has both a multiple-choice reference label and an optional judge response.",
        )
    return _boolean_row("mcqa_accuracy", samples, scored)


def _official_metric_rows(
    benchmark_id: str,
    metric_ids: tuple[str, ...],
    official_results_path: Path | None,
    *,
    official_requirements: Mapping[str, Mapping[str, JsonValue]],
    compute_official_scores,
    filter_official_records=None,
    generated_artifact_dir: Path | None = None,
    result_model_id: str | None = None,
) -> list[dict[str, JsonValue]]:
    records = _filter_official_records(
        _load_official_records(official_results_path),
        filter_fn=filter_official_records,
        generated_artifact_dir=generated_artifact_dir,
        result_model_id=result_model_id,
    )
    scores = compute_official_scores(records, official_results_path)
    return [
        _official_scored_metric_row(metric_id, scores[metric_id])
        if metric_id in scores
        else _blocked_metric_row(metric_id, official_requirements.get(benchmark_id, {}), official_results_path)
        for metric_id in metric_ids
    ]


def _filter_official_records(
    records: list[Mapping[str, JsonValue]],
    *,
    filter_fn=None,
    generated_artifact_dir: Path | None = None,
    result_model_id: str | None = None,
) -> list[Mapping[str, JsonValue]]:
    if filter_fn is None:
        return records
    return filter_fn(records, generated_artifact_dir=generated_artifact_dir, result_model_id=result_model_id)




def _choice(value: JsonValue) -> str | None:
    """Normalize a multiple-choice answer to its leading option letter.

    Args:
        value: Reference label or judge response value.
    """

    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    first = text[0].upper()
    return first if first in {"A", "B", "C", "D", "E"} else None


def _not_available_row(
    metric_id: str,
    reason: str,
    evidence: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Build a not-available metric row with explicit blocked reason.

    Args:
        metric_id: Metric id.
        reason: Human-readable blocked or unavailable reason.
        evidence: Optional evidence payload.
    """

    return {
        "metric_id": metric_id,
        "score": None,
        "value": None,
        "status": "not_available",
        "evidence": dict(evidence or {}),
        "blocked_reason": reason,
    }


def _official_scored_metric_row(metric_id: str, score: OfficialMetricScore) -> dict[str, JsonValue]:
    """Build a scored row imported from official judge/scorer outputs.

    Args:
        metric_id: Official leaderboard metric id.
        score: Normalized official metric score and provenance.
    """

    return {
        "metric_id": metric_id,
        "score": score.score,
        "value": score.raw_value,
        "status": "passed",
        "evidence": {"source": "official_results_import", **dict(score.evidence)},
        "blocked_reason": None,
    }


def _official_blocked_reason(requirements: Mapping[str, JsonValue]) -> str:
    """Return benchmark-specific official dependency text.

    Args:
        benchmark_id: Benchmark identifier.
    """

    spec = requirements
    inputs = spec.get("required_inputs")
    required = "; ".join(str(item) for item in inputs) if isinstance(inputs, list) else "official judge response or official normalized result file"
    return f"{OFFICIAL_BLOCKED_REASON} Required official inputs: {required}."


def _blocked_metric_row(
    metric_id: str,
    requirements: Mapping[str, JsonValue],
    official_results_path: Path | None = None,
) -> dict[str, JsonValue]:
    """Build a blocked row for official judge-backed metrics.

    Args:
        metric_id: Official leaderboard metric id.
        benchmark_id: Benchmark identifier.
        official_results_path: Optional official result file that was attempted.
    """

    spec = requirements
    required_inputs = spec.get("required_inputs")
    return {
        "metric_id": metric_id,
        "score": None,
        "value": None,
        "status": "blocked",
        "evidence": {
            "reason": spec.get("reason", "judge_required"),
            "required_inputs": required_inputs if isinstance(required_inputs, list) else ["official judge response or official normalized result file"],
            "official_results_path": None if official_results_path is None else str(official_results_path),
        },
        "blocked_reason": _official_blocked_reason(spec),
    }


def _per_metric(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Convert a raw metric row into scorecard per_metric metadata.

    Args:
        row: Raw metric row.
    """

    return {
        "available": row["status"] in {"passed", "failed"},
        "score": row.get("score"),
        "value": row.get("value"),
        "status": row.get("status"),
        "evidence": row.get("evidence", {}),
        "blocked_reason": row.get("blocked_reason"),
    }

def write_video_contract_evaluation(
    *,
    benchmark_id: str,
    display_name: str,
    official_metric_ids: tuple[str, ...],
    output_dir: str | Path,
    official_requirements: Mapping[str, Mapping[str, JsonValue]],
    compute_official_scores,
    evaluator_kind: str,
    prompt_manifest_keys: tuple[str, ...] = (),
    filter_official_records=None,
    generated_artifact_dir: str | Path | None = None,
    manifest: Mapping[str, JsonValue] | None = None,
    runner: str = "benchmark_zoo_contract_evaluator",
    mode: str = "contract",
    **kwargs: JsonValue,
) -> dict[str, JsonValue]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = None if generated_artifact_dir is None else Path(generated_artifact_dir)
    prompt_manifest_path = _prompt_manifest_path(benchmark_id, kwargs, prompt_manifest_keys)
    prompt_records = _load_records(prompt_manifest_path) if prompt_manifest_path is not None else []
    official_results_path = _path_value(kwargs, "official_results_path", "judge_results_path")
    judge_rows = _load_judge_rows(official_results_path)
    generated_files = _generated_video_files(artifact_root)
    samples = _samples(prompt_records, judge_rows, artifact_root, generated_files)
    local_rows = _local_metric_rows(samples, prompt_manifest_path, generated_files, kwargs)
    official_rows = _official_metric_rows(
        benchmark_id,
        official_metric_ids,
        official_results_path,
        official_requirements=official_requirements,
        compute_official_scores=compute_official_scores,
        filter_official_records=filter_official_records,
        generated_artifact_dir=artifact_root,
        result_model_id=_text_value(kwargs, "result_model_id", "official_results_model_id", "model_id"),
    )
    rows = local_rows + official_rows
    per_metric = {str(row["metric_id"]): _per_metric(row) for row in rows}
    local_available = [row for row in local_rows if row["status"] in {"passed", "failed"}]
    official_available = [row for row in official_rows if row["status"] in {"passed", "failed"}]
    available_rows = local_available + official_available
    blocked_rows = [row for row in rows if row["status"] in {"blocked", "not_available"}]
    official_results_normalized = bool(official_available)
    artifacts = {
        "scorecard": str((root / "scorecard.json").resolve()),
        "benchmark_contract": str((root / "benchmark_contract.json").resolve()),
        "raw_metric_table": str((root / "raw_metric_table.jsonl").resolve()),
    }
    video_exists_value = per_metric.get("generated_video_exists", {}).get("value")
    video_exists_count = (
        int(video_exists_value["passed"])
        if isinstance(video_exists_value, Mapping) and video_exists_value.get("passed") is not None
        else 0
    )
    requirements = official_requirements.get(benchmark_id, {})
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": official_results_normalized,
        "normalization_ok": official_results_normalized,
        "run": {
            "status": "official_results_normalized" if official_results_normalized else "in_tree_local_checks",
            "started_at": utc_now_iso(),
            "runner": runner,
            "mode": mode,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": display_name,
            "contract_only": not official_results_normalized,
            "evidence_level": "official_results_normalized" if official_results_normalized else "contract_fixture_only",
            "requires_upstream_runtime": True,
        },
        "dataset": {
            "generated_artifact_dir": None if artifact_root is None else str(artifact_root),
            "generated_file_count": len(generated_files),
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "manifest_sample_count": len(prompt_records),
            "official_results_path": None if official_results_path is None else str(official_results_path),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "local artifact checks are not official benchmark judge evidence",
                _official_blocked_reason(requirements),
            ],
        },
        "generation": {
            "successful": video_exists_count,
            "failed": max(len(samples) - video_exists_count, 0),
        },
        "metrics": {
            "leaderboard": {
                row["metric_id"]: row["score"]
                for row in official_available
                if row.get("score") is not None
            },
            "local": {
                row["metric_id"]: row["score"]
                for row in local_rows
                if row["status"] in {"passed", "failed"} and row.get("score") is not None
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(samples),
                "local_metric_count": len(local_rows),
                "local_available_count": len(local_available),
                "official_available_count": len(official_available),
                "blocked_metric_count": len(blocked_rows),
            },
        },
        "evaluation": {
            "available": bool(available_rows),
            "kind": "official_results_normalizer" if official_results_normalized else evaluator_kind,
            "evidence_level": "official_results_normalized" if official_results_normalized else "contract_fixture_only",
            "num_results": len(rows),
            "leaderboard_metrics": {
                row["metric_id"]: row["score"]
                for row in official_available
                if row.get("score") is not None
            },
            "skip_count": len(blocked_rows),
        },
        "skips": {
            "count": len(blocked_rows),
            "reasons": sorted({str(row.get("blocked_reason")) for row in blocked_rows if row.get("blocked_reason")}),
        },
        "artifacts": artifacts,
    }
    benchmark_contract = {
        "benchmark_id": benchmark_id,
        "display_name": display_name,
        "input_keys": ["generated_video_path", "prompt", "question", "reference_or_labels", "optional_judge_response"],
        "output_keys": ["metric_id", "score", "value", "status", "evidence", "blocked_reason"],
        "local_metric_ids": list(LOCAL_METRIC_IDS),
        "official_metric_ids": list(official_metric_ids),
        "manifest": dict(manifest or {}),
        "generated_files": [str(path) for path in generated_files],
    }
    write_json(root / "scorecard.json", scorecard)
    write_json(root / "benchmark_contract.json", benchmark_contract)
    write_jsonl(root / "raw_metric_table.jsonl", rows)
    return {
        "ok": True,
        "benchmark_id": benchmark_id,
        "output_dir": str(root),
        "contract_only": not official_results_normalized,
        "evidence_level": "official_results_normalized" if official_results_normalized else "contract_fixture_only",
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "artifacts": artifacts,
    }
