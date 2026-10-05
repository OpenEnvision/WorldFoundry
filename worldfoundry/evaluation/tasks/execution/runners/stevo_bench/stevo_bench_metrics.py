"""STEVO-Bench metric discovery, per-sample extraction, and aggregation.

The upstream evaluator (``eval/eval_cli.py`` plus ``eval/summarize_results.py``)
writes one run directory per output map::

    runs/<output_map_stem>/
    |-- summary.json
    `-- per_task/
        `-- <task_id>/
            |-- control_report[__<provider>__<model>].json
            |-- physics_report[__<provider>__<model>].json
            `-- se_report[__<provider>__<model>].json

``summary.json`` holds ``num_tasks``, ``overall``, ``by_level`` and a ``tasks``
list.  Every task entry carries ``task_id``/``task_level`` and, once a judge has
run, ``llm_evals[<provider>__<model>]`` with the raw boolean verdicts
``occlusion_done``, ``trigger_applied``, ``physical_inaccuracy`` and
``state_evol``.  Legacy runs store the same booleans flat on the task entry, and
human runs add an ``annotations`` mapping keyed by annotator name.

This module accepts a ``summary.json``, a run directory, a ``runs/`` parent
directory, or a bare ``per_task/`` tree and reproduces the official aggregation
from ``eval/summarize_results.py``:

* ``control_success = occlusion_done AND trigger_applied``
* ``task_success = state_evol AND NOT physical_inaccuracy``
* every reported metric is the mean over tasks where the field is present.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric

__all__ = [
    "METRIC_ORDER",
    "METRIC_SPECS",
    "PRIMARY_METRIC_ID",
    "StevoResultError",
    "aggregate_metrics",
    "discover_result_files",
    "load_sample_records",
    "metric_rows",
    "normalize_score",
    "summarize_run",
]


class StevoResultError(ValueError):
    """Raised when a STEVO-Bench results path holds nothing readable."""


PRIMARY_METRIC_ID = "task_success"

METRIC_ORDER = (
    "state_evol_success",
    "physical_inaccuracy",
    "task_success",
    "occlusion_done",
    "trigger_applied",
    "control_success",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "state_evol_success": {
        "name": "State Progress",
        "group": "state_evolution",
        "higher_is_better": True,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_state_evol_success",
        "record_key": "state_evol",
        "description": (
            "Fraction of tasks where the state-evolution judge confirmed that the hidden process "
            "kept progressing while it was unobserved."
        ),
    },
    "physical_inaccuracy": {
        "name": "Physical Inaccuracy",
        "group": "physics",
        "higher_is_better": False,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_physical_inaccuracy",
        "record_key": "physical_inaccuracy",
        "description": (
            "Fraction of tasks where the physics judge flagged a physically implausible artifact; "
            "lower is better and the normalized score is 1 - rate."
        ),
    },
    "task_success": {
        "name": "Task Success",
        "group": "aggregate",
        "higher_is_better": True,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_task_success",
        "record_key": "task_success",
        "primary": True,
        "description": (
            "Primary STEVO-Bench metric: fraction of tasks with confirmed state evolution and no "
            "physical inaccuracy (state_evol AND NOT physical_inaccuracy)."
        ),
    },
    "occlusion_done": {
        "name": "Observation Control",
        "group": "control",
        "higher_is_better": True,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_occlusion_done",
        "record_key": "occlusion_done",
        "description": (
            "Fraction of tasks where the requested observation control (occlusion, camera lookaway "
            "or illumination dimming) was actually applied in the generated video."
        ),
    },
    "trigger_applied": {
        "name": "Action Control",
        "group": "control",
        "higher_is_better": True,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_trigger_applied",
        "record_key": "trigger_applied",
        "description": "Fraction of tasks where the requested trigger action was applied in the generated video.",
    },
    "control_success": {
        "name": "Control Success",
        "group": "control",
        "higher_is_better": True,
        "native_scale": "fraction_of_tasks_0_1",
        "summary_key": "avg_control_success",
        "record_key": "control_success",
        "description": "Fraction of tasks where the observation control and the trigger action were both applied.",
    },
}

# Raw judge verdicts stored per task, in the order the official summarizer reads them.
VERDICT_KEYS = ("occlusion_done", "trigger_applied", "physical_inaccuracy", "state_evol")

_REPORT_KIND_RE = re.compile(r"^(?P<kind>control|physics|se)_report(?:__(?P<judge>.+))?\.json$")
_REPORT_KIND_FIELDS = {
    "control": ("occlusion_done", "trigger_applied"),
    "physics": ("physical_inaccuracy",),
    "se": ("state_evol",),
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_bool(value: Any) -> bool | None:
    """Coerce an upstream judge verdict to a boolean, preserving ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    return None


def discover_result_files(path: Path) -> dict[str, list[Path]]:
    """Locate upstream STEVO-Bench result files under ``path``.

    Returns a mapping with ``summaries`` (``summary.json`` files) and ``reports``
    (per-task judge reports).  ``path`` may be a ``summary.json`` file, a single
    run directory, a ``runs/`` parent holding several run directories, or a bare
    ``per_task`` tree.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise StevoResultError(f"STEVO-Bench results path does not exist: {path}")

    summaries: list[Path] = []
    reports: list[Path] = []
    if path.is_file():
        if path.name == "summary.json":
            summaries.append(path)
        elif _REPORT_KIND_RE.match(path.name):
            reports.append(path)
        else:
            raise StevoResultError(
                f"unrecognized STEVO-Bench result file: {path} "
                "(expected summary.json or <control|physics|se>_report*.json)"
            )
        return {"summaries": summaries, "reports": reports}

    summaries = sorted(path.rglob("summary.json"))
    reports = sorted(
        candidate
        for candidate in path.rglob("*_report*.json")
        if _REPORT_KIND_RE.match(candidate.name)
    )
    if not summaries and not reports:
        raise StevoResultError(
            f"no STEVO-Bench results found under {path}; expected summary.json or per_task/<task_id>/*_report*.json"
        )
    return {"summaries": summaries, "reports": reports}


def _records_from_summary(summary_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        return []
    tasks = payload.get("tasks")
    if not isinstance(tasks, Sequence):
        return []
    records: list[dict[str, Any]] = []
    for entry in tasks:
        if not isinstance(entry, Mapping):
            continue
        task_id = entry.get("task_id")
        if not task_id:
            continue
        llm_evals = entry.get("llm_evals")
        judges: list[tuple[str | None, Mapping[str, Any]]] = []
        if isinstance(llm_evals, Mapping) and llm_evals:
            judges = [
                (str(judge), values) for judge, values in llm_evals.items() if isinstance(values, Mapping)
            ]
        else:
            # Legacy flat-field runs keep the verdicts directly on the task entry.
            judges = [(None, entry)]
        for judge, values in judges:
            verdicts = {key: _as_bool(values.get(key)) for key in VERDICT_KEYS}
            if all(value is None for value in verdicts.values()):
                continue
            records.append(
                {
                    "task_id": str(task_id),
                    "task_level": entry.get("task_level"),
                    "judge": judge,
                    "source_path": str(summary_path),
                    "source_kind": "summary_json",
                    **verdicts,
                }
            )
    return records


def _records_from_reports(report_paths: Iterable[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str | None], dict[str, Any]] = {}
    for report_path in report_paths:
        match = _REPORT_KIND_RE.match(report_path.name)
        if match is None:
            continue
        payload = _read_json(report_path)
        if not isinstance(payload, Mapping):
            continue
        task_id = payload.get("task_id") or report_path.parent.name
        if not task_id:
            continue
        judge = match.group("judge")
        if judge is None and payload.get("provider") and payload.get("model"):
            judge = f"{payload['provider']}__{str(payload['model']).replace('.', '-')}"
        key = (str(task_id), judge)
        record = merged.setdefault(
            key,
            {
                "task_id": str(task_id),
                "task_level": None,
                "judge": judge,
                "source_path": str(report_path.parent),
                "source_kind": "per_task_report",
                **{field: None for field in VERDICT_KEYS},
            },
        )
        for field in _REPORT_KIND_FIELDS[match.group("kind")]:
            value = _as_bool(payload.get(field))
            if value is not None:
                record[field] = value
    return [record for record in merged.values() if any(record[field] is not None for field in VERDICT_KEYS)]


def _derive(record: Mapping[str, Any]) -> dict[str, bool | None]:
    """Apply the official derived-field rules from ``eval/summarize_results.py``."""
    occlusion = record.get("occlusion_done")
    trigger = record.get("trigger_applied")
    inaccuracy = record.get("physical_inaccuracy")
    state = record.get("state_evol")
    control_success = None if occlusion is None or trigger is None else bool(occlusion) and bool(trigger)
    task_success = None if state is None or inaccuracy is None else bool(state) and not bool(inaccuracy)
    return {"control_success": control_success, "task_success": task_success}


def load_sample_records(path: Path) -> list[dict[str, Any]]:
    """Return one per-sample record per (task, judge) pair with derived fields applied."""
    discovered = discover_result_files(path)
    records: list[dict[str, Any]] = []
    for summary_path in discovered["summaries"]:
        records.extend(_records_from_summary(summary_path))
    if not records:
        records = _records_from_reports(discovered["reports"])
    else:
        # Fill verdicts that summary.json has not been updated with yet.
        indexed = {(record["task_id"], record["judge"]): record for record in records}
        for report_record in _records_from_reports(discovered["reports"]):
            existing = indexed.get((report_record["task_id"], report_record["judge"]))
            if existing is None:
                records.append(report_record)
                indexed[(report_record["task_id"], report_record["judge"])] = report_record
                continue
            for field in VERDICT_KEYS:
                if existing.get(field) is None and report_record.get(field) is not None:
                    existing[field] = report_record[field]
    if not records:
        raise StevoResultError(f"no readable STEVO-Bench judge verdicts under {path}")
    for record in records:
        record.update(_derive(record))
        record["baseline"] = str(record["task_id"]).endswith("_00")
    records.sort(key=lambda record: (str(record["task_id"]), str(record.get("judge") or "")))
    return records


def _bucket(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for metric_id in METRIC_ORDER:
        record_key = METRIC_SPECS[metric_id]["record_key"]
        observed = [
            1.0 if record.get(record_key) else 0.0
            for record in records
            if record.get(record_key) is not None
        ]
        values[metric_id] = mean_numeric(observed)
    return values


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-sample records into the official mean-over-tasks metrics."""
    if not records:
        raise StevoResultError("cannot aggregate STEVO-Bench metrics from an empty record set")
    means = _bucket(records)
    sources = sorted({str(record.get("source_path")) for record in records if record.get("source_path")})
    metrics: dict[str, dict[str, Any]] = {}
    for metric_id, raw_score in means.items():
        record_key = METRIC_SPECS[metric_id]["record_key"]
        sample_count = sum(1 for record in records if record.get(record_key) is not None)
        metrics[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_score,
            "normalized_score": normalize_score(metric_id, raw_score),
            "sample_count": sample_count,
            "source": "stevo_bench_per_task_verdicts",
            "source_path": ";".join(sources[:5]),
        }
    return metrics


def summarize_run(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce the official baseline/occluded and per-level breakdown tables."""
    baseline = [record for record in records if record.get("baseline")]
    occluded = [record for record in records if not record.get("baseline")]
    by_level: dict[str, dict[str, float | None]] = {}
    for record in records:
        level = record.get("task_level")
        if level is None or level == "":
            continue
        by_level.setdefault(str(level), {})
    for level in list(by_level):
        by_level[level] = _bucket([record for record in records if str(record.get("task_level")) == level])
    return {
        "task_count": len({str(record["task_id"]) for record in records}),
        "record_count": len(records),
        "judges": sorted({str(record["judge"]) for record in records if record.get("judge")}),
        "overall": _bucket(records),
        "baseline": _bucket(baseline) if baseline else {},
        "occluded": _bucket(occluded) if occluded else {},
        "by_level": by_level,
    }


def normalize_score(metric_id: str, raw_score: float | None) -> float | None:
    """Map an upstream rate in [0, 1] to a higher-is-better unit score."""
    if raw_score is None:
        return None
    value = max(0.0, min(1.0, float(raw_score)))
    if not METRIC_SPECS[metric_id]["higher_is_better"]:
        return 1.0 - value
    return value


def metric_rows(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    official_runtime_executed: bool = False,
) -> list[dict[str, Any]]:
    """Return WorldFoundry metric rows in the declared metric order."""
    rows: list[dict[str, Any]] = []
    source_label = "stevo_bench_official_runtime" if official_runtime_executed else "stevo_bench_results_file"
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        item = metrics.get(metric_id) or {}
        normalized = item.get("normalized_score")
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": normalized is not None,
                "raw_score": item.get("raw_score"),
                "normalized_score": normalized,
                "score": normalized,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "native_scale": spec["native_scale"],
                "description": spec["description"],
                "sample_count": item.get("sample_count", 0),
                "source": item.get("source") or source_label,
                "source_path": item.get("source_path"),
                "reason": None if normalized is not None else "score_not_available_in_stevo_bench_results",
            }
        )
    return rows
