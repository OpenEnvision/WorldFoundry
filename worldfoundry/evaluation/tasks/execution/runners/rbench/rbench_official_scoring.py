"""RBench official summary-CSV normalization for the contract evaluator.

The zoo contract evaluator flattens caller-supplied files into a record list, which loses
the directory structure the runner relies on. This module therefore reads the *summary*
CSVs the upstream scripts emit, all of which are self-describing:

``score_summary_<tag>.csv``      one row per embodiment plus ``TOTAL_MEAN``
``all_models_summary_<tag>.csv`` one row per model, embodiment track (``overall_mean``)
                                 or task track (per-task columns + ``ALL_TASKS_MEAN``)
``summary_scores_<tag>.csv``     one row per task (``task``/``mean_score``)

Rows that carry plain ``metric_id``/``score`` pairs are accepted too. For the full
per-video recomputation, use the runner with ``--official-results-path`` pointing at the
``results/`` tree instead — that path reproduces the upstream aggregation exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import OfficialMetricScore
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_metrics import (
    COMPONENT_METRIC_IDS,
    DERIVED_METRIC_IDS,
    METRIC_ORDER,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_prompts import (
    EMBODIMENT_ORDER,
    TASK_METRIC_IDS,
)

JsonValue = Any

TOTAL_MEAN_ROW = "TOTAL_MEAN"
ALL_TASKS_MEAN_COLUMN = "ALL_TASKS_MEAN"
OVERALL_MEAN_COLUMN = "overall_mean"

# Upstream summary column -> WorldFoundry metric id.
_SUMMARY_COLUMN_METRICS: Mapping[str, str] = {
    **COMPONENT_METRIC_IDS,
    **DERIVED_METRIC_IDS,
}


def _to_float(value: JsonValue) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(value: float, source_path: Path | None, **evidence: JsonValue) -> OfficialMetricScore:
    return OfficialMetricScore(
        score=value,
        raw_value=value,
        evidence={
            "source_path": None if source_path is None else str(source_path),
            "source": "rbench_official_summary_csv",
            **evidence,
        },
    )


def _embodiment_summary_rows(records: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return [record for record in records if "Robot_Type" in record]


def _task_summary_rows(records: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return [record for record in records if "task" in record and "mean_score" in record]


def _all_models_rows(records: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return [record for record in records if "model" in record]


def _from_embodiment_summary(
    rows: Sequence[Mapping[str, JsonValue]],
    source_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Read ``score_summary_<tag>.csv``: per-embodiment rows plus TOTAL_MEAN."""
    scores: dict[str, OfficialMetricScore] = {}
    total_row = next((row for row in rows if str(row.get("Robot_Type")) == TOTAL_MEAN_ROW), None)
    embodiment_rows = [row for row in rows if str(row.get("Robot_Type")) in EMBODIMENT_ORDER]

    for column, metric_id in _SUMMARY_COLUMN_METRICS.items():
        value = _to_float(total_row.get(column)) if total_row else None
        if value is None:
            values = [
                number
                for row in embodiment_rows
                if (number := _to_float(row.get(column))) is not None
            ]
            value = _mean(values)
            aggregation = "mean_over_embodiment_rows"
        else:
            aggregation = "upstream_total_mean_row"
        if value is not None:
            scores[metric_id] = _score(
                value,
                source_path,
                column=column,
                aggregation=aggregation,
                embodiment_row_count=len(embodiment_rows),
            )

    pooled = [
        number
        for row in embodiment_rows
        for column in ("Task_Completion", "Visual_Quality")
        if (number := _to_float(row.get(column))) is not None
    ]
    overall = _mean(pooled)
    if overall is not None:
        scores["embodiment_overall"] = _score(
            overall,
            source_path,
            aggregation="pooled_task_completion_and_visual_quality",
            embodiment_row_count=len(embodiment_rows),
            complete=len(embodiment_rows) == len(EMBODIMENT_ORDER),
        )
    return scores


def _from_task_summary(
    rows: Sequence[Mapping[str, JsonValue]],
    source_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Read ``summary_scores_<tag>.csv``: one row per task plus ALL_TASKS_MEAN."""
    scores: dict[str, OfficialMetricScore] = {}
    for row in rows:
        task = str(row.get("task") or "").strip()
        value = _to_float(row.get("mean_score"))
        if value is None:
            continue
        if task == ALL_TASKS_MEAN_COLUMN:
            scores["task_track_overall"] = _score(value, source_path, aggregation="upstream_all_tasks_mean")
        elif task in TASK_METRIC_IDS:
            scores[TASK_METRIC_IDS[task]] = _score(value, source_path, split=task)
    return scores


def _from_all_models(
    rows: Sequence[Mapping[str, JsonValue]],
    source_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Read ``all_models_summary_<tag>.csv`` for either track."""
    scores: dict[str, OfficialMetricScore] = {}
    for row in rows:
        overall = _to_float(row.get(OVERALL_MEAN_COLUMN))
        if overall is not None:
            scores["embodiment_overall"] = _score(
                overall, source_path, model=row.get("model"), aggregation="upstream_overall_mean"
            )
        all_tasks = _to_float(row.get(ALL_TASKS_MEAN_COLUMN))
        if all_tasks is not None:
            scores["task_track_overall"] = _score(
                all_tasks, source_path, model=row.get("model"), aggregation="upstream_all_tasks_mean"
            )
        for task, metric_id in TASK_METRIC_IDS.items():
            value = _to_float(row.get(task))
            if value is not None:
                scores[metric_id] = _score(value, source_path, model=row.get("model"), split=task)
    return scores


def _from_metric_rows(
    rows: Sequence[Mapping[str, JsonValue]],
    source_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    scores: dict[str, OfficialMetricScore] = {}
    for row in rows:
        metric_id = str(row.get("metric_id") or row.get("metric") or row.get("Metric") or "").strip()
        if metric_id not in METRIC_ORDER:
            continue
        value = _to_float(row.get("score") if "score" in row else row.get("value"))
        if value is None:
            continue
        scores[metric_id] = _score(
            value / 100.0 if value > 1.0 else value, source_path, aggregation="summary_metric_row"
        )
    return scores


def _rbench_scores(
    records: list[Mapping[str, JsonValue]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Normalize RBench official summary artifacts into WorldFoundry metric scores."""
    scores: dict[str, OfficialMetricScore] = {}
    scores.update(_from_metric_rows(records, official_results_path))
    scores.update(_from_all_models(_all_models_rows(records), official_results_path))
    scores.update(_from_task_summary(_task_summary_rows(records), official_results_path))
    scores.update(_from_embodiment_summary(_embodiment_summary_rows(records), official_results_path))

    # The cross-track composite is WorldFoundry-defined and needs both official tracks.
    embodiment = scores.get("embodiment_overall")
    task = scores.get("task_track_overall")
    if embodiment is not None and task is not None and "rbench_overall" not in scores:
        value = (embodiment.score + task.score) / 2
        scores["rbench_overall"] = _score(
            value,
            official_results_path,
            aggregation="worldfoundry_cross_track_mean",
            components=["embodiment_overall", "task_track_overall"],
            worldfoundry_defined=True,
        )
    return scores


official_scores_from_records = _rbench_scores


__all__ = ["official_scores_from_records"]
