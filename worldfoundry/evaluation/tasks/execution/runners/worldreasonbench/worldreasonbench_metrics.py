"""Normalize WorldReasonBench and WorldRewardBench evaluator outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric, normalize_unit_score, scalar_number

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "score_pr": {"name": "Process-aware Reasoning Score", "group": "qa", "scale": "unit"},
    "qa_accuracy": {"name": "QA Accuracy", "group": "qa", "scale": "unit"},
    "state_score": {"name": "State Score", "group": "qa", "scale": "unit"},
    "process_score": {"name": "Process Score", "group": "qa", "scale": "unit"},
    "fidelity_score": {"name": "Fidelity Score", "group": "qa", "scale": "unit"},
    "mechanism_score": {"name": "Mechanism Score", "group": "qa", "scale": "unit"},
    "static_outcome_score": {"name": "Static Outcome Score", "group": "qa", "scale": "unit"},
    "dynamic_reasoning_score": {"name": "Dynamic Reasoning Score", "group": "qa", "scale": "unit"},
    "reasoning_gap": {"name": "Reasoning Gap", "group": "qa", "scale": "signed", "higher_is_better": False},
    "pointwise_score": {"name": "Pointwise S(v)", "group": "pointwise", "scale": "five"},
    "reasoning_correctness": {"name": "Reasoning Correctness", "group": "pointwise", "scale": "five"},
    "content_fidelity": {"name": "Content Fidelity", "group": "pointwise", "scale": "five"},
    "visual_aesthetics": {"name": "Visual Aesthetics", "group": "pointwise", "scale": "five"},
    "pointwise_spearman": {"name": "Pointwise Spearman Correlation", "group": "pointwise", "scale": "signed"},
    "induced_pairwise_accuracy": {"name": "Induced Pairwise Accuracy", "group": "pointwise", "scale": "unit"},
    "pairwise_accuracy_with_ties": {"name": "Pairwise Accuracy With Ties", "group": "pairwise", "scale": "unit"},
    "pairwise_accuracy_without_ties": {"name": "Pairwise Accuracy Without Ties", "group": "pairwise", "scale": "unit"},
}
METRIC_ORDER = tuple(METRIC_SPECS)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        for key in ("rows", "results", "per_video", "pairs", "videos"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, Mapping)]
        return [dict(value)]
    return []


def _dedupe(rows: Iterable[Mapping[str, Any]], *keys: str) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        identity = next((str(item[key]) for key in keys if item.get(key) not in (None, "")), "")
        if identity:
            indexed[identity] = item
        else:
            anonymous.append(item)
    return [*indexed.values(), *anonymous]


def _mean_field(rows: Iterable[Mapping[str, Any]], *keys: str) -> float | None:
    values: list[float | None] = []
    for row in rows:
        value = next((row[key] for key in keys if row.get(key) not in (None, "")), None)
        values.append(scalar_number(value))
    return mean_numeric(values)


def _summary_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = value.get("summary")
    return summary if isinstance(summary, Mapping) else value


def normalize_qa(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"QA result must be a JSON object: {path}")
    summary = _summary_mapping(value)
    process = summary.get("process_aware") if isinstance(summary.get("process_aware"), Mapping) else {}
    qa_accuracy = scalar_number(
        next(
            (summary[key] for key in ("qa_accuracy", "acc_qa", "overall_average_score", "accuracy") if key in summary),
            None,
        )
    )
    state = scalar_number(process.get("state_score"))
    process_score = scalar_number(process.get("process_satisfaction"))
    fidelity = scalar_number(process.get("fidelity_score"))
    mechanism = scalar_number(process.get("mechanism_score"))
    static_outcome = scalar_number(summary.get("static_outcome_score") or summary.get("s_out"))
    if static_outcome is None and state is not None and fidelity is not None:
        static_outcome = (state + fidelity) / 2.0
    dynamic = scalar_number(summary.get("dynamic_reasoning_score") or summary.get("s_dyn"))
    if dynamic is None and process_score is not None and mechanism is not None:
        dynamic = (process_score + mechanism) / 2.0
    reasoning_gap = scalar_number(summary.get("reasoning_gap") or summary.get("delta_rg"))
    if reasoning_gap is None and static_outcome is not None and dynamic is not None:
        reasoning_gap = static_outcome - dynamic
    score_pr = scalar_number(next((summary[key] for key in ("score_pr", "Score_PR") if key in summary), None))
    qa_unit = normalize_unit_score(qa_accuracy)
    dynamic_unit = normalize_unit_score(dynamic)
    if score_pr is None and qa_unit is not None and dynamic_unit is not None:
        score_pr = qa_unit**0.8 * dynamic_unit**0.2
    return {
        "protocol": "qa",
        "source_path": str(path),
        "sample_count": int(summary.get("num_videos_evaluated") or len(value.get("per_video") or [])),
        "metrics": {
            "score_pr": normalize_unit_score(score_pr),
            "qa_accuracy": qa_unit,
            "state_score": normalize_unit_score(state),
            "process_score": normalize_unit_score(process_score),
            "fidelity_score": normalize_unit_score(fidelity),
            "mechanism_score": normalize_unit_score(mechanism),
            "static_outcome_score": normalize_unit_score(static_outcome),
            "dynamic_reasoning_score": dynamic_unit,
            "reasoning_gap": reasoning_gap,
        },
    }


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for offset in range(start, end + 1):
            result[indexed[offset][0]] = rank
        start = end + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def normalize_pointwise(path: Path, induced_pairs_path: Path | None = None) -> dict[str, Any]:
    rows = _dedupe(_read_rows(path), "video_id", "id")
    parsed = [row for row in rows if row.get("parsed_scores") or row.get("score_weighted") is not None]
    weighted: list[float] = []
    reasoning: list[float | None] = []
    content: list[float | None] = []
    aesthetics: list[float | None] = []
    correlation_ai: list[float] = []
    correlation_human: list[float] = []
    for row in parsed:
        scores = row.get("parsed_scores")
        if isinstance(scores, list) and len(scores) >= 3:
            r, c, a = (scalar_number(scores[index]) for index in range(3))
        else:
            r = scalar_number(row.get("reasoning_score"))
            c = scalar_number(row.get("content_score"))
            a = scalar_number(row.get("aesthetics_score"))
        score = scalar_number(row.get("score_weighted"))
        if score is None and None not in (r, c, a):
            score = 0.4 * float(r) + 0.3 * float(c) + 0.3 * float(a)
        reasoning.append(r)
        content.append(c)
        aesthetics.append(a)
        if score is not None:
            weighted.append(score)
            human = scalar_number(row.get("human_score"))
            if human is not None:
                correlation_ai.append(score)
                correlation_human.append(human)

    induced_accuracy = None
    induced_count = 0
    if induced_pairs_path is not None:
        induced = _dedupe(_read_rows(induced_pairs_path), "pair_id", "id")
        scoreable = [row for row in induced if row.get("parsed_verdict")]
        induced_count = len(scoreable)
        induced_accuracy = mean_numeric(1.0 if row.get("is_correct") is True else 0.0 for row in scoreable)
    return {
        "protocol": "pointwise",
        "source_path": str(path),
        "sample_count": len(parsed),
        "metrics": {
            "pointwise_score": mean_numeric(weighted),
            "reasoning_correctness": mean_numeric(reasoning),
            "content_fidelity": mean_numeric(content),
            "visual_aesthetics": mean_numeric(aesthetics),
            "pointwise_spearman": spearman(correlation_ai, correlation_human),
            "induced_pairwise_accuracy": induced_accuracy,
        },
        "details": {"correlation_sample_count": len(correlation_ai), "induced_pair_count": induced_count},
    }


def _expected_verdict(row: Mapping[str, Any]) -> str | None:
    expected = str(row.get("expected_verdict") or "").strip()
    if expected:
        return "A=B" if expected.startswith("A=B") else expected
    score_1, score_2 = scalar_number(row.get("score_1")), scalar_number(row.get("score_2"))
    if score_1 is None or score_2 is None:
        return None
    if str(row.get("pair_type") or "").lower() == "tie" or math.isclose(score_1, score_2):
        return "A=B"
    return "A>B" if score_1 > score_2 else "B>A"


def normalize_pairwise(path: Path) -> dict[str, Any]:
    rows = _dedupe(_read_rows(path), "pair_id", "source_pair_id", "id")
    scoreable = [row for row in rows if row.get("parsed_verdict") and row.get("parsed_verdict") != "REFUSED"]
    judged: list[tuple[bool, bool]] = []
    for row in scoreable:
        expected = _expected_verdict(row)
        predicted = str(row.get("parsed_verdict") or "")
        if expected is None:
            if isinstance(row.get("is_correct"), bool):
                judged.append((bool(row["is_correct"]), False))
            continue
        is_tie = expected == "A=B"
        correct = predicted.startswith("A=B") if is_tie else predicted == expected
        judged.append((correct, is_tie))
    non_ties = [correct for correct, is_tie in judged if not is_tie]
    return {
        "protocol": "pairwise",
        "source_path": str(path),
        "sample_count": len(judged),
        "metrics": {
            "pairwise_accuracy_with_ties": mean_numeric(1.0 if correct else 0.0 for correct, _ in judged),
            "pairwise_accuracy_without_ties": mean_numeric(1.0 if correct else 0.0 for correct in non_ties),
        },
        "details": {"parsed_count": len(scoreable), "tie_count": sum(1 for _, tie in judged if tie)},
    }


def _find(directory: Path, names: tuple[str, ...], patterns: tuple[str, ...] = ()) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    for pattern in patterns:
        candidate = next((path for path in sorted(directory.glob(pattern)) if path.is_file()), None)
        if candidate is not None:
            return candidate
    return None


def normalize_results(path: Path, protocol: str = "auto") -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    requested = {part.strip().lower() for part in protocol.split(",") if part.strip()}
    if "all" in requested:
        requested = {"qa", "pointwise", "pairwise"}
    if "auto" in requested:
        requested = set()
    if path.is_file():
        selected = next(iter(requested), "")
        if not selected:
            name = path.name.lower()
            selected = (
                "pairwise" if "pair" in name else "pointwise" if "point" in name or path.suffix == ".jsonl" else "qa"
            )
        if selected == "qa":
            return [normalize_qa(path)]
        if selected == "pointwise":
            induced = path.with_name(f"{path.stem}.induced_pairs.jsonl")
            return [normalize_pointwise(path, induced if induced.is_file() else None)]
        if selected == "pairwise":
            return [normalize_pairwise(path)]
        raise ValueError(f"unsupported WorldReasonBench protocol: {selected}")
    if not path.is_dir():
        raise FileNotFoundError(path)

    qa = _find(path, ("summary.json", "full_results.json", "qa_summary.json"), ("**/summary.json",))
    pointwise = _find(path, ("pointwise_eval.jsonl", "video_level.jsonl"), ("*pointwise*.jsonl",))
    induced = _find(path, ("pointwise_eval.induced_pairs.jsonl", "induced_pairs.jsonl"), ("*induced*pairs*.jsonl",))
    pairwise = _find(path, ("pairwise_eval.jsonl",), ("*pairwise*.jsonl",))
    results = []
    if qa is not None and (not requested or "qa" in requested):
        results.append(normalize_qa(qa))
    if pointwise is not None and (not requested or "pointwise" in requested):
        results.append(normalize_pointwise(pointwise, induced))
    if pairwise is not None and pairwise != induced and (not requested or "pairwise" in requested):
        results.append(normalize_pairwise(pairwise))
    if not results:
        raise FileNotFoundError(f"no WorldReasonBench result files found under {path}")
    return results


def metric_rows(protocol_results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    for result in protocol_results:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        for metric_id, value in metrics.items():
            if metric_id in METRIC_SPECS and value is not None:
                observed[metric_id] = (value, result)
    rows = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        value, result = observed.get(metric_id, (None, {}))
        raw = scalar_number(value)
        scale = spec["scale"]
        normalized = (
            raw / 5.0 if raw is not None and scale == "five" else normalize_unit_score(raw) if scale == "unit" else raw
        )
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "group": spec["group"],
                "available": raw is not None,
                "raw_score": raw,
                "normalized_score": normalized,
                "score": normalized,
                "higher_is_better": spec.get("higher_is_better", True),
                "source": "worldreasonbench_official_output",
                "source_path": result.get("source_path"),
                "sample_count": result.get("sample_count"),
                "reason": None if raw is not None else "protocol_result_not_supplied",
            }
        )
    return rows
