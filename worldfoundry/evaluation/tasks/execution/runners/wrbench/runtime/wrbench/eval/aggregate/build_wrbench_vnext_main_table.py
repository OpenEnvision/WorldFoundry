#!/usr/bin/env python3
"""Build the WRBench vNext main-table candidate from the D1-D6 metric contract."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from . import latest_d1_d6_metrics as metric_contract

EXCLUDED_MAIN_TABLE_MODELS = set(
    metric_contract.MAIN_TABLE_HYPERPARAMETERS["excluded_main_table_models"]
)
DIMENSION_SPECS = metric_contract.latest_metric_specs()
D1_SPEC = metric_contract.metric_by_dimension("D1")
D2_SPEC = metric_contract.metric_by_dimension("D2")
GATE_FIELD = metric_contract.WORLDSTATE_GATE_FIELD

VIEWPOINT_CONDITION_BY_MODEL: dict[str, str] = {
    "recammaster": "source-video",
    "hydra": "source-video",
    "inspatio-world": "source-video",
    "inspatio_world_14b": "source-video",
    "gen3c": "geometry-cache",
    "spatia": "geometry-cache",
    "versecrafter": "geometry-cache",
    "wan21-fun-14b-cam": "model-inferred",
    "wan21_fun_14b_cam": "model-inferred",
    "wan21-fun-1p3b-cam": "model-inferred",
    "wan21_fun_1p3b_cam": "model-inferred",
    "wan22-fun-5b-cam": "model-inferred",
    "wan22_fun_5b_cam": "model-inferred",
    "wan22-fun-a14b-cam": "model-inferred",
    "wan22_fun_a14b_cam": "model-inferred",
    "lingbot-world": "model-inferred",
    "lingbot_world": "model-inferred",
    "lingbot-world-act": "model-inferred",
    "lingbot_world_act": "model-inferred",
    "liveworld": "model-inferred",
    "hunyuan-game-craft": "model-inferred",
    "hunyuan_game_craft": "model-inferred",
    "hunyuan-worldplay": "model-inferred",
    "hunyuan_worldplay": "model-inferred",
    "magicworld": "model-inferred",
    "hailuo-2.3": "prompt-only",
    "hailuo_2_3": "prompt-only",
    "happyhorse-1.0-i2v": "prompt-only",
    "happyhorse_1_0_i2v": "prompt-only",
    "kling-v2.6": "prompt-only",
    "kling_v2_6": "prompt-only",
    "wan2.2-i2v-plus": "prompt-only",
    "wan2_2_i2v_plus": "prompt-only",
    "wan2.6-i2v": "prompt-only",
    "wan2_6_i2v": "prompt-only",
    "wan2.7-i2v": "prompt-only",
    "wan2_7_i2v": "prompt-only",
    "wanx2.1-i2v-turbo": "prompt-only",
    "wanx2_1_i2v_turbo": "prompt-only",
}


def _viewpoint_condition(model: str) -> str:
    key = str(model).replace("_", "-")
    alt = str(model).replace("-", "_")
    return VIEWPOINT_CONDITION_BY_MODEL.get(model) or VIEWPOINT_CONDITION_BY_MODEL.get(key) or VIEWPOINT_CONDITION_BY_MODEL.get(alt) or "unknown"


def _gate_applicable(row: Mapping[str, Any]) -> bool | None:
    for field in (GATE_FIELD, "runtime_v2_shared_oov_gate_applicable"):
        if field in row:
            value = row.get(field)
            if value in (None, ""):
                return None
            return bool(value)
    return None


def _load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"score artifact does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "scores", "data"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
        if payload.get("video_id"):
            return [payload]
    raise ValueError(f"Unsupported record payload: {path}")


def _index(records: list[dict[str, Any]], *, artifact_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for offset, row in enumerate(records, start=1):
        video_id = row.get("video_id")
        if not video_id:
            raise ValueError(f"{artifact_name} row {offset}: missing video_id")
        key = str(video_id)
        if key in indexed:
            raise ValueError(f"{artifact_name}: duplicate video_id {key}")
        indexed[key] = row
    return indexed


def _index_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        model = row.get("model")
        if model and model not in EXCLUDED_MAIN_TABLE_MODELS:
            buckets[str(model)].append(row)
    return buckets


def build_table(
    *,
    runtime_records: list[dict[str, Any]],
    d1_records: list[dict[str, Any]],
    d2_records: list[dict[str, Any]],
    d1_camalign_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    camalign_spec = metric_contract.metric_by_dimension("D1-CamAlign")
    d1_by_id = _index(d1_records, artifact_name="D1 score artifact")
    d2_by_id = _index(d2_records, artifact_name="D2 score artifact")
    camalign_by_id = (
        _index(d1_camalign_records, artifact_name="D1 CamAlign artifact")
        if d1_camalign_records is not None
        else {}
    )
    d1_by_model = _index_by_model(d1_records)
    d2_by_model = _index_by_model(d2_records)
    camalign_by_model = _index_by_model(d1_camalign_records) if d1_camalign_records is not None else {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runtime_records:
        model = row.get("model")
        if not model or model in EXCLUDED_MAIN_TABLE_MODELS:
            continue
        buckets[str(model)].append(row)

    table: list[dict[str, Any]] = []
    runtime_models = set(buckets)
    d1_models = set(d1_by_model)
    d2_models = set(d2_by_model)
    models_without_d2_d3_d6_artifacts: set[str] = set()
    for model in sorted(runtime_models | d1_models | d2_models):
        rows = (
            buckets.get(model)
            or d1_by_model.get(model)
            or camalign_by_model.get(model)
            or d2_by_model.get(model, [])
        )
        out: dict[str, Any] = {"model": model, "n_records": len(rows), "viewpoint_condition_type": _viewpoint_condition(model)}
        ready = True
        gate_values: list[bool] = []
        for spec in DIMENSION_SPECS:
            column = spec.output_column
            values: list[float] = []
            eligible = 0
            for row in rows:
                video_id = str(row.get("video_id") or "")
                if spec.dimension_id == "D1":
                    if row.get("camera_type") == "uncontrolled":
                        continue
                    eligible += 1
                    d1_row = d1_by_id.get(video_id)
                    value = (
                        metric_contract.require_metric_value(
                            d1_row,
                            spec,
                            context=f"D1 score row video_id={video_id}",
                        )
                        if d1_row
                        else None
                    )
                elif spec.dimension_id == "D1-CamAlign":
                    if _viewpoint_condition(model) != "prompt-only":
                        value = None
                        continue
                    camera = str(row.get("camera_type") or row.get("camera") or "")
                    if camera not in {"yaw_LR", "yaw_RL", "static"}:
                        continue
                    eligible += 1
                    camalign_row = camalign_by_id.get(video_id)
                    value = (
                        metric_contract.require_metric_value(
                            camalign_row,
                            spec,
                            context=f"D1 CamAlign row video_id={video_id}",
                        )
                        if camalign_row
                        else None
                    )
                elif spec.dimension_id == "D2":
                    if video_id not in d2_by_id and model not in runtime_models:
                        continue
                    eligible += 1
                    d2_row = d2_by_id.get(video_id)
                    if d2_row is None:
                        value = None
                    elif "d2_status" in d2_row and d2_row.get("d2_status") != "ok":
                        value = None
                    else:
                        value = metric_contract.require_metric_value(
                            d2_row,
                            spec,
                            context=f"D2 score row video_id={video_id}",
                        )
                else:
                    if model not in runtime_models:
                        continue
                    eligible += 1
                    gate = _gate_applicable(row)
                    if gate is not None:
                        gate_values.append(gate)
                    value = metric_contract.require_metric_value(
                        row,
                        spec,
                        context=f"{spec.dimension_id} runtime row video_id={video_id}",
                    )
                if value is not None:
                    values.append(value)
            out[column] = mean(values) if values else None
            out[f"{column}_n"] = len(values)
            out[f"{column}_eligible"] = eligible
            if spec.dimension_id in {"D1", "D2"} and len(values) < eligible:
                ready = False
            if spec.dimension_id == "D6" and model not in runtime_models:
                models_without_d2_d3_d6_artifacts.add(model)
        if gate_values:
            out["reobservation_support"] = sum(1 for v in gate_values if v) / len(gate_values)
            out["reobservation_support_n"] = len(gate_values)
        else:
            out["reobservation_support"] = None
            out["reobservation_support_n"] = 0
        out["main_table_ready"] = ready
        table.append(out)

    main_table_ready = bool(table) and all(row["main_table_ready"] for row in table)
    pending_evidence_dimensions = list(metric_contract.pending_evidence_dimensions())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_contract_schema_version": metric_contract.SCHEMA_VERSION,
        "metric_source_fields": {spec.dimension_id: spec.score_field for spec in DIMENSION_SPECS},
        "forbidden_fallback_fields": {
            spec.dimension_id: list(spec.forbidden_fallback_fields) for spec in DIMENSION_SPECS
        },
        "metric_hyperparameters": {
            spec.dimension_id: dict(spec.hyperparameters or {}) for spec in DIMENSION_SPECS
        },
        "metric_evidence_status": metric_contract.metric_evidence_status(),
        "pending_evidence_dimensions": pending_evidence_dimensions,
        "table_hyperparameters": dict(metric_contract.MAIN_TABLE_HYPERPARAMETERS),
        "runtime_records": len(runtime_records),
        "d1_records": len(d1_records),
        "d2_records": len(d2_records),
        "usable_models": len(table),
        "main_table_ready": main_table_ready,
        "reviewed_current_truth_ready": main_table_ready and not pending_evidence_dimensions,
        "runtime_models": sorted(runtime_models),
        "d1_models_without_runtime_records": sorted(d1_models - runtime_models),
        "d2_models_without_runtime_records": sorted(d2_models - runtime_models),
        "models_without_d2_d3_d6_artifacts": sorted(models_without_d2_d3_d6_artifacts),
        "qa_notes": [
            "D1-D6 output columns and source fields come from latest_d1_d6_metrics.py.",
            "D2 accepts only d2_selected_visual_integrity_score; d2_dinov2_temporal_consistency is forbidden as a fallback.",
            "D2 accepts rows without d2_status; when d2_status is present it must be ok.",
        ],
        "blocked_dimensions": [
            spec.output_column
            for spec in (D1_SPEC, D2_SPEC)
            if sum(row[f"{spec.output_column}_n"] for row in table)
            < sum(row[f"{spec.output_column}_eligible"] for row in table)
        ],
    }
    return table, summary


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "viewpoint_condition_type", "n_records", "reobservation_support", "reobservation_support_n"]
    for spec in DIMENSION_SPECS:
        fields.extend([spec.output_column, f"{spec.output_column}_n"])
        if spec.dimension_id in {"D1", "D2"}:
            fields.append(f"{spec.output_column}_eligible")
    fields.append("main_table_ready")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_md(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# WRBench vNext Main Table Candidate",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Metric contract: `{summary['metric_contract_schema_version']}`",
        f"- Main table ready: `{summary['main_table_ready']}`",
        f"- Reviewed current truth ready: `{summary['reviewed_current_truth_ready']}`",
        f"- Blocked dimensions: `{', '.join(summary['blocked_dimensions']) or 'none'}`",
        f"- Pending evidence dimensions: `{', '.join(summary['pending_evidence_dimensions']) or 'none'}`",
        f"- Metric source fields: `{json.dumps(summary['metric_source_fields'], sort_keys=True)}`",
        f"- Forbidden fallback fields: `{json.dumps(summary['forbidden_fallback_fields'], sort_keys=True)}`",
        f"- D1-only / no runtime models: `{', '.join(summary['d1_models_without_runtime_records']) or 'none'}`",
        f"- Models without D2/D3-D6 artifacts: `{', '.join(summary['models_without_d2_d3_d6_artifacts']) or 'none'}`",
        "",
        "## QA Notes",
        "",
        *[f"- {note}" for note in summary["qa_notes"]],
        "",
        "## Table",
        "",
        "| model | n | D1 pose | D2 visual | D3 spatial-in | D4 state-in | D5 spatial-OoV | D6 state-OoV | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["n_records"]),
                    f"{_fmt(row['D1_camera_pose'])} ({row['D1_camera_pose_n']}/{row['D1_camera_pose_eligible']})",
                    f"{_fmt(row['D2_visual_integrity'])} ({row['D2_visual_integrity_n']}/{row['D2_visual_integrity_eligible']})",
                    f"{_fmt(row['D3_spatial_in'])} ({row['D3_spatial_in_n']})",
                    f"{_fmt(row['D4_state_in'])} ({row['D4_state_in_n']})",
                    f"{_fmt(row['D5_spatial_oov'])} ({row['D5_spatial_oov_n']})",
                    f"{_fmt(row['D6_state_oov'])} ({row['D6_state_oov_n']})",
                    str(row["main_table_ready"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-scores", type=Path, required=True)
    parser.add_argument("--d1-scores", type=Path)
    parser.add_argument("--d1-camalign-scores", type=Path)
    parser.add_argument("--d2-scores", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table, summary = build_table(
        runtime_records=_load_records(args.runtime_scores),
        d1_records=_load_records(args.d1_scores),
        d2_records=_load_records(args.d2_scores),
        d1_camalign_records=_load_records(args.d1_camalign_scores),
    )
    write_csv(args.out_csv, table)
    write_md(args.out_md, table, summary)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
