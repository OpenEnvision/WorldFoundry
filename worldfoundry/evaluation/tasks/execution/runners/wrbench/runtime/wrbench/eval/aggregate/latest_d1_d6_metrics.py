#!/usr/bin/env python3
"""Single source of truth for the current WRBench D1-D6 metric contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "wrbench_latest_d1_d6_metrics_v3"
MAIN_TABLE_HYPERPARAMETERS = {
    "excluded_main_table_models": ("hunyuanworld_voyager",),
}
PAPER_FACING_READY_STATUS = "paper_facing_current"
BENCHMARK_DEFAULT_STATUS = "benchmark_default_current"
CONTRACT_VERSION = "wrbench_paper_v1"
WORLDSTATE_CURRENT_RUN_TAG = CONTRACT_VERSION
WORLDSTATE_PROMPT_MODE = "visible_probe_logprob_slot_parse"
WORLDSTATE_D5D6_NATIVE_PROMPT_MODE = "returned_probe_logprob_native_d5d6"
WORLDSTATE_GATE_NAME = "reobservation_judgeability_gate"
WORLDSTATE_GATE_FIELD = "runtime_v2_evidence_shared_oov_applicable"
PAPER_REFERENCE = "WRBench: Current World Models Lack a Persistent State Core (preprint)"
CURRENT_EVIDENCE_STATUS_BY_DIMENSION = {
    "D1": {
        "promotion_status": PAPER_FACING_READY_STATUS,
        "current_source": "vggt_omega_d1_camprec_20260608",
        "current_run_tag": CONTRACT_VERSION,
        "raw_scored_rows": "7600/7600 ok",
        "reporting_denominator": 7500,
        "reporting_score": 0.7021636320348326,
        "reporting_exclusions": (
            {
                "model": "hunyuan_game_craft",
                "camera": "static",
                "rows": 100,
                "reason": "unsupported_static_hold_upstream_gamecraft",
                "kept_rows_for_model": 400,
            },
        ),
        "blocking_caveat": None,
        "benchmark_auxiliary_metrics": (
            "D1-CamAlign common-yaw for local/API prompt-camera alignment",
            "D1-CamAlign API static-hold sanity metric",
        ),
    },
    "D1-CamAlign": {
        "promotion_status": PAPER_FACING_READY_STATUS,
        "current_source": "d1_camalign_common_yaw_static_hold_v1",
        "current_run_tag": CONTRACT_VERSION,
        "blocking_caveat": None,
    },
    "D2": {
        "promotion_status": PAPER_FACING_READY_STATUS,
        "current_source": "d2_fullfov_dinov2_v2_20260605",
        "current_run_tag": "d2_lg_v2_candidate_e_as_selected_visual_integrity",
        "blocking_caveat": None,
    },
    "D3": {
        "promotion_status": BENCHMARK_DEFAULT_STATUS,
        "current_source": "p25_d3d4_slot_parse_latest_20260608",
        "current_run_tag": WORLDSTATE_CURRENT_RUN_TAG,
        "prompt_mode": WORLDSTATE_PROMPT_MODE,
        "coverage": "9600/9600",
        "blocking_caveat": None,
    },
    "D4": {
        "promotion_status": BENCHMARK_DEFAULT_STATUS,
        "current_source": "p25_d3d4_slot_parse_latest_20260608",
        "current_run_tag": WORLDSTATE_CURRENT_RUN_TAG,
        "prompt_mode": WORLDSTATE_PROMPT_MODE,
        "coverage": "9600/9600",
        "blocking_caveat": None,
    },
    "D5": {
        "promotion_status": BENCHMARK_DEFAULT_STATUS,
        "current_source": "p22_d5d6_score_with_shared_e14_gate_20260608",
        "current_run_tag": WORLDSTATE_CURRENT_RUN_TAG,
        "prompt_mode": WORLDSTATE_D5D6_NATIVE_PROMPT_MODE,
        "gate_name": WORLDSTATE_GATE_NAME,
        "gate_field": WORLDSTATE_GATE_FIELD,
        "post_gate_denominator": "2073/9600",
        "blocking_caveat": None,
    },
    "D6": {
        "promotion_status": BENCHMARK_DEFAULT_STATUS,
        "current_source": "p22_d5d6_score_with_shared_e14_gate_20260608",
        "current_run_tag": WORLDSTATE_CURRENT_RUN_TAG,
        "prompt_mode": WORLDSTATE_D5D6_NATIVE_PROMPT_MODE,
        "gate_name": WORLDSTATE_GATE_NAME,
        "gate_field": WORLDSTATE_GATE_FIELD,
        "post_gate_denominator": "2073/9600",
        "blocking_caveat": None,
    },
}


@dataclass(frozen=True)
class MetricSpec:
    dimension_id: str
    output_column: str
    display_name: str
    score_field: str
    source_artifact_role: str
    owner: str
    scale: str
    required_status_field: str | None = None
    required_status_value: str | None = None
    allow_null_score: bool = False
    fallback_score_fields: tuple[str, ...] = ()
    forbidden_fallback_fields: tuple[str, ...] = ()
    hyperparameters: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hyperparameters"] = dict(self.hyperparameters or {})
        return payload


LATEST_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        dimension_id="D1",
        output_column="D1_camera_pose",
        display_name="Requested-camera precision (D1-CamPrec)",
        score_field="d1_camera_accuracy",
        source_artifact_role="d1_requested_control_rows_jsonl",
        owner="wrbench.eval.d1.d1_camera",
        scale="0_to_1",
        required_status_field="d1_status",
        required_status_value="ok",
        hyperparameters={
            "metric_name": "D1-CamPrec",
            "target_policy": "requested_control_latest_benchmark_local_deployed_20260608",
            "eligible_camera_policy": "exclude_uncontrolled",
            "target_coordinate_convention": "opencv_c2w",
            "pose_backend": "vggt_omega",
            "api_prompt_camera_policy": "separate_D1_CamAlign_metric_not_merged",
            "reporting_exclusion_policy": "exclude unsupported static-hold rows from D1-CamPrec reporting denominator",
            "reporting_exclusions": (
                {
                    "model": "hunyuan_game_craft",
                    "camera": "static",
                    "rows": 100,
                    "reason": "unsupported_static_hold_upstream_gamecraft",
                },
            ),
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D1-CamAlign",
        output_column="D1_camalign",
        display_name="Prompt-camera alignment (D1-CamAlign)",
        score_field="d1_camalign_score",
        source_artifact_role="d1_camalign_rows_jsonl",
        owner="wrbench.eval.d1.d1_camalign",
        scale="0_to_1",
        required_status_field="d1_camalign_status",
        required_status_value="ok",
        hyperparameters={
            "metric_name": "D1-CamAlign",
            "common_yaw_intents": ("yaw_LR", "yaw_RL"),
            "static_hold_intent": "static",
            "viewpoint_condition_policy": "prompt_only_rows_for_main_table",
            "never_merge_with": "D1_camera_pose",
            "pose_backend": "vggt_omega",
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D2",
        output_column="D2_visual_integrity",
        display_name="Visual integrity (D2)",
        score_field="d2_selected_visual_integrity_score",
        source_artifact_role="d2_selected_visual_integrity_scores_json",
        owner="wrbench.eval.d2.extract_d2_dinov2_local_global_candidate",
        scale="0_to_1",
        forbidden_fallback_fields=("d2_dinov2_temporal_consistency",),
        hyperparameters={
            "method": "fixed_nonlearned_dinov2_fullfov_v2_candidate_e",
            "combine_rule": "min(cls_first_last_cosine, local_patch_token_low_p20)",
            "preprocessing": "full_fov_resize_pad_no_center_crop",
            "sampling": "time_fps_3_max24",
            "legacy_display_label": "D2_dinov2_temporal_consistency",
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D3",
        output_column="D3_spatial_in",
        display_name="Visible spatial consistency (D3)",
        score_field="vlm_spatial_fidelity",
        source_artifact_role="runtime_v2_score_probe_or_gate_masked_export",
        owner="wrbench.eval.scoring.run_local_qwen35_probe_logprob_scorer",
        scale="0_to_1",
        hyperparameters={
            "prompt_mode": WORLDSTATE_PROMPT_MODE,
            "task_context_mode": "none",
            "applicability": "always_in_view_dimension",
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D4",
        output_column="D4_state_in",
        display_name="Visible state consistency (D4)",
        score_field="vlm_state_fidelity",
        source_artifact_role="runtime_v2_score_probe_or_gate_masked_export",
        owner="wrbench.eval.scoring.run_local_qwen35_probe_logprob_scorer",
        scale="0_to_1",
        hyperparameters={
            "prompt_mode": WORLDSTATE_PROMPT_MODE,
            "task_context_mode": "none",
            "applicability": "always_in_view_dimension",
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D5",
        output_column="D5_spatial_oov",
        display_name="Re-observation spatial consistency (D5)",
        score_field="vlm_spatial_reasoning",
        source_artifact_role="runtime_v2_gate_masked_export",
        owner="wrbench.eval.scoring.export_runtime_v2_evidence_first",
        scale="0_to_1_or_null_when_not_applicable",
        allow_null_score=True,
        hyperparameters={
            "prompt_mode": WORLDSTATE_D5D6_NATIVE_PROMPT_MODE,
            "wrapper_prompt_mode": WORLDSTATE_PROMPT_MODE,
            "task_context_mode": "none",
            "applicability_gate": WORLDSTATE_GATE_NAME,
            "applicability_gate_field": WORLDSTATE_GATE_FIELD,
            "null_policy": "null_only_when_not_applicable",
            "score_range": (0.0, 1.0),
        },
    ),
    MetricSpec(
        dimension_id="D6",
        output_column="D6_state_oov",
        display_name="Re-observation event-state consistency (D6)",
        score_field="vlm_state_reasoning",
        source_artifact_role="runtime_v2_gate_masked_export",
        owner="wrbench.eval.scoring.export_runtime_v2_evidence_first",
        scale="0_to_1_or_null_when_not_applicable",
        allow_null_score=True,
        hyperparameters={
            "prompt_mode": WORLDSTATE_D5D6_NATIVE_PROMPT_MODE,
            "wrapper_prompt_mode": WORLDSTATE_PROMPT_MODE,
            "task_context_mode": "none",
            "applicability_gate": WORLDSTATE_GATE_NAME,
            "applicability_gate_field": WORLDSTATE_GATE_FIELD,
            "null_policy": "null_only_when_not_applicable",
            "score_range": (0.0, 1.0),
        },
    ),
)


def latest_metric_specs() -> tuple[MetricSpec, ...]:
    return LATEST_METRICS


def worldstate_metric_specs() -> tuple[MetricSpec, ...]:
    return tuple(spec for spec in LATEST_METRICS if spec.dimension_id in {"D3", "D4", "D5", "D6"})


def metric_output_columns() -> tuple[str, ...]:
    return tuple(spec.output_column for spec in LATEST_METRICS)


def metric_evidence_status() -> dict[str, dict[str, Any]]:
    return json.loads(json.dumps(CURRENT_EVIDENCE_STATUS_BY_DIMENSION, ensure_ascii=False))


def pending_evidence_dimensions() -> tuple[str, ...]:
    return tuple(
        dimension_id
        for dimension_id, status in CURRENT_EVIDENCE_STATUS_BY_DIMENSION.items()
        if str(status["promotion_status"]).startswith("pending")
    )


def metric_by_dimension(dimension_id: str) -> MetricSpec:
    for spec in LATEST_METRICS:
        if spec.dimension_id == dimension_id:
            return spec
    raise KeyError(f"unknown WRBench metric dimension: {dimension_id}")


def metric_by_output_column(output_column: str) -> MetricSpec:
    for spec in LATEST_METRICS:
        if spec.output_column == output_column:
            return spec
    raise KeyError(f"unknown WRBench metric output column: {output_column}")


def require_metric_value(row: Mapping[str, Any], spec: MetricSpec, *, context: str) -> float | None:
    missing_primary = spec.score_field not in row
    forbidden_present = [field for field in spec.forbidden_fallback_fields if field in row]
    if forbidden_present:
        raise ValueError(
            f"{context}: forbidden fallback field(s) present for {spec.dimension_id}: {forbidden_present}"
        )
    if spec.required_status_field:
        if spec.required_status_field not in row:
            raise ValueError(f"{context}: missing required status field {spec.required_status_field}")
        if str(row.get(spec.required_status_field)) != str(spec.required_status_value):
            return None
    if missing_primary:
        raise ValueError(f"{context}: missing required metric field {spec.score_field}")
    value = row.get(spec.score_field)
    if value in (None, ""):
        if spec.allow_null_score:
            return None
        raise ValueError(f"{context}: metric field {spec.score_field} must be non-null")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: metric field {spec.score_field} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context}: metric field {spec.score_field} is not finite: {value!r}")
    return numeric


def metric_contract_payload() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "fallback_score_fields_allowed": False,
            "missing_required_metric_field": "raise",
            "oov_null_scores": "allowed only for declared D5/D6 not-applicable rows",
            "paper_facing_ready_requires_no_pending_correction": True,
            "d3_d6_default_scorer": "visible + re-observation probe logprob with shared re-observation gate",
            "oov_gate_update_policy": "shared re-observation judgeability gate for D5/D6",
            "d1_camera_metric_policy": "requested-camera precision and prompt-camera alignment are separate metrics",
            "paper_reference": PAPER_REFERENCE,
        },
        "table_hyperparameters": dict(MAIN_TABLE_HYPERPARAMETERS),
        "evidence_status": metric_evidence_status(),
        "pending_evidence_dimensions": list(pending_evidence_dimensions()),
        "metrics": [spec.to_json() for spec in LATEST_METRICS],
    }
    return json.loads(json.dumps(payload, ensure_ascii=False))


def write_metric_contract(*, out_json: Path, out_md: Path | None = None) -> dict[str, Any]:
    payload = metric_contract_payload()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(metric_contract_markdown(payload), encoding="utf-8")
    return payload


def metric_contract_markdown(payload: Mapping[str, Any] | None = None) -> str:
    payload = payload or metric_contract_payload()
    lines = [
        "# Latest D1-D6 Metric Contract",
        "",
        f"- schema: `{payload['schema_version']}`",
        "- fallback score fields allowed: `false`",
        f"- pending evidence dimensions: `{', '.join(payload['pending_evidence_dimensions']) or 'none'}`",
        "",
        "| dimension | output column | source field | source role | promotion status | current run tag | forbidden fallback fields |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in payload["metrics"]:
        forbidden = ", ".join(item["forbidden_fallback_fields"]) or "none"
        evidence = payload["evidence_status"][item["dimension_id"]]
        current_run_tag = evidence.get("current_run_tag") or "none"
        lines.append(
            "| {dimension_id} | `{output_column}` | `{score_field}` | `{source_artifact_role}` | `{promotion_status}` | `{current_run_tag}` | `{forbidden}` |".format(
                forbidden=forbidden,
                promotion_status=evidence["promotion_status"],
                current_run_tag=current_run_tag,
                **item,
            )
        )
    d1 = metric_by_dimension("D1")
    d1_params = d1.hyperparameters or {}
    d1_exclusions = d1_params.get("reporting_exclusions") or ()
    if d1_exclusions:
        lines.extend(["", "## D1 Reporting Exclusions", ""])
        for exclusion in d1_exclusions:
            lines.append(
                "- D1-CamPrec excludes `{rows}` `{model}` `{camera}` rows from the "
                "reporting denominator (`{reason}`); raw row scoring remains auditable.".format(
                    **exclusion
                )
            )

    d2 = metric_by_dimension("D2")
    d2_params = d2.hyperparameters or {}
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- D2 uses the fixed full-FoV DINOv2 v2 visual-integrity score: "
            f"`{d2.score_field} = d2_lg_v2_candidate_e = "
            f"{d2_params['combine_rule']}`. Preprocessing is "
            f"`{d2_params['preprocessing']}`; sampling is "
            f"`{d2_params['sampling']}`. The legacy display label "
            "`D2_dinov2_temporal_consistency` is forbidden as a fallback.",
        ]
    )
    return "\n".join(lines) + "\n"
