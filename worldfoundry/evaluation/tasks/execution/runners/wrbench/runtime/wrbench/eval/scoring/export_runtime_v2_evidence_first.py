#!/usr/bin/env python3
"""Export score-probe and evidence-gate-masked Runtime V2 smoke score files."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


SCORE_PROBE_POLICY = "score_probe_export"
EVIDENCE_GATE_MASKED_POLICY = "evidence_gate_masked_export"
MANIFEST_METADATA_FIELDS = (
    "model",
    "scoring_video_surface",
    "scoring_video_path",
    "eval_video_path",
    "path",
    "original_concat_path",
    "evaluated_segment",
    "condition_frame_count",
    "generated_frame_count",
    "eval_video_frame_start",
    "eval_video_frame_end_exclusive",
    "source_video_path",
    "variant_id",
    "family_id",
    "camera_type",
    "reasoning_tier",
    "event_tier",
    "event_tag",
    "event_delta",
    "oov_gap",
    "scenario_type",
    "protocol",
    "visibility_gap_level",
    "prompt_id",
    "world_state_prompt",
    "expected_state",
    "expected_visibility",
    "expected_behavior",
    "motion_direction",
    "target_object",
    "camera_return_direction",
)
RAW_OOV_SCORE_FIELDS = ("runtime_v2_d5_raw_score", "runtime_v2_d6_raw_score")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _index_by_video_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("video_id")): row for row in rows if row.get("video_id") is not None}


def _manifest_ids(manifest: list[dict[str, Any]] | None) -> list[str] | None:
    if manifest is None:
        return None
    return [str(row["video_id"]) for row in manifest if row.get("video_id")]


def _manifest_by_video_id(manifest: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    return _index_by_video_id(manifest)


def _bool_or_false(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _evidence_value(evidence: dict[str, Any] | None, key: str, missing_value: Any = None) -> Any:
    if evidence is None:
        return missing_value
    return evidence.get(key, missing_value)


def _shared_oov_gate(evidence: dict[str, Any] | None) -> tuple[bool, str | None]:
    if evidence is None:
        return False, "missing_evidence"
    shared_value = evidence.get("evidence_shared_oov_applicable", evidence.get("shared_oov_applicable"))
    if isinstance(shared_value, bool):
        applicable = shared_value
    else:
        # Backward compatibility for pre-shared evidence rows.
        applicable = _bool_or_false(evidence.get("evidence_d5_applicable")) and _bool_or_false(
            evidence.get("evidence_d6_applicable")
        )
    reason = evidence.get("evidence_shared_oov_na_reason", evidence.get("shared_oov_na_reason"))
    if not applicable and reason in (None, ""):
        reason = evidence.get("evidence_d5_na_reason") or evidence.get("evidence_d6_na_reason") or "unclear"
    return bool(applicable), None if applicable else str(reason)


def _dimension_oov_gate(
    evidence: dict[str, Any] | None,
    *,
    applicable_key: str,
    reason_key: str,
) -> tuple[bool, str | None]:
    if evidence is None:
        return False, "missing_evidence"
    value = evidence.get(applicable_key)
    if isinstance(value, bool):
        applicable = value
        reason = evidence.get(reason_key)
        if not applicable and reason in (None, ""):
            reason = evidence.get("evidence_shared_oov_na_reason", evidence.get("shared_oov_na_reason")) or "unclear"
        return bool(applicable), None if applicable else str(reason)
    return _shared_oov_gate(evidence)


def _require_raw_oov_score_fields(row: dict[str, Any]) -> None:
    missing = [field for field in RAW_OOV_SCORE_FIELDS if field not in row]
    if missing:
        video_id = row.get("video_id")
        raise ValueError(f"missing required raw OOV score field(s) for video_id={video_id}: {missing}")


def _manifest_declares_no_oov(row: dict[str, Any]) -> bool:
    return row.get("oov_gap") == "none"


def attach_evidence(row: dict[str, Any], evidence: dict[str, Any] | None, *, policy: str) -> dict[str, Any]:
    out = deepcopy(row)
    _require_raw_oov_score_fields(out)
    shared_app, shared_reason = _shared_oov_gate(evidence)
    d5_app, d5_reason = _dimension_oov_gate(
        evidence,
        applicable_key="evidence_d5_applicable",
        reason_key="evidence_d5_na_reason",
    )
    d6_app, d6_reason = _dimension_oov_gate(
        evidence,
        applicable_key="evidence_d6_applicable",
        reason_key="evidence_d6_na_reason",
    )
    if evidence is not None and "evidence_shared_oov_applicable" not in evidence and "shared_oov_applicable" not in evidence:
        shared_app = bool(d5_app and d6_app)
        shared_reason = None if shared_app else d5_reason or d6_reason or "unclear"
    if _manifest_declares_no_oov(out):
        shared_app = False
        shared_reason = "no_oov"
        d5_app = False
        d5_reason = "no_oov"
        d6_app = False
        d6_reason = "no_oov"

    out["runtime_v2_evidence_export_policy"] = policy
    out["runtime_v2_evidence_schema"] = _evidence_value(evidence, "schema_version")
    out["runtime_v2_shared_oov_applicable"] = shared_app
    out["runtime_v2_shared_oov_na_reason"] = shared_reason
    out["runtime_v2_evidence_shared_oov_applicable"] = shared_app
    out["runtime_v2_evidence_shared_oov_na_reason"] = shared_reason
    out["runtime_v2_evidence_d5_applicable"] = d5_app
    out["runtime_v2_evidence_d6_applicable"] = d6_app
    out["runtime_v2_evidence_d5_na_reason"] = d5_reason
    out["runtime_v2_evidence_d6_na_reason"] = d6_reason
    out["runtime_v2_evidence_confidence"] = _evidence_value(evidence, "evidence_confidence", _evidence_value(evidence, "confidence"))

    if policy == EVIDENCE_GATE_MASKED_POLICY:
        out.setdefault("vlm_dimension_applicable", {})
        out["vlm_dimension_applicable"]["spatial_reasoning"] = d5_app
        out["vlm_dimension_applicable"]["state_reasoning"] = d6_app
        if not d5_app:
            out["vlm_spatial_reasoning"] = None
        if not d6_app:
            out["vlm_state_reasoning"] = None
    return out


def attach_manifest_metadata(
    row: dict[str, Any],
    manifest_row: dict[str, Any] | None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Carry benchmark grouping fields into Runtime V2 exports.

    Runtime V2 score rows intentionally contain only scorer outputs plus a
    minimal video identity. Section-5 table scripts need the benchmark
    grouping contract (`camera_type`, `reasoning_tier`, `event_delta`, etc.).
    The manifest is the authoritative source for those fields, so export rows
    should preserve score values while filling missing or stale metadata from
    the manifest.
    """
    if manifest_row is None:
        return row
    out = dict(row)
    conflicts: list[str] = []
    for key in MANIFEST_METADATA_FIELDS:
        value = manifest_row.get(key)
        if value not in (None, ""):
            existing = row.get(key)
            if strict and existing not in (None, "") and str(existing) != str(value):
                conflicts.append(key)
            out[key] = value
    if conflicts:
        video_id = row.get("video_id") or manifest_row.get("video_id")
        raise ValueError(
            "manifest metadata conflict for "
            f"video_id={video_id}: fields={conflicts[:10]}"
        )
    return out


def build_exports(
    *,
    scores: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    manifest_ids: list[str] | None = None,
    manifest_by_id: dict[str, dict[str, Any]] | None = None,
    strict_manifest_metadata: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    score_by_id = _index_by_video_id(scores)
    manifest_by_id = manifest_by_id or {}
    ordered_ids = manifest_ids or [str(row["video_id"]) for row in scores if row.get("video_id")]
    score_probe_rows: list[dict[str, Any]] = []
    masked_rows: list[dict[str, Any]] = []
    for video_id in ordered_ids:
        score = score_by_id.get(video_id)
        if score is None:
            continue
        score = attach_manifest_metadata(
            score,
            manifest_by_id.get(video_id),
            strict=strict_manifest_metadata,
        )
        evidence = evidence_by_id.get(video_id)
        score_probe_rows.append(attach_evidence(score, evidence, policy=SCORE_PROBE_POLICY))
        masked_rows.append(attach_evidence(score, evidence, policy=EVIDENCE_GATE_MASKED_POLICY))
    return {
        SCORE_PROBE_POLICY: score_probe_rows,
        EVIDENCE_GATE_MASKED_POLICY: masked_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-v7", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scores = load_json(args.scores_v7)
    if not isinstance(scores, list):
        raise TypeError("--scores-v7 must contain a JSON list")
    evidence_by_id = _index_by_video_id(load_jsonl(args.evidence_jsonl))
    manifest = load_json(args.manifest_path) if args.manifest_path.exists() else None
    ids = _manifest_ids(manifest)
    exports = build_exports(
        scores=scores,
        evidence_by_id=evidence_by_id,
        manifest_ids=ids,
        manifest_by_id=_manifest_by_video_id(manifest),
        strict_manifest_metadata=True,
    )

    score_probe_path = args.out_dir / "scores_v7_runtime_v2_evidence_first_score_probe_export.json"
    masked_path = args.out_dir / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json"
    summary_path = args.out_dir / "evidence_first_export_summary.json"
    write_json(score_probe_path, exports[SCORE_PROBE_POLICY])
    write_json(masked_path, exports[EVIDENCE_GATE_MASKED_POLICY])
    write_json(
        summary_path,
        {
            "schema_version": "runtime_v2_evidence_first_export_v1",
            "scores_v7": str(args.scores_v7),
            "evidence_jsonl": str(args.evidence_jsonl),
            "manifest_path": str(args.manifest_path),
            "score_probe_export": str(score_probe_path),
            "evidence_gate_masked_export": str(masked_path),
            "score_probe_records": len(exports[SCORE_PROBE_POLICY]),
            "evidence_gate_masked_records": len(exports[EVIDENCE_GATE_MASKED_POLICY]),
            "evidence_records": len(evidence_by_id),
        },
    )
    print(f"[wrote] {score_probe_path}")
    print(f"[wrote] {masked_path}")
    print(f"[wrote] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
