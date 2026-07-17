#!/usr/bin/env python3
"""Build a guarded-rescue manifest from Qwen3-VL binary gate N/A evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen3vl_binary_na_rescue_manifest_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_no} must contain a JSON object")
        rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_ids(manifest: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, row in enumerate(manifest):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"manifest row {index} is missing video_id")
        if video_id in seen:
            duplicates.append(video_id)
        seen.add(video_id)
        ids.append(video_id)
    if duplicates:
        raise ValueError(f"duplicate manifest video_id values: {sorted(set(duplicates))[:10]}")
    return ids


def _evidence_by_id(evidence_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, row in enumerate(evidence_rows):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"evidence row {index} is missing video_id")
        if video_id in rows_by_id:
            duplicates.append(video_id)
        rows_by_id[video_id] = row
    if duplicates:
        raise ValueError(f"duplicate evidence video_id values: {sorted(set(duplicates))[:10]}")
    return rows_by_id


def _gate_bool(row: dict[str, Any], video_id: str, key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"non-boolean {key} for video_id={video_id}: {value!r}")
    return value


def build_rescue_manifest(
    *,
    manifest: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = _manifest_ids(manifest)
    manifest_by_id = {str(row["video_id"]): row for row in manifest}
    evidence_by_id = _evidence_by_id(evidence_rows)
    missing = [video_id for video_id in manifest_ids if video_id not in evidence_by_id]
    if missing:
        raise ValueError(f"missing evidence for {len(missing)} manifest videos: {missing[:10]}")

    extra = sorted(set(evidence_by_id) - set(manifest_ids))
    selected_ids: list[str] = []
    selected_reasons: dict[str, list[str]] = {}
    request_mapping: list[dict[str, Any]] = []
    dim_counter: Counter[str] = Counter()
    for video_id in manifest_ids:
        manifest_row = manifest_by_id[video_id]
        evidence = evidence_by_id[video_id]
        d5_app = _gate_bool(evidence, video_id, "evidence_d5_applicable")
        d6_app = _gate_bool(evidence, video_id, "evidence_d6_applicable")
        reasons: list[str] = []
        if not d5_app:
            reasons.append("D5")
            dim_counter["D5"] += 1
        if not d6_app:
            reasons.append("D6")
            dim_counter["D6"] += 1
        if reasons:
            selected_ids.append(video_id)
            selected_reasons[video_id] = reasons
            request_mapping.append(
                {
                    "video_id": video_id,
                    "request_id": manifest_row.get("request_id"),
                    "row_index": manifest_row.get("row_index"),
                    "evidence_request_id": evidence.get("request_id"),
                    "evidence_row_index": evidence.get("row_index"),
                    "reasons": reasons,
                }
            )

    rescue_manifest = [dict(manifest_by_id[video_id]) for video_id in selected_ids]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_records": len(manifest_ids),
        "evidence_records": len(evidence_rows),
        "selected_records": len(rescue_manifest),
        "false_dimension_counts": {dim: int(dim_counter.get(dim, 0)) for dim in ("D5", "D6")},
        "selected_video_ids": selected_ids,
        "selected_request_ids": [row.get("request_id") for row in request_mapping],
        "selected_row_indices": [row.get("row_index") for row in request_mapping],
        "selected_reasons_by_video_id": selected_reasons,
        "request_mapping": request_mapping,
        "extra_evidence_video_ids": extra,
    }
    return rescue_manifest, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--binary-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise TypeError("--manifest-path must contain a JSON list")
    rescue_manifest, summary = build_rescue_manifest(
        manifest=manifest,
        evidence_rows=load_jsonl(args.binary_evidence_jsonl),
    )
    summary.update(
        {
            "manifest_path": str(args.manifest_path),
            "binary_evidence_jsonl": str(args.binary_evidence_jsonl),
            "out_manifest": str(args.out_manifest),
            "out_summary": str(args.out_summary),
        }
    )
    write_json(args.out_manifest, rescue_manifest)
    write_json(args.out_summary, summary)
    print(f"[wrote] {args.out_manifest}")
    print(f"[wrote] {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
