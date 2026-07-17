#!/usr/bin/env python3
"""Overlay Qwen3-VL rescue evidence onto complete baseline evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen3vl_rescue_evidence_overlay_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
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


def _index_by_video_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"{label} row {index} is missing video_id")
        if video_id in rows_by_id:
            duplicates.append(video_id)
        rows_by_id[video_id] = row
    if duplicates:
        raise ValueError(f"duplicate {label} video_id values: {sorted(set(duplicates))[:10]}")
    return rows_by_id


def overlay_evidence(
    *,
    manifest: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = _index_by_video_id(baseline_rows, label="baseline")
    rescue_by_id = _index_by_video_id(rescue_rows, label="rescue")
    manifest_ids = _manifest_ids(manifest)
    missing_baseline = [video_id for video_id in manifest_ids if video_id not in baseline_by_id]
    output_rows: list[dict[str, Any]] = []
    replaced_ids: list[str] = []
    request_mapping: list[dict[str, Any]] = []
    for video_id in manifest_ids:
        row = rescue_by_id.get(video_id) or baseline_by_id.get(video_id)
        if row is None:
            request_mapping.append(
                {
                    "video_id": video_id,
                    "request_id": None,
                    "row_index": None,
                    "baseline_present": False,
                    "rescue_present": False,
                    "overlay_source": None,
                }
            )
            continue
        out = dict(row)
        if video_id in rescue_by_id:
            out["evidence_overlay_source"] = "guarded_rescue"
            replaced_ids.append(video_id)
        else:
            out["evidence_overlay_source"] = "binary_baseline"
        output_rows.append(out)
        request_mapping.append(
            {
                "video_id": video_id,
                "request_id": out.get("request_id"),
                "row_index": out.get("row_index"),
                "baseline_present": video_id in baseline_by_id,
                "rescue_present": video_id in rescue_by_id,
                "overlay_source": out["evidence_overlay_source"],
            }
        )
    unused_rescue_ids = sorted(set(rescue_by_id) - set(manifest_ids))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_records": len(manifest_ids),
        "baseline_records": len(baseline_by_id),
        "rescue_records": len(rescue_by_id),
        "output_records": len(output_rows),
        "replaced_records": len(replaced_ids),
        "replaced_video_ids": replaced_ids,
        "missing_baseline_video_ids": missing_baseline,
        "unused_rescue_video_ids": unused_rescue_ids,
        "request_mapping": request_mapping,
    }
    return output_rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--baseline-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--rescue-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise TypeError("--manifest-path must contain a JSON list")
    output_rows, summary = overlay_evidence(
        manifest=manifest,
        baseline_rows=load_jsonl(args.baseline_evidence_jsonl),
        rescue_rows=load_jsonl(args.rescue_evidence_jsonl),
    )
    write_jsonl(args.out_evidence_jsonl, output_rows)
    summary.update(
        {
            "manifest_path": str(args.manifest_path),
            "baseline_evidence_jsonl": str(args.baseline_evidence_jsonl),
            "rescue_evidence_jsonl": str(args.rescue_evidence_jsonl),
            "out_evidence_jsonl": str(args.out_evidence_jsonl),
        }
    )
    write_json(args.out_summary_json, summary)
    print(f"[wrote] {args.out_evidence_jsonl}")
    print(f"[wrote] {args.out_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
