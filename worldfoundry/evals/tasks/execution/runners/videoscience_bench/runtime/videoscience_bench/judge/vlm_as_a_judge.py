from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .api_manager import judge_experiment
except ImportError:
    from judge.api_manager import judge_experiment

# ---------- scoring weighting (all 1–4) ----------
WEIGHTS = {
    "prompt_consistency": 0.20,
    "expected_phenomenon": 0.30,  # emphasize scientific reasoning
    "coherence": 0.20,
    "immutability": 0.15,
    "dynamism": 0.15,
}

REPORT_MD = """# VLM Judge Report

**When:** {when}

**Provider/Model:** {provider}/{model}

**Phenomenon:** {phenomenon}

**Overall:** {overall:.1f} / 4

{rubric_section}## Summary
{summary}

## Notable issues
{issues}

## Evidence (timestamps)
- Candidate: {cand_frames}
{ref_section}
"""


# ---------- Parsing helpers ----------
def _try_json_loads(blob: str) -> dict[str, Any]:
    return json.loads(blob)


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """
    Try, in order:
      1) direct JSON
      2) fenced ```json ... ```
      3) fenced ``` ... ```
      4) longest {...} substring that parses
    """
    obj = _try_json_loads(text)
    if obj:
        return obj

    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        obj = _try_json_loads(m.group(1))
        if obj:
            return obj

    for m in re.finditer(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        obj = _try_json_loads(m.group(1))
        if obj:
            return obj

    candidates = list(re.finditer(r"\{.*\}", text, flags=re.DOTALL))
    candidates.sort(key=lambda m: len(m.group(0)), reverse=True)
    for m in candidates:
        obj = _try_json_loads(m.group(0))
        if obj:
            return obj

    return {}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(x: Any) -> float | None:
    return float(x)


def _normalize_1to4(v: Any) -> float:
    """
    Robustly map provider values to 1–4.
    - If already in [1,4], clamp and return.
    - If looks like 0–100, map 0→1 and 100→4.
    - Missing → 1.0 by default (worst).
    """
    f = _to_float(v)
    if f is None:
        return 1.0
    if f < 1.0 or f > 4.0:
        # Assume 0–100 scale, map linearly to 1–4.
        f = _clamp(f, 0.0, 100.0)
        return 1.0 + 3.0 * (f / 100.0)
    return _clamp(f, 1.0, 4.0)


def _compute_overall_1to4(r: dict[str, Any], w: dict[str, float]) -> float:
    return (
        w["prompt_consistency"] * _normalize_1to4(r.get("prompt_consistency"))
        + w["expected_phenomenon"] * _normalize_1to4(r.get("expected_phenomenon"))
        + w["immutability"] * _normalize_1to4(r.get("immutability"))
        + w["dynamism"] * _normalize_1to4(r.get("dynamism"))
        + w["coherence"] * _normalize_1to4(r.get("coherence"))
    )


def _parse_output_text(output_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns (rubric_scores_1to4, explanations).
    rubric_scores_1to4 keys (all 1–4):
      - prompt_consistency
      - expected_phenomenon
      - immutability
      - dynamism
      - coherence
    explanations: {"summary": str, "issues": [str]}
    """
    data = _extract_json_from_text(output_text)

    scores = {}
    explanations = {}

    if isinstance(data, dict):
        scores = data.get("scores", {}) or data.get("rubric", {}) or {}
        explanations = data.get("explanations", {}) or {}

    # Tolerate case/spacing variants
    raw = {
        "prompt_consistency": scores.get("prompt_consistency") or scores.get("Prompt consistency"),
        "expected_phenomenon": scores.get("expected_phenomenon") or scores.get("Expected phenomenon"),
        "immutability": scores.get("immutability") or scores.get("Immutability"),
        "dynamism": scores.get("dynamism") or scores.get("Correct Dynamism") or scores.get("dynamism_other_laws"),
        "coherence": scores.get("coherence")
        or scores.get("Spatio-Temporal Continuity")
        or scores.get("spatio_temporal_continuity"),
    }

    # Normalize to 1–4
    norm = {k: _normalize_1to4(v) for k, v in raw.items()}

    # Explanations normalization
    if not isinstance(explanations, dict):
        explanations = {}
    summary = explanations.get("summary", "")
    issues = explanations.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    return norm, {"summary": str(summary), "issues": [str(x) for x in issues]}


def _format_rubric_section(r: dict, overall: float) -> str:
    if not r:
        return ""
    lines = [
        "## Rubric scores (1–4)",
        f"- Prompt consistency: {float(r.get('prompt_consistency', 1.0)):.1f} / 4",
        f"- Expected phenomenon: {float(r.get('expected_phenomenon', 1.0)):.1f} / 4",
        f"- Immutability: {float(r.get('immutability', 1.0)):.1f} / 4",
        f"- Dynamism (other physical laws): {float(r.get('dynamism', 1.0)):.1f} / 4",
        f"- Coherence (across frames): {float(r.get('coherence', 1.0)):.1f} / 4",
        f"- Overall (weighted): {overall:.1f} / 4",
    ]
    return "\n".join(lines) + "\n\n"


# add somewhere above main()
def _is_remote_or_missing(p: str | None) -> bool:
    if not p:
        return True
    u = urlparse(p)
    if u.scheme in {"http", "https", "gs"}:
        return True

    return not Path(p).exists()


def _load_checklist(path: str | None) -> dict[str, Any]:
    """
    Load a checklist JSON file if provided.

    Expected structure (example):

    {
      "phenomenon_congruency": ["..."],
      "correct_dynamism": ["..."],
      "spatio_temporal_continuity": ["..."],
      "immutability": ["..."],
      "interaction_realism": ["..."]
    }
    """
    if not path:
        return {}

    text = Path(path).read_text()
    data = json.loads(text)
    if isinstance(data, dict):
        return data
    else:
        print(f"[vlm_as_a_judge] Checklist JSON at {path} is not an object; ignoring.", file=sys.stderr)

    return {}


def _load_cv_metrics(path: str | None) -> dict[str, Any]:
    """
    Load CV-based metrics JSON produced by cv_metrics.py (optional).
    """
    if not path:
        return {}

    text = Path(path).read_text()
    data = json.loads(text)

    if isinstance(data, dict):
        return data

    print(f"[vlm_as_a_judge] CV metrics JSON at {path} is not an object; ignoring.", file=sys.stderr)
    return {}


def main():
    ap = argparse.ArgumentParser(description="Judge science experiment video quality with a VLM.")
    ap.add_argument("--provider", required=True, help="openai | gemini | anthropic")
    ap.add_argument(
        "--model",
        required=True,
        help="e.g., gpt-5-pro-2025-10-06, gpt-4o-2024-08-06, gemini-2.5-flash, claude-3-7-sonnet",
    )
    ap.add_argument("--video", required=True, help="Path to candidate video (or image)")
    ap.add_argument("--phenomenon", required=True, help="Ground-truth phenomenon name")
    ap.add_argument("--description", required=True, help="Authoritative description of expected behavior")
    ap.add_argument("--ref_video", default=None, help="Optional reference ground-truth video path")
    ap.add_argument("--max_frames", type=int, default=24)
    ap.add_argument("--fps", type=float, default=None, help="Optional sampling fps override")
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--json_out", default=None, help="Where to write full JSON result")
    ap.add_argument("--md_out", default=None, help="Where to write a Markdown report")
    ap.add_argument(
        "--checklist_json", default=None, help="Optional checklist JSON file with expected visual items per category"
    )
    ap.add_argument(
        "--cv_json",
        default=None,
        help="Optional CV metrics JSON (e.g., GroundingDINO/ByteTrack/RAFT/CLIP4Clip/LPIPS) for auxiliary evidence",
    )
    args = ap.parse_args()

    # --- Guard: if remote URL or missing path, pass None so providers can skip frame extraction ---
    video_arg = "" if _is_remote_or_missing(args.video) else args.video
    ref_arg = "" if _is_remote_or_missing(args.ref_video) else args.ref_video

    # Optional: small log to stderr so you can see why frames are missing
    if not video_arg:
        print(f"[vlm_as_a_judge] Skipping candidate frames (remote or missing): {args.video}", file=sys.stderr)
    if not ref_arg and args.ref_video:
        print(f"[vlm_as_a_judge] Skipping reference frames (remote or missing): {args.ref_video}", file=sys.stderr)

    checklist = _load_checklist(args.checklist_json)
    cv_metrics = _load_cv_metrics(args.cv_json)

    # Build extra payload
    extra: dict[str, Any] = {}
    if checklist:
        extra["checklist"] = checklist
    if cv_metrics:
        extra["cv_metrics"] = cv_metrics

    # Call provider via manager (returns raw output_text + evidence)
    res = judge_experiment(
        provider=args.provider,
        model=args.model,
        video_path=video_arg,
        phenomenon=args.phenomenon,
        gt_description=args.description,
        ref_video_path=ref_arg,
        max_frames=args.max_frames,
        fps=args.fps,
        timeout_s=args.timeout_s,
        extra=extra or None,
    )

    output_text = res.get("output_text", "") or ""
    evidence = res.get("evidence", {}) or {}

    # Parse model output and compute overall score (1–4)
    rubric_scores, explanations = _parse_output_text(output_text)
    overall = _compute_overall_1to4(rubric_scores, WEIGHTS)

    # Build normalized JSON result for optional export
    full_out = {
        "provider": res.get("provider"),
        "model": res.get("model"),
        "scores": {
            "overall": overall,  # 1–4
        },
        "explanations": {
            "summary": explanations.get("summary", ""),
            "issues": explanations.get("issues", []),
        },
        "evidence": {
            "candidate_frames": evidence.get("candidate_frames", []),
            "reference_frames": evidence.get("reference_frames", []),
        },
        "rubric": {
            **rubric_scores,  # all 1–4
            "overall_weighted": overall,
            "weights": dict(WEIGHTS),
        },
        # Include the checklist we used
        "checklist": checklist or {},
        "cv_metrics": cv_metrics or {},
        "output_text": output_text,
        "raw": res.get("raw", {}),
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(full_out, indent=2))
        print(f"Wrote JSON to: {args.json_out}")

    # Render Markdown report (timezone-aware UTC)
    md = REPORT_MD.format(
        when=datetime.now(timezone.utc).isoformat(),
        provider=res.get("provider"),
        model=res.get("model"),
        phenomenon=args.phenomenon,
        overall=overall,
        rubric_section=_format_rubric_section(rubric_scores, overall),
        summary=explanations.get("summary", "") or "(no summary parsed)",
        issues="\n".join(f"- {s}" for s in (explanations.get("issues") or [])) or "- (none reported)",
        cand_frames=", ".join(evidence.get("candidate_frames", [])) or "(frames hidden)",
        ref_section=("- Reference: " + ", ".join(evidence.get("reference_frames", [])))
        if evidence.get("reference_frames")
        else "",
    )

    if args.md_out:
        Path(args.md_out).write_text(md)
        print(f"Wrote report --> {args.md_out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
