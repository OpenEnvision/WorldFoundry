"""
Subject adherence — VLM-based evaluation of subject appearance and action.

Method:
1. Extract Turn-1 frames from the video (sampled at 2fps)
2. Ask VLM to evaluate two aspects:
   - Subject Appearance Maintenance (1-5): How well the subject's visual appearance is maintained
   - Subject Action (0 or 1): Whether the subject exhibits the described action
3. Score = (maintenance / 5 + action) / 2

Requires:
- Split character prompt data (appearance_part + action_part per case)
- VLM API endpoint (OpenAI-compatible)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from PIL import Image

from ..vlm.vlm_evaluator import VLMClient

logger = logging.getLogger(__name__)

METRIC_NAME = "subject_adherence"

EVAL_PROMPT = """You are a video analysis expert. You will see video frames from Turn 1 of a generated video (sampled at 2fps).

Evaluate the video on two aspects:

**1. Subject Appearance Maintenance (1-5)**
How well does the video maintain the subject's visual appearance (shape, color, clothing, equipment, category)?
Subject appearance: {appearance_part}

Scoring:
- 5: Subject appearance perfectly maintained throughout
- 4: Mostly maintained, minor visual changes
- 3: Some features maintained, noticeable changes
- 2: Few features maintained, significant appearance drift
- 1: Subject barely recognizable or completely changed

**2. Subject Action (0 or 1)**
Does the subject exhibit the described movement/action pattern in the video?
Expected action: {action_part}

- 1: The described action/movement pattern is clearly visible
- 0: The action/movement pattern is not visible or completely different

Answer in JSON format ONLY. Each reason must be within 20 words.
{{"maintenance": <1-5>, "maintenance_reason": "...", "action": <0 or 1>, "action_reason": "..."}}"""


def evaluate_case(
    frames: list[Image.Image] | None = None,
    appearance_part: str = "",
    action_part: str = "",
    vlm_client: VLMClient | None = None,
    video_url: str | None = None,
    **vlm_kwargs,
) -> dict[str, Any]:
    """Evaluate subject adherence for a single case.

    Args:
        frames: Video frames (fallback if video_url not provided)
        appearance_part: Subject appearance description
        action_part: Expected subject action/movement
        vlm_client: Pre-initialized VLM client (or pass vlm_kwargs to create one)
        video_url: Base64-encoded video data URL (preferred over frames)

    Returns:
        Dict with score, maintenance, action, and details
    """
    if not video_url and not frames:
        return {"score": None, "error": "no video_url or frames provided"}

    client = vlm_client or VLMClient(**vlm_kwargs)

    prompt = EVAL_PROMPT.format(
        appearance_part=appearance_part,
        action_part=action_part,
    )

    last_error = None
    for _attempt in range(3):
        try:
            raw_response = client.ask(prompt, frames, video_url=video_url)
            parsed = _parse_json_response(raw_response)

            maintenance = int(parsed["maintenance"])
            action = int(parsed["action"])
            score = round((maintenance / 5.0 + action) / 2.0, 4)

            return {
                "score": score,
                "details": {
                    "maintenance": maintenance,
                    "action": action,
                    "maintenance_reason": parsed.get("maintenance_reason", ""),
                    "action_reason": parsed.get("action_reason", ""),
                },
                "params": {"method": "vlm"},
                "error": None,
            }
        except Exception as e:
            last_error = e

    logger.warning(f"Subject adherence evaluation failed after 3 retries: {last_error}")
    return {"score": None, "details": None, "params": {"method": "vlm"}, "error": str(last_error)}


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from VLM response, handling markdown code blocks, extra text, and trailing commas."""
    import re

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start >= 0:
        depth, end = 0, start
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = raw[start:end]
        candidate = re.sub(r",\s*}", "}", candidate)
        return json.loads(candidate)
    raise json.JSONDecodeError("No JSON found", raw, 0)


def build_case_record(
    case_id: str,
    video_path: str,
    maintenance_score: float | None = None,
    action_score: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if error is None and maintenance_score is not None and action_score is not None:
        score = round((maintenance_score / 5.0 + action_score) / 2.0, 4)
    else:
        score = None
    return {
        "case_id": str(case_id),
        "video_path": str(video_path),
        "score": score,
        "details": {
            "maintenance_score": maintenance_score,
            "action_score": action_score,
        },
        "params": {"method": "vlm"},
        "error": error,
    }


def summarize_model_results(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in case_records if r.get("score") is not None]
    if not valid:
        return {"score": 0.0, "n_cases": 0}
    scores = [r["score"] for r in valid]
    maintenances = [
        r["details"]["maintenance_score"]
        for r in valid
        if r.get("details") and r["details"].get("maintenance_score") is not None
    ]
    actions = [
        r["details"]["action_score"] for r in valid if r.get("details") and r["details"].get("action_score") is not None
    ]
    return {
        "score": round(float(np.mean(scores)), 4),
        "maintenance_mean": round(float(np.mean(maintenances)), 2) if maintenances else None,
        "action_mean": round(float(np.mean(actions)), 2) if actions else None,
        "n_cases": len(valid),
    }
