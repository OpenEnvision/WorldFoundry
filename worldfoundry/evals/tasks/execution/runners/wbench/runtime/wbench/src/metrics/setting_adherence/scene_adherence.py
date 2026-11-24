"""
Scene adherence — VLM-based evaluation of environment maintenance.

Method:
1. Extract Turn-1 frames from the video (sampled at 2fps)
2. Ask VLM to evaluate two aspects:
   - Environment Maintenance (1-5): How well the video maintains scene elements
   - Offscreen Content Appearance (0 or 1): Whether off-screen elements appear
3. Score = (maintenance / 5 + offscreen) / 2

Requires:
- Split environment prompt data (visible_part + offscreen_part per case)
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

METRIC_NAME = "scene_adherence"

EVAL_PROMPT = """You are a video analysis expert. You will see video frames from Turn 1 of a generated video (sampled at 2fps).

Evaluate the video on two aspects:

**1. Environment Maintenance (1-5)**
How well does the video maintain these scene elements that were visible in the initial frame? Also consider whether the visual style remains consistent.
Scene description: {visible_part}
Visual style: {style}

Scoring:
- 5: All elements perfectly maintained, style fully consistent
- 4: Most elements maintained, style mostly consistent, minor issues
- 3: Some elements maintained or style partially drifts
- 2: Few elements maintained, significant changes or style mismatch
- 1: Scene barely matches the description or style completely wrong

**2. Offscreen Content Appearance (0 or 1)**
The following elements were described as existing OFF-SCREEN (not visible in the initial frame). Did ANY of them appear in the video as the camera moved?
Description: {offscreen_part}

- 1: At least one offscreen element appeared
- 0: None of the offscreen elements appeared

Answer in JSON format ONLY. Each reason must be within 20 words.
{{"maintenance": <1-5>, "maintenance_reason": "...", "offscreen": <0 or 1>, "offscreen_reason": "..."}}"""


def evaluate_case(
    frames: list[Image.Image] | None = None,
    visible_part: str = "",
    offscreen_part: str = "",
    style: str = "realistic",
    vlm_client: VLMClient | None = None,
    video_url: str | None = None,
    **vlm_kwargs,
) -> dict[str, Any]:
    """Evaluate scene adherence for a single case.

    Args:
        frames: Video frames (fallback if video_url not provided)
        visible_part: Scene elements visible in the initial frame
        offscreen_part: Scene elements described as off-screen
        style: Visual style of the scene
        vlm_client: Pre-initialized VLM client (or pass vlm_kwargs to create one)
        video_url: Base64-encoded video data URL (preferred over frames)

    Returns:
        Dict with score, maintenance, offscreen, and details
    """
    if not video_url and not frames:
        return {"score": None, "error": "no video_url or frames provided"}

    client = vlm_client or VLMClient(**vlm_kwargs)

    prompt = EVAL_PROMPT.format(
        visible_part=visible_part,
        offscreen_part=offscreen_part,
        style=style,
    )

    last_error = None
    for _attempt in range(3):
        try:
            raw_response = client.ask(prompt, frames, video_url=video_url)
            parsed = _parse_json_response(raw_response)

            maintenance = int(parsed["maintenance"])
            offscreen = int(parsed["offscreen"])
            score = round((maintenance / 5.0 + offscreen) / 2.0, 4)

            return {
                "score": score,
                "details": {
                    "maintenance": maintenance,
                    "offscreen": offscreen,
                    "maintenance_reason": parsed.get("maintenance_reason", ""),
                    "offscreen_reason": parsed.get("offscreen_reason", ""),
                },
                "params": {"method": "vlm", "style": style},
                "error": None,
            }
        except Exception as e:
            last_error = e

    logger.warning(f"Scene adherence evaluation failed after 3 retries: {last_error}")
    return {"score": None, "details": None, "params": {"method": "vlm"}, "error": str(last_error)}


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from VLM response, handling markdown code blocks, extra text, and trailing commas."""
    import re

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)
    # Try direct parse first
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract JSON by finding balanced braces
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
    offscreen_score: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if error is None and maintenance_score is not None and offscreen_score is not None:
        score = round((maintenance_score / 5.0 + offscreen_score) / 2.0, 4)
    else:
        score = None
    return {
        "case_id": str(case_id),
        "video_path": str(video_path),
        "score": score,
        "details": {
            "maintenance_score": maintenance_score,
            "offscreen_score": offscreen_score,
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
    offscreens = [
        r["details"]["offscreen_score"]
        for r in valid
        if r.get("details") and r["details"].get("offscreen_score") is not None
    ]
    return {
        "score": round(float(np.mean(scores)), 4),
        "maintenance_mean": round(float(np.mean(maintenances)), 2) if maintenances else None,
        "offscreen_mean": round(float(np.mean(offscreens)), 2) if offscreens else None,
        "n_cases": len(valid),
    }
