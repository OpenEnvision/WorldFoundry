"""WorldFoundry facade for NaturalScore (OpenS2V-Eval)."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.base_models.perception_core.video_text.opens2v_nexus.paths import ensure_opens2v_eval_path


def _normalize_gpt_score(raw: str) -> float:
    try:
        value = float(str(raw).strip())
    except ValueError:
        return 0.0
    value = max(1.0, min(5.0, value))
    return (value - 1.0) / 4.0


def compute_natural_score(
    *,
    input_video_folder: str | Path,
    api_key: str | None = None,
    model_name: str = "gpt-4o-2024-11-20",
    base_url: str | None = None,
    num_workers: int = 8,
    runs: int = 3,
) -> dict[str, Any]:
    """Run OpenS2V NaturalScore GPT-based naturalness evaluation."""
    ensure_opens2v_eval_path()
    process_folder = __import__("get_naturalscore", fromlist=["process_folder"]).process_folder

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("NaturalScore requires OPENAI_API_KEY or an explicit api_key argument")

    run_maps: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="worldfoundry-naturalscore-") as temp_dir:
        temp_path = Path(temp_dir)
        for index in range(1, runs + 1):
            output_json = temp_path / f"naturalscore_{index}.json"
            process_folder(
                str(input_video_folder),
                str(output_json),
                num_workers,
                api_key,
                model_name=model_name,
                base_url=base_url,
            )
            run_maps.append(json.loads(output_json.read_text(encoding="utf-8")))

    videos = sorted(set().union(*run_maps))
    per_video: dict[str, Any] = {}
    for video in videos:
        normalized = []
        for run_map in run_maps:
            if video in run_map:
                normalized.append(_normalize_gpt_score(run_map[video]))
        if normalized:
            per_video[video] = {
                "natural_score": float(np.mean(normalized)),
                "runs": normalized,
            }
    if not per_video:
        raise ValueError("NaturalScore produced no video scores")
    mean_score = float(np.mean([value["natural_score"] for value in per_video.values()]))
    return {"mean_natural_score": mean_score, "videos": per_video}


def compute_natural_score_from_results(results: Mapping[str, Mapping[str, float]]) -> float:
    scores = [float(payload["natural_score"]) for payload in results.values()]
    return float(np.mean(scores))


__all__ = ["compute_natural_score", "compute_natural_score_from_results"]
