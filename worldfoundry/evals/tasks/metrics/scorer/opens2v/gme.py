"""WorldFoundry facade for GmeScore (OpenS2V-Eval)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.base_models.perception_core.video_text.opens2v_nexus.paths import ensure_opens2v_eval_path


def compute_gme_score(
    *,
    input_video_folder: str | Path,
    input_json_file: str | Path,
    model_path: str | None = None,
    num_frames: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run OpenS2V GmeScore text-video relevance evaluation."""
    ensure_opens2v_eval_path()
    sample_video_frames = __import__("get_gmescore", fromlist=["sample_video_frames"]).sample_video_frames
    GmeQwen2VL = __import__("utils.gme.gme_model", fromlist=["GmeQwen2VL"]).GmeQwen2VL

    model_path = model_path or os.environ.get("WORLDFOUNDRY_GME_MODEL_PATH", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/")
    with open(input_json_file, encoding="utf-8") as handle:
        prompts = json.load(handle)
    gme = GmeQwen2VL(model_path, attn_model="flash_attention_2", device=device)
    video_root = Path(input_video_folder)
    results: dict[str, Any] = {}
    for video_path in sorted(video_root.glob("*.mp4")):
        video_name = video_path.stem
        if video_name not in prompts:
            continue
        text_prompt = prompts[video_name].get("prompt", "")
        frames = sample_video_frames(str(video_path), num_frames=num_frames)
        e_query = gme.get_text_embeddings(
            texts=[text_prompt] * len(frames),
            instruction="Find an image that matches the given text.",
        )
        e_corpus = gme.get_image_embeddings(images=frames, is_query=False, show_progress_bar=False)
        gme_scores = (e_query * e_corpus).sum(-1)
        score = float(gme_scores.mean().detach().item())
        results[video_name] = {"gme_score": score}
    if not results:
        raise ValueError("GmeScore found no videos with matching prompts")
    mean_score = float(np.mean([value["gme_score"] for value in results.values()]))
    return {"mean_gme_score": mean_score, "videos": results}


def compute_gme_score_from_results(results: Mapping[str, Mapping[str, float]]) -> float:
    scores = [float(payload["gme_score"]) for payload in results.values()]
    return float(np.mean(scores))


__all__ = ["compute_gme_score", "compute_gme_score_from_results"]
