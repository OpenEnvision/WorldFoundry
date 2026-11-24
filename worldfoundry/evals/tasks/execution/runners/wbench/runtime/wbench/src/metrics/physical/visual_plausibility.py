"""
Physical plausibility — PAVRM reward model scoring.

Method: PAVRM (qwen3vl_a3b) assigns a 0~5 physical plausibility score per video.
- raw_score: PAVRM raw score (0~5)
- score = raw_score / 5.0 (normalized to 0~1, higher = better)

Usage:
    from src.metrics.physical.visual_plausibility import PhysicalPlausibilityEvaluator
    evaluator = PhysicalPlausibilityEvaluator(model_path="path/to/qwen3vl_a3b")
    results = evaluator.score_videos(["video1.mp4", "video2.mp4"])
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.wbench_visual_plausibility import model_dir as pavrm_model_dir

METRIC_NAME = "visual_plausibility"

VIDEO_QUALITY_PROMPT = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos, "
    "please watch the above provided video and give scores for the video's truthfulness and "
    "rationality. i.e., whether the video's overall appreance and motion are consistent with "
    "our common-sense, physical principles.\n"
    "Your rating should be chosen from the following five catefories: Perfect, Good, Fair, Poor, "
    "and Bad. Now please rate this video:"
)

ANCHOR_TOKEN_IDS = [51041, 15216, 60795, 84103, 17082]
ANCHOR_WEIGHTS = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])


class PhysicalPlausibilityEvaluator:
    """Local PAVRM inference using Qwen3-VL reward model (transformers backend)."""

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if model_path is None:
            model_path = os.environ.get("PAVRM_MODEL_PATH") or os.environ.get("WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR")
        if not model_path:
            model_path = str(pavrm_model_dir())
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"PAVRM model path not found: {model_path}. "
                "Set WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR/PAVRM_MODEL_PATH or stage the WBench PAVRM base model."
            )

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except (ImportError, ValueError):
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    @torch.no_grad()
    def score_video(self, video_path: str, fps: float = 2.0) -> dict[str, Any]:
        """Score a single video for physical plausibility.

        Returns:
            Dict with raw_score (0-5), score (0-1), error
        """
        if not os.path.exists(video_path):
            return {"raw_score": None, "score": None, "error": f"Video not found: {video_path}"}

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path, "fps": fps, "max_pixels": 602112},
                        {"type": "text", "text": VIDEO_QUALITY_PROMPT},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True
            )

            logits = outputs.logits[0]
            token_ids = torch.tensor(ANCHOR_TOKEN_IDS, device=logits.device)
            logits_gathered = torch.gather(logits[0], dim=-1, index=token_ids)
            probs = torch.softmax(logits_gathered, dim=-1)
            raw_score = float(torch.sum(probs * ANCHOR_WEIGHTS.to(logits.device)))

            return {
                "raw_score": round(raw_score, 4),
                "score": round(raw_score / 5.0, 4),
                "error": None,
            }
        except Exception as e:
            return {"raw_score": None, "score": None, "error": str(e)}

    def score_videos(self, video_paths: list[str], fps: float = 2.0) -> list[dict[str, Any]]:
        """Score multiple videos sequentially.

        Args:
            video_paths: List of video file paths
            fps: Frames per second to sample from videos

        Returns:
            List of dicts with raw_score (0-5), score (0-1), error
        """
        results = []
        for vp in video_paths:
            results.append(self.score_video(vp, fps))
        return results


def score_video_api(video_path: str, endpoint: str | None = None) -> dict[str, Any]:
    """Score a video via PAVRM HTTP API endpoint (fallback)."""
    import requests

    endpoint = endpoint or os.environ.get("PAVRM_API_URL", "")
    if not endpoint:
        return {"raw_score": None, "score": None, "error": "PAVRM_API_URL not set. Set env var or pass endpoint."}

    abs_path = os.path.abspath(video_path)
    try:
        resp = requests.post(
            endpoint,
            json={"video_url": abs_path, "prompt": "", "dimension": "visual"},
            timeout=120,
        )
        data = resp.json()
        if "detail" in data:
            return {"raw_score": None, "score": None, "error": data["detail"]}
        raw = float(data["score"])
        return {"raw_score": round(raw, 4), "score": round(raw / 5.0, 4), "error": None}
    except Exception as e:
        return {"raw_score": None, "score": None, "error": str(e)}


def compute_case(
    video_path: str, model_path: str | None = None, endpoint: str | None = None, device: str = "cuda"
) -> dict[str, Any]:
    """Evaluate physical plausibility. Tries local model first, falls back to API."""
    model_path = (
        model_path
        or os.environ.get("PAVRM_MODEL_PATH")
        or os.environ.get("WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR")
        or str(pavrm_model_dir())
    )
    if not model_path:
        model_path = str(pavrm_model_dir())
    endpoint = endpoint or os.environ.get("PAVRM_API_URL", "")

    if model_path and os.path.isdir(model_path):
        evaluator = PhysicalPlausibilityEvaluator(model_path=model_path, device=device)
        result = evaluator.score_video(video_path)
    elif endpoint:
        result = score_video_api(video_path, endpoint)
    else:
        return {"score": None, "error": "WBench PAVRM model directory or PAVRM_API_URL is not available"}

    return {
        "score": result["score"],
        "details": {"raw_score": result["raw_score"]},
        "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5"},
        "error": result["error"],
    }


def build_case_record(
    case_id: str,
    video_path: str,
    raw_score: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    score = round(raw_score / 5.0, 4) if raw_score is not None and error is None else None
    return {
        "case_id": str(case_id),
        "video_path": str(video_path),
        "score": score,
        "details": {"raw_score": round(raw_score, 4) if raw_score is not None else None},
        "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5"},
        "error": error,
    }


def summarize_model_results(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in case_records if r.get("score") is not None]
    if not valid:
        return {"score": 0.0, "n_cases": 0}
    scores = [r["score"] for r in valid]
    raws = [r["details"]["raw_score"] for r in valid if r["details"].get("raw_score")]
    return {
        "score": round(float(np.mean(scores)), 4),
        "raw_mean": round(float(np.mean(raws)), 4) if raws else 0.0,
        "n_cases": len(valid),
    }
