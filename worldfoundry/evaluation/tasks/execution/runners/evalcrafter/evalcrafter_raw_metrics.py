"""Bounded raw-video metrics from the official EvalCrafter implementation.

The upstream repository evaluates three metrics with one CLIP ViT-B/32 model:
text/video alignment, adjacent-frame consistency, and first-frame consistency.
This module keeps those formulas while removing upstream cwd, logging, and
output-directory side effects.  It intentionally does not synthesize the
remaining EvalCrafter metrics or the 17-metric aggregate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.evaluation.tasks.execution.framework.io import write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
    resolve_official_metric_prompt_path,
)

OFFICIAL_SOURCE_REVISION = "1275bda05e74f295ad2585e3ebde8e95a81a15af"
OFFICIAL_SOURCE_FILE = "metrics/Scores_with_CLIP/Scores_with_CLIP.py"
SUPPORTED_RAW_METRICS = (
    "clip_score",
    "clip_temp_score",
    "face_consistency_score",
)
DEFAULT_RAW_METRICS = SUPPORTED_RAW_METRICS


def parse_raw_metrics(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return a validated, de-duplicated metric selection."""

    if value is None:
        return DEFAULT_RAW_METRICS
    raw_values = [value] if isinstance(value, str) else list(value)
    selected: list[str] = []
    for raw in raw_values:
        for metric in str(raw).replace(",", " ").split():
            if metric == "all":
                metric_values = SUPPORTED_RAW_METRICS
            else:
                metric_values = (metric,)
            for item in metric_values:
                if item not in SUPPORTED_RAW_METRICS:
                    supported = ", ".join(SUPPORTED_RAW_METRICS)
                    raise ValueError(f"unsupported EvalCrafter raw metric {item!r}; supported: {supported}")
                if item not in selected:
                    selected.append(item)
    if not selected:
        raise ValueError("at least one EvalCrafter raw metric is required")
    return tuple(selected)


def resolve_clip_model_path(explicit: Path | None = None, *, evalcrafter_root: Path | None = None) -> Path:
    """Resolve the local PyTorch CLIP checkpoint used by official metrics."""

    env_value = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CLIP_MODEL")
    shared_env_value = os.environ.get("WORLDFOUNDRY_CLIP_VIT_B32_MODEL_DIR")
    candidates = (
        explicit,
        Path(env_value).expanduser() if env_value else None,
        Path(shared_env_value).expanduser() if shared_env_value else None,
        hfd_root_path("openai--clip-vit-base-patch32"),
        checkpoint_root_path("evalcrafter", "checkpoints", "clip-vit-base-patch32"),
        evalcrafter_root / "checkpoints" / "clip-vit-base-patch32" if evalcrafter_root else None,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.expanduser().resolve()
        has_weights = (path / "pytorch_model.bin").is_file() or (path / "model.safetensors").is_file()
        config_path = path / "config.json"
        if not (path.is_dir() and config_path.is_file() and has_weights):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        vision_config = config.get("vision_config")
        if isinstance(vision_config, Mapping):
            try:
                if int(vision_config.get("patch_size") or 0) == 32:
                    return path
            except (TypeError, ValueError):
                continue
    expected = hfd_root_path("openai--clip-vit-base-patch32")
    raise FileNotFoundError(
        "EvalCrafter CLIP checkpoint is missing. Download openai-mirror/clip-vit-base-patch32 "
        f"from ModelScope into {expected}, set WORLDFOUNDRY_CLIP_VIT_B32_MODEL_DIR, or pass --clip-model."
    )


def _read_rgb_frames(video_path: Path) -> list[Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open EvalCrafter video: {video_path}")
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"EvalCrafter video contains no decodable frames: {video_path}")
    return frames


def _image_features(model: Any, pixel_values: Any, *, device: Any, batch_size: int) -> Any:
    import torch

    chunks = []
    with torch.inference_mode():
        for start in range(0, int(pixel_values.shape[0]), batch_size):
            batch = pixel_values[start : start + batch_size].to(device)
            output = model.get_image_features(pixel_values=batch)
            chunks.append(getattr(output, "pooler_output", output))
    features = torch.cat(chunks, dim=0)
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _alignment_features(model: Any, frames: Sequence[Any], *, device: Any, batch_size: int) -> Any:
    import cv2
    import numpy as np
    import torch

    resized = np.stack([cv2.resize(frame, (224, 224)) for frame in frames])
    pixels = torch.from_numpy(resized).permute(0, 3, 1, 2).float()
    return _image_features(model, pixels, device=device, batch_size=batch_size)


def _consistency_features(model: Any, frames: Sequence[Any], *, device: Any, batch_size: int) -> Any:
    import torch
    from torchvision.transforms import Resize

    resize = Resize([224, 224])
    pixels = torch.stack(
        [resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames]
    )
    return _image_features(model, pixels, device=device, batch_size=batch_size)


def _text_feature(model: Any, tokenizer: Any, prompt: str, *, device: Any) -> Any:
    import torch

    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.inference_mode():
        output = model.get_text_features(input_ids=tokens["input_ids"].to(device))
        feature = getattr(output, "pooler_output", output)
    return feature / feature.norm(p=2, dim=-1, keepdim=True)


def _score_video(
    *,
    video_path: Path,
    prompt: str,
    metrics: tuple[str, ...],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
) -> tuple[dict[str, float], int]:
    frames = _read_rgb_frames(video_path)
    scores: dict[str, float] = {}
    if "clip_score" in metrics:
        image_features = _alignment_features(model, frames, device=device, batch_size=batch_size)
        text_feature = _text_feature(model, tokenizer, prompt, device=device)
        scores["clip_score"] = float((image_features @ text_feature.T).mean().item())

    consistency_metrics = {"clip_temp_score", "face_consistency_score"}.intersection(metrics)
    if consistency_metrics:
        if len(frames) < 2:
            joined = ", ".join(sorted(consistency_metrics))
            raise ValueError(f"{video_path} needs at least two frames for {joined}")
        features = _consistency_features(model, frames, device=device, batch_size=batch_size)
        if "clip_temp_score" in metrics:
            scores["clip_temp_score"] = float((features[:-1] * features[1:]).sum(dim=-1).mean().item())
        if "face_consistency_score" in metrics:
            scores["face_consistency_score"] = float((features[1:] @ features[0].unsqueeze(1)).mean().item())
    return scores, len(frames)


def run_raw_video_metrics(
    *,
    videos_dir: Path,
    prompt_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    metrics: str | Sequence[str] | None = None,
    clip_model_path: Path | None = None,
    evalcrafter_root: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    use_official_metric_prompts: bool = True,
) -> dict[str, Any]:
    """Execute the supported official metric subset and write raw artifacts."""

    selected = parse_raw_metrics(metrics)
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    videos_dir = videos_dir.expanduser().resolve()
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"EvalCrafter videos directory not found: {videos_dir}")
    selected_records = [
        record
        for record in prompt_records
        if (videos_dir / f"{record['prompt_id']}.mp4").is_file()
    ]
    if not selected_records:
        raise FileNotFoundError(
            f"no prompt-matched ####.mp4 videos found under {videos_dir}; expected names such as 0000.mp4"
        )

    resolved_model = resolve_clip_model_path(clip_model_path, evalcrafter_root=evalcrafter_root)

    import torch
    from transformers import AutoTokenizer, CLIPModel

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {resolved_device!r}, but CUDA is unavailable")

    model = CLIPModel.from_pretrained(str(resolved_model), local_files_only=True).to(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model), local_files_only=True)
    model.eval()

    per_sample: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {metric: [] for metric in selected}
    for record in selected_records:
        prompt_id = str(record["prompt_id"])
        video_path = videos_dir / f"{prompt_id}.mp4"
        prompt = str(record["prompt"])
        prompt_source_path: Path | None = None
        if "clip_score" in selected and use_official_metric_prompts:
            prompt_source_path = resolve_official_metric_prompt_path(prompt_id, repo_root=evalcrafter_root)
            prompt = prompt_source_path.read_text(encoding="utf-8", errors="replace").strip()
        scores, frame_count = _score_video(
            video_path=video_path,
            prompt=prompt,
            metrics=selected,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            batch_size=batch_size,
        )
        for metric, score in scores.items():
            values[metric].append(score)
            per_sample.append(
                {
                    "sample_id": prompt_id,
                    "metric_id": metric,
                    "raw_score": score,
                    "raw_score_unit": "cosine_similarity_0_1",
                    "available": True,
                    "video_path": str(video_path),
                    "frame_count": frame_count,
                    "source": "evalcrafter_official_clip_metric",
                    "prompt_source_path": (
                        str(prompt_source_path) if metric == "clip_score" and prompt_source_path is not None else None
                    ),
                }
            )

    aggregate = {metric: sum(scores) / len(scores) for metric, scores in values.items()}
    official_export = {metric: round(score * 100.0, 2) for metric, score in aggregate.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_metrics.jsonl"
    raw_metrics_path = output_dir / "raw_metrics.json"
    final_result_path = output_dir / "final_result.txt"
    write_jsonl(per_sample_path, per_sample)
    raw_metrics_path.write_text(
        json.dumps(official_export, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_result_path.write_text(f"Metrics: {official_export!r}\n", encoding="utf-8")
    return {
        "backend": "raw_video",
        "evidence_scope": "bounded_official_clip_metric_subset",
        "results_path": str(final_result_path.resolve()),
        "raw_metrics_path": str(raw_metrics_path.resolve()),
        "per_sample_metrics_path": str(per_sample_path.resolve()),
        "executed_metrics": list(selected),
        "metric_sample_counts": {metric: len(scores) for metric, scores in values.items()},
        "per_sample_metric_unit": "cosine_similarity_0_1",
        "aggregate_metric_unit": "official_percent_0_100",
        "clip_prompt_mode": (
            "official_metric_prompt_files"
            if "clip_score" in selected and use_official_metric_prompts
            else "custom_prompt_manifest"
            if "clip_score" in selected
            else "not_used"
        ),
        "video_count": len(selected_records),
        "prompt_count": len(prompt_records),
        "clip_model_path": str(resolved_model),
        "device": str(resolved_device),
        "batch_size": batch_size,
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "official_source_file": OFFICIAL_SOURCE_FILE,
    }


__all__ = [
    "DEFAULT_RAW_METRICS",
    "OFFICIAL_SOURCE_FILE",
    "OFFICIAL_SOURCE_REVISION",
    "SUPPORTED_RAW_METRICS",
    "parse_raw_metrics",
    "resolve_clip_model_path",
    "run_raw_video_metrics",
]
