"""WorldFoundry facade for NexusScore (OpenS2V-Eval)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.base_models.perception_core.video_text.opens2v_nexus.paths import ensure_opens2v_eval_path


def compute_nexus_score(
    *,
    input_video_folder: str | Path,
    input_image_folder: str | Path,
    input_json_file: str | Path,
    model_config: str | None = None,
    yolo_model_path: str | None = None,
    yolo_clip_model_path: str | None = None,
    gme_model_path: str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run OpenS2V NexusScore subject-consistency evaluation."""
    import torch
    from PIL import Image

    eval_root = ensure_opens2v_eval_path()
    nexus_runtime = __import__(
        "get_nexusscore", fromlist=["load_model_and_config", "sample_video_frames", "yoloworld_inference"]
    )
    load_model_and_config = nexus_runtime.load_model_and_config
    sample_video_frames = nexus_runtime.sample_video_frames
    yoloworld_inference = nexus_runtime.yoloworld_inference
    GmeQwen2VL = __import__("utils.gme.gme_model", fromlist=["GmeQwen2VL"]).GmeQwen2VL

    resolved_model_config = model_config or str(
        eval_root / "utils/yoloworld/configs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py"
    )
    resolved_yolo_model_path = yolo_model_path or os.environ.get(
        "WORLDFOUNDRY_YOLO_WORLD_CKPT",
        "BestWishYsh/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth",
    )

    class _Args:
        model_config = resolved_model_config
        yolo_model_path = resolved_yolo_model_path
        work_dir = None
        cfg_options = None

    yolo_clip_model_path = yolo_clip_model_path or os.environ.get(
        "WORLDFOUNDRY_YOLO_CLIP_MODEL", "openai/clip-vit-base-patch32"
    )
    gme_model_path = gme_model_path or os.environ.get(
        "WORLDFOUNDRY_GME_MODEL_PATH", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/"
    )

    yolo_vision_model, yolo_processor, runner, txt_feats = load_model_and_config(_Args(), yolo_clip_model_path, device)
    gme = GmeQwen2VL(gme_model_path, attn_model="flash_attention_2", device=device)

    with open(input_json_file, encoding="utf-8") as handle:
        json_data = json.load(handle)
    video_root = Path(input_video_folder)
    image_root = Path(input_image_folder)
    results: dict[str, Any] = {}
    for video_path in sorted(video_root.glob("*.mp4")):
        prefix = video_path.stem
        prompt_image_paths = json_data.get(prefix, {}).get("img_paths", [])
        image_labels = json_data.get(prefix, {}).get("class_label", [])
        frames = sample_video_frames(str(video_path), num_frames=32)
        all_prompt_images: list[Image.Image] = []
        all_image_labels: list[str] = []
        all_local_images: list[Image.Image] = []
        all_yolo_world_conf: list[float] = []
        frame_obj = 0
        for frame in frames:
            frame_flag = True
            for prompt_image_path, image_label in zip(prompt_image_paths, image_labels, strict=False):
                label = str(image_label)
                if "face" in prefix and label in {"Man", "Woman"}:
                    continue
                if "human" in prefix and label in {"Man", "Woman"}:
                    label = "human"
                prompt_image = Image.open(image_root / prompt_image_path)
                pred_instances = yoloworld_inference(
                    runner,
                    yolo_vision_model,
                    yolo_processor,
                    txt_feats,
                    frame,
                    prompt_image,
                )
                bboxes = pred_instances["bboxes"]
                confidences = pred_instances["scores"]
                all_yolo_world_conf.extend(confidences)
                if len(bboxes) != 0 and frame_flag:
                    frame_obj += 1
                    frame_flag = False
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    all_local_images.append(frame.crop((x1, y1, x2, y2)))
                    all_prompt_images.append(prompt_image)
                    all_image_labels.append(label)
        if not all_local_images:
            results[prefix] = {"nexus_score": 0.0}
            continue
        e_main_image_corpus = gme.get_image_embeddings(
            images=all_prompt_images, is_query=False, show_progress_bar=False
        )
        e_query = gme.get_text_embeddings(
            texts=all_image_labels,
            instruction="Find an image that matches the given text.",
            show_progress_bar=False,
        )
        e_local_image_corpus = gme.get_image_embeddings(
            images=all_local_images, is_query=False, show_progress_bar=False
        )
        gme_image_score = (e_main_image_corpus * e_local_image_corpus).sum(-1)
        gme_text_score = (e_query * e_local_image_corpus).sum(-1)
        retrieval_score_list = []
        for bbox_conf, text_conf, score in zip(all_yolo_world_conf, gme_text_score, gme_image_score, strict=False):
            if bbox_conf > 0.6 and text_conf > 0.30 and score != 0:
                retrieval_score_list.append(float(score.item()))
        if retrieval_score_list and frame_obj > 0:
            nexus = float(torch.mean(torch.tensor(retrieval_score_list)).item() / frame_obj)
        else:
            nexus = 0.0
        results[prefix] = {"nexus_score": nexus}
    if not results:
        raise ValueError("NexusScore found no videos to score")
    mean_score = float(np.mean([value["nexus_score"] for value in results.values()]))
    return {"mean_nexus_score": mean_score, "videos": results}


def compute_nexus_score_from_results(results: Mapping[str, Mapping[str, float]]) -> float:
    scores = [float(payload["nexus_score"]) for payload in results.values()]
    return float(np.mean(scores))


__all__ = ["compute_nexus_score", "compute_nexus_score_from_results"]
