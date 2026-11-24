"""WorldFoundry facade for FaceSim-Cur (OpenS2V-Eval)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.base_models.perception_core.video_text.opens2v_nexus.paths import ensure_opens2v_eval_path


def compute_facesim_cur(
    *,
    input_video_folder: str | Path,
    input_image_folder: str | Path,
    input_json_file: str | Path,
    model_path: str | Path | None = None,
    num_frames: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run OpenS2V FaceSim-Cur evaluation and return per-video CurricularFace scores."""
    ensure_opens2v_eval_path()
    import torch

    get_facesim = __import__("get_facesim", fromlist=["get_image_path_from_json", "process_image", "process_video"])
    get_image_path_from_json = get_facesim.get_image_path_from_json
    process_image = get_facesim.process_image
    process_video = get_facesim.process_video
    FaceAnalysis = __import__("insightface.app", fromlist=["FaceAnalysis"]).FaceAnalysis
    get_model = __import__("utils.curricularface", fromlist=["get_model"]).get_model

    model_path = Path(model_path or os.environ.get("WORLDFOUNDRY_OPENS2V_WEIGHT_DIR", "BestWishYsh/OpenS2V-Weight"))
    face_arc_path = model_path / "face_extractor"
    face_cur_path = model_path / "glint360k_curricular_face_r101_backbone.bin"

    face_arc_model = FaceAnalysis(root=str(face_arc_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_arc_model.prepare(ctx_id=0, det_size=(320, 320))
    face_cur_model = get_model("IR_101")([112, 112])
    face_cur_model.load_state_dict(torch.load(face_cur_path, map_location="cpu"))
    face_cur_model = face_cur_model.to(device)
    face_cur_model.eval()

    with open(input_json_file, encoding="utf-8") as handle:
        json_data = json.load(handle)
    video_root = Path(input_video_folder)
    image_root = Path(input_image_folder)
    results: dict[str, Any] = {}
    for video_file in sorted(video_root.glob("*.mp4")):
        file_basename = video_file.name
        if not any(keyword in file_basename for keyword in ("singlehuman", "singleface", "faceobj", "humanobj")):
            continue
        image_path = get_image_path_from_json(json_data, video_file.name, str(image_root))
        if image_path is None:
            continue
        align_face_image, arcface_image_embedding = process_image(face_arc_model, image_path)
        if align_face_image is None:
            continue
        inference = __import__("get_facesim", fromlist=["inference"]).inference

        cur_image_embedding = inference(face_cur_model, align_face_image, device)
        cur_score, arc_score = process_video(
            video_path=str(video_file),
            face_arc_model=face_arc_model,
            face_cur_model=face_cur_model,
            arcface_image_embedding=arcface_image_embedding,
            cur_image_embedding=cur_image_embedding,
            device=device,
            num_frames=num_frames,
        )
        results[video_file.stem] = {
            "cur_score": float(cur_score),
            "arc_score": float(arc_score),
            "facesim_cur": float(cur_score),
        }
    if not results:
        raise ValueError("FaceSim-Cur found no eligible human-domain videos to score")
    mean_cur = float(np.mean([value["cur_score"] for value in results.values()]))
    return {"mean_facesim_cur": mean_cur, "videos": results}


def compute_facesim_cur_from_results(results: Mapping[str, Mapping[str, float]]) -> float:
    """Aggregate mean CurricularFace score from precomputed per-video dicts."""
    scores = [float(payload.get("cur_score", payload.get("facesim_cur", 0.0))) for payload in results.values()]
    if not scores:
        raise ValueError("results must contain at least one video score")
    return float(np.mean(scores))


__all__ = ["compute_facesim_cur", "compute_facesim_cur_from_results"]
