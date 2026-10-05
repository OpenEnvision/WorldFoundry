#!/usr/bin/env python3
"""Run physical, interaction, and 3D consistency scoring in one entrypoint.

 python worldeval/scripts/score_video_physical_3d.py \
    --video motor/motor_longlive.mp4 \
    --gt-video motor/motor_gt.mp4 \
    --camera-trajectory motor/motor_gt_da3_camera_trajectory.json \
    --physical-max-frames 64 \
    --three-d-max-frames 32 \
    --output outputs/motor_longlive_physical_3d.json

 python worldeval/scripts/score_video_physical_3d.py \
    --video worldeval/examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3.mp4 \
    --gt-video worldeval/examples/02BEoux44n8_part3/ref_02BEoux44n8_part3.mp4 \
    --prompt-json worldeval/examples/02BEoux44n8_part3/prompt.json \
    --chunk-json worldeval/examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3_chunk_timestamps.json \
    --skip-physical \
    --skip-3d \
    --vlm-backend local \
    --vlm-model worldeval/weights/QwenVL \
    --output outputs/example_interaction.json

    
CUDA_VISIBLE_DEVICES=5 QWENVL_DEVICE=cuda:0 python worldeval/scripts/score_video_physical_3d.py \
    --video motor/motor_longlive.mp4 \
    --gt-video motor/motor_gt.mp4 \
    --camera-trajectory motor/motor_gt_da3_camera_trajectory.json \
    --physical-max-frames 64 \
    --sam-device 0 \
    --three-d-max-frames 32 \
    --three-d-model-name worldeval/weights/da3 \
    --da-device 0 \
    --vlm-backend local \
    --vlm-model worldeval/weights/QwenVL \
    --three-d-vlm-backend local \
    --three-d-scoring-model worldeval/weights/QwenVL \
    --output outputs/motor_longlive_physical_3d.json


CUDA_VISIBLE_DEVICES=5 QWENVL_DEVICE=cuda:0 python worldeval/scripts/score_video_physical_3d.py \
    --video outputs/yume_embodied_case1/generated.mp4 \
    --gt-video outputs/yume_embodied_case1/reference_case1.mp4 \
    --physical-max-frames 64 \
    --sam-device 0 \
    --three-d-max-frames 64 \
    --three-d-model-name worldeval/weights/da3 \
    --da-device 0 \
    --vlm-backend local \
    --vlm-model worldeval/weights/QwenVL \
    --three-d-vlm-backend local \
    --three-d-scoring-model worldeval/weights/QwenVL \
    --output outputs/cosmos_predict2p5_stream_from_json_case1_physical_3d.json    

CUDA_VISIBLE_DEVICES=5 QWENVL_DEVICE=cuda:0 python worldeval/scripts/score_video_physical_3d.py \
    --video outputs/lingbot_case1/generated.mp4 \
    --gt-video outputs/lingbot_case1/reference_case1.mp4 \
    --physical-max-frames 64 \
    --skip-physical \
    --sam-device 0 \
    --three-d-max-frames 64 \
    --three-d-model-name worldeval/weights/da3 \
    --da-device 0 \
    --vlm-backend local \
    --vlm-model worldeval/weights/QwenVL \
    --three-d-vlm-backend local \
    --three-d-scoring-model worldeval/weights/QwenVL \
    --output outputs/lingbot_case1_physical_3d.json

export CUDA_VISIBLE_DEVICES=4
  export QWENVL_DEVICE=cuda:0

  python scripts/score_video_physical_3d.py \
    --video motor/motor_longlive.mp4 \
    --gt-video motor/motor_gt.mp4 \
    --camera-trajectory motor/motor_gt_da3_camera_trajectory.json \
    --physical-max-frames 64 \
    --three-d-max-frames 32 \
    --sam-device 0 \
    --da-device 0 \
    --vlm-backend local \
    --vlm-model weights/QwenVL \
    --three-d-vlm-backend local \
    --three-d-scoring-model weights/QwenVL \
    --output outputs/motor_longlive_physical_3d.json

export CUDA_VISIBLE_DEVICES=4
export QWENVL_DEVICE=cuda:0
export QWENVL_SERVER_URL=http://127.0.0.1:8008
python worldeval/scripts/score_video_physical_3d.py \
    --video outputs_batch/embodied/2ZjVNh154dc_part1/matrix_game2_gen_2ZjVNh154dc_part1.mp4 \
    --gt-video outputs_batch/embodied/2ZjVNh154dc_part1/ref_2ZjVNh154dc_part1.mp4 \
    --prompt-json outputs_batch/embodied/2ZjVNh154dc_part1/prompt.json \
    --chunk-json outputs_batch/embodied/2ZjVNh154dc_part1/matrix_game2_gen_2ZjVNh154dc_part1_chunk_timestamps.json \
    --physical-max-frames 64 \
    --physical-batch-mode dimension \
    --three-d-max-frames 32 \
    --sam-device 0 \
    --da-device 0 \
    --vlm-model worldeval/weights/QwenVL \
    --interaction-vlm-model worldeval/weights/QwenVL \
    --three-d-scoring-model worldeval/weights/QwenVL \
    --three-d-model-name worldeval/weights/da3 \
    --sam3-model worldeval/weights/sam3/sam3.pt \
    --run-clip-interaction \
    --clip-download-root worldeval/weights/clip \
    --output outputs_batch/embodied/2ZjVNh154dc_part1/score_physical_interaction_3d.json 

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_DIR = REPO_ROOT / "physical"
THREE_D_DIR = REPO_ROOT / "3d_metrics"
INTERACTION_DIR = REPO_ROOT / "interaction"

for path in (REPO_ROOT, PHYSICAL_DIR, THREE_D_DIR, INTERACTION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model.vlm import parse_json_content, resolve_model_name, resolve_vlm_backend, text_vlm_call


DEFAULT_VLM_MODEL: str | None = None
DEFAULT_3D_MODEL_NAME = "worldeval/weights/da3"
DEFAULT_SAM3_MODEL = "worldeval/weights/sam3/sam3.pt"
DEFAULT_SAM_DEVICE = "1"
DEFAULT_3D_MAX_FRAMES = 32
DEFAULT_PHYSICAL_MAX_FRAMES = 128
DEFAULT_SAM_MIN_MASK_AREA_RATIO = 0.005
DEFAULT_SAM_MAX_MASKS_PER_FRAME = 3
DEFAULT_PROCESS_RES = 504
DEFAULT_INTERACTION_CHUNK_FRAMES = 4
DEFAULT_INTERACTION_REF_FRAMES = 4
DEFAULT_INTERACTION_TRANSITION_SIDE_FRAMES = 3
DEFAULT_INTERACTION_GLOBAL_FRAMES = 12
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_CLIP_FRAMES_PER_CHUNK = 8
NON_H264_SAM_SUFFIXES = (
    "_mask.mp4",
    "_bbox.mp4",
    "_dynamic_binary_mask.mp4",
    "_dynamic_inpainted.mp4",
)

load_dotenv()


def resolve_prompt(
    video_path: str | os.PathLike[str],
    explicit_prompt: str | None,
    gt_video_path: str | os.PathLike[str] | None = None,
    *,
    required: bool = True,
) -> str:
    if explicit_prompt:
        return explicit_prompt

    candidate_dirs = []
    if gt_video_path:
        candidate_dirs.append(Path(gt_video_path).resolve().parent)
    candidate_dirs.append(Path(video_path).resolve().parent)

    for directory in candidate_dirs:
        for prompt_filename in ("prompt.txt", "prompt.json"):
            prompt_path = directory / prompt_filename
            if not prompt_path.exists():
                continue
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if prompt:
                print(f"Loaded prompt from {prompt_path}")
                return prompt

    if required:
        raise FileNotFoundError("No prompt provided and no sibling prompt.txt or prompt.json was found.")
    return ""


def summarize_prompt_for_3d(
    prompt: str,
    *,
    vlm_model: str | None = None,
    vlm_backend: str | None = None,
) -> str:
    """Rewrite the generation prompt into a static-scene prompt for 3D scoring."""
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    system_prompt = """You rewrite video generation prompts for static 3D reconstruction evaluation.
The reconstruction input has dynamic foreground actors masked and video-inpainted before DA3 reconstruction.
Return strict JSON only."""
    user_content = f"""Rewrite the following original video generation prompt into a concise prompt for judging only the static 3D scene reconstruction.

Requirements:
- Keep static background/environment/layout/materials/lighting/weather/terrain/large structures.
- Keep camera/view behavior if it is present.
- Remove or explicitly ignore dynamic foreground actors and actions, including people, animals, vehicles, limbs, clothing motion, object manipulation, galloping/running/turning, and other moving subjects.
- If the original prompt has no camera information, set camera_behavior to exactly: "camera is static".
- Mention that dynamic foreground actors may be absent because they were masked and video-inpainted before reconstruction.
- Return JSON with exactly these keys: static_scene_description, camera_behavior, ignore_for_3d.

Original prompt:
{prompt}"""

    try:
        response = text_vlm_call(
            system_prompt=system_prompt,
            user_content=user_content,
            model_name=vlm_model,
            backend=vlm_backend,
        )
        parsed = parse_json_content(response)
        if isinstance(parsed, dict):
            scene = str(parsed.get("static_scene_description", "")).strip()
            camera = str(parsed.get("camera_behavior", "")).strip() or "camera is static"
            ignore = str(parsed.get("ignore_for_3d", "")).strip()
            if scene:
                sections = [f"Static scene description: {scene}", f"Expected camera/view behavior: {camera}"]
                if ignore:
                    sections.append(f"Ignore for 3D scoring: {ignore}")
                sections.append(
                    "Dynamic foreground actors may be absent because they are masked and video-inpainted before 3D reconstruction."
                )
                return "\n".join(sections)
    except Exception as exc:
        print(f"[WARN] Failed to rewrite 3D prompt with VLM; falling back to rule-based extraction: {exc}")

    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt

    if not isinstance(parsed, list):
        return _extract_static_scene_prompt_from_text(prompt)

    scene_parts: list[str] = []
    camera_parts: list[str] = []
    fallback_parts: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            value_text = str(value).strip()
            if not value_text or value_text.lower() == "none":
                continue
            key_lower = str(key).lower()
            if "scene" in key_lower or "environment" in key_lower or "background" in key_lower:
                scene_parts.append(value_text)
            elif "camera" in key_lower:
                camera_parts.append(value_text)
            elif key_lower in {"caption", "prompt", "description"}:
                fallback_parts.append(_extract_static_scene_prompt_from_text(value_text))

    return _format_static_scene_prompt(scene_parts, camera_parts, fallback_parts or [prompt])


def _extract_static_scene_prompt_from_text(prompt: str) -> str:
    scene_matches = re.findall(
        r"Scene Dynamics:\s*(.*?)(?=\s*(?:Character Appearance|Character Actions|Camera Movement):|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    camera_matches = re.findall(
        r"Camera Movement:\s*(.*?)(?=\s*(?:Character Appearance|Character Actions|Scene Dynamics):|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _format_static_scene_prompt(scene_matches, camera_matches, [prompt])


def _format_static_scene_prompt(
    scene_parts: list[str],
    camera_parts: list[str],
    fallback_parts: list[str],
) -> str:
    scene_text = " ".join(part.strip() for part in scene_parts if part and part.strip())
    camera_text = " ".join(part.strip() for part in camera_parts if part and part.strip())
    if not scene_text:
        scene_text = " ".join(part.strip() for part in fallback_parts if part and part.strip())

    sections = []
    if scene_text:
        sections.append(f"Static scene description: {scene_text}")
    if camera_text:
        sections.append(f"Expected camera/view behavior: {camera_text}")
    sections.append(
        "Dynamic foreground actors and actions from the original generation prompt may be absent because "
        "they are masked and video-inpainted before 3D reconstruction."
    )
    return "\n".join(sections)


def warn_if_trajectory_mismatches_video(
    *,
    trajectory_path: str | os.PathLike[str] | None,
    reference_video_path: str | os.PathLike[str] | None,
) -> None:
    if not trajectory_path or not reference_video_path:
        return

    trajectory_stem = Path(trajectory_path).stem.lower().replace("_da3_camera_trajectory", "")
    reference_stem = Path(reference_video_path).stem.lower()
    if trajectory_stem and reference_stem and trajectory_stem != reference_stem:
        print(
            "[WARN] The selected camera trajectory may not match the reference video: "
            f"trajectory={Path(trajectory_path).name}, reference_video={Path(reference_video_path).name}"
        )


def default_bbox_video_path(video_path: str | os.PathLike[str]) -> Path:
    video_path = Path(video_path).resolve()
    return video_path.with_name(f"{video_path.stem}_bbox_h264.mp4")


def default_dynamic_masked_video_path(video_path: str | os.PathLike[str]) -> Path:
    video_path = Path(video_path).resolve()
    return video_path.with_name(f"{video_path.stem}_dynamic_inpainted_h264.mp4")


def cleanup_non_h264_sam_sidecars(video_path: str | os.PathLike[str]) -> None:
    video_path = Path(video_path).resolve()
    for suffix in NON_H264_SAM_SUFFIXES:
        sidecar_path = video_path.with_name(f"{video_path.stem}{suffix}")
        try:
            sidecar_path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[WARN] Failed to remove non-H264 SAM sidecar {sidecar_path}: {exc}")
        else:
            print(f"Removed non-H264 SAM sidecar: {sidecar_path}")


def normalize_sam_device(device: str | int) -> str:
    raw = str(device).strip()
    if raw.lower() == "cpu":
        return "cpu"
    if raw.startswith("cuda:"):
        return raw
    if raw.isdigit():
        return f"cuda:{raw}"
    raise ValueError(f"Unsupported --sam-device value: {device}. Use 0/1/2 or cpu.")


def normalize_da_device(device: str | int | None) -> str | None:
    if device is None:
        return None

    raw = str(device).strip()
    if not raw:
        return None
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    if raw.isdigit():
        return raw
    raise ValueError(f"Unsupported --da-device value: {device}. Use 0/1/2.")


def resolve_da_visible_devices(device: str | int | None) -> str | None:
    """Resolve DA device without clobbering an existing CUDA_VISIBLE_DEVICES mask.

    When CUDA_VISIBLE_DEVICES is already set, --da-device is interpreted as a
    visible CUDA slot inside that mask. For example, CUDA_VISIBLE_DEVICES=5 with
    --da-device 0 should keep physical GPU 5 visible as cuda:0.
    """
    normalized = normalize_da_device(device)
    if normalized is None:
        return None

    existing_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not existing_visible:
        return normalized

    visible_devices = [part.strip() for part in existing_visible.split(",") if part.strip()]
    if len(visible_devices) == 1:
        if normalized != "0":
            print(
                "[WARN] CUDA_VISIBLE_DEVICES already exposes a single GPU "
                f"({existing_visible}); ignoring --da-device {normalized} and using visible cuda:0."
            )
        return existing_visible

    device_idx = int(normalized)
    if device_idx >= len(visible_devices):
        raise ValueError(
            f"--da-device {normalized} is out of range for CUDA_VISIBLE_DEVICES={existing_visible}. "
            f"Use a visible slot from 0 to {len(visible_devices) - 1}."
        )
    return visible_devices[device_idx]


def ensure_sam_bbox_video(
    video_path: str | os.PathLike[str],
    *,
    prompt: str,
    vlm_model: str | None,
    vlm_backend: str | None,
    sam3_model: str,
    sam_device: str,
    max_frames: int,
    min_mask_area_ratio: float,
    max_masks_per_frame: int,
    explicit_bbox_video: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> tuple[str, dict[str, Any]]:
    if explicit_bbox_video:
        resolved = str(Path(explicit_bbox_video).resolve())
        return resolved, {"bbox_video": resolved, "source": "explicit"}

    bbox_path = default_bbox_video_path(video_path)
    dynamic_masked_path = default_dynamic_masked_video_path(video_path)
    if bbox_path.exists() and not force:
        print(f"Reusing existing SAM bbox video: {bbox_path}")
        metadata = {"bbox_video": str(bbox_path), "source": "reused"}
        if dynamic_masked_path.exists():
            metadata["dynamic_masked_video"] = str(dynamic_masked_path)
        cleanup_non_h264_sam_sidecars(video_path)
        return str(bbox_path), metadata

    from sam_server_client import run_sam_process_via_server, sam3_server_url

    server_url = sam3_server_url()
    if server_url:
        result = run_sam_process_via_server(
            server_url=server_url,
            video_path=str(Path(video_path).resolve()),
            video_prompt=prompt,
            model_name=vlm_model,
            backend=vlm_backend,
            sam3_model_path=sam3_model,
            device=normalize_sam_device(sam_device),
            max_frames=max_frames,
            min_mask_area_ratio=min_mask_area_ratio,
            max_masks_per_frame=max_masks_per_frame,
        )
        bbox_video = result["outputs"]["bbox_video"]
        cleanup_non_h264_sam_sidecars(video_path)
        return bbox_video, {"bbox_video": bbox_video, "source": "sam3_server", "result": result}

    from sam_process import run_sam_process

    result = run_sam_process(
        video_path=str(Path(video_path).resolve()),
        video_prompt=prompt,
        model_name=vlm_model,
        backend=vlm_backend,
        sam3_model_path=sam3_model,
        device=normalize_sam_device(sam_device),
        max_frames=max_frames,
        min_mask_area_ratio=min_mask_area_ratio,
        max_masks_per_frame=max_masks_per_frame,
    )
    bbox_video = result["outputs"]["bbox_video"]
    cleanup_non_h264_sam_sidecars(video_path)
    return bbox_video, {"bbox_video": bbox_video, "source": "generated", "result": result}


def ensure_dynamic_masked_video(
    video_path: str | os.PathLike[str],
    *,
    prompt: str,
    vlm_model: str | None,
    vlm_backend: str | None,
    sam3_model: str,
    sam_device: str,
    max_frames: int,
    min_mask_area_ratio: float,
    max_masks_per_frame: int,
    explicit_dynamic_masked_video: str | os.PathLike[str] | None = None,
    existing_sam_info: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[str, dict[str, Any]]:
    if explicit_dynamic_masked_video:
        resolved = str(Path(explicit_dynamic_masked_video).resolve())
        return resolved, {"dynamic_masked_video": resolved, "source": "explicit"}

    if existing_sam_info:
        outputs = existing_sam_info.get("result", {}).get("outputs", {})
        dynamic_masked_video = outputs.get("dynamic_masked_video") or existing_sam_info.get("dynamic_masked_video")
        if (
            dynamic_masked_video
            and Path(dynamic_masked_video).exists()
            and Path(dynamic_masked_video).name.endswith("_dynamic_inpainted_h264.mp4")
            and not force
        ):
            cleanup_non_h264_sam_sidecars(video_path)
            return str(Path(dynamic_masked_video).resolve()), {
                "dynamic_masked_video": str(Path(dynamic_masked_video).resolve()),
                "source": existing_sam_info.get("source", "sam"),
            }

    dynamic_masked_path = default_dynamic_masked_video_path(video_path)
    if dynamic_masked_path.exists() and not force:
        print(f"Reusing existing dynamic-masked video for 3D reconstruction: {dynamic_masked_path}")
        cleanup_non_h264_sam_sidecars(video_path)
        return str(dynamic_masked_path), {"dynamic_masked_video": str(dynamic_masked_path), "source": "reused"}

    _, sam_info = ensure_sam_bbox_video(
        video_path,
        prompt=prompt,
        vlm_model=vlm_model,
        vlm_backend=vlm_backend,
        sam3_model=sam3_model,
        sam_device=sam_device,
        max_frames=max_frames,
        min_mask_area_ratio=min_mask_area_ratio,
        max_masks_per_frame=max_masks_per_frame,
        force=True,
    )
    outputs = sam_info.get("result", {}).get("outputs", {})
    dynamic_masked_video = outputs.get("dynamic_masked_video") or sam_info.get("dynamic_masked_video")
    if not dynamic_masked_video:
        raise RuntimeError("SAM processing did not produce a dynamic_masked_video output.")
    cleanup_non_h264_sam_sidecars(video_path)
    return str(Path(dynamic_masked_video).resolve()), {
        "dynamic_masked_video": str(Path(dynamic_masked_video).resolve()),
        "source": sam_info.get("source", "generated"),
        "sam_info": sam_info,
    }


def compute_physical_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    related_results = [item for item in results if item.get("related") is True]

    def score_item(item: dict[str, Any]) -> float:
        confidence = float(item.get("confidence", 0.0))
        return confidence if item.get("compliant") is True else 0.0

    if related_results:
        score_values = [score_item(item) for item in related_results]
        physical_score = sum(score_values) / len(score_values)
    else:
        score_values = []
        physical_score = 0.5

    dimension_scores: dict[str, float | None] = {}
    for dimension in ("mechanics", "thermotics", "material"):
        dim_results = [item for item in related_results if item.get("dimension") == dimension]
        if dim_results:
            dimension_scores[dimension] = sum(score_item(item) for item in dim_results) / len(dim_results)
        else:
            dimension_scores[dimension] = None

    return {
        "score": round(float(physical_score), 4),
        "score_scale": "0-1",
        "default_if_no_related": 0.5,
        "question_count": len(results),
        "related_question_count": len(related_results),
        "compliant_question_count": sum(1 for item in related_results if item.get("compliant") is True),
        "dimension_scores": dimension_scores,
        "results": results,
    }


def run_physical_pipeline(
    *,
    video_bbox: str,
    gt_bbox: str,
    prompt: str,
    vlm_model: str | None,
    vlm_backend: str | None,
    question_ids: list[str] | None,
    batch_mode: str,
) -> dict[str, Any]:
    from vlm_judge_prompt_engineering import evaluate_video

    results = evaluate_video(
        gen_video=video_bbox,
        gt_video=gt_bbox,
        video_prompt=prompt,
        model_name=vlm_model,
        backend=vlm_backend,
        question_ids=question_ids,
        batch_mode=batch_mode,
    )
    return compute_physical_summary(results)


def ensure_camera_trajectory(
    *,
    camera_trajectory_path: str | os.PathLike[str] | None,
    trajectory_video_path: str | os.PathLike[str] | None,
    model_name: str,
    max_frames: int,
    process_res: int,
    da_device: str | None,
    force: bool,
) -> tuple[Any, str | None, str | None]:
    from reward_3d_client import extract_camera_trajectory_via_server, reward_3d_server_url

    if camera_trajectory_path:
        resolved = str(Path(camera_trajectory_path).resolve())
        trajectory = json.loads(Path(resolved).read_text(encoding="utf-8"))
        return trajectory, resolved, "explicit"

    if trajectory_video_path is None:
        return None, None, None

    trajectory_video_path = Path(trajectory_video_path).resolve()
    default_path = trajectory_video_path.parent / f"{trajectory_video_path.stem}_da3_camera_trajectory.json"
    if default_path.exists() and not force:
        print(f"Reusing existing DA3 camera trajectory: {default_path}")
        trajectory = json.loads(default_path.read_text(encoding="utf-8"))
        return trajectory, str(default_path), "reused"

    server_url = reward_3d_server_url()
    if server_url:
        print(f"Generating DA3 camera trajectory via persistent server: {server_url}")
        trajectory, saved_path = extract_camera_trajectory_via_server(
            server_url=server_url,
            video_path=trajectory_video_path,
            output_path=default_path,
            max_frames=max_frames,
            process_res=process_res,
        )
        return trajectory, str(saved_path), "reward_3d_server"

    from extract_da3_camera_trajectory import extract_camera_trajectory

    print(f"Generating DA3 camera trajectory for reference video: {trajectory_video_path}")
    trajectory, saved_path = extract_camera_trajectory(
        trajectory_video_path,
        output_path=default_path,
        model_name=model_name,
        max_frames=max_frames,
        process_res=process_res,
        gpus=resolve_da_visible_devices(da_device),
    )
    return trajectory, str(saved_path), "generated"


def normalize_3d_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_score = float(result.get("final_score", 0.0))
    normalized_score = max(0.0, min(1.0, raw_score / 3.0))
    enriched = dict(result)
    enriched["final_score_raw"] = raw_score
    enriched["final_score_raw_scale"] = "0-3"
    enriched["final_score_normalized"] = round(normalized_score, 4)
    enriched["final_score_normalized_scale"] = "0-1"
    return enriched


def run_3d_pipeline(
    *,
    video_path: str,
    prompt: str,
    model_name: str,
    scoring_model_name: str | None,
    scorer_backend: str | None,
    num_workers: int,
    max_frames: int,
    camera_trajectory: Any,
    da_device: str | None,
) -> dict[str, Any]:
    from reward_3d_client import reward_3d_server_url, score_video_file_via_server

    server_url = reward_3d_server_url()
    if server_url:
        result = score_video_file_via_server(
            server_url=server_url,
            video_path=video_path,
            prompt=prompt,
            max_frames=max_frames,
            camera_trajectory=camera_trajectory,
        )
        return normalize_3d_result(result)

    resolved_da_visible_devices = resolve_da_visible_devices(da_device)
    if resolved_da_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = resolved_da_visible_devices

    from reward_3d import score_video_file

    result = score_video_file(
        video_path=video_path,
        prompt=prompt,
        model_name=model_name,
        scorer_type=scorer_backend,
        scoring_model_name=scoring_model_name,
        use_lpips=False,
        num_workers=num_workers,
        max_frames=max_frames,
        camera_trajectory=camera_trajectory,
    )
    return normalize_3d_result(result)


def offload_qwenvl_model_cache_before_3d() -> dict[str, Any]:
    """Release local QwenVL weights held by the main process before starting 3D workers."""
    try:
        from model.qwenvl import clear_qwenvl_model_cache
    except Exception as exc:
        return {"attempted": True, "released_bundles": None, "error": str(exc)}

    released_bundles = clear_qwenvl_model_cache(empty_cuda_cache=True)
    return {"attempted": True, "released_bundles": released_bundles, "error": None}


def normalize_interaction_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    raw_score = summary.get("overall")
    normalized_score = None
    if raw_score is not None:
        normalized_score = max(0.0, min(1.0, float(raw_score) / 5.0))

    enriched = dict(result)
    enriched["overall_raw"] = None if raw_score is None else float(raw_score)
    enriched["overall_raw_scale"] = "0-5"
    enriched["score"] = None if normalized_score is None else round(normalized_score, 4)
    enriched["score_scale"] = "0-1"
    return enriched


def run_interaction_pipeline(
    *,
    video_path: str,
    ref_video_path: str,
    prompt_json: str,
    chunk_json: str,
    vlm_model: str | None,
    vlm_backend: str | None,
    chunk_frames: int,
    ref_frames: int,
    transition_side_frames: int,
    global_frames: int,
    skip_chunk: bool,
    skip_transition: bool,
    skip_global: bool,
) -> dict[str, Any]:
    from vlm_inter import evaluate_interaction

    result = evaluate_interaction(
        gen_video=video_path,
        ref_video=ref_video_path,
        prompt_json=prompt_json,
        chunk_json=chunk_json,
        model_name=vlm_model,
        backend=vlm_backend,
        chunk_frames=chunk_frames,
        ref_frames=ref_frames,
        transition_side_frames=transition_side_frames,
        global_frames=global_frames,
        skip_chunk=skip_chunk,
        skip_transition=skip_transition,
        skip_global=skip_global,
    )
    return normalize_interaction_result(result)


def run_clip_interaction_pipeline(
    *,
    video_path: str,
    prompt_json: str,
    chunk_json: str,
    model_name: str,
    device: str | None,
    frames_per_chunk: int,
    download_root: str | None,
) -> dict[str, Any]:
    from clip_score import evaluate_clip_semantic_adherence

    return evaluate_clip_semantic_adherence(
        gen_video=video_path,
        prompt_json=prompt_json,
        chunk_json=chunk_json,
        model_name=model_name,
        device=device,
        frames_per_chunk=frames_per_chunk,
        download_root=download_root,
    )


def combine_scores(
    physical_summary: dict[str, Any] | None,
    interaction_summary: dict[str, Any] | None,
    three_d_summary: dict[str, Any] | None,
) -> float | None:
    available_scores: list[float] = []
    if physical_summary is not None:
        available_scores.append(float(physical_summary["score"]))
    if interaction_summary is not None and interaction_summary.get("score") is not None:
        available_scores.append(float(interaction_summary["score"]))
    if three_d_summary is not None:
        available_scores.append(float(three_d_summary["final_score_normalized"]))
    if not available_scores:
        return None
    return round(sum(available_scores) / len(available_scores), 4)


def print_score_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FINAL SCORES")
    print("=" * 80)

    physical = summary.get("physical")
    if physical is None:
        print("Physical consistency: skipped")
    else:
        print(
            "Physical consistency: "
            f"{physical['score']:.4f} "
            f"(related={physical['related_question_count']}, compliant={physical['compliant_question_count']})"
        )

    three_d = summary.get("three_d")
    if three_d is None:
        print("3D consistency: skipped")
    else:
        print(
            "3D consistency: "
            f"{three_d['final_score_raw']:.4f}/3.0000 "
            f"(normalized={three_d['final_score_normalized']:.4f})"
        )

    interaction = summary.get("interaction")
    if interaction is None:
        print("Physical interaction 3D: skipped")
    elif interaction.get("score") is None:
        print("Physical interaction 3D: N/A")
    else:
        print(
            "Physical interaction 3D: "
            f"{interaction['overall_raw']:.4f}/5.0000 "
            f"(normalized={interaction['score']:.4f})"
        )

    clip_interaction = summary.get("clip_interaction")
    if clip_interaction is not None:
        clip_summary = clip_interaction.get("summary", {})
        print(
            "CLIP interaction semantic adherence: "
            f"{clip_summary.get('semantic_adherence')} "
            f"(chunks={clip_summary.get('num_scored_chunks')}/{clip_summary.get('num_chunks')})"
        )

    combined = summary.get("combined_score")
    if combined is None:
        print("Combined consistency: N/A")
    else:
        print(f"Combined consistency: {combined:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physical, interaction, and 3D consistency pipelines together")
    parser.add_argument("--video", required=True, help="Candidate/generated video to evaluate")
    parser.add_argument("--gt-video", help="Reference video for the physical pipeline")
    parser.add_argument("--trajectory-video", help="Video used to extract a DA3 camera trajectory if none is provided")
    parser.add_argument("--prompt", help="Prompt text. Defaults to prompt.txt next to --gt-video or --video")
    parser.add_argument("--prompt-json", help="JSON file containing per-chunk interaction prompts/actions/captions")
    parser.add_argument("--chunk-json", help="JSON file containing generated chunk timestamps")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--vlm-backend",
        default=None,
        help="VLM backend for the physical pipeline: api/openrouter, local/qwenvl, or qwenvl_server.",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help="VLM model name or local path for the physical pipeline. Defaults to the selected backend's default model.",
    )
    parser.add_argument(
        "--interaction-vlm-backend",
        default=None,
        help="VLM backend for interaction scoring. Defaults to --vlm-backend.",
    )
    parser.add_argument(
        "--interaction-vlm-model",
        default=None,
        help="VLM model for interaction scoring. Defaults to --vlm-model when the backend is shared.",
    )
    parser.add_argument(
        "--three-d-model-name",
        default=DEFAULT_3D_MODEL_NAME,
        help="DA3 reconstruction model name or local model path for the 3D pipeline",
    )
    parser.add_argument(
        "--three-d-vlm-backend",
        default=os.getenv("REWARD_3D_SCORER"),
        help="VLM backend for 3D GS/meta scoring: api/openrouter, local/qwenvl, or qwenvl_server.",
    )
    parser.add_argument(
        "--three-d-scoring-model",
        default=os.getenv("REWARD_3D_SCORING_MODEL") or DEFAULT_VLM_MODEL,
        help="VLM model name or local path used to score GS/meta renderings. Defaults to the selected backend's default model.",
    )
    parser.add_argument("--sam3-model", default=DEFAULT_SAM3_MODEL, help="Path to SAM3 weights")
    parser.add_argument(
        "--sam-device",
        default=DEFAULT_SAM_DEVICE,
        help="GPU id for SAM3, e.g. 0/1/2. Also accepts cpu.",
    )
    parser.add_argument(
        "--da-device",
        help="GPU id for DA3 trajectory extraction and 3D scoring, e.g. 0/1/2",
    )
    parser.add_argument("--sam-video", help="Precomputed bbox video for --video")
    parser.add_argument("--gt-sam-video", help="Precomputed bbox video for --gt-video")
    parser.add_argument(
        "--three-d-dynamic-masked-video",
        help="Precomputed video with dynamic objects inpainted for the 3D reconstruction pipeline",
    )
    parser.add_argument(
        "--three-d-mask-dynamic-objects",
        dest="three_d_mask_dynamic_objects",
        action="store_true",
        help="Mask dynamic objects with SAM, inpaint the background, then run 3D reconstruction (default)",
    )
    parser.add_argument(
        "--three-d-no-mask-dynamic-objects",
        dest="three_d_mask_dynamic_objects",
        action="store_false",
        help="Run 3D reconstruction on the original candidate video",
    )
    parser.add_argument("--camera-trajectory", help="Existing reference camera trajectory JSON file")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of local DA3 worker processes / GPUs")
    parser.add_argument(
        "--physical-max-frames",
        type=int,
        default=DEFAULT_PHYSICAL_MAX_FRAMES,
        help="Max frames kept by the physical SAM pipeline; <=0 disables temporal sampling",
    )
    parser.add_argument(
        "--sam-min-mask-area-ratio",
        type=float,
        default=DEFAULT_SAM_MIN_MASK_AREA_RATIO,
        help="Drop SAM masks smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--sam-max-masks-per-frame",
        type=int,
        default=DEFAULT_SAM_MAX_MASKS_PER_FRAME,
        help="Keep only the largest N SAM masks per frame after confidence and area filtering; <=0 keeps all.",
    )
    parser.add_argument(
        "--three-d-max-frames",
        "--max-frames",
        dest="three_d_max_frames",
        type=int,
        default=DEFAULT_3D_MAX_FRAMES,
        help="Max frames for DA3 extraction and 3D scoring",
    )
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES, help="DA3 process resolution")
    parser.add_argument("--question-ids", nargs="+", help="Optional subset of physics question ids")
    parser.add_argument(
        "--physical-batch-mode",
        choices=["none", "dimension", "all"],
        default="none",
        help="Batch physical questions into fewer VLM calls. 'dimension' is recommended for bulk scoring.",
    )
    parser.add_argument(
        "--interaction-chunk-frames",
        type=int,
        default=DEFAULT_INTERACTION_CHUNK_FRAMES,
        help="Frames sampled per generated chunk for VLM interaction scoring",
    )
    parser.add_argument(
        "--interaction-ref-frames",
        type=int,
        default=DEFAULT_INTERACTION_REF_FRAMES,
        help="Reference context frames sampled for VLM interaction scoring",
    )
    parser.add_argument(
        "--interaction-transition-side-frames",
        type=int,
        default=DEFAULT_INTERACTION_TRANSITION_SIDE_FRAMES,
        help="Frames sampled on each side of a chunk boundary for interaction transition scoring",
    )
    parser.add_argument(
        "--interaction-global-frames",
        type=int,
        default=DEFAULT_INTERACTION_GLOBAL_FRAMES,
        help="Frames sampled from the full generated video for global interaction scoring",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="CLIP model used when --run-clip-interaction is set",
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Torch device for --run-clip-interaction. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--clip-frames-per-chunk",
        type=int,
        default=DEFAULT_CLIP_FRAMES_PER_CHUNK,
        help="Frames sampled per chunk when --run-clip-interaction is set",
    )
    parser.add_argument(
        "--clip-download-root",
        default=None,
        help="CLIP weight cache directory. Defaults to CLIP_DOWNLOAD_ROOT or /tmp/clip_cache.",
    )
    parser.add_argument("--force-sam", action="store_true", help="Re-run SAM even if *_bbox_h264.mp4 already exists")
    parser.add_argument(
        "--force-da3",
        action="store_true",
        help="Re-run DA3 trajectory extraction even if *_da3_camera_trajectory.json already exists",
    )
    parser.add_argument("--skip-physical", action="store_true", help="Skip the physical pipeline")
    parser.add_argument(
        "--skip-interaction",
        action="store_true",
        help="Skip VLM physical interaction scoring even when --prompt-json and --chunk-json are provided",
    )
    parser.add_argument("--skip-interaction-chunk", action="store_true", help="Skip per-chunk interaction scoring")
    parser.add_argument(
        "--skip-interaction-transition",
        action="store_true",
        help="Skip adjacent chunk transition scoring",
    )
    parser.add_argument("--skip-interaction-global", action="store_true", help="Skip whole-video interaction scoring")
    parser.add_argument(
        "--run-clip-interaction",
        action="store_true",
        help="Also run CLIP chunk semantic-adherence scoring and include it as an auxiliary output",
    )
    parser.add_argument(
        "--offload-qwenvl-before-3d",
        action="store_true",
        help="Clear local QwenVL model weights from the main process before starting the 3D reward worker.",
    )
    parser.add_argument("--skip-3d", action="store_true", help="Skip the 3D pipeline")
    parser.set_defaults(three_d_mask_dynamic_objects=True)
    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    gt_video_path = str(Path(args.gt_video).resolve()) if args.gt_video else None
    prompt_json_path = str(Path(args.prompt_json).resolve()) if args.prompt_json else None
    chunk_json_path = str(Path(args.chunk_json).resolve()) if args.chunk_json else None
    run_vlm_interaction = not args.skip_interaction and (args.prompt_json is not None or args.chunk_json is not None)
    run_clip_interaction = args.run_clip_interaction
    run_any_interaction = run_vlm_interaction or run_clip_interaction

    if args.skip_physical and args.skip_3d and not run_any_interaction:
        raise ValueError(
            "Nothing to run: physical and 3D were skipped and no interaction inputs were provided. "
            "Pass --prompt-json and --chunk-json or unset a skip flag."
        )
    if run_any_interaction and (not prompt_json_path or not chunk_json_path):
        raise ValueError("Interaction scoring requires both --prompt-json and --chunk-json.")
    if run_vlm_interaction and gt_video_path is None:
        raise ValueError("Interaction scoring requires --gt-video as the reference/context video.")
    if (
        run_vlm_interaction
        and args.skip_interaction_chunk
        and args.skip_interaction_transition
        and args.skip_interaction_global
    ):
        raise ValueError("Nothing to run for interaction: all VLM interaction components were skipped.")

    args.vlm_backend = resolve_vlm_backend(args.vlm_backend)
    args.vlm_model = resolve_model_name(args.vlm_model, args.vlm_backend)
    args.interaction_vlm_backend = resolve_vlm_backend(args.interaction_vlm_backend or args.vlm_backend)
    default_interaction_model = (
        args.vlm_model if args.interaction_vlm_backend == args.vlm_backend else None
    )
    args.interaction_vlm_model = resolve_model_name(
        args.interaction_vlm_model or default_interaction_model,
        args.interaction_vlm_backend,
    )
    args.three_d_vlm_backend = resolve_vlm_backend(args.three_d_vlm_backend)
    args.three_d_scoring_model = resolve_model_name(args.three_d_scoring_model, args.three_d_vlm_backend)

    prompt = resolve_prompt(
        args.video,
        args.prompt,
        args.gt_video,
        required=(
            not args.skip_physical
            or (
                not args.skip_3d
                and args.three_d_mask_dynamic_objects
                and not args.three_d_dynamic_masked_video
            )
        ),
    )

    physical_summary = None
    video_bbox = None
    gt_bbox = None
    video_sam_info = None


    if not args.skip_physical:
        if gt_video_path is None and args.gt_sam_video is None:
            raise ValueError("Physical scoring requires --gt-video or --gt-sam-video.")

        print("\nPreparing SAM outputs for the physical pipeline...")
        video_bbox, video_sam_info = ensure_sam_bbox_video(
            video_path,
            prompt=prompt,
            vlm_model=args.vlm_model,
            vlm_backend=args.vlm_backend,
            sam3_model=args.sam3_model,
            sam_device=args.sam_device,
            max_frames=args.physical_max_frames,
            min_mask_area_ratio=args.sam_min_mask_area_ratio,
            max_masks_per_frame=args.sam_max_masks_per_frame,
            explicit_bbox_video=args.sam_video,
            force=args.force_sam,
        )
        gt_bbox, _ = ensure_sam_bbox_video(
            gt_video_path if gt_video_path is not None else args.gt_sam_video,
            prompt=prompt,
            vlm_model=args.vlm_model,
            vlm_backend=args.vlm_backend,
            sam3_model=args.sam3_model,
            sam_device=args.sam_device,
            max_frames=args.physical_max_frames,
            min_mask_area_ratio=args.sam_min_mask_area_ratio,
            max_masks_per_frame=args.sam_max_masks_per_frame,
            explicit_bbox_video=args.gt_sam_video,
            force=args.force_sam,
        )

        print("\nRunning the physical consistency judge...")
        physical_summary = run_physical_pipeline(
            video_bbox=video_bbox,
            gt_bbox=gt_bbox,
            prompt=prompt,
            vlm_model=args.vlm_model,
            vlm_backend=args.vlm_backend,
            question_ids=args.question_ids,
            batch_mode=args.physical_batch_mode,
        )

    interaction_summary = None
    clip_interaction_summary = None

    if run_vlm_interaction:
        print("\nRunning the physical interaction 3D judge...")
        interaction_summary = run_interaction_pipeline(
            video_path=video_path,
            ref_video_path=gt_video_path,
            prompt_json=prompt_json_path,
            chunk_json=chunk_json_path,
            vlm_model=args.interaction_vlm_model,
            vlm_backend=args.interaction_vlm_backend,
            chunk_frames=args.interaction_chunk_frames,
            ref_frames=args.interaction_ref_frames,
            transition_side_frames=args.interaction_transition_side_frames,
            global_frames=args.interaction_global_frames,
            skip_chunk=args.skip_interaction_chunk,
            skip_transition=args.skip_interaction_transition,
            skip_global=args.skip_interaction_global,
        )

    if run_clip_interaction:
        print("\nRunning the auxiliary CLIP interaction semantic-adherence scorer...")
        clip_interaction_summary = run_clip_interaction_pipeline(
            video_path=video_path,
            prompt_json=prompt_json_path,
            chunk_json=chunk_json_path,
            model_name=args.clip_model,
            device=args.clip_device,
            frames_per_chunk=args.clip_frames_per_chunk,
            download_root=args.clip_download_root,
        )

    three_d_summary = None
    three_d_input_video = video_path
    three_d_dynamic_masked_video = None
    three_d_dynamic_masked_source = None
    trajectory = None
    trajectory_path = None
    trajectory_source = None
    qwen_offload_info = None

    if not args.skip_3d:
        trajectory_video = args.trajectory_video or gt_video_path
        if args.camera_trajectory or trajectory_video:
            print("\nPreparing DA3 trajectory for the 3D pipeline...")
        else:
            print("\nNo reference camera trajectory provided. 3D camera-motion score will default to 0.")

        trajectory, trajectory_path, trajectory_source = ensure_camera_trajectory(
            camera_trajectory_path=args.camera_trajectory,
            trajectory_video_path=trajectory_video,
            model_name=args.three_d_model_name,
            max_frames=args.three_d_max_frames,
            process_res=args.process_res,
            da_device=args.da_device,
            force=args.force_da3,
        )
        warn_if_trajectory_mismatches_video(
            trajectory_path=trajectory_path,
            reference_video_path=trajectory_video,
        )

        if args.three_d_mask_dynamic_objects:
            print("\nPreparing dynamic-object mask and video-inpainted background for the 3D pipeline...")
            three_d_input_video, dynamic_mask_info = ensure_dynamic_masked_video(
                video_path,
                prompt=prompt,
                vlm_model=args.vlm_model,
                vlm_backend=args.vlm_backend,
                sam3_model=args.sam3_model,
                sam_device=args.sam_device,
                max_frames=args.three_d_max_frames,
                min_mask_area_ratio=args.sam_min_mask_area_ratio,
                max_masks_per_frame=args.sam_max_masks_per_frame,
                explicit_dynamic_masked_video=args.three_d_dynamic_masked_video,
                existing_sam_info=video_sam_info,
                force=args.force_sam,
            )
            three_d_dynamic_masked_video = dynamic_mask_info["dynamic_masked_video"]
            three_d_dynamic_masked_source = dynamic_mask_info["source"]
            print(f"3D reconstruction input after video inpainting: {three_d_input_video}")
        else:
            print("\nDynamic-object masking disabled for the 3D pipeline; using original video.")

        print("\nRunning the 3D consistency pipeline...")
        three_d_prompt = summarize_prompt_for_3d(
            prompt,
            vlm_model=args.three_d_scoring_model,
            vlm_backend=args.three_d_vlm_backend,
        )
        if args.offload_qwenvl_before_3d:
            print("\nOffloading local QwenVL weights from the main process before 3D scoring...")
            qwen_offload_info = offload_qwenvl_model_cache_before_3d()
            if qwen_offload_info.get("error"):
                print(f"[WARN] Failed to offload QwenVL cache before 3D: {qwen_offload_info['error']}")
            else:
                print(f"Released QwenVL model bundle(s): {qwen_offload_info['released_bundles']}")

        three_d_summary = run_3d_pipeline(
            video_path=three_d_input_video,
            prompt=three_d_prompt,
            model_name=args.three_d_model_name,
            scoring_model_name=args.three_d_scoring_model,
            scorer_backend=args.three_d_vlm_backend,
            num_workers=args.num_workers,
            max_frames=args.three_d_max_frames,
            camera_trajectory=trajectory,
            da_device=args.da_device,
        )

    combined_score = combine_scores(physical_summary, interaction_summary, three_d_summary)

    summary = {
        "video": video_path,
        "gt_video": gt_video_path,
        "prompt": prompt,
        "prompt_json": prompt_json_path,
        "chunk_json": chunk_json_path,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "interaction_vlm_backend": args.interaction_vlm_backend,
        "interaction_vlm_model": args.interaction_vlm_model,
        "three_d_model_name": args.three_d_model_name,
        "three_d_vlm_backend": args.three_d_vlm_backend,
        "three_d_scoring_model": args.three_d_scoring_model,
        "physical_max_frames": args.physical_max_frames,
        "physical_batch_mode": args.physical_batch_mode,
        "interaction_chunk_frames": args.interaction_chunk_frames,
        "interaction_ref_frames": args.interaction_ref_frames,
        "interaction_transition_side_frames": args.interaction_transition_side_frames,
        "interaction_global_frames": args.interaction_global_frames,
        "three_d_max_frames": args.three_d_max_frames,
        "three_d_mask_dynamic_objects": args.three_d_mask_dynamic_objects,
        "sam_device": normalize_sam_device(args.sam_device),
        "da_device": normalize_da_device(args.da_device),
        "da_visible_devices": resolve_da_visible_devices(args.da_device),
        "artifacts": {
            "video_bbox": video_bbox,
            "gt_bbox": gt_bbox,
            "three_d_input_video": three_d_input_video,
            "three_d_dynamic_masked_video": three_d_dynamic_masked_video,
            "three_d_dynamic_masked_source": three_d_dynamic_masked_source,
            "camera_trajectory_path": trajectory_path,
            "camera_trajectory_source": trajectory_source,
            "qwen_offload_before_3d": qwen_offload_info,
        },
        "physical": physical_summary,
        "interaction": interaction_summary,
        "clip_interaction": clip_interaction_summary,
        "three_d": three_d_summary,
        "combined_score": combined_score,
    }

    print_score_summary(summary)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
