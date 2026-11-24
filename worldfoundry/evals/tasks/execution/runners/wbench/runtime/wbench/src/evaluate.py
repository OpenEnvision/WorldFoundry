"""
WBench Unified Evaluation Entry Point

Usage:
    # Full evaluation: video + case JSON → all applicable metrics
    python evaluate.py --video path/to/video.mp4 --case path/to/case.json

    # Video-only metrics (no case needed)
    python evaluate.py --video path/to/video.mp4 --metrics video_quality segment_continuity

    # Batch evaluation
    python evaluate.py --video_dir work_dirs/model/videos --case_dir data/cases --metrics all

    # With pre-computed data (MegaSAM poses, SAM2 masks, DA3 depth)
    python evaluate.py --video path/to/video.mp4 --case path/to/case.json \
        --poses poses.npz --mask_dir masks/case_1 --depth depth.npy
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

import time  # noqa: E402 - Follows the warnings filters set above.
from pathlib import Path  # noqa: E402 - Follows the warnings filters set above.
from typing import Any  # noqa: E402 - Follows the warnings filters set above.

import src.compat  # noqa: E402, F401 — stub optional deps before any third-party imports

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Individual metric evaluators
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_video_quality(
    video_path: str, device: str = "cuda", metrics: list[str] | None = None, evaluator=None
) -> dict[str, Any]:
    from src.metrics.video_quality.evaluator import VideoQualityEvaluator

    if evaluator is None:
        evaluator = VideoQualityEvaluator(device=device, metrics=metrics)
    return evaluator.evaluate(video_path)


def evaluate_background_consistency(video_path: str, device: str = "cuda", metric=None) -> dict[str, Any]:
    import cv2
    from PIL import Image

    from src.metrics.consistency.background_consistency import BackgroundConsistencyMetric

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / 2.0))
    frames, fid = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fid % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        fid += 1
    cap.release()

    if len(frames) < 2:
        return {"score": None, "error": "too few frames"}

    if metric is None:
        metric = BackgroundConsistencyMetric(device=device)
    result = metric.compute(frames)
    return {"score": result.get("background_consistency_score"), "num_frames": len(frames)}


def evaluate_segment_continuity(video_path: str, case_id: str) -> dict[str, Any]:
    from src.metrics.consistency.segment_continuity import compute_case

    return compute_case(video_path, case_id)


def evaluate_perspective_consistency(mask_dir: str, depth_path: str) -> dict[str, Any] | None:
    from src.metrics.consistency.perspective_consistency import compute_case

    return compute_case(mask_dir, depth_path)


def evaluate_navigation(poses_path: str, case_data: dict) -> dict[str, Any]:
    import numpy as np

    from src.metrics.interaction.navigation_trajectory import evaluate_navigation as _eval_nav

    NAVI_ACTIONS = {
        "W",
        "A",
        "S",
        "D",
        "left",
        "right",
        "up",
        "down",
        "W+A",
        "W+D",
        "S+D",
        "W+left",
        "W+right",
        "W+up",
        "W+down",
    }

    npz = np.load(poses_path)
    poses = npz["cam_c2w"]

    all_actions = [t["action"] for t in case_data["interactions"]]
    n_turns = len(all_actions)
    n_poses = len(poses)
    per_turn = n_poses // n_turns

    # Split all turns equally, then filter to nav-only turns
    all_bounds = [(i * per_turn, min((i + 1) * per_turn, n_poses)) for i in range(n_turns)]
    nav_mask = [a in NAVI_ACTIONS for a in all_actions]
    nav_actions = [a for a, m in zip(all_actions, nav_mask, strict=False) if m]
    nav_bounds = [b for b, m in zip(all_bounds, nav_mask, strict=False) if m]

    if not nav_actions:
        return {"NavScore": None, "error": "no navigation actions"}

    perspective = case_data.get("settings", {}).get("perspective", "first_person")
    return _eval_nav(poses, nav_bounds, nav_actions, perspective)


def evaluate_spatial_consistency(
    video_path: str, npz_path: str, n_turns: int, device: str = "cuda", ds_model=None, ds_preprocess=None
) -> dict[str, Any] | None:
    from src.metrics.consistency.spatial_consistency import evaluate_case

    if ds_model is None:
        from worldfoundry.base_models.perception_core.video_quality.dreamsim import load_model

        ds_model, ds_preprocess = load_model(device=device)
        ds_model = ds_model.to(device).eval()
    return evaluate_case(video_path, npz_path, n_turns, ds_model, ds_preprocess, device)


def evaluate_subject_consistency_cross(
    video_path: str, mask_dir: str, device: str = "cuda", metric=None
) -> dict[str, Any] | None:
    import cv2
    import numpy as np
    from PIL import Image

    from src.metrics.consistency.subject_consistency import SubjectConsistencyMetric

    if metric is None:
        metric = SubjectConsistencyMetric(device=device)

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if len(mask_files) < 2:
        return None

    frame_indices = [int(f.replace(".png", "")) for f in mask_files]
    max_fid = max(frame_indices)

    cap = cv2.VideoCapture(video_path)
    frames, valid_indices, masks = [], [], {}
    fid = 0
    while fid <= max_fid:
        ret, frame = cap.read()
        if not ret:
            break
        if fid in frame_indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            valid_indices.append(fid)
            m = np.array(Image.open(os.path.join(mask_dir, f"{fid:05d}.png")).convert("L")) > 127
            masks[fid] = m
        fid += 1
    cap.release()

    if len(frames) < 2:
        return None

    result = metric.compute(frames, frame_indices=valid_indices, masks=masks)
    return {
        "score": result.get("subject_consistency_score"),
        "dinov2_adj_mean": result.get("dinov2_adj_mean"),
        "clip_first_mean": result.get("clip_first_mean"),
    }


def _get_turn1_video_url(video_path: str, case_data: dict, max_short_side: int = 0) -> str:
    """Encode Turn-1 video clip as base64 URL."""
    import cv2

    from src.metrics.vlm.vlm_evaluator import clip_video_to_b64url, encode_video_to_b64url

    n_turns = len(case_data.get("interactions", []))
    if n_turns > 1:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        turn1_end = duration / n_turns
        return clip_video_to_b64url(video_path, 0, turn1_end, max_short_side=max_short_side)
    else:
        return encode_video_to_b64url(video_path, max_short_side=max_short_side)


def evaluate_scene_adherence(video_path: str, case_data: dict, device: str = "cuda") -> dict[str, Any]:
    from src.metrics.setting_adherence.scene_adherence import evaluate_case

    sa = case_data.get("scene_adherence", {})
    if not sa:
        return {"score": None, "skipped": True, "reason": "no scene_adherence data in case"}

    style = case_data.get("settings", {}).get("scene", {}).get("attribute", "realistic")

    video_url = _get_turn1_video_url(video_path, case_data)
    result = evaluate_case(
        video_url=video_url,
        visible_part=sa.get("visible_part", ""),
        offscreen_part=sa.get("offscreen_part", ""),
        style=style,
    )
    if result.get("error") and "413" in str(result["error"]):
        video_url = _get_turn1_video_url(video_path, case_data, max_short_side=480)
        result = evaluate_case(
            video_url=video_url,
            visible_part=sa.get("visible_part", ""),
            offscreen_part=sa.get("offscreen_part", ""),
            style=style,
        )
    return result


def evaluate_subject_adherence(video_path: str, case_data: dict, device: str = "cuda") -> dict[str, Any]:
    from src.metrics.setting_adherence.subject_adherence import evaluate_case

    sa = case_data.get("subject_adherence", {})
    if not sa:
        return {"score": None, "skipped": True, "reason": "no subject_adherence data in case"}

    video_url = _get_turn1_video_url(video_path, case_data)
    result = evaluate_case(
        video_url=video_url,
        appearance_part=sa.get("appearance_part", ""),
        action_part=sa.get("action_part", ""),
    )
    if result.get("error") and "413" in str(result["error"]):
        video_url = _get_turn1_video_url(video_path, case_data, max_short_side=480)
        result = evaluate_case(
            video_url=video_url,
            appearance_part=sa.get("appearance_part", ""),
            action_part=sa.get("action_part", ""),
        )
    return result


def evaluate_causal_fidelity(video_path: str, case_data: dict, device: str = "cuda") -> dict[str, Any]:
    import cv2
    from PIL import Image

    from src.metrics.physical.causal_fidelity import evaluate_case
    from src.metrics.vlm.vlm_evaluator import VLMClient

    if not case_data.get("causal_fidelity"):
        return {"score": None, "skipped": True, "reason": "no causal_fidelity annotation"}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / 3.0))
    frames = []
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fid % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        fid += 1
    cap.release()

    if len(frames) < 2:
        return {"score": None, "error": "too few frames"}

    client = VLMClient()
    return evaluate_case(client, frames, case_data)


def evaluate_vlm_interaction(video_path: str, case_data: dict, device: str = "cuda") -> dict[str, Any]:
    from src.metrics.interaction.vlm_interaction import (
        evaluate_event_edit,
        evaluate_perspective_switch,
        evaluate_subject_action,
    )
    from src.metrics.vlm.vlm_evaluator import VLMClient

    interactions = case_data.get("interactions", [])
    types = {t.get("type", "") for t in interactions}

    client = VLMClient()
    results = {}

    if "event_edit" in types:
        results["event_edit_adherence"] = evaluate_event_edit(client, video_path, case_data)
    if "subject_action" in types:
        results["subject_action_adherence"] = evaluate_subject_action(client, video_path, case_data)
    if "perspective_switch" in types:
        results["perspective_switch_adherence"] = evaluate_perspective_switch(client, video_path, case_data)

    if not results:
        results["skipped"] = True
        results["reason"] = "no event_edit/subject_action/perspective_switch interactions"

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Full case evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_full(
    video_path: str,
    case_data: dict,
    device: str = "cuda",
    poses_path: str | None = None,
    mask_dir: str | None = None,
    depth_path: str | None = None,
) -> dict[str, Any]:
    """Run all applicable metrics on a single video + case."""
    results = {"case_id": case_data.get("id"), "video_path": video_path}
    timings = {}

    # --- 1. Video Quality (6 sub-metrics) ---
    _header("Video Quality")
    t0 = time.time()
    results["video_quality"] = evaluate_video_quality(video_path, device)
    timings["video_quality"] = round(time.time() - t0, 1)

    # --- 2. Background Consistency ---
    _header("Background Consistency")
    t0 = time.time()
    results["background_consistency"] = evaluate_background_consistency(video_path, device)
    timings["background_consistency"] = round(time.time() - t0, 1)

    # --- 3. Segment Continuity ---
    _header("Segment Continuity")
    t0 = time.time()
    case_id = str(case_data.get("id", "0"))
    results["segment_continuity"] = evaluate_segment_continuity(video_path, case_id)
    timings["segment_continuity"] = round(time.time() - t0, 1)

    # --- 4. Navigation Trajectory (requires MegaSAM poses) ---
    navi_actions = {
        "W",
        "S",
        "A",
        "D",
        "left",
        "right",
        "up",
        "down",
        "forward",
        "backward",
        "cam_left",
        "cam_right",
        "cam_up",
        "cam_down",
    }
    interactions = case_data.get("interactions", [])
    is_navi = any(t.get("action", "") in navi_actions or t.get("type") == "navigation" for t in interactions)

    if is_navi and poses_path and os.path.exists(poses_path):
        _header("Navigation Trajectory")
        t0 = time.time()
        results["navigation_trajectory"] = evaluate_navigation(poses_path, case_data)
        timings["navigation_trajectory"] = round(time.time() - t0, 1)
    else:
        results["navigation_trajectory"] = {
            "skipped": True,
            "reason": "not navi case" if not is_navi else "poses not provided",
        }

    # --- 5. Spatial Consistency (requires MegaSAM poses, loop case) ---
    if is_navi and poses_path and os.path.exists(poses_path):
        from src.metrics.consistency.spatial_consistency import SYMMETRIC_PAIRS

        actions = [t.get("action", "") for t in interactions]
        n = len(actions)
        is_loop = n >= 2 and all((actions[i], actions[n - 1 - i]) in SYMMETRIC_PAIRS for i in range(n // 2))
        if is_loop:
            _header("Spatial Consistency")
            t0 = time.time()
            results["spatial_consistency"] = evaluate_spatial_consistency(
                video_path, poses_path, len(interactions), device
            )
            timings["spatial_consistency"] = round(time.time() - t0, 1)
        else:
            results["spatial_consistency"] = {"skipped": True, "reason": "not a loop case"}
    else:
        results["spatial_consistency"] = {"skipped": True, "reason": "poses not provided"}

    # --- 6. Perspective Consistency (requires SAM2 masks + DA3 depth) ---
    if mask_dir and depth_path and os.path.isdir(mask_dir) and os.path.exists(depth_path):
        _header("Perspective Consistency")
        t0 = time.time()
        results["perspective_consistency"] = evaluate_perspective_consistency(mask_dir, depth_path)
        timings["perspective_consistency"] = round(time.time() - t0, 1)
    else:
        results["perspective_consistency"] = {"skipped": True, "reason": "masks/depth not provided"}

    # --- 7. Subject Consistency (requires SAM2 masks) ---
    if mask_dir and os.path.isdir(mask_dir):
        _header("Subject Consistency")
        t0 = time.time()
        results["subject_consistency"] = evaluate_subject_consistency_cross(video_path, mask_dir, device)
        timings["subject_consistency"] = round(time.time() - t0, 1)
    else:
        results["subject_consistency"] = {"skipped": True, "reason": "masks not provided"}

    # --- 8. Reconstruction Consistency (geometric + photometric, requires DA3 cache) ---
    if depth_path and os.path.exists(depth_path):
        da3_dir = os.path.dirname(depth_path)
        _header("Reconstruction Consistency (geometric + photometric)")
        t0 = time.time()
        from src.metrics.consistency.reconstruction_consistency import compute_case as _rc_compute

        results["reconstruction_consistency"] = _rc_compute(video_path, da3_dir, fps=3.0, device=device)
        timings["reconstruction_consistency"] = round(time.time() - t0, 1)
    else:
        results["reconstruction_consistency"] = {"skipped": True, "reason": "DA3 depth not provided"}

    # --- 9. Scene Adherence (VLM) ---
    _header("Scene Adherence")
    t0 = time.time()
    try:
        results["scene_adherence"] = evaluate_scene_adherence(video_path, case_data, device)
    except Exception as e:
        results["scene_adherence"] = {"score": None, "error": str(e)}
    timings["scene_adherence"] = round(time.time() - t0, 1)

    # --- 9. Subject Adherence (VLM) ---
    _header("Subject Adherence")
    t0 = time.time()
    try:
        results["subject_adherence"] = evaluate_subject_adherence(video_path, case_data, device)
    except Exception as e:
        results["subject_adherence"] = {"score": None, "error": str(e)}
    timings["subject_adherence"] = round(time.time() - t0, 1)

    # --- 10. Physics Fidelity (VLM) ---
    _header("Physics Fidelity")
    t0 = time.time()
    try:
        results["causal_fidelity"] = evaluate_causal_fidelity(video_path, case_data, device)
    except Exception as e:
        results["causal_fidelity"] = {"score": None, "error": str(e)}
    timings["causal_fidelity"] = round(time.time() - t0, 1)

    # --- 11. VLM Interaction (event_edit / subject_action / perspective_switch) ---
    _header("VLM Interaction Adherence")
    t0 = time.time()
    try:
        results["vlm_interaction"] = evaluate_vlm_interaction(video_path, case_data, device)
    except Exception as e:
        results["vlm_interaction"] = {"error": str(e)}
    timings["vlm_interaction"] = round(time.time() - t0, 1)

    # --- 12. Physical Plausibility (PAVRM: local model or API) ---
    from worldfoundry.base_models.llm_mllm_core.mllm.qwen.wbench_visual_plausibility import model_dir as pavrm_model_dir

    pavrm_path = (
        os.environ.get("PAVRM_MODEL_PATH")
        or os.environ.get("WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR")
        or str(pavrm_model_dir())
    )
    pavrm_api = os.environ.get("PAVRM_API_URL", "")
    if os.path.isdir(pavrm_path) or pavrm_api:
        _header("Physical Plausibility (PAVRM)")
        t0 = time.time()
        try:
            from src.metrics.physical.visual_plausibility import compute_case as _pp_compute

            results["visual_plausibility"] = _pp_compute(video_path, model_path=pavrm_path, device=device)
        except Exception as e:
            results["visual_plausibility"] = {"score": None, "error": str(e)}
        timings["visual_plausibility"] = round(time.time() - t0, 1)
    else:
        results["visual_plausibility"] = {
            "skipped": True,
            "reason": "WBench PAVRM model directory or PAVRM_API_URL not available",
        }

    results["timings"] = timings
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def _worker_fn(args_tuple):
    gpu_id, video_paths, metrics, output_dir = args_tuple
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.set_num_threads(4)

    device = "cuda"
    results = {}
    for i, video_path in enumerate(video_paths):
        case_id = Path(video_path).name.replace("case_", "").replace("_combined.mp4", "")
        case_result = {"case_id": case_id, "video_path": video_path}

        if "video_quality" in metrics or "all" in metrics:
            case_result["video_quality"] = evaluate_video_quality(video_path, device)
        if "segment_continuity" in metrics or "all" in metrics:
            case_result["segment_continuity"] = evaluate_segment_continuity(video_path, case_id)
        if "background_consistency" in metrics or "all" in metrics:
            case_result["background_consistency"] = evaluate_background_consistency(video_path, device)

        results[case_id] = case_result
        out_path = os.path.join(output_dir, f"case_{case_id}.json")
        with open(out_path, "w") as f:
            json.dump(case_result, f, indent=2, default=str)
        print(f"  [GPU {gpu_id}] ({i + 1}/{len(video_paths)}) case_{case_id} done")
    return results


def batch_evaluate(
    video_dir: str, metrics: list[str], device: str = "cuda", output_dir: str = "results", gpus: str | None = None
) -> dict[str, Any]:
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    videos = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.startswith("case_") and f.endswith("_combined.mp4")
        ]
    )
    print(f"Found {len(videos)} videos in {video_dir}")

    if gpus:
        gpu_ids = [int(g) for g in gpus.split(",")]
    else:
        gpu_ids = [int(device.replace("cuda:", ""))] if ":" in device else [0]

    if len(gpu_ids) > 1:
        from multiprocessing import Pool

        chunks = [[] for _ in gpu_ids]
        for i, v in enumerate(videos):
            chunks[i % len(gpu_ids)].append(v)
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        worker_args = [(gpu_ids[i], chunks[i], metrics, output_dir) for i in range(len(gpu_ids))]
        with Pool(len(gpu_ids)) as pool:
            chunk_results = pool.map(_worker_fn, worker_args)
        all_results = {}
        for cr in chunk_results:
            all_results.update(cr)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        all_results = {}
        for i, video_path in enumerate(videos):
            case_id = Path(video_path).name.replace("case_", "").replace("_combined.mp4", "")
            print(f"\n[{i + 1}/{len(videos)}] case_{case_id}")
            case_result = {"case_id": case_id, "video_path": video_path}
            if "video_quality" in metrics or "all" in metrics:
                case_result["video_quality"] = evaluate_video_quality(video_path, device)
            if "segment_continuity" in metrics or "all" in metrics:
                case_result["segment_continuity"] = evaluate_segment_continuity(video_path, case_id)
            if "background_consistency" in metrics or "all" in metrics:
                case_result["background_consistency"] = evaluate_background_consistency(video_path, device)
            all_results[case_id] = case_result
            out_path = os.path.join(output_dir, f"case_{case_id}.json")
            with open(out_path, "w") as f:
                json.dump(case_result, f, indent=2, default=str)

    summary = {}
    for _case_id, res in all_results.items():
        vq = res.get("video_quality", {}).get("summary", {})
        for metric_name, score in vq.items():
            if score is not None:
                summary.setdefault(f"vq_{metric_name}", []).append(score)
        for key in ["segment_continuity", "background_consistency"]:
            sc = res.get(key, {}).get("score")
            if sc is not None:
                summary.setdefault(key, []).append(sc)

    report = {
        "n_cases": len(all_results),
        "overall": {k: round(float(np.mean(v)), 4) for k, v in summary.items()},
    }
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_dir}/report.json")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="WBench: Unified Evaluation for Video World Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Full evaluation (all metrics)
  python evaluate.py --video video.mp4 --case case.json

  # Video-only metrics
  python evaluate.py --video video.mp4 --metrics video_quality segment_continuity

  # Batch mode
  python evaluate.py --video_dir videos/ --metrics video_quality --gpus 0,1,2,3
""",
    )
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory of videos (batch mode)")
    parser.add_argument("--case", type=str, help="Case JSON file (enables full evaluation)")
    parser.add_argument("--case_dir", type=str, help="Case directory (batch mode)")
    parser.add_argument("--poses", type=str, help="MegaSAM poses .npz")
    parser.add_argument("--mask_dir", type=str, help="SAM2 mask directory")
    parser.add_argument("--depth", type=str, help="DA3 depth .npy file")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics (default: all if --case provided, video_quality otherwise)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs for multi-GPU batch")
    parser.add_argument("--output", type=str, default="results.json", help="Output file (single) or directory (batch)")
    args = parser.parse_args()

    # Batch mode
    if args.video_dir:
        metrics = args.metrics or ["video_quality", "segment_continuity", "background_consistency"]
        batch_evaluate(args.video_dir, metrics, args.device, args.output, args.gpus)
        return

    if not args.video:
        parser.print_help()
        return

    # Full case evaluation mode
    if args.case:
        with open(args.case) as f:
            case_data = json.load(f)

        print("\n  WBench Full Evaluation")
        print(f"  Video: {args.video}")
        print(f"  Case:  {args.case} (id={case_data.get('id')})")
        print(f"  Poses: {args.poses or 'N/A'}")
        print(f"  Masks: {args.mask_dir or 'N/A'}")
        print(f"  Depth: {args.depth or 'N/A'}")

        results = evaluate_full(
            video_path=args.video,
            case_data=case_data,
            device=args.device,
            poses_path=args.poses,
            mask_dir=args.mask_dir,
            depth_path=args.depth,
        )

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        _header("Summary")
        for key, val in results.items():
            if key in ("case_id", "video_path", "timings"):
                continue
            if isinstance(val, dict):
                score = val.get("score")
                summary = val.get("summary")
                skipped = val.get("skipped")
                error = val.get("error")
                if skipped:
                    print(f"  {key:35s} | SKIPPED ({val.get('reason', '')})")
                elif error:
                    print(f"  {key:35s} | ERROR: {error}")
                elif summary:
                    scores_str = ", ".join(f"{k}={v:.4f}" for k, v in summary.items() if isinstance(v, (int, float)))
                    print(f"  {key:35s} | {scores_str}")
                elif score is not None:
                    print(f"  {key:35s} | score={score:.4f}")
                else:
                    print(f"  {key:35s} | {val}")

        t = results.get("timings", {})
        total = sum(t.values())
        print(f"\n  Total time: {total:.1f}s")
        print(f"\n  Results saved to: {args.output}")
        return

    # Simple per-metric mode (no case file)
    metrics = args.metrics or ["video_quality"]
    results = {}

    if "video_quality" in metrics or "all" in metrics:
        _header(f"Video Quality — {args.video}")
        results["video_quality"] = evaluate_video_quality(args.video, args.device)

    if "background_consistency" in metrics or "all" in metrics:
        _header(f"Background Consistency — {args.video}")
        results["background_consistency"] = evaluate_background_consistency(args.video, args.device)

    if "segment_continuity" in metrics or "all" in metrics:
        _header(f"Segment Continuity — {args.video}")
        case_id = Path(args.video).stem.replace("case_", "").replace("_combined", "")
        results["segment_continuity"] = evaluate_segment_continuity(args.video, case_id)

    if "subject_consistency" in metrics and args.mask_dir:
        _header("Subject Consistency")
        results["subject_consistency"] = evaluate_subject_consistency_cross(args.video, args.mask_dir, args.device)

    if ("navigation" in metrics or "all" in metrics) and args.poses and args.case:
        _header("Navigation Trajectory")
        with open(args.case) as f:
            case_data = json.load(f)
        results["navigation"] = evaluate_navigation(args.poses, case_data)

    if "perspective_consistency" in metrics and args.mask_dir and args.depth:
        _header("Perspective Consistency")
        results["perspective_consistency"] = evaluate_perspective_consistency(args.mask_dir, args.depth)

    if "spatial_consistency" in metrics and args.video and args.poses and args.case:
        _header("Spatial Consistency")
        with open(args.case) as f:
            n_turns = len(json.load(f).get("interactions", []))
        results["spatial_consistency"] = evaluate_spatial_consistency(args.video, args.poses, n_turns, args.device)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
