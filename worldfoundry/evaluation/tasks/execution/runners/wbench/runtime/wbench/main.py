# -*- coding: utf-8 -*-
"""
WBench Evaluation Pipeline - 3 independent phases.

Usage:
    # Run all 3 phases
    python main.py --model kling

    # Phase 1: Precompute (SAM2 masks, DA3 depth, MegaSAM poses)
    python main.py --model kling --phase precompute

    # Phase 2: GPU metrics (per-metric, one model at a time across all cases)
    python main.py --model kling --phase gpu

    # Phase 3: VLM metrics (API-based, concurrent)
    python main.py --model kling --phase vlm

    # Single video
    python main.py --video path/to/video.mp4 --case data/cases/case_1.json

Directory convention:
    work_dirs/{model}/
    ├── videos/          ← input
    ├── megasam/         ← Phase 1 output (camera poses)
    ├── masks/           ← Phase 1 output (SAM2 tracked masks)
    ├── da3_cache/       ← Phase 1 output (depth maps)
    └── evaluation/      ← Phase 2+3 output
        ├── case_{id}.json   (per-case merged results)
        └── report.json      (aggregated scores)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("WBENCH_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_gpus():
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


def find_videos(video_dir):
    return sorted(glob.glob(os.path.join(video_dir, "case_*_combined.mp4")))


def video_to_case_id(video_path):
    return os.path.basename(video_path).replace("case_", "").replace("_combined.mp4", "")


def load_case(case_id, cases_dir):
    path = os.path.join(cases_dir, f"case_{case_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def is_navi_case(case_data):
    if not case_data:
        return False
    NAVI_ACTIONS = {"W", "A", "S", "D", "left", "right", "up", "down",
                    "forward", "backward", "cam_left", "cam_right", "cam_up", "cam_down"}
    interactions = case_data.get("interactions", [])
    return any(t.get("action", "") in NAVI_ACTIONS or t.get("type") == "navigation"
               for t in interactions)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Precompute (SAM2, DA3, MegaSAM)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_precompute(model, video_dir, gpus, skip_sam2=False, skip_da3=False, skip_megasam=False):
    model_dir = os.path.dirname(video_dir)
    data_dir = DATA_DIR
    gpu_str = ",".join(str(g) for g in gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    conda_lib = os.environ.get("CONDA_PREFIX", "")
    if conda_lib:
        env["LD_LIBRARY_PATH"] = os.path.join(conda_lib, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    # SAM2 Tracking
    if not skip_sam2:
        masks_dir = os.path.join(model_dir, "masks")
        print(f"\n{'='*60}\n  Phase 1a: SAM2 Mask Tracking → {masks_dir}\n{'='*60}")
        t0 = time.time()
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_sam2_track.py"),
            "--video_dir", video_dir,
            "--case_dir", os.path.join(data_dir, "cases"),
            "--mask_dir", os.path.join(data_dir, "masks"),
            "--output_base", masks_dir,
            "--gpus", gpu_str, "--fps", "5.0",
        ]
        subprocess.run(cmd, env=env, check=True)
        print(f"  SAM2 done in {time.time()-t0:.0f}s")

    # DA3 Depth
    if not skip_da3:
        da3_dir = os.path.join(model_dir, "da3_cache")
        print(f"\n{'='*60}\n  Phase 1b: DA3 Depth Estimation → {da3_dir}\n{'='*60}")
        t0 = time.time()
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_da3_depth.py"),
            "--video_dir", video_dir,
            "--output_base", da3_dir,
            "--gpus", gpu_str, "--fps", "3",
        ]
        subprocess.run(cmd, env=env, check=True)
        print(f"  DA3 done in {time.time()-t0:.0f}s")

    # MegaSAM Poses (navi cases only)
    if not skip_megasam:
        megasam_dir = os.path.join(model_dir, "megasam")
        print(f"\n{'='*60}\n  Phase 1c: MegaSAM Camera Poses → {megasam_dir}\n{'='*60}")
        t0 = time.time()
        cases_dir = os.path.join(data_dir, "cases")
        navi_videos = []
        for vp in find_videos(video_dir):
            cid = video_to_case_id(vp)
            cd = load_case(cid, cases_dir)
            if is_navi_case(cd):
                navi_videos.append(vp)

        navi_video_dir = os.path.join(model_dir, "_navi_videos_tmp")
        os.makedirs(navi_video_dir, exist_ok=True)
        for vp in navi_videos:
            dst = os.path.join(navi_video_dir, os.path.basename(vp))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(vp), dst)

        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_megasam.py"),
            "--video_dir", navi_video_dir,
            "--output_dir", megasam_dir,
            "--gpus", gpu_str, "--target_fps", "15",
        ]
        subprocess.run(cmd, env=env, check=True)
        print(f"  MegaSAM done in {time.time()-t0:.0f}s")
    else:
        print("\n  Phase 1c: MegaSAM Camera Poses skipped by --skip_megasam.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: GPU Metrics (per-metric mode)
# ═══════════════════════════════════════════════════════════════════════════════

# 5 dimensions → individual metrics mapping
DIMENSION_MAP = {
    "quality": [
        "aesthetic_quality",
        "imaging_quality",
        "temporal_flickering",
        "dynamic_degree",
        "motion_smoothness",
        "hpsv3_quality",
    ],
    "consistency": [
        "background_consistency",
        "segment_continuity",
        "perspective_consistency",
        "subject_consistency",
        "geometric_consistency",
        "photometric_consistency",
        "spatial_consistency",
        "gated_spatial_consistency",
    ],
    "interaction": [
        "navigation_trajectory",
        "event_edit_adherence",
        "subject_action_adherence",
        "perspective_switch_adherence",
    ],
    "setting": [
        "scene_adherence",
        "subject_adherence",
    ],
    "physical": [
        "visual_plausibility",
        "causal_fidelity",
    ],
}

# GPU compute units — each is one model load + evaluation pass
GPU_COMPUTE_UNITS = [
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "dynamic_degree",
    "motion_smoothness",
    "hpsv3_quality",
    "background_consistency",
    "segment_continuity",
    "perspective_consistency",
    "subject_consistency",
    "reconstruction_consistency",  # computes: geometric_consistency + photometric_consistency
    "navigation_trajectory",
    "spatial_consistency",       # computes: spatial_consistency + gated_spatial_consistency
    "visual_plausibility",
]

DEFAULT_GPU_COMPUTE_UNITS = list(GPU_COMPUTE_UNITS)

# Mapping: user-facing metric name → GPU compute unit
METRIC_TO_COMPUTE_UNIT = {
    "aesthetic_quality": "aesthetic_quality",
    "imaging_quality": "imaging_quality",
    "temporal_flickering": "temporal_flickering",
    "dynamic_degree": "dynamic_degree",
    "motion_smoothness": "motion_smoothness",
    "hpsv3_quality": "hpsv3_quality",
    "background_consistency": "background_consistency",
    "segment_continuity": "segment_continuity",
    "perspective_consistency": "perspective_consistency",
    "subject_consistency": "subject_consistency",
    "geometric_consistency": "reconstruction_consistency",
    "photometric_consistency": "reconstruction_consistency",
    "navigation_trajectory": "navigation_trajectory",
    "spatial_consistency": "spatial_consistency",
    "gated_spatial_consistency": "spatial_consistency",
    "visual_plausibility": "visual_plausibility",
}

# All GPU-facing metric names (user-visible)
GPU_METRICS = list(METRIC_TO_COMPUTE_UNIT.keys())

# All VLM/API metrics
VLM_METRICS = [
    "scene_adherence",
    "subject_adherence",
    "causal_fidelity",
    "event_edit_adherence",
    "subject_action_adherence",
    "perspective_switch_adherence",
]

# All known metric names (for validation)
ALL_METRICS = GPU_METRICS + VLM_METRICS


def resolve_metrics(raw_list):
    """Expand dimension names and metric names into a flat list.

    Supports:
        --metrics video_quality,segment_continuity   (individual metrics)
        --metrics consistency                         (dimension → expand)
        --metrics renderer,interaction               (mix of dimensions)
        --metrics all                                (everything)
    """
    if raw_list is None:
        return None  # means "run all"

    resolved = []
    for item in raw_list:
        item = item.strip()
        if item == "all":
            return None
        elif item in DIMENSION_MAP:
            resolved.extend(DIMENSION_MAP[item])
        elif item in ALL_METRICS:
            resolved.append(item)
        else:
            print(f"  [WARN] Unknown metric/dimension: '{item}', skipping")
    return resolved if resolved else None


def _get_applicable_cases(metric, videos, cases_dir, model_dir):
    """Return list of (case_id, video_path, case_data) applicable to this metric."""
    from src.metrics.consistency.spatial_consistency import SYMMETRIC_PAIRS

    applicable = []
    for vp in videos:
        cid = video_to_case_id(vp)
        cd = load_case(cid, cases_dir)

        # Video quality sub-metrics + universal metrics: all cases
        VQ_METRICS = {"aesthetic_quality", "imaging_quality", "temporal_flickering",
                      "dynamic_degree", "motion_smoothness", "hpsv3_quality"}
        # Check conditions per metric
        if metric in VQ_METRICS or metric in ("background_consistency", "segment_continuity", "visual_plausibility"):
            applicable.append((cid, vp, cd))

        elif metric == "perspective_consistency":
            mask_dir = os.path.join(model_dir, "masks", f"case_{cid}")
            depth_file = os.path.join(model_dir, "da3_cache", f"case_{cid}", "depth.npy")
            if os.path.isdir(mask_dir) and os.path.exists(depth_file):
                applicable.append((cid, vp, cd))

        elif metric == "subject_consistency":
            mask_dir = os.path.join(model_dir, "masks", f"case_{cid}")
            if os.path.isdir(mask_dir):
                applicable.append((cid, vp, cd))

        elif metric == "reconstruction_consistency":
            depth_file = os.path.join(model_dir, "da3_cache", f"case_{cid}", "depth.npy")
            if os.path.exists(depth_file):
                applicable.append((cid, vp, cd))

        elif metric == "navigation_trajectory":
            poses_file = os.path.join(model_dir, "megasam", f"case_{cid}_combined.npz")
            if is_navi_case(cd) and os.path.exists(poses_file):
                applicable.append((cid, vp, cd))

        elif metric == "spatial_consistency":
            poses_file = os.path.join(model_dir, "megasam", f"case_{cid}_combined.npz")
            if not (is_navi_case(cd) and os.path.exists(poses_file)):
                continue
            interactions = cd.get("interactions", [])
            nav_acts = [t.get("action", "") for t in interactions if t.get("type") == "navigation"]
            n = len(nav_acts)
            if n >= 2 and all((nav_acts[i], nav_acts[n-1-i]) in SYMMETRIC_PAIRS for i in range(n//2)):
                applicable.append((cid, vp, cd))

    return applicable


def _run_metric_on_gpu(gpu_id, metric, tasks, eval_dir, model_dir):
    """Single GPU worker for one metric: load model, process assigned cases, save."""
    import torch
    import cv2

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    cv2.setNumThreads(4)

    warnings.filterwarnings("ignore")
    sys.path.insert(0, PROJECT_ROOT)
    import src.compat  # noqa: F401

    device = "cuda"
    tag = f"[GPU{gpu_id}]"

    # Load metric-specific model
    VQ_METRICS = {"aesthetic_quality", "imaging_quality", "temporal_flickering",
                  "dynamic_degree", "motion_smoothness", "hpsv3_quality"}
    evaluator = None
    if metric in VQ_METRICS:
        from src.metrics.video_quality.evaluator import VideoQualityEvaluator
        evaluator = VideoQualityEvaluator(device=device, metrics=[metric])
    elif metric == "background_consistency":
        from src.metrics.consistency.background_consistency import BackgroundConsistencyMetric
        evaluator = BackgroundConsistencyMetric(device=device)
    elif metric == "subject_consistency":
        from src.metrics.consistency.subject_consistency import SubjectConsistencyMetric
        evaluator = SubjectConsistencyMetric(device=device)
    elif metric == "spatial_consistency":
        from worldfoundry.base_models.perception_core.video_quality.dreamsim import load_model
        ds_model, ds_preprocess = load_model(device=device)
        evaluator = (ds_model.to(device).eval(), ds_preprocess)
    elif metric == "visual_plausibility":
        from src.metrics.physical.visual_plausibility import PhysicalPlausibilityEvaluator
        evaluator = PhysicalPlausibilityEvaluator(device=device)

    print(f"  {tag} Model loaded for '{metric}'. Processing {len(tasks)} cases.", flush=True)

    done, fail = 0, 0
    for i, (cid, vp, cd) in enumerate(tasks):
        out_dir = os.path.join(eval_dir, metric)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"case_{cid}.json")

        # Skip if already computed
        if os.path.exists(out_file):
            try:
                existing = json.load(open(out_file))
                if existing.get("score") is not None or existing.get("summary"):
                    done += 1
                    continue
            except Exception:
                pass

        t0 = time.time()
        try:
            result = {"case_id": cid, "video_path": vp}

            if metric in VQ_METRICS:
                from src.evaluate import evaluate_video_quality
                r = evaluate_video_quality(vp, device, metrics=[metric], evaluator=evaluator)
                result.update(r)

            elif metric == "background_consistency":
                from src.evaluate import evaluate_background_consistency
                r = evaluate_background_consistency(vp, device, metric=evaluator)
                result.update(r)

            elif metric == "segment_continuity":
                from src.evaluate import evaluate_segment_continuity
                r = evaluate_segment_continuity(vp, cid)
                result.update(r)

            elif metric == "perspective_consistency":
                from src.evaluate import evaluate_perspective_consistency
                mask_dir = os.path.join(model_dir, "masks", f"case_{cid}")
                depth_file = os.path.join(model_dir, "da3_cache", f"case_{cid}", "depth.npy")
                r = evaluate_perspective_consistency(mask_dir, depth_file)
                result.update(r)

            elif metric == "subject_consistency":
                from src.evaluate import evaluate_subject_consistency_cross
                mask_dir = os.path.join(model_dir, "masks", f"case_{cid}")
                r = evaluate_subject_consistency_cross(vp, mask_dir, device, metric=evaluator)
                result.update(r)

            elif metric == "reconstruction_consistency":
                from src.metrics.consistency.reconstruction_consistency import compute_case
                da3_dir = os.path.join(model_dir, "da3_cache", f"case_{cid}")
                r = compute_case(vp, da3_dir, fps=3.0, device=device)
                result.update(r)

            elif metric == "navigation_trajectory":
                from src.evaluate import evaluate_navigation
                poses_file = os.path.join(model_dir, "megasam", f"case_{cid}_combined.npz")
                r = evaluate_navigation(poses_file, cd)
                result.update(r)

            elif metric == "spatial_consistency":
                from src.evaluate import evaluate_spatial_consistency
                poses_file = os.path.join(model_dir, "megasam", f"case_{cid}_combined.npz")
                interactions = cd.get("interactions", [])
                ds_model, ds_preprocess = evaluator
                r = evaluate_spatial_consistency(
                    vp, poses_file, len(interactions), device,
                    ds_model=ds_model, ds_preprocess=ds_preprocess)
                result.update(r)

            elif metric == "visual_plausibility":
                r = evaluator.score_video(vp)
                result.update({
                    "score": r["score"],
                    "details": {"raw_score": r["raw_score"]},
                    "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5", "fps": 2.0},
                    "error": r["error"],
                })

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            elapsed = time.time() - t0
            score = result.get("score", result.get("summary", {}).get("aesthetic_quality", "?"))
            print(f"  {tag} [{i+1}/{len(tasks)}] case_{cid}: {elapsed:.1f}s", flush=True)
            done += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {tag} [{i+1}/{len(tasks)}] case_{cid}: FAIL ({elapsed:.1f}s) {e}", flush=True)
            with open(out_file, "w") as f:
                json.dump({"case_id": cid, "score": None, "error": str(e)}, f)
            fail += 1

        torch.cuda.empty_cache()

    print(f"  {tag} Finished '{metric}': {done} ok, {fail} fail", flush=True)


def _print_metric_summary(metric, eval_dir, cases_dir):
    """Print aggregated score after a metric finishes."""
    import numpy as np

    metric_dir = os.path.join(eval_dir, metric)
    if not os.path.isdir(metric_dir):
        return

    full_scores, navi_scores, non_navi_scores = [], [], []
    extra_scores = []  # for spatial_consistency ungated (ret_sim)
    n_total, n_error = 0, 0
    for f in os.listdir(metric_dir):
        if not f.endswith(".json"):
            continue
        n_total += 1
        cid = f.replace("case_", "").replace(".json", "")
        try:
            data = json.load(open(os.path.join(metric_dir, f)))
        except Exception:
            n_error += 1
            continue

        # Extract score
        score = data.get("score")
        if score is None:
            score = data.get("NavScore")
        if score is None:
            summary = data.get("summary", {})
            score = summary.get(metric)
        if score is None:
            metrics_dict = data.get("metrics", {})
            if metric in metrics_dict:
                score = metrics_dict[metric].get("score")
        if score is None:
            n_error += 1
            continue

        full_scores.append(score)
        cd = load_case(cid, cases_dir)
        if is_navi_case(cd):
            navi_scores.append(score)
        else:
            non_navi_scores.append(score)

        # Also collect ret_sim for spatial_consistency (ungated)
        if metric == "spatial_consistency" and data.get("ret_sim") is not None:
            extra_scores.append(data["ret_sim"])

    if not full_scores and n_total == 0:
        return

    print(f"     ┌─ {metric}: {len(full_scores)} ok, {n_error} fail, {n_total} total")
    if metric == "spatial_consistency" and extra_scores:
        print(f"     │  spatial_consistency (ungated):  {np.mean(extra_scores):.4f} (n={len(extra_scores)})")
        print(f"     │  gated_spatial_consistency:      {np.mean(full_scores):.4f} (n={len(full_scores)})")
    else:
        print(f"     │  Full:     {np.mean(full_scores):.4f} (n={len(full_scores)})")
        if navi_scores:
            print(f"     │  Navi:     {np.mean(navi_scores):.4f} (n={len(navi_scores)})")
        if non_navi_scores:
            print(f"     └  Non-navi: {np.mean(non_navi_scores):.4f} (n={len(non_navi_scores)})")
            return
    print(f"     └")


def run_phase_gpu(model, video_dir, gpus, metrics=None):
    """Phase 2: Run GPU metrics one at a time (per-metric mode)."""
    import multiprocessing as mp

    model_dir = os.path.dirname(video_dir)
    data_dir = DATA_DIR
    cases_dir = os.path.join(data_dir, "cases")
    eval_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    videos = find_videos(video_dir)
    if metrics:
        vlm_in_gpu = [m for m in metrics if m in VLM_METRICS]
        if vlm_in_gpu:
            print(f"  [WARN] These are VLM metrics, skipping in GPU phase (use --phase vlm): {vlm_in_gpu}")
        unknown = [m for m in metrics if m not in GPU_METRICS and m not in VLM_METRICS]
        if unknown:
            print(f"  [ERROR] Unknown metrics: {unknown}")
        # Convert user-facing metrics to unique compute units
        gpu_metrics = [m for m in metrics if m in GPU_METRICS]
        compute_units = list(dict.fromkeys(METRIC_TO_COMPUTE_UNIT[m] for m in gpu_metrics))
    else:
        compute_units = DEFAULT_GPU_COMPUTE_UNITS

    if not compute_units:
        print(f"  [SKIP] No GPU metrics to run.")
        return

    print(f"\n{'='*60}")
    print(f"  Phase 2: GPU Metrics (per-metric mode)")
    print(f"  Videos: {len(videos)} | GPUs: {gpus}")
    print(f"  Compute units: {compute_units}")
    print(f"{'='*60}")

    for metric in compute_units:
        if metric not in GPU_COMPUTE_UNITS:
            print(f"  [SKIP] Unknown compute unit: {metric}")
            continue

        tasks = _get_applicable_cases(metric, videos, cases_dir, model_dir)

        # Filter already completed
        pending = []
        for cid, vp, cd in tasks:
            out_file = os.path.join(eval_dir, metric, f"case_{cid}.json")
            if os.path.exists(out_file):
                try:
                    existing = json.load(open(out_file))
                    if existing.get("score") is not None or existing.get("summary"):
                        continue
                except Exception:
                    pass
            pending.append((cid, vp, cd))

        print(f"\n  ── {metric} ── ({len(tasks)} applicable, {len(tasks)-len(pending)} done, {len(pending)} pending)")

        if not pending:
            print(f"     All done, skipping.")
            continue

        t0 = time.time()
        n_workers = min(len(gpus), len(pending))

        if n_workers <= 1:
            _run_metric_on_gpu(gpus[0], metric, pending, eval_dir, model_dir)
        else:
            ctx = mp.get_context("spawn")
            worker_tasks = [[] for _ in range(n_workers)]
            for i, t in enumerate(pending):
                worker_tasks[i % n_workers].append(t)

            processes = []
            for w in range(n_workers):
                if not worker_tasks[w]:
                    continue
                p = ctx.Process(
                    target=_run_metric_on_gpu,
                    args=(gpus[w], metric, worker_tasks[w], eval_dir, model_dir),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        elapsed = time.time() - t0
        print(f"     {metric} done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Print summary for this metric
        _print_metric_summary(metric, eval_dir, cases_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: VLM Metrics (API-based, concurrent)
# ═══════════════════════════════════════════════════════════════════════════════


def _vlm_eval_case(task):
    """Thread worker for VLM metrics on a single case."""
    case_id, video_path, case_data, eval_dir, metrics_filter = task

    sys.path.insert(0, PROJECT_ROOT)
    warnings.filterwarnings("ignore")

    from src.evaluate import (evaluate_scene_adherence, evaluate_subject_adherence,
                              evaluate_causal_fidelity)

    def _save_metric(metric_name, result_data):
        """Save per-metric result to evaluation/{metric_name}/case_{id}.json"""
        metric_dir = os.path.join(eval_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        out = {"case_id": case_id, "video_path": video_path}
        out.update(result_data)
        with open(os.path.join(metric_dir, f"case_{case_id}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    def _has_valid_score(metric_name):
        """Check if metric already has a valid score."""
        fpath = os.path.join(eval_dir, metric_name, f"case_{case_id}.json")
        if not os.path.exists(fpath):
            return False
        try:
            data = json.load(open(fpath))
            return data.get("score") is not None
        except Exception:
            return False

    if metrics_filter is None or "scene_adherence" in metrics_filter:
        if "scene_adherence" in case_data and not _has_valid_score("scene_adherence"):
            try:
                result = evaluate_scene_adherence(video_path, case_data, "cpu")
                _save_metric("scene_adherence", result)
            except Exception as e:
                _save_metric("scene_adherence", {"score": None, "error": str(e)})

    if metrics_filter is None or "subject_adherence" in metrics_filter:
        if "subject_adherence" in case_data and not _has_valid_score("subject_adherence"):
            try:
                result = evaluate_subject_adherence(video_path, case_data, "cpu")
                _save_metric("subject_adherence", result)
            except Exception as e:
                _save_metric("subject_adherence", {"score": None, "error": str(e)})

    if metrics_filter is None or "causal_fidelity" in metrics_filter:
        if "causal_fidelity" in case_data and not _has_valid_score("causal_fidelity"):
            try:
                result = evaluate_causal_fidelity(video_path, case_data, "cpu")
                _save_metric("causal_fidelity", result)
            except Exception as e:
                _save_metric("causal_fidelity", {"score": None, "error": str(e)})

    from src.metrics.interaction.vlm_interaction import (
        evaluate_event_edit, evaluate_subject_action, evaluate_perspective_switch,
    )
    from src.metrics.vlm.vlm_evaluator import VLMClient

    interactions = case_data.get("interactions", [])
    itypes = {t.get("type", "") for t in interactions}
    _vlm_client = None

    def _get_vlm_client():
        nonlocal _vlm_client
        if _vlm_client is None:
            _vlm_client = VLMClient()
        return _vlm_client

    if metrics_filter is None or "event_edit_adherence" in metrics_filter:
        if "event_edit" in itypes and not _has_valid_score("event_edit_adherence"):
            try:
                result = evaluate_event_edit(_get_vlm_client(), video_path, case_data)
                _save_metric("event_edit_adherence", result)
            except Exception as e:
                _save_metric("event_edit_adherence", {"score": None, "error": str(e)})

    if metrics_filter is None or "subject_action_adherence" in metrics_filter:
        if "subject_action" in itypes and not _has_valid_score("subject_action_adherence"):
            try:
                result = evaluate_subject_action(_get_vlm_client(), video_path, case_data)
                _save_metric("subject_action_adherence", result)
            except Exception as e:
                _save_metric("subject_action_adherence", {"score": None, "error": str(e)})

    if metrics_filter is None or "perspective_switch_adherence" in metrics_filter:
        if "perspective_switch" in itypes and not _has_valid_score("perspective_switch_adherence"):
            try:
                result = evaluate_perspective_switch(_get_vlm_client(), video_path, case_data)
                _save_metric("perspective_switch_adherence", result)
            except Exception as e:
                _save_metric("perspective_switch_adherence", {"score": None, "error": str(e)})

    return case_id, "ok"


def run_phase_vlm(model, video_dir, vlm_workers=8, metrics=None):
    """Phase 3: Run VLM/API metrics with thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model_dir = os.path.dirname(video_dir)
    data_dir = DATA_DIR
    cases_dir = os.path.join(data_dir, "cases")
    eval_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Validate metrics
    if metrics:
        gpu_in_vlm = [m for m in metrics if m in GPU_METRICS]
        if gpu_in_vlm:
            print(f"  [WARN] These are GPU metrics, skipping in VLM phase (use --phase gpu): {gpu_in_vlm}")
        unknown = [m for m in metrics if m not in VLM_METRICS and m not in GPU_METRICS]
        if unknown:
            print(f"  [ERROR] Unknown metrics: {unknown}")
        metrics = [m for m in metrics if m in VLM_METRICS]
        if not metrics:
            print(f"  [SKIP] No VLM metrics to run.")
            return

    videos = find_videos(video_dir)

    # Build task list with pre-filtering: only include cases that have
    # relevant annotation data for the requested metrics
    def _case_has_metric_data(cd, metrics_filter):
        """Check if case has annotation data for at least one requested metric."""
        if metrics_filter is None:
            return True
        for m in metrics_filter:
            if m == "scene_adherence" and cd.get("scene_adherence"):
                return True
            if m == "subject_adherence" and cd.get("subject_adherence"):
                return True
            if m == "causal_fidelity" and cd.get("causal_fidelity"):
                return True
            if m == "event_edit_adherence":
                itypes = {t.get("type", "") for t in cd.get("interactions", [])}
                if "event_edit" in itypes:
                    return True
            if m == "subject_action_adherence":
                itypes = {t.get("type", "") for t in cd.get("interactions", [])}
                if "subject_action" in itypes:
                    return True
            if m == "perspective_switch_adherence":
                itypes = {t.get("type", "") for t in cd.get("interactions", [])}
                if "perspective_switch" in itypes:
                    return True
        return False

    tasks = []
    for vp in videos:
        cid = video_to_case_id(vp)
        cd = load_case(cid, cases_dir)
        if not cd:
            continue
        if not _case_has_metric_data(cd, metrics):
            continue
        tasks.append((cid, vp, cd, eval_dir, metrics))

    print(f"\n{'='*60}")
    print(f"  Phase 3: VLM Metrics (API-based)")
    print(f"  Cases: {len(tasks)} | Workers: {vlm_workers}")
    if metrics:
        print(f"  Metrics filter: {metrics}")
    print(f"{'='*60}")

    t0 = time.time()
    done, fail = 0, 0
    with ThreadPoolExecutor(max_workers=vlm_workers) as executor:
        futures = {executor.submit(_vlm_eval_case, t): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                cid, status = future.result()
                done += 1
            except Exception as e:
                fail += 1
            if (done + fail) % 1 == 0:
                print(f"  [VLM] Progress: {done+fail}/{len(tasks)}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  VLM done: {done} ok, {fail} fail in {elapsed:.0f}s ({elapsed/60:.1f}min)")


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(model, eval_dir, video_dir, cases_dir):
    """Merge per-metric results into per-case JSONs and generate report."""
    import numpy as np

    videos = find_videos(video_dir)

    # Determine navi cases
    navi_ids = set()
    for vp in videos:
        cid = video_to_case_id(vp)
        cd = load_case(cid, cases_dir)
        if is_navi_case(cd):
            navi_ids.add(cid)

    # Collect scores from per-metric directories
    all_scores = {}  # case_id → {metric: score}
    VQ_METRICS_SET = {"aesthetic_quality", "imaging_quality", "temporal_flickering",
                      "dynamic_degree", "motion_smoothness", "hpsv3_quality"}

    for metric in GPU_COMPUTE_UNITS:
        metric_dir = os.path.join(eval_dir, metric)
        if not os.path.isdir(metric_dir):
            continue
        for f in os.listdir(metric_dir):
            if not f.endswith(".json"):
                continue
            cid = f.replace("case_", "").replace(".json", "")
            try:
                data = json.load(open(os.path.join(metric_dir, f)))
            except Exception:
                continue

            if cid not in all_scores:
                all_scores[cid] = {}

            if metric in VQ_METRICS_SET:
                # Individual VQ metric: score in summary or top-level
                summary = data.get("summary", {})
                if metric in summary and isinstance(summary[metric], (int, float)):
                    all_scores[cid][metric] = summary[metric]
                elif data.get("score") is not None:
                    all_scores[cid][metric] = data["score"]
            elif metric == "reconstruction_consistency":
                details = data.get("details", {})
                if details.get("geometric_consistency") is not None:
                    all_scores[cid]["geometric_consistency"] = details["geometric_consistency"]
                if details.get("photometric_psnr") is not None:
                    psnr = details["photometric_psnr"]
                    all_scores[cid]["photometric_consistency"] = 1.0 - 10**(-psnr/20.0)
            elif metric == "navigation_trajectory":
                if data.get("NavScore") is not None:
                    all_scores[cid]["navigation_trajectory"] = data["NavScore"]
            elif metric == "spatial_consistency":
                if data.get("score") is not None:
                    all_scores[cid]["gated_spatial_consistency"] = data["score"]
                if data.get("ret_sim") is not None:
                    all_scores[cid]["spatial_consistency"] = data["ret_sim"]
            else:
                if data.get("score") is not None:
                    all_scores[cid][metric] = data["score"]

    # VLM scores (per-metric directories)
    VLM_METRIC_DIRS = ["scene_adherence", "subject_adherence", "causal_fidelity",
                       "event_edit_adherence", "subject_action_adherence",
                       "perspective_switch_adherence"]
    for metric in VLM_METRIC_DIRS:
        metric_dir = os.path.join(eval_dir, metric)
        if not os.path.isdir(metric_dir):
            continue
        for f in os.listdir(metric_dir):
            if not f.endswith(".json") or f == "report.json":
                continue
            cid = f.replace("case_", "").replace(".json", "")
            try:
                data = json.load(open(os.path.join(metric_dir, f)))
            except Exception:
                continue
            if data.get("score") is not None:
                if cid not in all_scores:
                    all_scores[cid] = {}
                all_scores[cid][metric] = data["score"]

    # Aggregate
    full_metrics = {}
    navi_metrics = {}
    for cid, scores in all_scores.items():
        for k, v in scores.items():
            full_metrics.setdefault(k, []).append(v)
            if cid in navi_ids:
                navi_metrics.setdefault(k, []).append(v)

    def agg(d):
        return {k: {"mean": round(float(np.mean(v)), 4), "n": len(v)}
                for k, v in sorted(d.items())}

    report = {
        "model": model,
        "n_cases": len(all_scores),
        "n_navi": len(navi_ids),
        "full": agg(full_metrics),
        "navi": agg(navi_metrics),
    }

    report_path = os.path.join(eval_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Single Video Mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_video(video_path, case_path=None, poses_path=None, mask_dir=None, depth_path=None):
    from src.evaluate import evaluate_full
    result = evaluate_full(video_path, case_path=case_path,
                           poses_path=poses_path, mask_dir=mask_dir,
                           depth_path=depth_path)
    print(json.dumps(result, indent=2, default=str))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WBench Evaluation Pipeline (3-phase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py --model kling                          # all phases
  python main.py --model kling --phase precompute       # SAM2 + DA3 + MegaSAM
  python main.py --model kling --phase gpu              # GPU metrics (per-metric)
  python main.py --model kling --phase vlm              # VLM metrics (API)
  python main.py --model kling --phase gpu --metrics video_quality,segment_continuity
  python main.py --video path/to/video.mp4 --case data/cases/case_1.json
""",
    )
    # Single video mode
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--case", type=str, help="Case JSON path (single video)")
    parser.add_argument("--poses", type=str, help="MegaSAM poses .npz (single video)")
    parser.add_argument("--mask_dir", type=str, help="SAM2 mask directory (single video)")
    parser.add_argument("--depth", type=str, help="DA3 depth .npy (single video)")

    # Batch model mode
    parser.add_argument("--model", type=str, help="Model name (directory under work_dirs/)")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs (default: all)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "precompute", "gpu", "vlm", "report"],
                        help="Run specific phase")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated metrics or dimensions. "
                             "Dimensions: renderer,consistency,interaction,setting,physical. "
                             "Metrics: video_quality,segment_continuity,... Use 'all' for everything.")
    parser.add_argument("--skip_megasam", action="store_true")
    parser.add_argument("--enable_megasam", action="store_true",
                        help="Backward-compatible no-op; MegaSAM runs unless --skip_megasam is set.")
    parser.add_argument("--skip_sam2", action="store_true")
    parser.add_argument("--skip_da3", action="store_true")
    parser.add_argument("--work_dir", type=str, default="work_dirs")
    parser.add_argument("--vlm_workers", type=int, default=8, help="VLM concurrent threads")
    args = parser.parse_args()

    # Single video mode
    if args.video:
        run_single_video(args.video, case_path=args.case,
                         poses_path=args.poses, mask_dir=args.mask_dir,
                         depth_path=args.depth)
        return

    # Batch model mode
    if not args.model:
        parser.error("Either --video or --model is required")

    video_dir = os.path.join(PROJECT_ROOT, args.work_dir, args.model, "videos")
    if not os.path.isdir(video_dir):
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)

    videos = find_videos(video_dir)
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = get_all_gpus()

    metrics_list = resolve_metrics(args.metrics.split(",") if args.metrics else None)

    print(f"\n{'='*60}")
    print(f"  WBench Pipeline")
    print(f"  Model:  {args.model}")
    print(f"  Videos: {len(videos)}")
    print(f"  GPUs:   {gpus}")
    print(f"  Phase:  {args.phase}")
    print(f"{'='*60}")

    t_start = time.time()

    if args.phase in ("all", "precompute"):
        run_phase_precompute(args.model, video_dir, gpus,
                            skip_sam2=args.skip_sam2, skip_da3=args.skip_da3,
                            skip_megasam=args.skip_megasam)

    if args.phase in ("all", "gpu"):
        run_phase_gpu(args.model, video_dir, gpus, metrics=metrics_list)

    if args.phase in ("all", "vlm"):
        vlm_filter = [m for m in metrics_list if m in VLM_METRICS] if metrics_list else None
        run_phase_vlm(args.model, video_dir, vlm_workers=args.vlm_workers, metrics=vlm_filter)

    if args.phase in ("all", "report"):
        model_dir = os.path.dirname(video_dir)
        eval_dir = os.path.join(model_dir, "evaluation")
        cases_dir = os.path.join(DATA_DIR, "cases")
        report = generate_report(args.model, eval_dir, video_dir, cases_dir)

        # Print summary
        print(f"\n{'='*70}")
        print(f"  Report: {args.model} | {report['n_cases']} cases | {report['n_navi']} navi")
        print(f"{'='*70}")
        all_m = sorted(set(list(report.get("full", {}).keys()) + list(report.get("navi", {}).keys())))
        print(f"\n  {'Metric':<34s} {'Full':<18s} {'Navi':<18s}")
        print(f"  {'-'*34} {'-'*18} {'-'*18}")
        for m in all_m:
            fi = report.get("full", {}).get(m)
            ni = report.get("navi", {}).get(m)
            fs = f"{fi['mean']:.4f} (n={fi['n']})" if fi else "-"
            ns = f"{ni['mean']:.4f} (n={ni['n']})" if ni else "-"
            print(f"  {m:<34s} {fs:<18s} {ns:<18s}")

    total = time.time() - t_start
    print(f"\n  Total: {total:.0f}s ({total/60:.1f}min)\n")


if __name__ == "__main__":
    main()
