import argparse
import csv
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from tqdm import tqdm

from evaluation import metrics_calculator
from worldfoundry.core.io import extract_video_frames_to_directory


THREE_D_METRICS = frozenset(
    {
        "3D_consistency_num_pts",
        "3D_consistency_num_inliers_F",
        "3D_consistency_keep_ratio",
        "3D_consistency_mean_err",
        "3D_consistency_rmse",
    }
)
TEXT_VIDEO_METRICS = frozenset(
    {
        "camera_alignment",
        "main_object_alignment",
        "background_alignment",
        "style_alignment",
        "overall_consistency",
    }
)


def extract_frames(video_path, store_image_folder):
    timeout = float(os.environ.get("WORLDFOUNDRY_MIRABENCH_FRAME_TIMEOUT_SECONDS", "300"))
    return extract_video_frames_to_directory(
        video_path,
        store_image_folder,
        threads=1,
        timeout_seconds=timeout,
    )


def metric_groups(metrics):
    groups = []
    consumed = set()
    for metric in metrics:
        if metric in consumed:
            continue
        if metric in THREE_D_METRICS:
            shared = THREE_D_METRICS
        elif metric in TEXT_VIDEO_METRICS:
            shared = TEXT_VIDEO_METRICS
        else:
            shared = {metric}
        group = tuple(candidate for candidate in metrics if candidate in shared and candidate not in consumed)
        groups.append(group)
        consumed.update(group)
    return groups


def optional_value(value):
    return None if pd.isna(value) else value


def frame_worker_count(sample_count):
    configured = os.environ.get("WORLDFOUNDRY_MIRABENCH_FRAME_WORKERS", "").strip()
    if configured:
        workers = int(configured)
        if workers < 1:
            raise ValueError("WORLDFOUNDRY_MIRABENCH_FRAME_WORKERS must be positive")
        return min(workers, max(sample_count, 1))
    return min(4, os.cpu_count() or 1, max(sample_count, 1))


def write_video_scores(path, samples, metrics, values):
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", *metrics])
            for sample_index, sample in enumerate(samples):
                writer.writerow(
                    [sample["video_id"], *(values[metric][sample_index] for metric in metrics)]
                )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def write_average_score(path, payload):
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


parser = argparse.ArgumentParser()
parser.add_argument("--meta_file", type=str,default="data/evaluation_example/meta_generated.csv")
parser.add_argument("--frame_dir", type=str,default="data/evaluation_example/frames_generated")
parser.add_argument("--gt_meta_file", type=str,default="data/evaluation_example/meta_gt.csv")
parser.add_argument("--gt_frame_dir", type=str,default="data/evaluation_example/frames_gt")
parser.add_argument("--output_folder", type=str,default="data/evaluation_example/results")
parser.add_argument("--ckpt_path", type=str,default="data/ckpt")
parser.add_argument("--device", type=str,default="cuda")
parser.add_argument("--metrics", type=str,nargs='+',default=[
        # temporal consistency
    'temporal_dino_consistency', # ↑
    'temporal_clip_consistency', # ↑
    'temporal_motion_smoothness', # ↑
        # temporal motion strength
    'dynamic_degree', # ↑
    'tracking_strength', # ↑
        # 3D consistency
    '3D_consistency_num_pts', # ↑
    '3D_consistency_num_inliers_F', # ↑
    '3D_consistency_keep_ratio', # ↑
    '3D_consistency_mean_err', # ↓
    '3D_consistency_rmse', # ↓
        # video frame quality
    'aesthetic_quality', # ↑
    'imaging_quality', # ↑
        # text-video alignment
    'camera_alignment', # ↑
    'main_object_alignment', # ↑
    'background_alignment', # ↑
    'style_alignment', # ↑
    'overall_consistency', # ↑
        # distribution consistency
    'fvd&kvd', # ↓
    'fid&kid', # ↓

])

args = parser.parse_args()
meta_file=args.meta_file
frame_dir=args.frame_dir
output_folder=args.output_folder
metrics=args.metrics
device=args.device
gt_meta_file=args.gt_meta_file
ckpt_path=args.ckpt_path
gt_frame_dir=args.gt_frame_dir

meta_info=pd.read_csv(meta_file)

if "fid&kid" in metrics:
    calculate_fid=True
    metrics.remove("fid&kid")
else:
    calculate_fid=False

if "fvd&kvd" in metrics:
    calculate_fvd=True
    metrics.remove("fvd&kvd")
else:
    calculate_fvd=False

os.makedirs(output_folder, exist_ok=True)
video_score_path=os.path.join(output_folder,"video_score.csv")

samples=[]
for row_idx in range(meta_info.shape[0]):
    present_test_case=meta_info.iloc[row_idx]
    video_idx=present_test_case["video_idx"]
    samples.append(
        {
            "video_id": video_idx,
            "video_path": present_test_case["video_path"],
            "frame_dir": os.path.join(frame_dir, str(video_idx)),
            "short_caption": optional_value(present_test_case["short_caption"]),
            "dense_caption": optional_value(present_test_case["dense_caption"]),
            "main_object_caption": optional_value(present_test_case["main_object_caption"]),
            "background_caption": optional_value(present_test_case["background_caption"]),
            "style_caption": optional_value(present_test_case["style_caption"]),
            "camera_caption": optional_value(present_test_case["camera_caption"]),
        }
    )

print(f"Extracting frames for {len(samples)} videos...")
with ThreadPoolExecutor(max_workers=frame_worker_count(len(samples))) as executor:
    list(executor.map(lambda sample: extract_frames(sample["video_path"], sample["frame_dir"]), samples))
print("Finished extracting frames")

metric_values={metric: [None] * len(samples) for metric in metrics}
write_video_scores(video_score_path, samples, metrics, metric_values)
for group in metric_groups(metrics):
    print(f"================ Calculating metric group: {', '.join(group)} ================")
    calculator=metrics_calculator(list(group),ckpt_path=ckpt_path,device=device)
    try:
        for sample_index, sample in enumerate(tqdm(samples)):
            for metric in group:
                try:
                    metric_values[metric][sample_index] = calculator(
                        metric,
                        sample["frame_dir"],
                        sample["video_path"],
                        sample["short_caption"],
                        sample["dense_caption"],
                        sample["main_object_caption"],
                        sample["background_caption"],
                        sample["style_caption"],
                        sample["camera_caption"],
                    )
                except Exception as exc:
                    print(f"Error in calculating metric {metric} for video {sample['video_id']}: {exc}")
    finally:
        del calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_video_scores(video_score_path, samples, metrics, metric_values)
    print(f"Saved metric-group checkpoint in {video_score_path}")

metrics_result_pd=pd.DataFrame(metric_values)
mean_metrics_result_dict={
    metric: None if pd.isna(value) else float(value)
    for metric, value in metrics_result_pd.apply(pd.to_numeric, errors="coerce").mean().items()
}

average_score_path=os.path.join(output_folder,"average_score.csv")
write_average_score(average_score_path, mean_metrics_result_dict)
print(f'Saved average score in {average_score_path}')

if calculate_fvd or calculate_fid:
    gt_meta_info=pd.read_csv(gt_meta_file)
    gt_samples=[]
    for row_idx in range(gt_meta_info.shape[0]):
        present_test_case=gt_meta_info.iloc[row_idx]
        video_idx=present_test_case["video_idx"]
        gt_samples.append(
            {
                "video_path": present_test_case["video_path"],
                "frame_dir": os.path.join(gt_frame_dir, str(video_idx)),
            }
        )
    print(f"Extracting frames for {len(gt_samples)} ground-truth videos...")
    with ThreadPoolExecutor(max_workers=frame_worker_count(len(gt_samples))) as executor:
        list(
            executor.map(
                lambda sample: extract_frames(sample["video_path"], sample["frame_dir"]),
                gt_samples,
            )
        )
    print("Finished extracting ground-truth frames")

if calculate_fvd:
    try:
        print(f"calculating metrics fvd kvd")
        from evaluation.fvd import EvaluateFVD
        mean_metrics_result_dict["fvd"], mean_metrics_result_dict["kvd"]=EvaluateFVD(frame_dir,gt_frame_dir, ckpt_path, device)
    except Exception as exc:
        print(f"Error in calculating metrics fvd kvd: {exc}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_average_score(average_score_path, mean_metrics_result_dict)
    print(f'Saved average score in {average_score_path}')

if calculate_fid:
    try:
        print(f"calculating metrics fid kid")
        from evaluation.fid import EvaluateFID
        mean_metrics_result_dict["fid"], mean_metrics_result_dict["kid"]=EvaluateFID(frame_dir, gt_frame_dir, ckpt_path, device)
    except Exception as exc:
        print(f"Error in calculating metrics fid kid: {exc}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_average_score(average_score_path, mean_metrics_result_dict)
    print(f'Saved average score in {average_score_path}')

print("Finish")
