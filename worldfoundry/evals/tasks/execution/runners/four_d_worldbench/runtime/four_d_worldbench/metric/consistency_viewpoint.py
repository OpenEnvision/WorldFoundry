import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.slam.droid_slam.droid import Droid

from ..paths import droid_checkpoint_path
from .base_metrics import BaseMetric

try:
    from metric.utils import load_dimension_info
except ImportError:
    from .utils import load_dimension_info
import contextlib
import json


def image_stream(image_list, stride, calib):
    """image generator"""

    fx, fy, cx, cy = calib

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = image_list[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        yield t, image[None], intrinsics


def split_video_into_clips(image_list: list[str], frames_per_clip: int = 10, min_clip_size: int = 5) -> list[list[str]]:
    total_frames = len(image_list)
    if total_frames < frames_per_clip:
        if total_frames >= min_clip_size:
            print(
                f"Warning: Total frames({total_frames}) less than standard clip size({frames_per_clip}), but greater than or equal to minimum clip size({min_clip_size}), using single clip"
            )
            return [image_list]
        else:
            print(
                f"Warning: Total frames({total_frames}) less than minimum clip size({min_clip_size}), skipping processing"
            )
            return []

    clips = []
    start_idx = 0

    while start_idx < total_frames:
        end_idx = min(start_idx + frames_per_clip, total_frames)
        clip_frames = image_list[start_idx:end_idx]

        # Check clip size
        if len(clip_frames) >= min_clip_size:
            clips.append(clip_frames)
            print(f"Clip {len(clips)}: Frames {start_idx}-{end_idx - 1} (total {len(clip_frames)} frames)")
        else:
            print(
                f"Ignoring last clip: Frames {start_idx}-{end_idx - 1} (total {len(clip_frames)} frames, less than minimum size {min_clip_size})"
            )
            break

        start_idx = end_idx

    print(f"Total {len(clips)} valid clips created, each with up to {frames_per_clip} frames")
    return clips


def compute_clip_reprojection_error(clip_frames: list[str], args) -> float:
    if len(clip_frames) < 2:
        print(f"Warning: Insufficient frames in clip ({len(clip_frames)}), skipping")
        return float("inf")

    droid = None

    try:
        # SLAM tracking
        for t, image, intrinsics in tqdm(
            image_stream(clip_frames, args.stride, args.calib),
            desc=f"SLAM Tracking ({len(clip_frames)} frames)",
            leave=False,
        ):
            if t < args.t0:
                continue

            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)

            if torch.cuda.is_available():
                image = image.cuda()
                intrinsics = intrinsics.cuda()

            try:
                droid.track(t, image, intrinsics=intrinsics)
            except Exception as e:
                print(f"Warning: Failed to track frame {t}: {e}")

        # Calculate reprojection error
        traj_est, valid_errors = droid.terminate(image_stream(clip_frames, args.stride, args.calib))

        if isinstance(valid_errors, torch.Tensor) and valid_errors.numel() > 0:
            clip_error = float(valid_errors.mean().item())
            print(f"Clip reprojection error: {clip_error:.4f}")
        else:
            print("Warning: No valid reprojection errors for this clip")
            clip_error = float("inf")

    except Exception as e:
        print(f"Error: Failed to calculate clip reprojection error: {e}")
        import traceback

        traceback.print_exc()
        clip_error = float("inf")

    finally:
        # Clean up resources
        if droid is not None:
            with contextlib.suppress(BaseException):
                del droid

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return clip_error


class ReprojectionErrorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        args = {
            "t0": 0,
            "stride": 1,
            "weights": droid_checkpoint_path(),
            "buffer": 512,
            "beta": 0.3,
            "filter_thresh": 0.01,
            "warmup": 8,
            "keyframe_thresh": 4.0,
            "frontend_thresh": 16.0,
            "frontend_window": 25,
            "frontend_radius": 2,
            "frontend_nms": 1,
            "backend_thresh": 22.0,
            "backend_radius": 2,
            "backend_nms": 3,
            "upsample": True,
            "stereo": False,
            "calib": [500.0, 500.0, 256.0, 256.0],
            "disable_vis": True,
        }

        self._args = argparse.Namespace(**args)
        self.frames_per_clip = 10
        self.min_clip_size = 5

        try:
            torch.multiprocessing.set_start_method("spawn")
        except Exception as e:
            print(f"Error setting start method: {e}")

    def _compute_scores(
        self,
        rendered_images: list[str],
    ) -> float:
        if not rendered_images:
            return float("inf")

        total_frames = len(rendered_images)
        remaining_frames = total_frames % self.frames_per_clip

        if remaining_frames > 0:
            if remaining_frames >= self.min_clip_size:
                print(f"  - Last clip: {remaining_frames} frames (kept)")
            else:
                print(f"  - Last clip: {remaining_frames} frames (ignored, less than {self.min_clip_size} frames)")

        # Split video into clips
        clips = split_video_into_clips(
            rendered_images, frames_per_clip=self.frames_per_clip, min_clip_size=self.min_clip_size
        )

        if not clips:
            print("Error: No valid clips to process")
            return float("inf")

        valid_errors = []

        # Calculate reprojection error for each clip separately
        for i, clip_frames in enumerate(clips):
            print(f"\n=== Processing clip {i + 1}/{len(clips)} ===")
            print(f"Clip {i + 1} contains {len(clip_frames)} frames")

            # Create independent copy of parameters for each clip
            clip_args = argparse.Namespace(**vars(self._args))

            clip_error = compute_clip_reprojection_error(clip_frames, clip_args)

            if not np.isinf(clip_error) and not np.isnan(clip_error):
                valid_errors.append(clip_error)
                print(f"Clip {i + 1} reprojection error: {clip_error:.4f} ✓")
            else:
                print(f"Clip {i + 1} calculation failed, skipping ✗")

        # Calculate average error
        if valid_errors:
            final_score = np.mean(valid_errors)
            print("\n=== Final Results ===")
            print(f"Valid clips: {len(valid_errors)}/{len(clips)}")
            print(f"Error per clip: {[f'{e:.4f}' for e in valid_errors]}")
            print(f"Average reprojection error: {final_score:.4f}")
        else:
            print("Error: All clip calculations failed")
            final_score = float("inf")

        return final_score

    def set_clip_params(self, frames_per_clip: int = 10, min_clip_size: int = 5):
        if frames_per_clip > 0:
            self.frames_per_clip = frames_per_clip
            print(f"Set frames per clip to: {frames_per_clip}")
        else:
            print("Warning: Frames per clip must be greater than 0")

        if min_clip_size > 0:
            self.min_clip_size = min_clip_size
            print(f"Set minimum clip size to: {min_clip_size}")
        else:
            print("Warning: Minimum clip size must be greater than 0")


def _extract_frames_to_dir(video_path: str, out_root: str, max_frames: int = 300) -> list[str]:
    os.makedirs(out_root, exist_ok=True)

    # Use full path with replaced separators to create unique directory
    safe_path = video_path.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_dir = os.path.join(out_root, safe_path)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")])
    if existing:
        return existing

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    indices = list(range(total))
    if max_frames > 0 and total > max_frames:
        step = total / float(max_frames)
        indices = [int(i * step) for i in range(max_frames)]

    saved: list[str] = []
    idx = 0
    next_set = set(indices)
    cur = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur in next_set:
            out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            idx += 1
        cur += 1
    cap.release()
    return saved


def compute_consistency_viewpoint(json_dir, device, submodules_dict, **kwargs):
    dimension = os.path.splitext(os.path.basename(json_dir))[0]
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension, lang="en")

    dataset_json = kwargs.get("dataset_json", "")
    dataset_base_dir = os.environ.get("DATASET_BASE_DIR", "")
    if not dataset_base_dir and dataset_json:
        parts = dataset_json.split("/condition_to_4D/")
        if len(parts) > 1:
            dataset_base_dir = parts[0]
    if dataset_base_dir:
        for item in prompt_dict_ls:
            resolved = []
            for vp in item.get("video_list", []):
                if not os.path.isabs(vp) and not os.path.exists(vp):
                    full = os.path.join(dataset_base_dir, vp)
                    resolved.append(full)
                else:
                    resolved.append(vp)
            item["video_list"] = resolved

    metric = ReprojectionErrorMetric()
    details = []
    scores: list[float] = []
    out_root = os.path.join(os.path.dirname(json_dir), "frames_cache")

    for item in prompt_dict_ls:
        for video_path in item.get("video_list", []) or []:
            frames = _extract_frames_to_dir(video_path, out_root)
            if len(frames) < 2:
                continue
            score = metric._compute_scores(frames)
            scores.append(float(score))
            details.append(
                {
                    "video_path": video_path,
                    "num_frames": len(frames),
                    "reprojection_error": float(score),
                }
            )

    final = float(sum(scores) / len(scores)) if scores else float("inf")

    # Save detailed results JSON
    try:
        output_dir = os.path.dirname(json_dir)
        dim_name = os.path.splitext(os.path.basename(json_dir))[0]
        model = kwargs.get("model", "")
        dataset_json = kwargs.get("dataset_json", "")
        dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else "dataset"
        suffix = f"{dim_name}__{model}__{dataset_base}_results.json" if model else f"{dim_name}_results.json"
        output_file = os.path.join(output_dir, suffix)
        detailed_output = {
            "evaluation_summary": {
                "total_videos": len(details),
                "average_score": final,
            },
            "video_details": details,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e!s}")

    return final, details
