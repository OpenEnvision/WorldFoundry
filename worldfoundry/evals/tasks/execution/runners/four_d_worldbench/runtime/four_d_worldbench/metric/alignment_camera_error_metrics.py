import argparse
import json
import os

import cv2
import cvxpy as cp
import numpy as np
import torch
from lietorch import SE3
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.slam.droid_slam.droid import Droid

from ..paths import droid_checkpoint_path
from .base_metrics import BaseMetric
from .utils import load_dimension_info


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


def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts):
    """Align predicted pose to gt pose and print cameras accuracy"""

    # find rotation
    n = pred_Rs.shape[0]

    R_opt = torch.diag(torch.tensor([1.0, 1, 1.0], device=gt_Rs.device)).double()
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    try:
        c_opt = cp.Variable()
        constraints = []
        obj = cp.Minimize(cp.sum(cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy()), axis=1)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        t_fixed = c_opt.value * pred_ts.numpy()
    except Exception as e:
        print(f"Optimization failed, using fallback method: {e!s}")
        # Fallback: Use simple scaling based on mean distances
        gt_distances = np.linalg.norm(gt_ts.numpy(), axis=1).mean()
        pred_distances = np.linalg.norm(pred_ts.numpy(), axis=1).mean()
        scale = gt_distances / (pred_distances + 1e-8)
        t_fixed = scale * pred_ts.numpy() if scale != 0 else pred_ts.numpy()

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_score = np.mean(t_error).item()

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)
    R_error = R_error.numpy()

    r_score = np.mean(R_error).item()
    print(f"Rotation error: {r_score}")
    print(f"Translation error: {t_score}")
    return (r_score, t_score)


class CameraErrorMetric(BaseMetric):
    """

    return: (R_score, T_score)

    Range: [0, 1] higher the better
    """

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
            # need high resolution depths
            "upsample": True,
            "stereo": False,
            "calib": [500.0, 500.0, 256.0, 256.0],
            "disable_vis": True,
        }
        args = argparse.Namespace(**args)

        self._args = args
        self.droid = None
        try:
            torch.multiprocessing.set_start_method("spawn")
        except Exception as e:
            print(f"Error setting start method: {e}")

    def _compute_scores(self, rendered_images: list[str], cameras_gt: torch.Tensor, scale: float = 1.0) -> float:
        for t, image, intrinsics in tqdm(image_stream(rendered_images, self._args.stride, self._args.calib)):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        traj_est, _ = self.droid.terminate(image_stream(rendered_images, self._args.stride, self._args.calib))
        # import pdb; pdb.set_trace()
        cameras_pred = SE3(torch.as_tensor(traj_est)).matrix()  ## Convert to 4x4 homogeneous matrix
        # cameras_gt = torch.inverse(cameras_gt) #problem: camera pred and camera gt have inconsistent c2w and w2c coordinate systems

        to_blender = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], device=cameras_pred.device))
        cameras_pred = to_blender @ cameras_pred  # c2w

        to_blender = torch.diag(torch.tensor([1.0, 1.0, -1.0, 1.0], device=cameras_pred.device))
        w2c_cameras_pred = torch.inverse(cameras_pred)
        w2c_cameras_pred = to_blender @ w2c_cameras_pred
        cameras_pred = torch.inverse(w2c_cameras_pred)  # Convert back to c2w format

        pred_Rs = cameras_pred[:, :3, :3].cpu().double()
        pred_Ts = cameras_pred[:, :3, 3].cpu().double()

        gt_Rs = cameras_gt[:, :3, :3].cpu().double()
        gt_Ts = cameras_gt[:, :3, 3].cpu().double() / scale
        camera_error = get_cameras_accuracy(pred_Rs, gt_Rs, pred_Ts, gt_Ts)

        self.droid = None
        return camera_error


def _extract_frames_to_dir(video_path: str, out_root: str, max_frames: int = 64) -> list[str]:
    """Extract frames from video to directory."""
    os.makedirs(out_root, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, base)
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


def compute_camera_error_metrics(json_dir, device, submodules_dict, **kwargs):
    """
    Compute Camera Error (alignment) metrics for videos.

    Args:
        json_dir: Path to the dimension JSON file
        device: Device to run the model on (e.g., 'cuda:0')
        submodules_dict: Dictionary of submodules (not used currently)
        **kwargs: Additional arguments (model, dataset_json, etc.)

    Returns:
        tuple: (final_score, details_list)
    """
    dimension = os.path.splitext(os.path.basename(json_dir))[0]
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension, lang="en")

    metric = CameraErrorMetric()

    details = []
    r_scores: list[float] = []
    t_scores: list[float] = []
    out_root = os.path.join(os.path.dirname(json_dir), "frames_cache")

    for item in prompt_dict_ls:
        for video_path in item.get("video_list", []) or []:
            # Extract frames from video
            frames = _extract_frames_to_dir(video_path, out_root)
            if len(frames) == 0:
                print(f"Warning: No frames extracted from {video_path}")
                continue

            # Get camera ground truth and scale from auxiliary_info
            auxiliary_info = item.get("auxiliary_info", {})

            # cameras_gt should be provided as a list or tensor in auxiliary_info
            if "cameras_gt" not in auxiliary_info:
                print(f"Warning: No cameras_gt found for {video_path}, skipping")
                continue

            cameras_gt = auxiliary_info["cameras_gt"]
            if isinstance(cameras_gt, list) or not isinstance(cameras_gt, torch.Tensor):
                cameras_gt = torch.tensor(cameras_gt)

            scale = auxiliary_info.get("scale", 1.0)

            # Compute camera error scores
            try:
                r_score, t_score = metric._compute_scores(frames, cameras_gt, scale)
                r_scores.append(float(r_score))
                t_scores.append(float(t_score))
                details.append(
                    {
                        "video_path": video_path,
                        "num_frames": len(frames),
                        "rotation_error": float(r_score),
                        "translation_error": float(t_score),
                    }
                )
            except Exception as e:
                print(f"Error computing camera error for {video_path}: {e!s}")
                continue

    # Compute average scores
    avg_r_score = float(sum(r_scores) / len(r_scores)) if r_scores else 0.0
    avg_t_score = float(sum(t_scores) / len(t_scores)) if t_scores else 0.0

    # Combined score (lower is better for both metrics)
    # We can use a weighted combination or report separately
    final = (avg_r_score + avg_t_score) / 2.0

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
                "average_rotation_error": avg_r_score,
                "average_translation_error": avg_t_score,
                "combined_error": final,
            },
            "video_details": details,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e!s}")

    return final, details
