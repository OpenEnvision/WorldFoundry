from utils.utils import crop_video_frames, CACHE_DIR
from utils.vipe_utils import extract_traj
import utils.vipe_utils as vipe_utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def action_accuracy_metric(pred_vid_path, gt_vid_path, mark_time, actions, cache_dir = None,
                           delta = 1, max_rot_deg = 179.0, max_trans = 1e9, min_valid_steps = 8, per_action = True, max_frames = 200, gt_data_dir = None, verbose_prefix="", gpu_id: int = 0):
    """
    Compute action accuracy metrics for a single video pair.

    Args:
        pred_vid_path: Path to predicted/generated video
        gt_vid_path: Path to ground truth video
        actions: List of action strings
        cache_dir: Directory for caching ViPE outputs
        delta: Frame delta for RPE computation
        max_rot_deg: Maximum rotation error threshold (filter outliers)
        max_trans: Maximum translation error threshold (filter outliers)
        min_valid_steps: Minimum number of valid steps required
        per_action: Whether to include per-action breakdown
        max_frames: Maximum number of frames to use from videos (None = use all frames)
        gt_data_dir: GT data directory (for caching images.txt)
        verbose_prefix: 前缀标识，用于日志输出
        gpu_id: GPU ID for ViPE inference (default: 0)

    Returns:
        Dictionary with action accuracy metrics in JSON format, or None if failed
    """

    if cache_dir is None:
        cache_dir = Path(CACHE_DIR) / "vipe"

    tqdm.write(f"{verbose_prefix}[1/5] Cropping videos to {max_frames} frames...")
    pred_vid_path = crop_video_frames(pred_vid_path, max_frames, cache_dir)
    gt_vid_path = crop_video_frames(gt_vid_path, max_frames, cache_dir, start_frame=mark_time)

    tqdm.write(f"{verbose_prefix}[2/5] Extracting GT trajectory...")
    gt_T = extract_traj(
        gt_vid_path, cache_dir,
        gt_cache_path=gt_data_dir,
        expected_frames=max_frames,
        verbose_prefix=verbose_prefix,
        gpu_id=gpu_id
    )

    tqdm.write(f"{verbose_prefix}[3/5] Extracting generated trajectory...")
    gen_T = extract_traj(
        pred_vid_path, cache_dir,
        verbose_prefix=verbose_prefix,
        gpu_id=gpu_id
    )

    # Align lengths to actions (+delta poses)
    n_steps = min(len(actions), len(gt_T) - delta, len(gen_T) - delta)
    if n_steps <= 0:
        tqdm.write(f"Not enough steps after alignment for data {pred_vid_path}: n_steps={n_steps}")
        return None
    gt_use = gt_T[:(n_steps + delta)]
    gen_use = gen_T[:(n_steps + delta)]
    actions = actions[:n_steps]

    tqdm.write(f"{verbose_prefix}[4/5] Aligning trajectories (Sim3)...")
    # Sim(3) align gen->gt (positions)
    s, R, t = vipe_utils.umeyama_sim3(gen_use[:, :3, 3], gt_use[:, :3, 3])
    gen_aligned = vipe_utils.apply_sim3_to_poses(gen_use, s, R, t)

    # RPE
    te, re = vipe_utils.compute_rpe(gt_use, gen_aligned, delta=delta)

    # Filter obvious pose failures
    valid = np.ones_like(te, dtype=bool)
    if max_rot_deg > 0:
        valid &= (re <= max_rot_deg)
    if max_trans > 0:
        valid &= (te <= max_trans)

    te = te[valid]
    re = re[valid]
    actions_valid = [a for (a, ok) in zip(actions, valid.tolist()) if ok]

    if te.size < min_valid_steps:
        tqdm.write(f"Too few valid RPE samples for data {pred_vid_path}: te.size={te.size}, min_valid_steps={min_valid_steps}")
        return None

    tqdm.write(f"{verbose_prefix}[5/5] Computing metrics ({te.size} valid samples)...")

    # Helper function to compute statistics
    def compute_stats(te_arr, re_arr):
        return {
            "count": int(te_arr.size),
            "rpe_trans_mean": float(te_arr.mean()),
            "rpe_trans_median": float(np.median(te_arr)),
            "rpe_rot_mean_deg": float(re_arr.mean()),
            "rpe_rot_median_deg": float(np.median(re_arr)),
        }

    # Create result dictionary
    result = {}

    # Overall
    result["__overall__"] = compute_stats(te, re)

    # Group buckets (translation/rotation/other)
    groups = [vipe_utils.bucket_for_action(a) for a in actions_valid]
    for grp in {"translation", "rotation", "other"}:
        idx = [i for i, g in enumerate(groups) if g == grp]
        if idx:
            result[grp] = compute_stats(te[idx], re[idx])

    # Optional per-action breakdown
    if per_action:
        uniq = sorted(set(actions_valid))
        for a in uniq:
            idx = [i for i, aa in enumerate(actions_valid) if aa == a]
            result[f"act:{a}"] = compute_stats(te[idx], re[idx])
            
    return result