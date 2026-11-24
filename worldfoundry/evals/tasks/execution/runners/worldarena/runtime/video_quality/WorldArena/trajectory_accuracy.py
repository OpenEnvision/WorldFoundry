#!/usr/bin/env python3
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff, euclidean

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None
import os

from tqdm import tqdm


def one_traj_interpo_fill(data):
    invaild_traj = False

    mask = (data != [-1.0, -1.0]).any(axis=1)

    invalid_ratio = 1 - np.mean(mask)

    if invalid_ratio > 0.90:
        invaild_traj = True
        # raise ValueError("All points are missing. Cannot perform interpolation.")
        print("Warning: This metric has all invalid values ​​and a score of zero！！！！Interpolation will not work.")
        return data, invaild_traj

    if not invaild_traj:
        n = data.shape[0]
        prev = np.full(n, -1, dtype=int)
        next_ = np.full(n, n, dtype=int)

        last = -1
        for i in range(n):
            if mask[i]:
                last = i
            prev[i] = last

        last = n
        for i in range(n - 1, -1, -1):
            if mask[i]:
                last = i
            next_[i] = last

        missing = np.where(~mask)[0]
        if len(missing) == 0:
            return data, invaild_traj

        p_vals = prev[missing]
        q_vals = next_[missing]

        mask_p_invalid = p_vals == -1
        mask_q_invalid = q_vals == n
        mask_both_valid = ~mask_p_invalid & ~mask_q_invalid

        if np.any(mask_p_invalid):
            data[missing[mask_p_invalid]] = data[q_vals[mask_p_invalid]]

        if np.any(mask_q_invalid):
            data[missing[mask_q_invalid]] = data[p_vals[mask_q_invalid]]

        if np.any(mask_both_valid):
            valid_missing = missing[mask_both_valid]
            p = p_vals[mask_both_valid]
            q = q_vals[mask_both_valid]

            alpha = (valid_missing - p) / (q - p).astype(float)
            alpha = alpha[:, np.newaxis]

            interpolated = (1 - alpha) * data[p] + alpha * data[q]
            data[valid_missing] = interpolated

    return data, invaild_traj


def traj_interpo_fill(traj):
    n_traj = traj.shape[1]
    filled_trajs = []
    invaild_trajs = []
    for i in range(n_traj):
        filled_traj, invaild_traj = one_traj_interpo_fill(traj[:, i].copy())
        filled_trajs.append(filled_traj)
        invaild_trajs.append(invaild_traj)

    filled_trajs = np.stack(filled_trajs, axis=1)
    return filled_trajs, invaild_trajs


def hausdorff_distance(traj1, traj2):
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    d1 = directed_hausdorff(traj1, traj2)[0]
    d2 = directed_hausdorff(traj2, traj1)[0]
    return max(d1, d2)


def select_farthest_traj_index(traj_gt, invalid_gt_trajs):
    """Return index of GT trajectory with largest spatial extent; fallback to 0."""
    n_traj = traj_gt.shape[1]
    gt_max_distance_list = []
    valid_gt_indices = []
    for i in range(n_traj):
        if invalid_gt_trajs[i]:
            continue
        gt_max_distance = farthest_distance(traj_gt[:, i])
        gt_max_distance_list.append(gt_max_distance)
        valid_gt_indices.append(i)

    if len(gt_max_distance_list) == 0:
        return 0

    max_idx_in_list, _ = max(enumerate(gt_max_distance_list), key=lambda x: x[1])
    return valid_gt_indices[max_idx_in_list]


def farthest_distance(points):
    if len(points) < 2:
        return 0.0

    # Degenerate point sets are handled before ConvexHull, which needs a non-flat 2D point set.
    unique_points = np.unique(points, axis=0)

    if len(unique_points) < 2:
        return 0.0

    if len(unique_points) == 2:
        return np.linalg.norm(unique_points[0] - unique_points[1])

    # A point set with a negligible extent counts as a single point.
    diff = points.max(axis=0) - points.min(axis=0)
    if np.all(diff < 1e-6):
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception as e:
        # Collinear points make ConvexHull fail; trajectories are short, so a brute-force pair search is cheap.
        print(f"Warning: ConvexHull failed, falling back to brute force. Error: {e}")
        max_d = 0.0
        for i in range(len(unique_points)):
            for j in range(i + 1, len(unique_points)):
                dist = np.linalg.norm(unique_points[i] - unique_points[j])
                if dist > max_d:
                    max_d = dist
        return max_d

    def rotating_calipers(vertices):
        n = len(vertices)
        max_dist = 0.0
        k = 1
        pts = np.vstack([vertices, vertices[0]])
        for i in range(n):
            j = (i + 1) % n
            while True:
                next_k = (k + 1) % n
                cross = (pts[j, 0] - pts[i, 0]) * (pts[next_k, 1] - pts[k, 1]) - (pts[j, 1] - pts[i, 1]) * (
                    pts[next_k, 0] - pts[k, 0]
                )
                if cross < 0:
                    k = next_k
                else:
                    break
            max_dist = max(max_dist, np.linalg.norm(pts[i] - pts[k]), np.linalg.norm(pts[i] - pts[next_k]))
        return max_dist

    return rotating_calipers(hull_points)


def dtw_distance(traj1, traj2):
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    if fastdtw is not None:
        distance, dpath = fastdtw(traj1, traj2, dist=euclidean)
        return distance, dpath

    n, m = len(traj1), len(traj2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    back: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min(
                ((cost[i - 1, j], (i - 1, j)), (cost[i, j - 1], (i, j - 1)), (cost[i - 1, j - 1], (i - 1, j - 1))),
                key=lambda item: item[0],
            )
            cost[i, j] = euclidean(traj1[i - 1], traj2[j - 1]) + prev[0]
            back[(i, j)] = prev[1]
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = back[(i, j)]
    path.reverse()
    return float(cost[n, m]), path


def NDTW(traj_pred, traj_gt, invaild_pred_trajs, invalid_gt_trajs, max_distance_index):
    """
    traj_pred: np arr N,k,2

    traj_gt: np arr M,k,2
    """

    if invaild_pred_trajs[max_distance_index]:
        ds = 0.0
    else:
        di, pi = dtw_distance(traj_pred[:, max_distance_index], traj_gt[:, max_distance_index])
        di = di / len(pi)

        ds = 1 / di
    return ds


def trim_trajectory(traj):
    non_zero = np.where(np.any(np.array(traj) != [0, 0], axis=1))[0]
    return traj[: non_zero[-1] + 1] if len(non_zero) > 0 else []


def eval_traj(traj_pred_file, traj_gt_file):
    traj_pred = np.load(traj_pred_file).astype("float32")
    traj_gt = np.load(traj_gt_file).astype("float32")
    traj_pred, invaild_pred_trajs = traj_interpo_fill(traj_pred)
    traj_gt, invalid_gt_trajs = traj_interpo_fill(traj_gt)

    # print(traj_gt_file, invaild_pred_trajs, invalid_gt_trajs)

    # NDTW is evaluated on the ground-truth trajectory with the largest spatial extent.
    max_distance_index = select_farthest_traj_index(traj_gt, invalid_gt_trajs)
    ndtw = NDTW(traj_pred, traj_gt, invaild_pred_trajs, invalid_gt_trajs, max_distance_index)

    # hsd_max = 16.052  # best dataset (95th percentile of ctrlworld scores)
    # dyn_max = 1.419
    # ndtw_max = 48.2
    # min = 0.0

    # if hsd>hsd_max:
    #     hsd = 1.0
    # else:
    #     hsd = (hsd-min)/(hsd_max-min)

    # if dyn>dyn_max:
    #     dyn = 1.0
    # else:
    #     dyn = (dyn-min)/(dyn_max-min)

    # if ndtw>ndtw_max:
    #     ndtw = 1.0
    # else:
    #     ndtw = (ndtw-min)/(ndtw_max-min)

    res = {"ndtw": f"{ndtw:.3f}"}

    return res


def calculate_means_and_variances(data):
    metrics = ["ndtw"]
    values = {metric: [] for metric in metrics}

    for _task_name, samples in data.items():
        for _sample_id, groups in samples.items():
            for _gid, traj_eval_res in groups.items():
                for metric in metrics:
                    if metric in traj_eval_res:
                        values[metric].append(float(traj_eval_res[metric]))

    means = {}
    variances = {}
    for metric in metrics:
        means[metric] = np.mean(values[metric]) if values[metric] else None
        variances[metric] = np.var(values[metric]) if values[metric] else None

    return means, variances


def compute_trajectory_accuracy(gt_path, data_base):
    res = {}

    for task_id in sorted(os.listdir(data_base)):
        task_path = os.path.join(data_base, task_id)

        res[task_id] = {}
        for episode_id in tqdm(sorted(os.listdir(task_path))):
            if episode_id.endswith((".png", ".json")):
                continue
            res[task_id][episode_id] = {}
            gt_traj_file = os.path.join(gt_path, task_id, episode_id, "traj", "traj.npy")

            episode_path = os.path.join(task_path, episode_id)

            for gid in sorted(os.listdir(episode_path)):
                pred_traj_file = os.path.join(episode_path, gid, "traj", "traj.npy")
                traj_eval_res = eval_traj(pred_traj_file, gt_traj_file)

                res[task_id][episode_id][gid] = traj_eval_res

    return res
