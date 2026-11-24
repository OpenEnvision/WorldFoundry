#!/usr/bin/env python3
"""
（ index_ta + index_td + index_npz_sim）
-  index_ta（Trajectory_Acc） index_td（Trajectory_Diff）
-  index_npz_sim（Trajectory_NPZ_Sim）：/NPZNPZ
- / index_revise
"""

import csv
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import matplotlib
import numpy as np

matplotlib.use("Agg")
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger("npz_renamer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ==========================================


# ==========================================
class MetricConfig:
    _PACKAGE_ROOT = Path(__file__).resolve().parent

    EPS = 1e-8
    SUPPORTED_FORMATS = (".npz",)
    EXP_DECAY_ALPHA = -0.2
    MAX_WORKERS = 64

    FILE_PREFIX = "camera_"
    TARGET_CAMERA_DIR = str(_PACKAGE_ROOT / "camera_trajectories" / "inference_txt")
    NORM_SCALE = 1.0
    TRAJ_ACCURACY_THRESHOLDS: ClassVar[dict[int, float]] = {
        0: 0.55,
        1: 0.55,
        2: 0.55,
        3: 0.55,
        4: 0.55,
    }
    DEFAULT_THRESHOLD = 0.7

    SOURCE_NPZ_DIR = str(_PACKAGE_ROOT / "camera_trajectories" / "reference_npz")
    GEN_NPZ_DIR = ""
    TRAJ_NPZ_SIMILARITY_THRESHOLDS: ClassVar[dict[int, float]] = {
        0: 0.55,
        1: 0.55,
        2: 0.55,
        3: 0.55,
        4: 0.55,
    }
    DEFAULT_NPZ_THRESHOLD = 0.7

    ENABLE_PLOTTING = False
    PLOT_FIGSIZE = (12, 8)
    PLOT_DPI = 100


cfg = MetricConfig()

# ==========================================

# ==========================================
EPS = cfg.EPS
SUPPORTED_FORMATS = cfg.SUPPORTED_FORMATS


def _cosine_similarity_abs(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1) + EPS
    norm2 = np.linalg.norm(vec2) + EPS
    return abs(dot_product / (norm1 * norm2))


def _rotation_matrix_to_euler(R):
    r00, r01, _r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, _r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    sin_beta = -r20
    cos_beta = np.sqrt(r00**2 + r10**2) + EPS
    beta = np.arctan2(sin_beta, cos_beta)

    if abs(sin_beta) > 0.9999:
        alpha = 0.0
        gamma = np.arctan2(-r01, r11)
    else:
        alpha = np.arctan2(r10, r00)
        gamma = np.arctan2(r21, r22)

    return np.array([alpha, beta, gamma])


def _12d_to_6dof(vec_12d):
    """126）"""
    R = np.array(
        [
            [vec_12d[0], vec_12d[1], vec_12d[2]],
            [vec_12d[4], vec_12d[5], vec_12d[6]],
            [vec_12d[8], vec_12d[9], vec_12d[10]],
        ]
    )
    t_xyz = np.array([vec_12d[3], vec_12d[7], vec_12d[11]])
    euler_angles = _rotation_matrix_to_euler(R)

    vec_6dof = np.concatenate([t_xyz, euler_angles])
    return vec_6dof


def _6dof_cosine_similarity(vec1_6dof, vec2_6dof):
    t1, r1 = vec1_6dof[:3], vec1_6dof[3:]
    t2, r2 = vec2_6dof[:3], vec2_6dof[3:]

    t1_norm = np.linalg.norm(t1)
    t2_norm = np.linalg.norm(t2)
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    t_sim = _cosine_similarity_abs(t1, t2) if (t1_norm > EPS and t2_norm > EPS) else 0.0
    r_sim = _cosine_similarity_abs(r1, r2) if (r1_norm > EPS and r2_norm > EPS) else 0.0

    if t1_norm < EPS or t2_norm < EPS:
        return r_sim
    elif r1_norm < EPS or r2_norm < EPS:
        return t_sim
    else:
        return (t_sim + r_sim) / 2.0


def _exponential_decay_weighted_sum(similarity_array, alpha=None):
    """index_revise （npz_sim，）"""
    alpha = alpha if alpha is not None else cfg.EXP_DECAY_ALPHA
    if not isinstance(similarity_array, np.ndarray):
        similarity_array = np.array(similarity_array)
    N = len(similarity_array)
    if N < 2:
        return 1.0 if N == 1 else 0.0
    frame_scores = similarity_array[1:]
    frame_distances = np.arange(1, N)
    weights = np.exp(-alpha * frame_distances)
    weight_sum = np.sum(weights) + EPS
    return np.clip(np.dot(weights / weight_sum, frame_scores), 0.0, 1.0)


def g(n: int, filename: str | None = None) -> int:
    """g(n)"""
    if n == 0:
        return 0
    elif 1 <= n <= 4:
        return 1
    elif 5 <= n <= 8:
        return 2
    else:
        if filename:
            logger.warning(f"{n}g(n)| ：{Path(filename).name}")
        else:
            logger.warning(f"{n}g(n)（0-8），0")
        return 0


def calculate_correct_x(y: int, z: int, filename: str | None = None) -> int:
    """yx"""
    if y == 0 and z == 0:
        return 1
    return g(y, filename) + g(z, filename)


def _parse_file_id(filename: str) -> tuple:
    stem = Path(filename).stem
    original_x, original_y, original_z = 0, 0, 0

    temp_pattern = re.search(r"_(\d+)_(\d+)_(\d+)_temp$", stem)
    if temp_pattern:
        try:
            original_x = int(temp_pattern.group(1))
            original_y = int(temp_pattern.group(2))
            original_z = int(temp_pattern.group(3))
        except (IndexError, ValueError):
            pass

    if original_x == 0:
        end_pattern = re.search(r"_(\d+)_(\d+)_(\d+)$", stem)
        if end_pattern:
            try:
                original_x = int(end_pattern.group(1))
                original_y = int(end_pattern.group(2))
                original_z = int(end_pattern.group(3))
            except (IndexError, ValueError):
                pass

    if original_x == 0 and stem.startswith("camera_"):
        stem_sub = stem[len("camera_") :]
        parts = stem_sub.split("_")
        try:
            original_x = int(parts[-3]) if len(parts) >= 3 else 0
            original_y = int(parts[-2]) if len(parts) >= 2 else 0
            original_z = int(parts[-1]) if len(parts) >= 1 else 0
        except (IndexError, ValueError):
            pass

    if original_x == 0:
        batch_pattern = re.search(r"_batch\d+_(\d+)_(\d+)_(\d+)", stem)
        if batch_pattern:
            try:
                original_x = int(batch_pattern.group(1))
                original_y = int(batch_pattern.group(2))
                original_z = int(batch_pattern.group(3))
            except (IndexError, ValueError):
                pass

    if original_x != 0:
        return (original_x, original_y, original_z)

    if original_x == 0:
        p_pattern = re.search(r"P(\d+)", stem)
        group_pattern = re.search(r"group(\d+)", stem)
        mem_pattern = re.search(r"mem(\d+)", stem)

        original_x = int(p_pattern.group(1)) if p_pattern else 0
        original_y = int(group_pattern.group(1)) if group_pattern else 0
        original_z = int(mem_pattern.group(1)) if mem_pattern else 0

    correct_x = calculate_correct_x(original_y, original_z, filename)

    if original_x != correct_x:
        logger.info(f"📝  {filename}：x={original_x} → x={correct_x}（y={original_y}, z={original_z}）")

    return (correct_x, original_y, original_z)


def _load_npz_trajectory(npz_path: str) -> tuple:
    R_trans = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_trans_T = R_trans.T

    with np.load(npz_path) as npz_data:
        keys = list(npz_data.keys())
        data_key = "data" if "data" in npz_data else keys[0]
        poses = npz_data[data_key]

        if len(poses.shape) != 3 or poses.shape[1:] not in [(3, 4), (4, 4)]:
            raise ValueError(f"NPZ：{poses.shape}")
        poses_3x4 = poses[:, :3, :4]
        num_frames = poses_3x4.shape[0]

        flattened_12d = poses_3x4.reshape(num_frames, 12)

        R_origin_batch = np.zeros((num_frames, 3, 3), dtype=np.float64)
        R_origin_batch[:, 0, :] = flattened_12d[:, 0:3]
        R_origin_batch[:, 1, :] = flattened_12d[:, 4:7]
        R_origin_batch[:, 2, :] = flattened_12d[:, 8:11]

        t_origin_batch = np.zeros((num_frames, 3, 1), dtype=np.float64)
        t_origin_batch[:, 0, 0] = flattened_12d[:, 3]
        t_origin_batch[:, 1, 0] = flattened_12d[:, 7]
        t_origin_batch[:, 2, 0] = flattened_12d[:, 11]

        R_target_batch = np.matmul(R_origin_batch, R_trans_T)
        t_target_batch = np.matmul(R_trans, t_origin_batch)

        transformed_12d = np.zeros((num_frames, 12), dtype=np.float64)
        transformed_12d[:, 0:3] = R_target_batch[:, 0, :]
        transformed_12d[:, 3] = t_target_batch[:, 0, 0]
        transformed_12d[:, 4:7] = R_target_batch[:, 1, :]
        transformed_12d[:, 7] = t_target_batch[:, 1, 0]
        transformed_12d[:, 8:11] = R_target_batch[:, 2, :]
        transformed_12d[:, 11] = t_target_batch[:, 2, 0]

        return transformed_12d, num_frames


def _get_level_threshold(level: int) -> float:
    return cfg.TRAJ_ACCURACY_THRESHOLDS.get(level, cfg.DEFAULT_THRESHOLD)


def _parse_memory_id(filename: str):
    match = re.search(r"(^|_)memory_(\d+)(_|$)", Path(filename).stem)
    if match:
        return int(match.group(2))
    return None


def _find_matched_target_file(file_id: tuple, filename: str | None = None) -> str:
    """）"""
    if filename:
        memory_id = _parse_memory_id(filename)
        if memory_id is not None:
            target_filename = f"memory_{memory_id}.txt"
            target_path = Path(cfg.TARGET_CAMERA_DIR) / target_filename
            if target_path.exists():
                return str(target_path)
    x, y, z = file_id
    target_filename = f"{cfg.FILE_PREFIX}{x}_{y}_{z}.txt"
    target_path = Path(cfg.TARGET_CAMERA_DIR) / target_filename
    if target_path.exists():
        return str(target_path)
    raise FileNotFoundError(f"：{target_filename}")


def _load_target_trajectory(target_path: str) -> np.ndarray:
    data_19d = np.loadtxt(target_path, dtype=np.float64)
    data_12d = data_19d[:, 7:] if len(data_19d.shape) == 2 else data_19d[7:].reshape(1, 12)

    if len(data_12d.shape) != 2 or data_12d.shape[1] != 12:
        raise ValueError(f"（N×12）：{data_12d.shape}")

    return data_12d


def _compute_trajectory_derivative(trajectory: np.ndarray) -> np.ndarray:
    """（index_ta / index_npz_sim ）"""
    if trajectory.shape[0] < 2:
        raise ValueError(f"（≥2）：{trajectory.shape[0]}")
    return trajectory[1:] - trajectory[:-1]


def _normalize_trajectory(trajectory: np.ndarray, scale: float | None = None) -> np.ndarray:
    """）"""
    scale = scale if scale is not None else cfg.NORM_SCALE
    trans_xyz = trajectory[:, [3, 7, 11]]
    max_val = np.max(np.abs(trans_xyz)) + EPS
    return trans_xyz * scale / max_val


def _calculate_trajectory_accuracy(vipe_traj: np.ndarray, target_traj: np.ndarray) -> float:
    vipe_frames = vipe_traj.shape[0]
    target_frames = target_traj.shape[0]
    min_frames = min(vipe_frames, target_frames)

    vipe_trunc = vipe_traj[:min_frames, :]
    target_trunc = target_traj[:min_frames, :]

    if min_frames < 2:
        print(f" （{min_frames}），，0.0")
        return 0.0

    vipe_deriv_12d = _compute_trajectory_derivative(vipe_trunc)
    target_deriv_12d = _compute_trajectory_derivative(target_trunc)

    cos_sims = []
    max_idx = len(vipe_deriv_12d)
    for t in range(max_idx):
        if t >= len(target_deriv_12d):
            print(f" {t}，")
            continue

        vipe_deriv_6dof = _12d_to_6dof(vipe_deriv_12d[t])
        target_deriv_6dof = _12d_to_6dof(target_deriv_12d[t])
        sim = _6dof_cosine_similarity(vipe_deriv_6dof, target_deriv_6dof)
        cos_sims.append(sim)

    return np.mean(cos_sims) if cos_sims else 0.0


def _extract_translation_components(trajectory: np.ndarray) -> np.ndarray:
    """（index_td ）"""
    return trajectory[:, [3, 7, 11]]


def _compute_displacement_vectors(translation: np.ndarray) -> np.ndarray:
    """（index_td ）"""
    if translation.shape[0] < 2:
        raise ValueError(f"（≥2）：{translation.shape[0]}")
    return translation[1:] - translation[:-1]


def _normalize_displacement_vectors(displacement: np.ndarray, scale: float | None = None) -> np.ndarray:
    """）"""
    scale = scale if scale is not None else cfg.NORM_SCALE
    max_val = np.max(np.abs(displacement)) + EPS
    return displacement * scale / max_val


def _calculate_trajectory_difference(vipe_traj: np.ndarray) -> tuple:
    translation = _extract_translation_components(vipe_traj)

    v_t = _compute_displacement_vectors(translation)
    T_v = len(v_t)

    if T_v < 2:
        return 0.0 if T_v < 1 else 1.0, v_t, np.array([])

    half_T = min(np.floor(T_v / 2).astype(int), T_v - 1)

    similarity_scores = []
    for t in range(half_T):
        v_rev_idx = T_v - t - 1
        if v_rev_idx < 0:
            print(f"⚠️  {v_rev_idx}，")
            continue

        if t + 1 >= len(vipe_traj) or T_v - t >= len(vipe_traj) or T_v - t - 1 < 0:
            print(f"⚠️  {t}，")
            continue

        frame_forward_12d = vipe_traj[t + 1] - vipe_traj[t]
        frame_backward_12d = vipe_traj[T_v - t] - vipe_traj[T_v - t - 1]
        frame_backward_12d = -frame_backward_12d

        v_forward_6dof = _12d_to_6dof(frame_forward_12d)
        v_backward_6dof = _12d_to_6dof(frame_backward_12d)

        sim = _6dof_cosine_similarity(v_forward_6dof, v_backward_6dof)
        similarity_scores.append(sim)

    diff_score = np.mean(similarity_scores) if similarity_scores else 0.0

    v_rev = -v_t[::-1] if T_v > 0 else np.array([])

    return diff_score, v_t, v_rev


def _get_npz_similarity_threshold(level: int) -> float:
    return cfg.TRAJ_NPZ_SIMILARITY_THRESHOLDS.get(level, cfg.DEFAULT_NPZ_THRESHOLD)


def _find_matched_source_npz(npz_filename: str, source_npz_dir: Path | None = None) -> str:
    source_npz_dir = Path(source_npz_dir or cfg.SOURCE_NPZ_DIR)
    source_npz_path = source_npz_dir / npz_filename
    if source_npz_path.exists():
        return str(source_npz_path)
    memory_id = _parse_memory_id(npz_filename)
    if memory_id is not None:
        memory_npz_path = source_npz_dir / f"memory_{memory_id}.npz"
        if memory_npz_path.exists():
            return str(memory_npz_path)
    x, y, z = _parse_file_id(npz_filename)
    camera_npz_filename = f"{cfg.FILE_PREFIX}{x}_{y}_{z}.npz"
    camera_npz_path = source_npz_dir / camera_npz_filename
    if camera_npz_path.exists():
        return str(camera_npz_path)
    raise FileNotFoundError(f"NPZ：{npz_filename}  {camera_npz_filename}（：{cfg.SOURCE_NPZ_DIR}）")


def _calculate_trajectory_similarity_between_npz(gen_traj: np.ndarray, source_traj: np.ndarray) -> float:
    """ """
    gen_frames = gen_traj.shape[0]
    source_frames = source_traj.shape[0]
    min_frames = min(gen_frames, source_frames)

    if min_frames < 2:
        print(f"  （{gen_frames}，{source_frames}），，0.0")
        return 0.0

    gen_trunc = gen_traj[:min_frames, :]
    source_trunc = source_traj[:min_frames, :]

    gen_deriv_12d = _compute_trajectory_derivative(gen_trunc)
    source_deriv_12d = _compute_trajectory_derivative(source_trunc)

    if len(gen_deriv_12d) != len(source_deriv_12d):
        print(f"  （{len(gen_deriv_12d)}，{len(source_deriv_12d)}），0.0")
        return 0.0

    cos_sims = []
    for t in range(len(gen_deriv_12d)):
        try:
            gen_deriv_6dof = _12d_to_6dof(gen_deriv_12d[t])
            source_deriv_6dof = _12d_to_6dof(source_deriv_12d[t])
        except Exception as e:
            print(f"  {t}：{str(e)[:50]}，")
            continue

        sim = _6dof_cosine_similarity(gen_deriv_6dof, source_deriv_6dof)
        cos_sims.append(sim)

    return np.mean(cos_sims) if cos_sims else 0.0


# ==========================================


# ==========================================
class TrajectoryMetric(ABC):
    """）"""

    def __init__(self, name, short_name, save_base_dir):
        self.name = name
        self.short_name = short_name
        self.metric_root = os.path.join(save_base_dir, self.name)
        self.values_dir = os.path.join(self.metric_root, "values")
        self.log_dir = os.path.join(save_base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, f"{self.name}.txt")

        if cfg.ENABLE_PLOTTING:
            os.makedirs(self.values_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_processed_files(self):
        if not os.path.exists(self.log_file):
            return set()
        with open(self.log_file, encoding="utf-8") as f:
            return {line.strip() for line in f}

    def mark_as_done(self, file_name):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{file_name}\n")

    @abstractmethod
    def visualize(self, file_name, *args):
        pass

    @abstractmethod
    def calculate(self, npz_path, **kwargs):
        pass


# ==========================================


# ==========================================
class TrajectoryAccuracyImpl(TrajectoryMetric):
    """（index_ta ）"""

    def __init__(self, save_base_dir):
        super().__init__("trajectory_accuracy_cosine", "Trajectory_Acc", save_base_dir)

    def visualize(self, file_name, vipe_traj, target_traj):
        """）"""
        if not cfg.ENABLE_PLOTTING:
            return

        save_name = os.path.splitext(file_name)[0]

        np.save(os.path.join(self.values_dir, f"{save_name}_vipe.npy"), vipe_traj)
        np.save(os.path.join(self.values_dir, f"{save_name}_target.npy"), target_traj[: vipe_traj.shape[0], :])

        T = vipe_traj.shape[0]
        target_trunc = target_traj[:T, :]

        vipe_xyz = _normalize_trajectory(vipe_traj)
        target_xyz = _normalize_trajectory(target_trunc)

        fig = plt.figure(figsize=cfg.PLOT_FIGSIZE, dpi=cfg.PLOT_DPI)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(vipe_xyz[:, 0], "b-", label="VIPE Trajectory", linewidth=2, alpha=0.8)
        ax1.plot(target_xyz[:, 0], "r--", label="Target Trajectory", linewidth=2, alpha=0.8)
        ax1.set_title(f"X Dimension (Normalized) - {file_name}", fontsize=12)
        ax1.set_ylabel("X Value", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(vipe_xyz[:, 1], "b-", label="VIPE Trajectory", linewidth=2, alpha=0.8)
        ax2.plot(target_xyz[:, 1], "r--", label="Target Trajectory", linewidth=2, alpha=0.8)
        ax2.set_title(f"Y Dimension (Normalized) - {file_name}", fontsize=12)
        ax2.set_ylabel("Y Value", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(vipe_xyz[:, 2], "b-", label="VIPE Trajectory", linewidth=2, alpha=0.8)
        ax3.plot(target_xyz[:, 2], "r--", label="Target Trajectory", linewidth=2, alpha=0.8)
        ax3.set_title(f"Z Dimension (Normalized) - {file_name}", fontsize=12)
        ax3.set_xlabel("Frame Index", fontsize=10)
        ax3.set_ylabel("Z Value", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.values_dir, f"{save_name}.png"), bbox_inches="tight")
        plt.close()

    def calculate(self, npz_path, **kwargs):
        """（index_ta ）"""
        file_id = _parse_file_id(os.path.basename(npz_path))
        level = file_id[0]

        vipe_traj, num_frames = _load_npz_trajectory(npz_path)

        target_path = _find_matched_target_file(file_id, os.path.basename(npz_path))
        target_traj = _load_target_trajectory(target_path)

        raw_accuracy = _calculate_trajectory_accuracy(vipe_traj, target_traj)

        threshold = _get_level_threshold(level)
        binary_accuracy = 1.0 if raw_accuracy >= threshold else 0.0

        return binary_accuracy, raw_accuracy, vipe_traj, target_traj


class TrajectoryDifferenceImpl(TrajectoryMetric):
    """（index_td ）"""

    def __init__(self, save_base_dir):
        super().__init__("trajectory_difference", "Trajectory_Diff", save_base_dir)

    def visualize(self, file_name, v_t, v_rev):
        """）"""
        if not cfg.ENABLE_PLOTTING:
            return

        save_name = os.path.splitext(file_name)[0]

        np.save(os.path.join(self.values_dir, f"{save_name}_v_t.npy"), v_t)
        np.save(os.path.join(self.values_dir, f"{save_name}_v_rev.npy"), v_rev)

        v_t_norm = _normalize_displacement_vectors(v_t)
        v_rev_norm = _normalize_displacement_vectors(v_rev)

        fig = plt.figure(figsize=cfg.PLOT_FIGSIZE, dpi=cfg.PLOT_DPI)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(v_t_norm[:, 0], "b-", label="vec{v}_Forward", linewidth=2, alpha=0.8)
        ax1.plot(v_rev_norm[:, 0], "r--", label="vec{v}_Mirror", linewidth=2, alpha=0.8)
        ax1.set_title(f"X Dimension (Normalized) - {file_name}", fontsize=12)
        ax1.set_ylabel("X Value", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(v_t_norm[:, 1], "b-", label="vec{v}_Forward", linewidth=2, alpha=0.8)
        ax2.plot(v_rev_norm[:, 1], "r--", label="vec{v}_Mirror", linewidth=2, alpha=0.8)
        ax2.set_title(f"Y Dimension (Normalized) - {file_name}", fontsize=12)
        ax2.set_ylabel("Y Value", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(v_t_norm[:, 2], "b-", label="vec{v}_Forward", linewidth=2, alpha=0.8)
        ax3.plot(v_rev_norm[:, 2], "r--", label="vec{v}_Mirror)", linewidth=2, alpha=0.8)
        ax3.set_title(f"Z Dimension (Normalized) - {file_name}", fontsize=12)
        ax3.set_xlabel("Frame Index", fontsize=10)
        ax3.set_ylabel("Z Value", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.values_dir, f"{save_name}.png"), bbox_inches="tight")
        plt.close()

    def calculate(self, npz_path, **kwargs):
        """（index_td ）"""
        # Parsing logs inferred trajectory-index corrections.
        _parse_file_id(os.path.basename(npz_path))

        vipe_traj, num_frames = _load_npz_trajectory(npz_path)

        diff_score, v_t, v_rev = _calculate_trajectory_difference(vipe_traj)

        return diff_score, v_t, v_rev


class TrajectoryNPZSimilarityImpl(TrajectoryMetric):
    def __init__(self, save_base_dir):
        super().__init__("trajectory_npz_similarity", "Trajectory_NPZ_Sim", save_base_dir)

    def visualize(self, file_name, gen_traj, source_traj):
        if not cfg.ENABLE_PLOTTING:
            return

        save_name = os.path.splitext(file_name)[0]

        np.save(os.path.join(self.values_dir, f"{save_name}_gen.npy"), gen_traj)
        np.save(os.path.join(self.values_dir, f"{save_name}_source.npy"), source_traj)

        min_frames = min(gen_traj.shape[0], source_traj.shape[0])
        gen_trunc = gen_traj[:min_frames, :]
        source_trunc = source_traj[:min_frames, :]

        gen_xyz = _normalize_trajectory(gen_trunc)
        source_xyz = _normalize_trajectory(source_trunc)

        fig = plt.figure(figsize=cfg.PLOT_FIGSIZE, dpi=cfg.PLOT_DPI)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(gen_xyz[:, 0], "b-", label="Generated Video Trajectory", linewidth=2, alpha=0.8)
        ax1.plot(source_xyz[:, 0], "r--", label="Reference Trajectory", linewidth=2, alpha=0.8)
        ax1.set_title(f"X Dimension (Normalized) - {file_name}", fontsize=12)
        ax1.set_ylabel("X Value", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(gen_xyz[:, 1], "b-", label="Generated Video Trajectory", linewidth=2, alpha=0.8)
        ax2.plot(source_xyz[:, 1], "r--", label="Reference Trajectory", linewidth=2, alpha=0.8)
        ax2.set_title(f"Y Dimension (Normalized) - {file_name}", fontsize=12)
        ax2.set_ylabel("Y Value", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(gen_xyz[:, 2], "b-", label="Generated Video Trajectory", linewidth=2, alpha=0.8)
        ax3.plot(source_xyz[:, 2], "r--", label="Reference Trajectory", linewidth=2, alpha=0.8)
        ax3.set_title(f"Z Dimension (Normalized) - {file_name}", fontsize=12)
        ax3.set_xlabel("Frame Index", fontsize=10)
        ax3.set_ylabel("Z Value", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.values_dir, f"{save_name}.png"), bbox_inches="tight")
        plt.close()

    def calculate(self, npz_path, **kwargs):
        npz_filename = os.path.basename(npz_path)

        file_id = _parse_file_id(npz_filename)
        level = file_id[0]

        gen_traj, gen_num_frames = _load_npz_trajectory(npz_path)

        source_npz_path = _find_matched_source_npz(npz_filename)
        source_traj, source_num_frames = _load_npz_trajectory(source_npz_path)

        raw_similarity = _calculate_trajectory_similarity_between_npz(gen_traj, source_traj)

        threshold = _get_npz_similarity_threshold(level)
        binary_similarity = 1.0 if raw_similarity >= threshold else 0.0

        return binary_similarity, raw_similarity, gen_traj, source_traj


# ==========================================


# ==========================================
class AtomicProcessor:
    """（index_ta + index_td + index_npz_sim ）"""

    def __init__(self, video_dir, save_base_dir, max_workers=None):
        self.max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS
        self.video_dir = video_dir
        self.save_base_dir = save_base_dir
        self.report_dir = os.path.join(save_base_dir, "reports")

        path_parts = os.path.normpath(video_dir).split(os.sep)
        if len(path_parts) >= 2:
            self.dataset_name = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            self.dataset_name = path_parts[-1]

        self.lock = threading.Lock()
        self.done = 0
        os.makedirs(self.report_dir, exist_ok=True)

    def run(self, metric_obj, compute_func, params_tag, is_accuracy=False):
        files = [
            f for f in os.listdir(self.video_dir) if not f.startswith(".") and f.lower().endswith(SUPPORTED_FORMATS)
        ]
        total = len(files)
        self.done = 0
        start_t = time.time()
        plot_status = ""
        print(f"\n>>> : {metric_obj.short_name} | : {self.dataset_name} | : {total} | : {plot_status}")
        print(f">>> ：video_dirNPZ: {self.video_dir}")

        def _worker(file_name):
            """"""
            try:
                npz_path = os.path.join(self.video_dir, file_name)
                if file_name in metric_obj.get_processed_files():
                    with self.lock:
                        self.done += 1
                    return None

                if is_accuracy:
                    binary_score, raw_score, traj1, traj2 = compute_func(npz_path)
                    metric_obj.visualize(file_name, traj1, traj2)
                    level = _parse_file_id(file_name)[0]
                    result = (file_name, round(binary_score, 6), round(raw_score, 6), level)
                else:
                    diff_score, v_t, v_rev = compute_func(npz_path)
                    metric_obj.visualize(file_name, v_t, v_rev)
                    result = (file_name, round(diff_score, 6))

                metric_obj.mark_as_done(file_name)

                with self.lock:
                    self.done += 1
                    pct = (self.done / total) * 100
                    progress_bar = (int(pct // 2) * "=") + (int(50 - pct // 2) * " ")
                    print(
                        f"\r: [{progress_bar}] {pct:.1f}% | {self.done}/{total} | : {time.time() - start_t:.1f}s",
                        end="",
                    )

                return result
            except Exception as e:
                print(f"\n[] {file_name}: {str(e)[:100]}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(executor.map(_worker, files))

        if is_accuracy:
            new_data_dict = {}
            level_data_dict = {}
            for res in new_results:
                if res:
                    new_data_dict[res[0]] = (res[1], res[2])
                    level_data_dict[res[0]] = (res[1], res[2], res[3])
            self._merge_accuracy_csv(metric_obj, new_data_dict, level_data_dict, params_tag)
        else:
            new_data_dict = {res[0]: res[1] for res in new_results if res}
            self._merge_difference_csv(metric_obj, new_data_dict, params_tag)

    def _merge_accuracy_csv(self, metric_obj, new_data, level_data, params_tag):
        """index_ta / index_npz_sim ）"""
        v_csv = os.path.join(self.report_dir, f"video_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")
        s_csv = os.path.join(self.report_dir, f"summary_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")
        level_csv = os.path.join(
            self.report_dir, f"level_success_rate_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv"
        )

        all_data = {}
        all_level_data = {}
        if os.path.exists(v_csv):
            try:
                with open(v_csv, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            file_name = row[0]
                            binary_score = float(row[1])
                            raw_sim = float(row[2]) if len(row) >= 3 else 0.0
                            all_data[file_name] = (binary_score, raw_sim)
                            level = _parse_file_id(file_name)[0]
                            all_level_data[file_name] = (binary_score, raw_sim, level)
            except Exception as e:
                print(f"\n[] CSV，: {e}")

        all_data.update(new_data)
        all_level_data.update(level_data)

        with open(v_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["NPZ_Name", f"{metric_obj.short_name}_Score", "Raw_Similarity"])
            for name in sorted(all_data.keys()):
                binary_score, raw_sim = all_data[name]
                writer.writerow([name, binary_score, raw_sim])

        with open(s_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{metric_obj.short_name}_Success_Rate"])
            binary_scores = [v[0] for v in all_data.values()]
            success_rate = np.mean(binary_scores) if binary_scores else 0.0
            writer.writerow([round(success_rate, 6)])

        level_groups = {}
        for _file_name, (binary_score, _raw_score, level) in all_level_data.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(binary_score)

        level_success_rates = {}
        for level in sorted(level_groups.keys()):
            scores = level_groups[level]
            success_rate = np.mean(scores) if scores else 0.0
            level_success_rates[level] = {
                "success_rate": round(success_rate, 6),
                "total_files": len(scores),
                "success_files": sum(1 for s in scores if s == 1.0),
                "failure_files": sum(1 for s in scores if s == 0.0),
            }

        with open(level_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Level", "Success_Rate", "Total_Files", "Success_Files", "Failure_Files"])
            for level in sorted(level_success_rates.keys()):
                data = level_success_rates[level]
                writer.writerow(
                    [level, data["success_rate"], data["total_files"], data["success_files"], data["failure_files"]]
                )

        print("\n[] ：")
        print(f"  - : {v_csv}")
        print(f"  - （）: {s_csv}")
        print(f"  - : {level_csv}")

    def _merge_difference_csv(self, metric_obj, new_data, params_tag):
        """index_td CSV"""
        v_csv = os.path.join(self.report_dir, f"video_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")
        s_csv = os.path.join(self.report_dir, f"summary_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")

        all_data = {}
        if os.path.exists(v_csv):
            try:
                with open(v_csv, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            all_data[row[0]] = float(row[1])
            except Exception as e:
                print(f"\n[] CSV，: {e}")

        all_data.update(new_data)

        with open(v_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["NPZ_Name", f"{metric_obj.short_name}_Score"])
            for name, score in sorted(all_data.items()):
                writer.writerow([name, score])

        with open(s_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{metric_obj.short_name}_Avg_Score"])
            avg_score = np.mean(list(all_data.values())) if all_data else 0.0
            writer.writerow([round(avg_score, 6)])

        print(f"\n[] ：={v_csv} | ={s_csv}")


# ==========================================


# ==========================================
def calculate_trajectory_accuracy(
    video_dir, save_dir, max_workers=None, auto_select_threshold=False, target_camera_dir=None
):
    """
    （index_ta）
    :param video_dir: NPZ
    :param save_dir:
    :param max_workers:
    :param auto_select_threshold: （，）
    """
    original_target_camera_dir = cfg.TARGET_CAMERA_DIR
    try:
        if target_camera_dir is not None:
            cfg.TARGET_CAMERA_DIR = target_camera_dir
        max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

        print(f"\n>>> ：{cfg.TRAJ_ACCURACY_THRESHOLDS}")
        print(f">>> TXT：{cfg.TARGET_CAMERA_DIR}")
        obj = TrajectoryAccuracyImpl(save_dir)
        engine = AtomicProcessor(video_dir, save_dir, max_workers)

        def _compute_logic(npz_path):
            return obj.calculate(npz_path)

        engine.run(obj, _compute_logic, f"MAX_WORKERS{max_workers}", is_accuracy=True)
    finally:
        cfg.TARGET_CAMERA_DIR = original_target_camera_dir


def calculate_trajectory_difference(video_dir, save_dir, max_workers=None):
    """
    （index_td）
    :param video_dir: NPZ
    :param save_dir:
    :param max_workers:
    """
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    obj = TrajectoryDifferenceImpl(save_dir)
    engine = AtomicProcessor(video_dir, save_dir, max_workers)

    def _compute_logic(npz_path):
        return obj.calculate(npz_path)

    engine.run(obj, _compute_logic, f"MAX_WORKERS{max_workers}", is_accuracy=False)


def calculate_trajectory_npz_similarity(video_dir, save_dir, max_workers=None, auto_select_threshold=False):
    """
    :param video_dir: NPZ
    :param save_dir:
    :param max_workers:
    :param auto_select_threshold: （，）
    """
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    print(f"\n>>> : Trajectory_NPZ_Sim | ：{cfg.TRAJ_NPZ_SIMILARITY_THRESHOLDS}")
    print(f">>> /NPZ：{cfg.SOURCE_NPZ_DIR}")
    print(f">>> NPZ：{video_dir}")

    obj = TrajectoryNPZSimilarityImpl(save_dir)
    engine = AtomicProcessor(video_dir, save_dir, max_workers)

    def _compute_logic(npz_path):
        return obj.calculate(npz_path)

    engine.run(obj, _compute_logic, f"MAX_WORKERS{max_workers}", is_accuracy=True)


def calculate_trajectory_npz_similarity_v2(
    video_dir, save_dir, source_npz_dir=None, max_workers=None, auto_select_threshold=False
):
    """
    :param source_npz_dir: /NPZ（，cfg）
    """
    original_source_dir = cfg.SOURCE_NPZ_DIR

    try:
        if source_npz_dir is not None:
            cfg.SOURCE_NPZ_DIR = source_npz_dir

        calculate_trajectory_npz_similarity(
            video_dir=video_dir, save_dir=save_dir, max_workers=max_workers, auto_select_threshold=auto_select_threshold
        )
    finally:
        cfg.SOURCE_NPZ_DIR = original_source_dir


def calculate_all_trajectory_metrics(
    video_dir, save_dir, max_workers=None, auto_select_threshold=False, target_camera_dir=None, source_npz_dir=None
):
    """
    ：（index_ta + index_td + index_npz_sim）
    :param video_dir: NPZ
    :param save_dir:
    :param max_workers:
    :param auto_select_threshold: （）
    """
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    calculate_trajectory_accuracy(video_dir, save_dir, max_workers, auto_select_threshold, target_camera_dir)

    calculate_trajectory_difference(video_dir, save_dir, max_workers)
    calculate_trajectory_npz_similarity_v2(video_dir, save_dir, source_npz_dir, max_workers, auto_select_threshold)


# ==========================================

# ==========================================
if __name__ == "__main__":
    raise SystemExit("Import this module or use run_iworldbench_evaluation.py.")
