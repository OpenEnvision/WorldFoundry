"""
Navigation trajectory evaluation — adaptive GT alignment (nATE-based NavScore).

Ported from main repo: tools/eval_navi_score_megasam.py + src/metrics/interaction/navigation_adaptive.py

Core idea:
1. Pure translation turns: GT = straight line in expected direction, length = pred displacement
2. Rotation turns: GT = adaptive orbit, radius R and angle theta inferred from pred
3. All alignment is GT -> pred; pred poses remain unchanged
4. Error normalization: divide by trajectory length/rotation to remove amplitude dependency

NavScore formula (ATE-only, no RPE):
    Accuracy    = 1 - (nATE_t + nATE_r) / 2
    Consistency = 1 - (cnATE_t + cnATE_r) / 2
    NavScore    = (Accuracy + Consistency) / 2
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("navigation_trajectory")

# Action -> expected motion mapping (local coordinate: X-right, Y-up, Z-forward)
ACTION_TO_MOTION = {
    "W": {"translation": (0, 0, 1), "rotation": None},
    "S": {"translation": (0, 0, -1), "rotation": None},
    "A": {"translation": (-1, 0, 0), "rotation": None},
    "D": {"translation": (1, 0, 0), "rotation": None},
    "left": {"translation": None, "rotation": (-1, 0, 0)},
    "right": {"translation": None, "rotation": (1, 0, 0)},
    "up": {"translation": None, "rotation": (0, 0, 1)},
    "down": {"translation": None, "rotation": (0, 0, -1)},
    "W+A": {"translation": (-0.707, 0, 0.707), "rotation": None},
    "W+D": {"translation": (0.707, 0, 0.707), "rotation": None},
    "S+A": {"translation": (-0.707, 0, -0.707), "rotation": None},
    "S+D": {"translation": (0.707, 0, -0.707), "rotation": None},
    "W+left": {"translation": (0, 0, 1), "rotation": (-1, 0, 0)},
    "W+right": {"translation": (0, 0, 1), "rotation": (1, 0, 0)},
    "W+up": {"translation": (0, 0, 1), "rotation": (0, 0, 1)},
    "W+down": {"translation": (0, 0, 1), "rotation": (0, 0, -1)},
    "S+left": {"translation": (0, 0, -1), "rotation": (-1, 0, 0)},
    "S+right": {"translation": (0, 0, -1), "rotation": (1, 0, 0)},
    "A+left": {"translation": (-1, 0, 0), "rotation": (-1, 0, 0)},
    "A+right": {"translation": (-1, 0, 0), "rotation": (1, 0, 0)},
    "D+left": {"translation": (1, 0, 0), "rotation": (-1, 0, 0)},
    "D+right": {"translation": (1, 0, 0), "rotation": (1, 0, 0)},
}

ACTION_ALIASES = {
    "forward": "W", "backward": "S",
    "cam_left": "left", "cam_right": "right",
    "cam_up": "up", "cam_down": "down",
    "look_left": "left", "look_right": "right",
    "look_up": "up", "look_down": "down",
    "pitch_up": "up", "pitch_down": "down",
    "yaw_left": "left", "yaw_right": "right",
}

MIN_DISP_NORM = 0.5
MIN_ROT_NORM = 10.0
MIN_DISP_THRESHOLD = 0.1
MIN_ROT_THRESHOLD = 3.0
ARC_SAMPLES_PER_TURN = 20
FALLBACK_DISP = 1.0
FALLBACK_ROT_DEG = 30.0
FALLBACK_DEPTH = 1.0

_SYMMETRIC_GROUPS = {
    "W": "forward_back", "S": "forward_back",
    "A": "lateral", "D": "lateral",
    "left": "yaw", "right": "yaw",
    "up": "pitch", "down": "pitch",
}


def normalize_action(action: str) -> str:
    """Normalize action aliases to canonical form."""
    parts = [p.strip() for p in str(action).strip().replace(",", "+").split("+") if p.strip()]
    normalized = []
    for p in parts:
        alias = ACTION_ALIASES.get(p, p)
        if alias in {"w", "a", "s", "d"}:
            alias = alias.upper()
        normalized.append(alias)
    return "+".join(sorted(normalized)) if len(normalized) > 1 else normalized[0] if normalized else action


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _rot_angle_deg(R: np.ndarray) -> float:
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))


def _total_rotation_deg(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for i in range(len(poses) - 1):
        total += _rot_angle_deg(poses[i, :3, :3].T @ poses[i + 1, :3, :3])
    return total


def _compute_ate(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    n = min(len(gt), len(pred))
    trans_err = np.linalg.norm(gt[:n, :3, 3] - pred[:n, :3, 3], axis=1)
    rot_err = np.array([_rot_angle_deg(gt[i, :3, :3].T @ pred[i, :3, :3]) for i in range(n)])
    return float(trans_err.mean()), float(rot_err.mean())


def _compute_rpe(gt: np.ndarray, pred: np.ndarray, delta: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    m = n - delta
    if m <= 0:
        return np.array([0.0]), np.array([0.0])
    trans = np.zeros(m, dtype=np.float64)
    rot = np.zeros(m, dtype=np.float64)
    for i in range(m):
        dQ = _inv_T(gt[i]) @ gt[i + delta]
        dP = _inv_T(pred[i]) @ pred[i + delta]
        E = _inv_T(dQ) @ dP
        trans[i] = np.linalg.norm(E[:3, 3])
        rot[i] = _rot_angle_deg(E[:3, :3])
    return trans, rot


def _is_pure_translation(action: str) -> bool:
    act = normalize_action(action)
    m = ACTION_TO_MOTION.get(act)
    return m is not None and m["rotation"] is None


def _get_symmetry_key(action: str) -> str:
    act = normalize_action(action)
    if "+" in act:
        parts = act.split("+")
        keys = [_SYMMETRIC_GROUPS.get(p, p) for p in parts]
        return "+".join(sorted(set(keys)))
    return _SYMMETRIC_GROUPS.get(act, act)


# ── Arc-length resampling ─────────────────────────────────────────────────────

def _resample_by_arc(poses: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample poses uniformly along arc-length."""
    from scipy.spatial.transform import Rotation as ScipyR, Slerp

    n = len(poses)
    if n <= 2 or n_samples <= 2:
        idx = np.linspace(0, max(n - 1, 0), min(n, n_samples)).astype(int)
        return poses[idx]

    diffs = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total = arc[-1]

    if total < 1e-8:
        idx = np.linspace(0, n - 1, n_samples).astype(int)
        return poses[idx]

    target = np.linspace(0, total, n_samples)
    new_pos = np.column_stack([
        np.interp(target, arc, poses[:, :3, 3][:, d]) for d in range(3)
    ])

    arc_norm = arc / total
    for i in range(1, len(arc_norm)):
        if arc_norm[i] <= arc_norm[i - 1]:
            arc_norm[i] = arc_norm[i - 1] + 1e-12
    target_norm = np.clip(target / total, arc_norm[0], arc_norm[-1])

    rots = ScipyR.from_matrix(poses[:, :3, :3])
    slerp = Slerp(arc_norm, rots)
    new_rots = slerp(target_norm)

    out = np.zeros((n_samples, 4, 4), dtype=np.float64)
    out[:, :3, :3] = new_rots.as_matrix()
    out[:, :3, 3] = new_pos
    out[:, 3, 3] = 1.0
    return out


def _compute_ate_arc(gt: np.ndarray, pred: np.ndarray,
                     n_samples: int = ARC_SAMPLES_PER_TURN) -> Tuple[float, float]:
    gt_s = _resample_by_arc(gt, n_samples)
    pred_s = _resample_by_arc(pred, n_samples)
    return _compute_ate(gt_s, pred_s)


# ── GT trajectory generation ─────────────────────────────────────────────────

def _build_gt_trajectory(
    action: str,
    n_frames: int,
    perspective: str = "first_person",
    rotation_angle_deg: float = 60.0,
    step_size: float = 2.0,
    subject_depth: float = 2.0,
) -> Optional[np.ndarray]:
    """Generate GT trajectory based on action and perspective."""
    from scipy.spatial.transform import Rotation as ScipyR

    action = normalize_action(action)
    expected = ACTION_TO_MOTION.get(action)
    if expected is None:
        return None

    target_R = np.eye(3)
    target_T = np.zeros(3)

    if expected["translation"] is not None:
        target_T = np.array(expected["translation"], dtype=np.float64) * step_size
    if expected["rotation"] is not None:
        rot_dir = np.array(expected["rotation"], dtype=np.float64)
        angle = np.deg2rad(rotation_angle_deg)
        if rot_dir[0] != 0:
            target_R = ScipyR.from_euler("y", rot_dir[0] * angle).as_matrix()
        elif rot_dir[2] != 0:
            target_R = ScipyR.from_euler("x", rot_dir[2] * angle).as_matrix()

    total_rotvec = ScipyR.from_matrix(target_R).as_rotvec()
    has_trans = expected["translation"] is not None
    has_rot = expected["rotation"] is not None
    n = max(n_frames - 1, 1)
    poses = []

    if has_trans and has_rot:
        if perspective == "first_person":
            R_cur = np.eye(3, dtype=np.float64)
            pos_cur = np.zeros(3, dtype=np.float64)
            rotvec_step = total_rotvec / n
            trans_step = target_T / n
            for idx in range(n_frames):
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = R_cur.copy()
                pose[:3, 3] = pos_cur.copy()
                poses.append(pose)
                if idx < n_frames - 1:
                    R_inc = ScipyR.from_rotvec(rotvec_step).as_matrix()
                    R_cur = R_cur @ R_inc
                    pos_cur += R_cur @ trans_step
        else:
            subject_base = np.array([0.0, 0.0, subject_depth], dtype=np.float64)
            offset = np.array([0.0, 0.0, -subject_depth], dtype=np.float64)
            for idx in range(n_frames):
                t = idx / n
                subject_pos = subject_base + t * target_T
                R_t = ScipyR.from_rotvec(t * total_rotvec).as_matrix()
                T_t = subject_pos + R_t @ offset
                fwd = subject_pos - T_t
                fwd /= (np.linalg.norm(fwd) + 1e-8)
                up = np.array([0.0, -1.0, 0.0])
                right = np.cross(fwd, up)
                right /= (np.linalg.norm(right) + 1e-8)
                up = np.cross(right, fwd)
                R_cam = np.stack([right, -up, fwd], axis=1)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = R_cam
                pose[:3, 3] = T_t
                poses.append(pose)

    elif has_rot and perspective != "first_person":
        subject_pos = np.array([0.0, 0.0, subject_depth], dtype=np.float64)
        offset = np.array([0.0, 0.0, -subject_depth], dtype=np.float64)
        for idx in range(n_frames):
            t = idx / n
            R_t = ScipyR.from_rotvec(t * total_rotvec).as_matrix()
            T_t = subject_pos + R_t @ offset
            fwd = subject_pos - T_t
            fwd /= (np.linalg.norm(fwd) + 1e-8)
            up = np.array([0.0, -1.0, 0.0])
            right = np.cross(fwd, up)
            right /= (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, fwd)
            R_cam = np.stack([right, -up, fwd], axis=1)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R_cam
            pose[:3, 3] = T_t
            poses.append(pose)

    elif has_rot:
        for idx in range(n_frames):
            t = idx / n
            R_cam = ScipyR.from_rotvec(t * total_rotvec).as_matrix()
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R_cam
            poses.append(pose)

    else:
        for idx in range(n_frames):
            t = idx / n
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = t * target_T
            poses.append(pose)

    return np.array(poses, dtype=np.float64)


def _estimate_orbit_radius(pred_turn: np.ndarray) -> Optional[float]:
    chord = np.linalg.norm(pred_turn[-1, :3, 3] - pred_turn[0, :3, 3])
    theta = np.deg2rad(_total_rotation_deg(pred_turn))
    if theta < np.deg2rad(5) or np.sin(theta / 2) < 1e-6:
        return None
    return float(chord / (2 * np.sin(theta / 2)))


# ── Adaptive GT builders ──────────────────────────────────────────────────────

def _build_gt_translation(pred_turn: np.ndarray, action: str, R_world: np.ndarray) -> np.ndarray:
    """Pure translation GT: straight line in expected direction, length matches pred."""
    nf = len(pred_turn)
    act_n = normalize_action(action)
    direction = np.array(ACTION_TO_MOTION[act_n]["translation"], dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    world_dir = R_world @ direction
    pred_disp = np.linalg.norm(pred_turn[-1, :3, 3] - pred_turn[0, :3, 3])
    gt_disp = pred_disp if pred_disp > MIN_DISP_THRESHOLD else FALLBACK_DISP

    seg = np.zeros((nf, 4, 4))
    start_pos = pred_turn[0, :3, 3]
    for i in range(nf):
        t = i / max(nf - 1, 1)
        seg[i, :3, :3] = R_world
        seg[i, :3, 3] = start_pos + t * gt_disp * world_dir
        seg[i, 3, 3] = 1.0
    return seg


def _build_gt_rotation(pred_turn: np.ndarray, action: str, perspective: str) -> np.ndarray:
    """Rotation GT: adaptive orbit with R and theta inferred from pred."""
    nf = len(pred_turn)
    pred_rot = _total_rotation_deg(pred_turn)
    est_R = _estimate_orbit_radius(pred_turn)

    if pred_rot < MIN_ROT_THRESHOLD:
        kwargs = {"rotation_angle_deg": FALLBACK_ROT_DEG, "subject_depth": FALLBACK_DEPTH}
    elif est_R is not None:
        depth = est_R
        if perspective != "first_person":
            depth = max(depth, FALLBACK_DEPTH)
        kwargs = {"rotation_angle_deg": max(pred_rot, FALLBACK_ROT_DEG), "subject_depth": depth}
    else:
        kwargs = {"rotation_angle_deg": FALLBACK_ROT_DEG, "subject_depth": FALLBACK_DEPTH}

    gt_seg = _build_gt_trajectory(action, nf, perspective, **kwargs)
    if gt_seg is None:
        return np.tile(np.eye(4), (nf, 1, 1))

    R_gt0 = gt_seg[0, :3, :3]
    R_pred0 = pred_turn[0, :3, :3]
    R_align = R_pred0 @ R_gt0.T

    gt_pos = gt_seg[:, :3, 3] - gt_seg[0, :3, 3]
    pred_pos = pred_turn[:, :3, 3] - pred_turn[0, :3, 3]
    gt_range = np.linalg.norm(gt_pos, axis=1).mean() + 1e-12
    pred_range = np.linalg.norm(pred_pos, axis=1).mean() + 1e-12

    if pred_rot < MIN_ROT_THRESHOLD or pred_range < MIN_DISP_THRESHOLD:
        scale = 1.0
    else:
        scale = pred_range / gt_range

    gt_pos_rot = (R_align @ gt_pos.T).T
    start_pos = pred_turn[0, :3, 3]

    seg = np.zeros_like(gt_seg)
    for i in range(len(gt_seg)):
        seg[i, :3, :3] = R_align @ gt_seg[i, :3, :3]
        seg[i, :3, 3] = scale * gt_pos_rot[i] + start_pos
        seg[i, 3, 3] = 1.0
    return seg


# ── Consistency (trajectory shape similarity between same-group turns) ────────

def _need_mirror(act1: str, act2: str) -> Optional[str]:
    a1, a2 = normalize_action(act1), normalize_action(act2)
    if a1 == a2:
        return None
    pairs = {
        frozenset({"right", "left"}): "yaw",
        frozenset({"up", "down"}): "pitch",
        frozenset({"W", "S"}): "forward",
        frozenset({"A", "D"}): "lateral",
    }
    return pairs.get(frozenset({a1, a2}))


_MIRROR_MATRICES = {
    "yaw": np.diag([-1.0, 1.0, 1.0]),
    "pitch": np.diag([1.0, -1.0, 1.0]),
    "forward": np.diag([1.0, 1.0, -1.0]),
    "lateral": np.diag([-1.0, 1.0, 1.0]),
}


def _normalize_turn(poses: np.ndarray) -> np.ndarray:
    R0_inv = poses[0, :3, :3].T
    t0 = poses[0, :3, 3]
    out = np.zeros_like(poses)
    for i in range(len(poses)):
        out[i, :3, :3] = R0_inv @ poses[i, :3, :3]
        out[i, :3, 3] = R0_inv @ (poses[i, :3, 3] - t0)
        out[i, 3, 3] = 1.0
    return out


def _mirror_turn(poses: np.ndarray, mirror_type: str) -> np.ndarray:
    M = _MIRROR_MATRICES[mirror_type]
    out = np.zeros_like(poses)
    for i in range(len(poses)):
        out[i, :3, :3] = M @ poses[i, :3, :3] @ M
        out[i, :3, 3] = M @ poses[i, :3, 3]
        out[i, 3, 3] = 1.0
    return out


def _resample_poses(poses: np.ndarray, n: int) -> np.ndarray:
    return poses[np.linspace(0, len(poses) - 1, n).astype(int)]


def _compute_turn_pair_error(poses1: np.ndarray, poses2: np.ndarray,
                             mirror_type: Optional[str] = None) -> Dict[str, float]:
    """Compute normalized ATE between two turns (for consistency)."""
    n1 = _normalize_turn(poses1)
    n2 = _normalize_turn(poses2)
    if mirror_type:
        n2 = _mirror_turn(n2, mirror_type)

    nc = min(len(n1), len(n2))
    n1 = _resample_poses(n1, nc)
    n2 = _resample_poses(n2, nc)

    trans_err = np.linalg.norm(n1[:, :3, 3] - n2[:, :3, 3], axis=1)
    ate_t = float(trans_err.mean())
    rot_err = np.array([_rot_angle_deg(n1[i, :3, :3].T @ n2[i, :3, :3]) for i in range(nc)])
    ate_r = float(rot_err.mean())

    path = (np.sum(np.linalg.norm(np.diff(n1[:, :3, 3], axis=0), axis=1)) +
            np.sum(np.linalg.norm(np.diff(n2[:, :3, 3], axis=0), axis=1))) / 2
    rot = (_total_rotation_deg(n1) + _total_rotation_deg(n2)) / 2
    norm_path = max(path, MIN_DISP_NORM)
    norm_rot = max(rot, MIN_ROT_NORM)

    return {
        "nATE_t": min(ate_t / norm_path, 1.0),
        "nATE_r": min(ate_r / norm_rot, 1.0),
    }


def _compute_consistency_trajectory(
    poses: np.ndarray,
    turn_bounds: List[Tuple[int, int]],
    actions: List[str],
) -> Dict[str, float]:
    """Trajectory shape consistency: cnATE across same-group turn pairs."""
    turn_data = []
    for (s, e), act in zip(turn_bounds, actions):
        turn_data.append((poses[s:e], act, _get_symmetry_key(act)))

    all_errs: List[Dict[str, float]] = []
    for i, j in combinations(range(len(turn_data)), 2):
        p1, a1, k1 = turn_data[i]
        p2, a2, k2 = turn_data[j]
        if k1 != k2:
            continue
        mt = _need_mirror(a1, a2)
        all_errs.append(_compute_turn_pair_error(p1, p2, mirror_type=mt))

    if not all_errs:
        return {"cnATE_t": 0.0, "cnATE_r": 0.0, "n_pairs": 0}

    return {
        "cnATE_t": float(np.mean([e["nATE_t"] for e in all_errs])),
        "cnATE_r": float(np.mean([e["nATE_r"] for e in all_errs])),
        "n_pairs": len(all_errs),
    }


# ── Main evaluation entry point ──────────────────────────────────────────────

def evaluate_navigation(
    poses: np.ndarray,
    turn_bounds: List[Tuple[int, int]],
    actions: List[str],
    perspective: str = "first_person",
) -> Dict[str, float]:
    """
    Evaluate navigation trajectory quality using nATE-based NavScore.

    Args:
        poses: Camera c2w poses (N, 4, 4)
        turn_bounds: Per-turn pose index ranges [(start, end), ...]
        actions: Per-turn actions ["W", "right", ...]
        perspective: "first_person" or "third_person"

    Returns:
        Dict with: NavScore, nATE_t, nATE_r, cnATE_t, cnATE_r, etc.
    """
    assert len(turn_bounds) == len(actions)

    R_world = poses[turn_bounds[0][0], :3, :3].copy()
    gt_segments = []
    current_pos = poses[turn_bounds[0][0], :3, 3].copy()

    for (s, e), act in zip(turn_bounds, actions):
        pred_turn = poses[s:e]

        if _is_pure_translation(act):
            gt_seg = _build_gt_translation(pred_turn, act, R_world)
        else:
            gt_seg = _build_gt_rotation(pred_turn, act, perspective)

        offset = current_pos - gt_seg[0, :3, 3]
        gt_seg_global = gt_seg.copy()
        gt_seg_global[:, :3, 3] += offset

        current_pos = gt_seg_global[-1, :3, 3].copy()
        if not _is_pure_translation(act):
            R_world = gt_seg_global[-1, :3, :3].copy()

        gt_segments.append(gt_seg_global)

    # Per-turn arc-length sampled ATE (equal weight per turn)
    gt_sampled_parts, pred_sampled_parts = [], []
    for (s, e), gt_seg in zip(turn_bounds, gt_segments):
        pred_turn = poses[s:e]
        nt = min(len(gt_seg), len(pred_turn))
        gt_sampled_parts.append(_resample_by_arc(gt_seg[:nt], ARC_SAMPLES_PER_TURN))
        pred_sampled_parts.append(_resample_by_arc(pred_turn[:nt], ARC_SAMPLES_PER_TURN))
    gt_sampled = np.concatenate(gt_sampled_parts, axis=0)
    pred_sampled = np.concatenate(pred_sampled_parts, axis=0)
    ate_t, ate_r = _compute_ate(gt_sampled, pred_sampled)

    # Normalize by pred motion magnitude
    total_s, total_e = turn_bounds[0][0], turn_bounds[-1][1]
    pred_full = poses[total_s:total_e]
    total_path = _path_length(pred_full)
    total_rot = _total_rotation_deg(pred_full)
    norm_path = max(total_path, MIN_DISP_NORM)
    norm_rot = max(total_rot, MIN_ROT_NORM)

    nate_t = min(ate_t / norm_path, 1.0)
    nate_r = min(ate_r / norm_rot, 1.0)

    # Trajectory consistency (same-group turn pairs)
    ct = _compute_consistency_trajectory(poses, turn_bounds, actions)

    # NavScore = (Accuracy + Consistency) / 2
    accuracy = 1.0 - (nate_t + nate_r) / 2.0
    consistency = 1.0 - (ct["cnATE_t"] + ct["cnATE_r"]) / 2.0
    nav_score = (accuracy + consistency) / 2.0

    return {
        "NavScore": float(nav_score),
        "accuracy": float(accuracy),
        "consistency": float(consistency),
        "nATE_t": float(nate_t),
        "nATE_r": float(nate_r),
        "cnATE_t": ct["cnATE_t"],
        "cnATE_r": ct["cnATE_r"],
        "consistency_pairs": ct["n_pairs"],
        "ATE_t": float(ate_t),
        "ATE_r": float(ate_r),
        "total_path_length": float(total_path),
        "total_rotation_deg": float(total_rot),
    }
