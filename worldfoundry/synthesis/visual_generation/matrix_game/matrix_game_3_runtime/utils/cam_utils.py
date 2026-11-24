import math

import numpy as np
import torch
import trimesh
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.networks.wan.camera_geometry import interpolate_camera_poses

try:
    import torch_npu
    from opensora.npu_config import npu_config
except BaseException:
    torch_npu = None
    npu_config = None


def create_camera_frustum(K, R, t, width, height, near=0.5, far=3.0):
    Kinv = np.linalg.inv(K)

    def pixel_to_ray(u, v):
        pix = np.array([u, v, 1.0])
        ray = Kinv @ pix
        return ray

    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    pts = []
    for d in [near, far]:
        for u, v in corners:
            r = pixel_to_ray(u, v)
            p_cam = r * d
            p_world = R @ p_cam + t
            pts.append(p_world)
    C = t
    pts = np.vstack([pts, C])
    hull = trimesh.convex.convex_hull(pts)
    return hull


def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid


def prompt_cleaning(cap):
    assert isinstance(cap, str)
    return cap.replace("\n", " ").replace("\t", " ")


def crop_frame(frame, target_w, target_h):
    w, h = frame.size
    crop_x = w // 2 - target_w // 2
    crop_y = h // 2 - target_h // 2
    cropped_frame = frame.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    resized_frame = cropped_frame.resize((w, h), Image.BICUBIC)
    return resized_frame


def zoomed_in(frame, zoom_factor=0.2, num_frames=49):
    width, height = frame.size
    smallest_w, smallest_h = int(width * (1 - zoom_factor)), int(height * (1 - zoom_factor))
    width_per_frame = np.linspace(width, smallest_w, num_frames)
    height_per_frame = np.linspace(height, smallest_h, num_frames)
    result_frames = [np.array(frame)]
    for i in range(1, num_frames):
        frame_i = crop_frame(frame, width_per_frame[i], height_per_frame[i])
        result_frames.append(np.array(frame_i))
    result_frames = np.stack(result_frames)
    return result_frames


def normalize_to_neg_one_to_one(x):
    """Map values from [0, 1] to [-1, 1]."""
    return 2.0 * x - 1.0


def get_K(height, width):
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)

    video_h, video_w = height, width
    fx = video_w / (2 * np.tan(fov_rad / 2))
    fy = video_h / (2 * np.tan(fov_rad / 2))
    cx = video_w / 2
    cy = video_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def get_convert_data(video_all, extrinsics_all):
    total_len = extrinsics_all.shape[0]
    assert total_len % 4 == 1 and video_all.shape[0] == total_len, (
        "total_len: {total_len} must be 4k + 1 and video_all.shape[0]: {video_all.shape[0]} must be equal to total_len"
    )

    num_actions = total_len - 1
    half_actions = num_actions // 2

    video_all[0] = video_all[-1]
    extrinsics_all[0] = extrinsics_all[-1]
    video_all[1 : half_actions + 1] = torch.flip(video_all[-half_actions:], dims=[0])
    extrinsics_all[1 : half_actions + 1] = torch.flip(extrinsics_all[-half_actions:], dims=[0])

    return video_all, extrinsics_all


def generate_points_in_sphere(n_points, radius, min_radius=None):
    """Generate uniformly distributed points within a sphere or spherical shell.

    Args:
        n_points: Number of points to generate
        radius: Maximum radius (outer boundary)
        min_radius: Minimum radius (inner boundary). If None, sample in full sphere.
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    if min_radius is None or min_radius <= 0:
        r = radius * torch.pow(samples_r, 1 / 3)
    else:
        r_cubed = min_radius**3 + (radius**3 - min_radius**3) * samples_r
        r = torch.pow(r_cubed, 1 / 3)

    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def is_inside_fov_3d_direct(points, position, rotation_matrix, fov_half_h, fov_half_v):
    """
    Check whether points are within a given 3D field of view (FOV).
    Directly uses rotation matrix to avoid euler angle order issues.

    Args:
        points: (N, 3) Sample point coordinates in world space
        position: (3,) Camera position in world space
        rotation_matrix: (3, 3) Camera-to-world rotation matrix (c2w)
        fov_half_h: Horizontal half-FOV angle (in degrees), scalar or tensor
        fov_half_v: Vertical half-FOV angle (in degrees), scalar or tensor
    Returns:
        Boolean tensor (N,), indicating whether each point is inside the FOV
    """
    vectors = points - position

    vectors_cam = vectors @ rotation_matrix

    x_cam = vectors_cam[..., 0]
    y_cam = vectors_cam[..., 1]
    z_cam = vectors_cam[..., 2]
    azimuth = torch.atan2(x_cam, z_cam) * (180 / math.pi)
    elevation = torch.atan2(y_cam, torch.sqrt(x_cam**2 + z_cam**2)) * (180 / math.pi)

    in_front = z_cam > 0
    in_h_fov = torch.abs(azimuth) < fov_half_h
    in_v_fov = torch.abs(elevation) < fov_half_v

    return in_front & in_h_fov & in_v_fov


def select_memory_idx_fov(
    extrinsics_all, current_start_frame_idx, selected_index_base, return_confidence=False, use_gpu=False
):
    if use_gpu:
        device = (
            extrinsics_all.device
            if isinstance(extrinsics_all, torch.Tensor)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if isinstance(extrinsics_all, np.ndarray):
            extrinsics_tensor = torch.from_numpy(extrinsics_all).to(device).float()
        else:
            extrinsics_tensor = extrinsics_all.to(device).float()

        video_w, video_h = 1280, 720
        fov_rad = np.deg2rad(90)
        fx = video_w / (2 * np.tan(fov_rad / 2))
        fy = video_h / (2 * np.tan(fov_rad / 2))

        K = torch.tensor([[fx, 0, video_w / 2], [0, fy, video_h / 2], [0, 0, 1]], device=device)

        if current_start_frame_idx <= 1:
            return (
                ([0] * len(selected_index_base), [0.0] * len(selected_index_base))
                if return_confidence
                else [0] * len(selected_index_base)
            )

        candidate_indices = torch.arange(1, current_start_frame_idx, device=device)

        R_cand = extrinsics_tensor[candidate_indices, :3, :3]
        t_cand = extrinsics_tensor[candidate_indices, :3, 3:4]

        R_cand_inv = R_cand.transpose(1, 2)
        t_cand_inv = -torch.bmm(R_cand_inv, t_cand)  # [N, 3, 1]

        selected_index = []
        selected_confidence = []

        near, far = 0.1, 30.0
        num_side = 10
        z_samples = torch.linspace(near, far, num_side, device=device)
        x_samples = torch.linspace(-1, 1, num_side, device=device)
        y_samples = torch.linspace(-1, 1, num_side, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(x_samples, y_samples, z_samples, indexing="ij")

        points_cam_base = torch.stack(
            [
                grid_x.reshape(-1) * grid_z.reshape(-1) * (video_w / (2 * fx)),
                grid_y.reshape(-1) * grid_z.reshape(-1) * (video_h / (2 * fy)),
                grid_z.reshape(-1),
            ],
            dim=0,
        )

        for i in selected_index_base:
            E_base = extrinsics_tensor[i]
            points_world = E_base[:3, :3] @ points_cam_base + E_base[:3, 3:4]

            points_world_batched = points_world.unsqueeze(0)
            points_in_cands = (
                torch.bmm(R_cand_inv, points_world_batched.expand(len(candidate_indices), -1, -1)) + t_cand_inv
            )

            x = points_in_cands[:, 0, :]
            y = points_in_cands[:, 1, :]
            z = points_in_cands[:, 2, :]

            u = (x * fx / torch.clamp(z, min=1e-6)) + video_w / 2
            v = (y * fy / torch.clamp(z, min=1e-6)) + video_h / 2

            in_view = (z > near) & (z < far) & (u >= 0) & (u <= video_w) & (v >= 0) & (v <= video_h)

            ratios = in_view.float().mean(dim=1)
            best_idx = torch.argmax(ratios)

            selected_index.append(candidate_indices[best_idx].item())
            selected_confidence.append(ratios[best_idx].item())

        return (selected_index, selected_confidence) if return_confidence else selected_index
    else:
        video_w = 1280
        video_h = 720
        fov_deg = 90
        fov_rad = np.deg2rad(fov_deg)

        fx = video_w / (2 * np.tan(fov_rad / 2))
        fy = video_h / (2 * np.tan(fov_rad / 2))
        cx = video_w / 2
        cy = video_h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        selected_index = []
        selected_confidence = []
        for i in selected_index_base:
            ratios = []
            for j in range(1, current_start_frame_idx):
                ratio = cal_intersection_ratio(K, extrinsics_all[i], extrinsics_all[j], 720, 1280, near=0.0, far=30)
                ratios.append(ratio)
            ratios_arr = np.array(ratios)
            sorted_idx = np.argsort(ratios_arr)[::-1]
            selected_index.append(sorted_idx[0])
            selected_confidence.append(ratios_arr[sorted_idx[0]])

        if return_confidence:
            return selected_index, selected_confidence
        return selected_index


def cal_intersection_ratio(K, E1, E2, height, width, near=0.0, far=50):
    R1 = E1[:3, :3]
    t1 = E1[:3, 3]

    R2 = E2[:3, :3]
    t2 = E2[:3, 3]

    frustum2 = create_camera_frustum(K, R2, t2, width, height, near, far)
    frustum1 = create_camera_frustum(K, R1, t1, width, height, near, far)
    intersection = frustum1.intersection(frustum2)
    if intersection.is_empty:
        return 0
    else:
        return intersection.volume / frustum1.volume


def cal_intersection_ratio_gpu(K, E1, E2, height, width, near=0.0, far=50):
    """
    Approximate frustum overlap on the GPU by sampling.

    trimesh has no GPU support, so this samples points inside the frustum of E1 and returns the
    fraction of them that falls inside the frustum of E2.
    """
    device = E1.device

    z_samples = torch.linspace(near + 0.1, far, 10, device=device)  # Depth samples
    x_samples = torch.linspace(-1, 1, 10, device=device)
    y_samples = torch.linspace(-1, 1, 10, device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x_samples, y_samples, z_samples, indexing="ij")
    points_cam1 = torch.stack(
        [
            grid_x.reshape(-1) * grid_z.reshape(-1) * (width / (2 * K[0, 0])),
            grid_y.reshape(-1) * grid_z.reshape(-1) * (height / (2 * K[1, 1])),
            grid_z.reshape(-1),
        ],
        dim=-1,
    )  # [N, 3]

    points_world = points_cam1 @ E1[:3, :3].T + E1[:3, 3]

    points_cam2 = (points_world - E2[:3, 3]) @ E2[:3, :3]

    x2 = points_cam2[:, 0]
    y2 = points_cam2[:, 1]
    z2 = points_cam2[:, 2]

    u2 = (x2 * K[0, 0] / torch.clamp(z2, min=1e-6)) + K[0, 2]
    v2 = (y2 * K[1, 1] / torch.clamp(z2, min=1e-6)) + K[1, 2]

    in_view = (z2 > near) & (z2 < far) & (u2 >= 0) & (u2 <= width) & (v2 >= 0) & (v2 <= height)

    return torch.mean(in_view.float()).item()


def select_memory_idx(
    c2w_all,
    chunk_start_idx,
    chunk_size,
    memory_end_idx,
    fov_half_h=45.0,
    fov_half_v=45.0,
    memory_num=5,
    return_confidence=False,
    is_train=False,
):
    """
    Select memory frames for a single chunk using multi-center FOV strategy.
    Uses frames 3, 7, 11 of the chunk as centers, for each center selects the closest memory frame.

    Args:
        c2w_all: (N, 4, 4) all camera to world matrices (numpy array or tensor)
        chunk_start_idx: absolute start frame index of this chunk
        chunk_size: number of frames in this chunk
        memory_end_idx: end index for selection, select from [0, memory_end_idx)
        memory_num: number of memory frames to select (should be 3)
        fov_half_h: horizontal half FOV in degrees
        fov_half_v: vertical half FOV in degrees
        return_confidence: if True, also return confidence scores for each selected frame
    Returns:
        selected_indices: list of selected frame indices (length = memory_num)
        confidences (optional): list of confidence scores for each selected frame
    """

    if memory_end_idx < memory_num * 4:
        if return_confidence:
            return [0] * memory_num, [0] * memory_num
        return [0] * memory_num

    assert memory_end_idx % 4 == 1, f"memory_end_idx:{memory_end_idx} must be 4k + 1 form (1, 5, 9, 13, ...)"

    c2w_all = torch.from_numpy(c2w_all).float() if isinstance(c2w_all, np.ndarray) else c2w_all.float()
    device = c2w_all.device

    positions = c2w_all[:, :3, 3]  # (N, 3)
    rotations = c2w_all[:, :3, :3]  # (N, 3, 3)

    if is_train:
        num_samples = 3000
        radius = 20
    else:
        num_samples = 10000
        radius = 30
    points = generate_points_in_sphere(num_samples, radius).to(device)

    chunk_end_idx = chunk_start_idx + chunk_size
    chunk_pos = positions[chunk_start_idx:chunk_end_idx]
    chunk_rot = rotations[chunk_start_idx:chunk_end_idx]

    center_indices = [chunk_size - 1, chunk_size - 9, chunk_size - 17, chunk_size - 25, chunk_size - 33]

    candidates = torch.arange(1, memory_end_idx, 4, device=device)

    def in_fov_batch(points_world, positions_batch, rotations_batch):
        vectors = points_world.unsqueeze(0) - positions_batch.unsqueeze(1)  # (F, S, 3)
        vectors_cam = torch.matmul(vectors, rotations_batch)  # (F, S, 3)
        x_cam = vectors_cam[..., 0]
        y_cam = vectors_cam[..., 1]
        z_cam = vectors_cam[..., 2]
        azimuth = torch.atan2(x_cam, z_cam) * (180 / math.pi)
        elevation = torch.atan2(y_cam, torch.sqrt(x_cam**2 + z_cam**2)) * (180 / math.pi)
        in_front = z_cam > 0
        in_h_fov = torch.abs(azimuth) < fov_half_h
        in_v_fov = torch.abs(elevation) < fov_half_v
        return in_front & in_h_fov & in_v_fov  # (F, S)

    selected_indices = []
    selected_confidences = []
    selected_mask = torch.zeros(memory_end_idx, dtype=torch.bool, device=device)
    selected_mask[0] = True

    for center_idx in center_indices:
        # Points relative to current center frame
        points_world = points + chunk_pos[center_idx : center_idx + 1]  # (num_samples, 3)

        # Check which points are in chunk's FOV (union of all frames in chunk)
        in_fov_chunk = in_fov_batch(points_world, chunk_pos, chunk_rot).any(dim=0)

        # Compute FOV masks for candidate frames only
        cand_pos = positions[candidates]
        cand_rot = rotations[candidates]
        in_fov_frames = in_fov_batch(points_world, cand_pos, cand_rot)  # (num_candidates, num_samples)

        # Compute overlap ratio for each candidate frame
        overlap_ratio = (in_fov_chunk.unsqueeze(0) & in_fov_frames).sum(1).float() / max(1, in_fov_chunk.sum().item())
        confidence = overlap_ratio

        # Mask already selected indices and their aligned 4-frame chunks
        candidate_mask = selected_mask[candidates]
        confidence[candidate_mask] = -1e10

        # Select best frame (highest overlap)
        if torch.all(candidate_mask):
            best_idx = candidates[-1].item()
            best_confidence = -1e10
        else:
            best_cand_idx = confidence.argmax().item()
            best_idx = candidates[best_cand_idx].item()
            best_confidence = confidence[best_cand_idx].item()

        # If confidence is 0 or negative, select the largest unselected index (most recent in time)
        if best_confidence <= 0:
            unselected_candidates = candidates[~candidate_mask]
            if len(unselected_candidates) > 0:
                best_idx = unselected_candidates.max().item()
                best_confidence = 0.0

        selected_indices.append(best_idx)
        selected_confidences.append(best_confidence)
        selected_mask[best_idx] = True

        # Mark aligned 4-frame chunk as selected to avoid re-selecting
        # Aligned chunk: [1 + (index - 1) // 4 * 4, 1 + (index - 1) // 4 * 4 + 4)
        aligned_start = 1 + ((best_idx - 1) // 4) * 4
        aligned_end = aligned_start + 4
        for frame_idx in range(aligned_start, aligned_end):
            selected_mask[frame_idx] = True

    if return_confidence:
        return selected_indices, selected_confidences
    return selected_indices


def get_extrinsics(video_rotation, video_position):
    num_frames = len(video_rotation)
    Extrinsics_vid = []
    for idx in range(num_frames):
        frame_rotation = video_rotation[idx]
        frame_position = video_position[idx]
        roll = frame_rotation[0]
        pitch = frame_rotation[1]
        yaw = frame_rotation[2]

        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R = Rz @ Ry @ Rx
        Extrinsics = np.eye(4)
        Extrinsics[:3, :3] = R
        Extrinsics[:3, 3] = frame_position
        Extrinsics_vid.append(Extrinsics)
    R_init = np.array(
        [
            [0, 0, 1],  # X_cam -> Y_world
            [1, 0, 0],  # Y_cam -> Z_world
            [0, -1, 0],  # Z_cam -> X_world
        ]
    )
    Extrinsics = torch.from_numpy(np.array(Extrinsics_vid))
    Extrinsics[:, :3, :3] = Extrinsics[:, :3, :3] @ R_init
    Extrinsics[:, :3, 3] = Extrinsics[:, :3, 3] * 0.01
    return Extrinsics


def get_intrinsics(height, width):
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)

    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2

    K = torch.tensor([fx, fy, cx, cy])
    return K


def _interpolate_camera_poses_handedness(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # Convert left-handed rotations to right-handed for SciPy, then convert back.
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat
    c2ws = interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=src_rot_mat,
        src_trans_vec=src_trans_vec,
        tgt_indices=tgt_indices,
    )
    if flip_handedness:
        flip_mat_t = torch.from_numpy(flip_mat).to(c2ws.device, dtype=c2ws.dtype)
        c2ws[:, :3, :3] = c2ws[:, :3, :3] @ flip_mat_t
    return c2ws
