import math
import os

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default


def get_default_intrinsics(
    fov_rad=DEFAULT_FOV_RAD,
    aspect_ratio=1.0,
):
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor([fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad)
    if aspect_ratio >= 1.0:  # W >= H
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio
    else:  # W < H
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio
    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, torch.eye(3, device=focal_x.device, dtype=bool)] = torch.stack(
        [focal_x, focal_y, torch.ones_like(focal_x)], dim=-1
    )
    intrinsics[:, :, -1] = torch.tensor([0.5, 0.5, 1.0], device=focal_x.device, dtype=focal_x.dtype)
    return intrinsics


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],  # noqa: B006 - Keeps the upstream default object and signature.
):
    # Support for batch dimension
    has_batch_dim = len(extrinsics.shape) == 4

    if has_batch_dim:
        # [B, N, 4, 4] -> reshape to handle batch
        batch_size, num_cameras = extrinsics.shape[0:2]
        extrinsics_flat = extrinsics.reshape(-1, *extrinsics.shape[2:])

        # Handle extrinsics_src appropriately
        if len(extrinsics_src.shape) == 3:  # [B, 4, 4]
            extrinsics_src_expanded = extrinsics_src.unsqueeze(1).expand(-1, num_cameras, -1, -1)
            extrinsics_src_flat = extrinsics_src_expanded.reshape(-1, *extrinsics_src.shape[1:])
        else:  # [4, 4] - single extrinsics_src for all batches
            extrinsics_src_flat = extrinsics_src.expand(batch_size * num_cameras, -1, -1)

        # Handle intrinsics for batch
        if intrinsics is None:
            intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
            intrinsics = intrinsics.expand(batch_size * num_cameras, -1, -1)
        elif len(intrinsics.shape) == 3:  # [N, 3, 3]
            if intrinsics.shape[0] == num_cameras:
                intrinsics = intrinsics.expand(batch_size, -1, -1, -1).reshape(-1, *intrinsics.shape[1:])
            else:
                intrinsics = intrinsics.expand(batch_size * num_cameras, -1, -1)
        elif len(intrinsics.shape) == 4:  # [B, N, 3, 3]
            intrinsics = intrinsics.reshape(-1, *intrinsics.shape[2:])
    else:
        # Original behavior for non-batch input
        extrinsics_flat = extrinsics
        extrinsics_src_flat = extrinsics_src
        if intrinsics is None:
            intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)

    # Process intrinsics normalization
    if not (torch.all(intrinsics[:, :2, -1] >= 0) and torch.all(intrinsics[:, :2, -1] <= 1)):
        intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8

    # Ensure normalized intrinsics
    assert torch.all(intrinsics[:, :2, -1] >= 0) and torch.all(intrinsics[:, :2, -1] <= 1), (
        "Intrinsics should be expressed in resolution-independent normalized image coordinates."
    )

    c2w_src = torch.linalg.inv(extrinsics_src_flat)
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum("vnm,vmp->vnp", extrinsics_flat, c2w_src)

    intrinsics[:, :2] *= extrinsics_flat.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)

    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)

    # Reshape back to batch dimension if needed
    if has_batch_dim:
        plucker = plucker.reshape(batch_size, num_cameras, *plucker.shape[1:])

    return plucker


def camera_outlier_filtered_center(ref_c2ws: torch.Tensor) -> torch.Tensor:
    """Compute outlier-filtered camera translation center deterministically."""
    positions = ref_c2ws[:, :3, 3]
    # CUDA median with dim= lacks a deterministic kernel; compute median over the small camera axis on CPU.
    median_pos = (
        positions.detach().cpu().median(0, keepdim=True).values.to(device=positions.device, dtype=positions.dtype)
    )
    camera_dist_2med = torch.norm(positions - median_pos, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med.float(), 0.97) * 10,
        max=1e6,
    )
    return positions[valid_mask].mean(0, keepdim=True)


def get_value_dict(
    curr_imgs,
    curr_imgs_clip,
    curr_input_frame_indices,
    curr_c2ws,
    curr_Ks,
    curr_input_camera_indices,
    all_c2ws,
    camera_scale,
):
    assert sorted(curr_input_camera_indices) == sorted(range(len(curr_input_camera_indices)))
    H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

    value_dict = {}
    value_dict["cond_frames_without_noise"] = curr_imgs_clip[curr_input_frame_indices]
    value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
    value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["cond_frames_mask"][curr_input_frame_indices] = True
    value_dict["cond_aug"] = 0.0

    c2w = to_hom_pose(curr_c2ws.float()) if curr_c2ws.shape[-1] == 3 else curr_c2ws
    w2c = torch.linalg.inv(c2w)

    # camera centering
    c2w[:, :3, 3] -= camera_outlier_filtered_center(all_c2ws)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor
    value_dict["plucker_coordinate"] = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=curr_Ks.float().clone(),
        target_size=(H // F, W // F),
    )

    value_dict["c2w"] = c2w
    value_dict["K"] = curr_Ks
    value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["camera_mask"][curr_input_camera_indices] = True

    return value_dict


def parse_meta_data(file_path, image_height=288, image_width=512):
    with open(file_path) as file:
        lines = file.readlines()

    # First line is the video URL
    _video_url = lines[0].strip()

    line = lines[1]
    data = line.strip().split()
    # Construct the camera intrinsics matrix K
    focal_length_x = float(data[1])
    focal_length_y = float(data[2])
    principal_point_x = float(data[3])
    principal_point_y = float(data[4])

    original_K = [[focal_length_x, 0, principal_point_x], [0, focal_length_y, principal_point_y], [0, 0, 1]]

    K = [
        [focal_length_x * image_width, 0, principal_point_x * image_width],
        [0, focal_length_y * image_height, principal_point_y * image_height],
        [0, 0, 1],
    ]

    # Initialize a list to store frame data
    timestamp_to_c2ws = {}
    timestamps = []
    # Process each frame line
    for line in lines[1:]:
        data = line.strip().split()
        timestamp = int(data[0])
        R_t = [float(x) for x in data[7:]]
        P = [R_t[0:4], R_t[4:8], R_t[8:12], [0, 0, 0, 1]]
        timestamp_to_c2ws[timestamp] = np.array(P)
        timestamps.append(timestamp)
    return timestamps, np.array(K), timestamp_to_c2ws, original_K


def get_wh_with_fixed_shortest_side(w, h, size):
    # size is smaller or equal to zero, we return original w h
    if size is None or size <= 0:
        return w, h
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return new_w, new_h


def get_resizing_factor(
    target_shape: tuple[int, int],  # H, W
    current_shape: tuple[int, int],  # H, W
    cover_target: bool = True,
    # If True, the output shape will fully cover the target shape.
    # If No, the target shape will fully cover the output shape.
) -> float:
    r_bound = target_shape[1] / target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def transform_img_and_K(
    image: torch.Tensor,
    size: int | tuple[int, int],
    scale: float = 1.0,
    center: tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | np.ndarray | None = None,
    size_stride: int = 1,
    mode: str = "crop",
):
    assert mode in [
        "crop",
        "pad",
        "stretch",
    ], f"mode should be one of ['crop', 'pad', 'stretch'], got {mode}"

    h, w = image.shape[-2:]
    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    if mode == "stretch":
        rh, rw = H, W
    else:
        rfs = get_resizing_factor(
            (H, W),
            (h, w),
            cover_target=mode != "pad",
        )
        (rh, rw) = [int(np.ceil(rfs * s)) for s in (h, w)]

    rh, rw = int(rh / scale), int(rw / scale)
    image = torch.nn.functional.interpolate(image, (rh, rw), mode="area", antialias=False)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if mode != "pad":
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)
        pl, pt = 0, 0
    else:
        pt = max(0, H // 2 - cy_center)
        pl = max(0, W // 2 - cx_center)
        pb = max(0, H - pt - image.shape[-2])
        pr = max(0, W - pl - image.shape[-1])
        image = TF.pad(
            image,
            [pl, pt, pr, pb],
        )
        cl, ct = 0, 0

    if K is not None:
        K = K.clone()
        # K[:, :2, 2] += K.new_tensor([pl, pt])
        if torch.all(K[:, :2, -1] >= 0) and torch.all(K[:, :2, -1] <= 1):
            K[:, :2] *= K.new_tensor([rw, rh])[None, :, None]  # normalized K
        else:
            K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]  # unnormalized K
        K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

    return image, K


def load_img_and_K(
    image_path_or_size: str | torch.Size,
    size: int | tuple[int, int] | None,
    scale: float = 1.0,
    center: tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | np.ndarray | None = None,
    size_stride: int = 1,
    center_crop: bool = False,
    image_as_tensor: bool = True,
    context_rgb: np.ndarray | None = None,
    device: str = "cuda",
):
    if isinstance(image_path_or_size, torch.Size):
        image = Image.new("RGBA", image_path_or_size[::-1])
    else:
        image = Image.open(image_path_or_size).convert("RGBA")

    w, h = image.size
    if size is None:
        size = (w, h)

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3:]
        image = rgb * alpha + context_rgb * (1 - alpha) if context_rgb is not None else rgb * alpha + (1 - alpha)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    rfs = get_resizing_factor((math.floor(H * scale), math.floor(W * scale)), (h, w))
    resize_size = rh, rw = [int(np.ceil(rfs * s)) for s in (h, w)]
    image = torch.nn.functional.interpolate(image, resize_size, mode="area", antialias=False)
    if scale < 1.0:
        pw = math.ceil((W - resize_size[1]) * 0.5)
        ph = math.ceil((H - resize_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if center_crop:
        side = min(H, W)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, image.shape[-2] - side)
        cl = min(cl, image.shape[-1] - side)
        image = TF.crop(image, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)

    if K is not None:
        K = K.clone()
        if torch.all(K[:2, -1] >= 0) and torch.all(K[:2, -1] <= 1):
            K[:2] *= K.new_tensor([rw, rh])[:, None]  # normalized K
        else:
            K[:2] *= K.new_tensor([rw / w, rh / h])[:, None]  # unnormalized K
        K[:2, 2] -= K.new_tensor([cl, ct])

    if image_as_tensor:
        # tensor of shape (1, 3, H, W) with values ranging from (-1, 1)
        image = image.to(device) * 2.0 - 1.0
    else:
        # PIL Image with values ranging from (0, 255)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
    return image, K


def geodesic_distance(
    extrinsic1: np.ndarray | torch.Tensor,
    extrinsic2: np.ndarray | torch.Tensor,
    weight_translation: float = 0.01,
):
    """
    Computes the geodesic distance between two camera poses in SE(3).

    Parameters:
        extrinsic1 (Union[np.ndarray, torch.Tensor]): 4x4 extrinsic matrix of the first pose.
        extrinsic2 (Union[np.ndarray, torch.Tensor]): 4x4 extrinsic matrix of the second pose.

    Returns:
        Union[float, torch.Tensor]: Geodesic distance between the two poses.
    """
    if torch.is_tensor(extrinsic1):
        # Extract the rotation and translation components
        R1 = extrinsic1[:3, :3]
        t1 = extrinsic1[:3, 3]
        R2 = extrinsic2[:3, :3]
        t2 = extrinsic2[:3, 3]

        # Compute the translation distance (Euclidean distance)
        translation_distance = torch.norm(t1 - t2)

        # Compute the relative rotation matrix
        R_relative = torch.matmul(R1.T, R2)

        # Compute the angular distance from the trace of the relative rotation matrix
        trace_value = torch.trace(R_relative)
        # Clamp the trace value to avoid numerical issues
        trace_value = torch.clamp(trace_value, -1.0, 3.0)
        angular_distance = torch.acos((trace_value - 1) / 2)

    else:
        # Extract the rotation and translation components
        R1 = extrinsic1[:3, :3]
        t1 = extrinsic1[:3, 3]
        R2 = extrinsic2[:3, :3]
        t2 = extrinsic2[:3, 3]

        # Compute the translation distance (Euclidean distance)
        translation_distance = np.linalg.norm(t1 - t2)

        # Compute the relative rotation matrix
        R_relative = np.dot(R1.T, R2)

        # Compute the angular distance from the trace of the relative rotation matrix
        trace_value = np.trace(R_relative)
        # Clamp the trace value to avoid numerical issues
        trace_value = np.clip(trace_value, -1.0, 3.0)
        angular_distance = np.arccos((trace_value - 1) / 2)

    # Combine the two distances
    geodesic_dist = translation_distance * weight_translation + angular_distance

    return geodesic_dist


def inverse_geodesic_distance(extrinsic1, extrinsic2, weight_translation=0.01):
    """
    Computes the inverse geodesic distance between two camera poses in SE(3).

    Parameters:
        extrinsic1 (np.ndarray): 4x4 extrinsic matrix of the first pose.
        extrinsic2 (np.ndarray): 4x4 extrinsic matrix of the second pose.

    Returns:
        float: Inverse geodesic distance between the two poses.
    """
    # Compute the geodesic distance
    geodesic_dist = geodesic_distance(extrinsic1, extrinsic2, weight_translation)

    # Compute the inverse geodesic distance
    inverse_geodesic_dist = 1.0 / (geodesic_dist + 1e-6)

    return inverse_geodesic_dist


def average_camera_pose(camera_poses):
    """
    Compute a better average of camera poses in SE(3).

    Args:
        camera_poses: List or array of camera poses, each a 4x4 matrix

    Returns:
        Average camera pose as a 4x4 matrix
    """
    # Extract rotation and translation components
    rotations = camera_poses[:, :3, :3].detach().cpu().numpy()
    translations = camera_poses[:, :3, 3].detach().cpu().numpy()

    # Average translation with simple mean
    avg_translation = np.mean(translations, axis=0)

    # Convert rotations to quaternions for better averaging
    import scipy.spatial.transform as transform

    quats = [transform.Rotation.from_matrix(R).as_quat() for R in rotations]

    # Ensure quaternions are in the same hemisphere to avoid issues with averaging
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]

    # Average the quaternions and convert back to rotation matrix
    avg_quat = np.mean(quats, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
    avg_rotation = transform.Rotation.from_quat(avg_quat).as_matrix()

    # Construct the average pose
    avg_pose = np.eye(4)
    avg_pose[:3, :3] = avg_rotation
    avg_pose[:3, 3] = avg_translation

    return avg_pose


def encode_image(
    image,
    image_encoder,
    device,
    dtype,
) -> torch.Tensor:
    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image)

    return image_embeddings


def encode_vae_image(
    image,
    vae,
    device,
    dtype,
):
    image = image.to(device=device, dtype=dtype)
    image_latents = vae.encode(image, 1)

    return image_latents


def do_sample(
    model,
    ae,
    denoiser,
    sampler,
    c,
    uc,
    c2w,
    K,
    cond_frames_mask,
    H=576,
    W=768,
    C=4,
    F=8,
    T=8,
    cfg=2.0,
    decoding_t=1,
    verbose=True,
    global_pbar=None,
    return_latents=False,
    device: str = "cuda",
    **_,
):
    num_samples = [1, T]
    with torch.inference_mode(), torch.autocast("cuda"):
        additional_model_inputs = {"num_frames": T}
        additional_sampler_inputs = {
            "c2w": c2w.to("cuda"),
            "K": K.to("cuda"),
            "input_frame_mask": cond_frames_mask.to("cuda"),
        }
        if global_pbar is not None:
            additional_sampler_inputs["global_pbar"] = global_pbar

        shape = (math.prod(num_samples), C, H // F, W // F)
        randn = torch.randn(shape).to(device)

        samples_z = sampler(
            lambda input, sigma, c: denoiser(
                model,
                input,
                sigma,
                c,
                **additional_model_inputs,
            ),
            randn,
            scale=cfg,
            cond=c,
            uc=uc,
            verbose=verbose,
            **additional_sampler_inputs,
        )
        if samples_z is None:
            return

        samples = ae.decode(samples_z, decoding_t)
    if return_latents:
        return samples, samples_z

    return samples


def decode_output(
    samples,
    T,
    indices=None,
):
    # decode model output into dict if it is not
    if isinstance(samples, dict):
        # model with postprocessor and outputs dict q``
        for sample, value in samples.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.tensor(value)

            if indices is not None and value.shape[0] == T:
                value = value[indices]
            samples[sample] = value
    else:
        # model without postprocessor and outputs tensor (rgb)
        samples = samples.detach().cpu()

        if indices is not None and samples.shape[0] == T:
            samples = samples[indices]
        samples = {"samples-rgb/image": samples}

    return samples


def select_frames(timestamps, min_num_frames=2, skip_frame=10, random_start=False):
    """
    Select frames from a video sequence based on defined criteria.

    Args:
        timestamps: List of timestamps for the frames
        min_num_frames: Minimum number of frames required
        skip_frame: Number of frames to skip between selections
        random_start: If True, start from a random frame

    Returns:
        tuple: (selected_frame_indices, selected_frame_timestamps) or (None, None) if criteria not met
    """

    num_frames = len(timestamps)
    if num_frames < min_num_frames:
        print(f"[Worker PID={os.getpid()}] Episode has less than {min_num_frames} frames")
        return None, None

    # Decide on start/end frames
    if num_frames < 2:
        print(f"[Worker PID={os.getpid()}] Episode has less than 2 frames")
        return None, None
    elif num_frames < skip_frame:
        cur_skip_frame = num_frames - 1
    else:
        cur_skip_frame = skip_frame

    start_frame = np.random.randint(0, skip_frame) if random_start else 0

    # Gather frame indices
    selected_frame_indices = list(range(start_frame, num_frames, cur_skip_frame))
    selected_frame_timestamps = [timestamps[i] for i in selected_frame_indices]

    return selected_frame_indices, selected_frame_timestamps


class LatentStorer:
    def __init__(self):
        self.latent = None

    def __call__(self, i, t, latent):
        self.latent = latent


def _write_video(path, video, fps, video_codec, video_options):
    try:
        from torchvision.io import write_video

        write_video(str(path), video, fps=fps, video_codec=video_codec, options=video_options)
        return
    except Exception:
        pass

    import imageio.v2 as imageio

    array = video.detach().cpu().numpy() if hasattr(video, "detach") else np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"save_video expects a 4D video array, got shape {array.shape!r}")
    if array.shape[-1] not in (1, 3, 4) and array.shape[1] in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))
    if np.issubdtype(array.dtype, np.floating):
        if array.size and array.min() >= -1.0 and array.max() <= 1.0:
            array = (array + 1.0) * 127.5 if array.min() < 0.0 else array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    output_params = []
    for key, value in video_options.items():
        output_params.extend([f"-{key}", str(value)])
    output_dir = os.path.dirname(str(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=float(fps),
        codec=video_codec or "libx264",
        macro_block_size=None,
        output_params=output_params or None,
    ) as writer:
        for frame in array:
            writer.append_data(frame)


def save_video(video, path, fps=10):
    video = video.permute(0, 2, 3, 1)
    video_codec = "libx264"
    video_options = {
        "crf": "23",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",
    }
    _write_video(path, video, fps, video_codec, video_options)


def tensor_to_pil(image):
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).detach().cpu().numpy()

        # Detect the range of the input tensor
        if image.min() < -0.1:  # If we have negative values, assume [-1, 1] range
            image = (image + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        # Otherwise, assume it's already in [0, 1] range

        image = image * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    return Image.fromarray(image)


def center_crop_pil_image(input_image, target_width=1024, target_height=576):
    w, h = input_image.size
    h_ratio = h / target_height
    w_ratio = w / target_width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < target_height:
            h = target_height
        input_image = input_image.resize((target_width, h), Image.Resampling.LANCZOS)
    else:
        w = int(w / h_ratio)
        if w < target_width:
            w = target_width
        input_image = input_image.resize((w, target_height), Image.Resampling.LANCZOS)

    return ImageOps.fit(input_image, (target_width, target_height), Image.BICUBIC)


def resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if long_edge_size < S:
        interp = PIL.Image.LANCZOS
    elif long_edge_size >= S:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    return img.resize(new_size, interp)


class Surfel:
    def __init__(self, position, normal, radius=1.0, color=None):
        """
        position: (x, y, z)
        normal:   (nx, ny, nz)
        radius:   scalar
        color:    (r, g, b) or None
        """
        self.position = position
        self.normal = normal
        self.radius = radius
        self.color = color

    def __repr__(self):
        return f"Surfel(position={self.position}, normal={self.normal}, radius={self.radius}, color={self.color})"


class Octree:
    def __init__(self, points, indices=None, bbox=None, max_points=10):
        self.points = points
        if indices is None:
            indices = np.arange(points.shape[0])
        self.indices = indices

        if bbox is None:
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            center = (min_bound + max_bound) / 2
            half_size = np.max(max_bound - min_bound) / 2
            bbox = (center, half_size)
        self.center, self.half_size = bbox

        self.children = []  # Child nodes
        self.max_points = max_points

        if len(self.indices) > self.max_points:
            self.subdivide()

    def subdivide(self):
        cx, cy, cz = self.center
        hs = self.half_size / 2

        offsets = np.array([[dx, dy, dz] for dx in (-hs, hs) for dy in (-hs, hs) for dz in (-hs, hs)])
        for offset in offsets:
            child_center = self.center + offset
            child_indices = []

            for idx in self.indices:
                p = self.points[idx]
                if np.all(np.abs(p - child_center) <= hs):
                    child_indices.append(idx)
            child_indices = np.array(child_indices)
            if len(child_indices) > 0:
                child = Octree(self.points, indices=child_indices, bbox=(child_center, hs), max_points=self.max_points)
                self.children.append(child)

        self.indices = None

    def sphere_intersects_node(self, center, r):
        diff = np.abs(center - self.center)
        max_diff = diff - self.half_size
        max_diff = np.maximum(max_diff, 0)
        dist_sq = np.sum(max_diff**2)
        return dist_sq <= r * r

    def query_ball_point(self, point, r):
        results = []
        if not self.sphere_intersects_node(point, r):
            return results

        if len(self.children) == 0:
            if self.indices is not None:
                for idx in self.indices:
                    if np.linalg.norm(self.points[idx] - point) <= r:
                        results.append(idx)
            return results
        else:
            for child in self.children:
                results.extend(child.query_ball_point(point, r))
            return results
