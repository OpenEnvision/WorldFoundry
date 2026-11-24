"""
Merge moving-object depth into a static scene point cloud.

Two-image mode:
Input: two images taken from the same camera pose that differ only in the depth of moving objects.
Output: the first-frame point cloud with the moving-object points of the second frame merged in.

Video mode:
Input: a video whose background is static and whose foreground objects move.
Output: the foreground points of every frame merged into the first-frame point cloud, rendered as a
video from an offset viewpoint.

Steps:
1. Estimate depth and intrinsics with STream3R.
2. Segment the foreground (moving objects) and background with SAM3.
3. Estimate the depth scale between frames on the background.
4. Merge the scale-aligned foreground points into the first-frame point cloud.
5. (Video mode) Render the accumulated point cloud from an offset viewpoint.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.models.stream3r import STream3R
from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.stream_session import StreamSession
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import (
    load_and_preprocess_images as load_and_preprocess_images,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# isort: split
from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import (
    Sam3VideoSegmenter,
)

DEFAULT_CONF_THRESHOLD = 0.10
MIN_CONF_THRESHOLD = 0.0
MAX_CONF_THRESHOLD = 0.60
MIN_SCALE_VALID_PIXELS = 100


def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_video_frames(video_path: str) -> list:
    """Load all video frames as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def get_video_fps(video_path: str) -> float:
    """Return the video frame rate."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _normalize_conf_threshold(conf_threshold: float, verbose: bool = True) -> float:
    """Validate confidence threshold for reconstruction."""
    if conf_threshold is None or not np.isfinite(conf_threshold):
        raise ValueError(f"Invalid conf_threshold={conf_threshold}")

    conf_threshold = float(conf_threshold)
    if conf_threshold < MIN_CONF_THRESHOLD or conf_threshold > MAX_CONF_THRESHOLD:
        raise ValueError(
            f"conf_threshold={conf_threshold:.3f} out of range [{MIN_CONF_THRESHOLD:.2f}, {MAX_CONF_THRESHOLD:.2f}]"
        )
    return conf_threshold


def load_stream3r_model(device: str = "cuda", model_path: str = "ckpts/yslan--STream3R"):
    """Load a STream3R model in eval mode on ``device``."""
    model = STream3R.from_pretrained(model_path)
    model = model.to(device=torch.device(device))
    model.eval()
    return model


def _prepare_stream3r_images_official(
    frames: list[np.ndarray],
    device: str = "cuda",
    mode: str = "crop",
) -> torch.Tensor:
    """Use the shared in-tree STream3R preprocessing path."""
    if not frames:
        raise ValueError("frames must be non-empty")

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i, frame in enumerate(frames):
            frame_np = np.asarray(frame)
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            path = Path(tmp_dir) / f"frame_{i:05d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            paths.append(str(path))
        images = load_and_preprocess_images(paths, mode=mode)
    return images.to(device=torch.device(device))


def estimate_depth_and_intrinsics(
    image: np.ndarray,
    model,
    device: str = "cuda",
    process_res: int = 518,
    conf_thresh_percentile: float = 40.0,
):
    """
    Estimate depth and intrinsics of a single image with STream3R, without a session.

    Args:
        image: RGB image [H, W, 3]
        model: Loaded STream3R model
        process_res: Ignored (with a warning) unless 518; the official preprocessing sets the size.
        conf_thresh_percentile: Unused.

    Returns:
        depth: [H, W] float32, Z-depth
        intrinsics: [3, 3] float32
        pts3d: None
        c2w: [4, 4] float32, camera to world transform
        conf: [H, W] float32, confidence map
    """

    # Keep process_res for API compatibility; official loader uses fixed target size.
    if process_res != 518:
        print(f"  [Stream3R] Warning: process_res={process_res} ignored; using official preprocessing")

    image_tensor = _prepare_stream3r_images_official([image], device=device, mode="crop")

    with torch.no_grad():
        predictions = model(image_tensor, mode="full")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_tensor.shape[-2:])

    c2w = _c2w_from_extrinsic(extrinsic[0, 0])
    intrinsics = intrinsic[0, 0].cpu().numpy().astype(np.float32)

    wp = predictions.get("world_points")
    if isinstance(wp, torch.Tensor):
        wp = wp.cpu().numpy()
    if wp is None:
        raise RuntimeError("[Stream3R] No world_points in predictions")
    if wp.ndim == 5:
        wp = wp[0, 0]
    elif wp.ndim == 4:
        wp = wp[0]

    conf = predictions.get("world_points_conf")
    if isinstance(conf, torch.Tensor):
        conf = conf.cpu().numpy()
    if conf is None:
        conf = np.ones(wp.shape[:2], dtype=np.float32)
    if conf.ndim == 4:
        conf = conf[0, 0]
    elif conf.ndim == 3:
        conf = conf[0]

    depth = _world_points_to_depth(wp, c2w)

    pts3d = None

    return depth, intrinsics, pts3d, c2w, conf.astype(np.float32)


def _c2w_from_extrinsic(extrinsic: torch.Tensor) -> np.ndarray:
    """Invert a [3, 4] world-to-camera extrinsic into a float32 [4, 4] camera-to-world matrix."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = extrinsic.cpu().numpy()
    return np.linalg.inv(w2c).astype(np.float32)


def _world_points_to_depth(world_points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Return the float32 camera Z-depth [H, W] of world points [H, W, 3] seen from ``c2w``."""
    proc_h, proc_w = world_points.shape[:2]
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ world_points.reshape(-1, 3).T).T + w2c[:3, 3]
    return pts_cam[:, 2].reshape(proc_h, proc_w).astype(np.float32)


def _extract_depth_from_preds(preds: dict, frame_count: int):
    """Extract depth, intrinsics, conf for the latest frame from session predictions.

    Args:
        preds: Dict of numpy predictions (batch dim already squeezed).
        frame_count: Total frames fed so far (1-indexed).

    Returns:
        Same as estimate_depth_and_intrinsics: (depth, intrinsics, None, c2w, conf)
    """

    # Extract current (last) frame predictions from accumulated outputs.
    cur = {}
    for key, val in preds.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == frame_count:
            cur[key] = val[-1:]
        else:
            cur[key] = val

    # pose_enc is a torch tensor in raw predictions; it was already converted.
    # We need the raw tensor for pose_encoding_to_extri_intri. Store it separately.
    pose_enc = cur.get("_pose_enc_tensor")
    if pose_enc is not None:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, cur["_image_shape"])
        c2w = _c2w_from_extrinsic(extrinsic[0, -1])  # last frame
        intrinsics = intrinsic[0, -1].cpu().numpy().astype(np.float32)
    else:
        c2w = np.eye(4, dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)

    wp = cur.get("world_points")
    if wp is None:
        raise RuntimeError("[Stream3R] No world_points in predictions")
    if wp.ndim == 4:
        wp = wp[-1:]
    if wp.ndim == 3:
        wp = wp[np.newaxis]

    conf = cur.get("world_points_conf", np.ones(wp.shape[:-1], dtype=np.float32))
    if conf.ndim >= 2 and conf.shape[0] > 1:
        conf = conf[-1:]
    if conf.ndim == 3:
        conf = conf[0]
    elif conf.ndim == 4:
        conf = conf[0, 0]

    depth = _world_points_to_depth(wp[0], c2w)

    return depth, intrinsics, None, c2w, conf.astype(np.float32)


def estimate_all_depths_with_session(
    frames: list,
    model,
    device: str = "cuda",
    process_res: int = 518,
    verbose: bool = False,
    session=None,
    initial_frames_fed: int = 0,
):
    """Estimate depth for all frames using a StreamSession for temporal consistency.

    If ``session`` is provided, frames are fed into that shared session
    (no creation / clearing). Otherwise a local session is created and
    cleared after use.

    Args:
        frames: List of RGB images [H, W, 3] uint8.
        model: Pre-loaded STream3R model (used only if session is None).
        device: Compute device string.
        process_res: Stream3R processing resolution.
        verbose: Print per-frame info.
        session: Optional pre-existing StreamSession to reuse.
        initial_frames_fed: How many frames the session has already consumed
            (needed to correctly index the accumulated predictions).

    Returns:
        Tuple of (results, final_frames_fed):
        - results: list of (depth, intrinsics, c2w, conf) per frame.
        - final_frames_fed: total frames fed after this call.
    """
    _log = print if verbose else (lambda *a, **k: None)

    # Keep process_res for API compatibility; official loader uses fixed target size.
    if process_res != 518:
        _log(f"  [Stream3R] Warning: process_res={process_res} ignored; using official preprocessing")

    # Preprocess all frames with official Stream3R loader.
    images = _prepare_stream3r_images_official(frames, device=device, mode="crop")

    owns_session = session is None
    if owns_session:
        session = StreamSession(model, mode="causal")
        initial_frames_fed = 0
        _log("  [Session] Created local session")
    else:
        _log(f"  [Session] Reusing shared session (already fed {initial_frames_fed} frames)")

    results = []
    frames_fed = initial_frames_fed

    with torch.no_grad():
        for i in range(images.shape[0]):
            image = images[i : i + 1]
            predictions = session.forward_stream(image)
            frames_fed += 1

            # Convert tensors to numpy, keeping pose_enc as tensor for later.
            preds = {}
            pose_enc_tensor = None
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    if key == "pose_enc":
                        pose_enc_tensor = val
                    preds[key] = val.cpu().numpy().squeeze(0)
                else:
                    preds[key] = val

            if pose_enc_tensor is not None:
                preds["_pose_enc_tensor"] = pose_enc_tensor
                preds["_image_shape"] = image.shape[-2:]

            depth, intrinsics, _, c2w, conf = _extract_depth_from_preds(preds, frames_fed)
            results.append((depth, intrinsics, c2w, conf))
            _log(f"  [Session] Frame {i} (global #{frames_fed}): depth {depth.shape}")

    if owns_session:
        session.clear()
        _log(f"  [Session] Local session cleared after {len(results)} frames")
    else:
        _log(f"  [Session] Shared session now at {frames_fed} total frames")

    return results, frames_fed


def load_sam3_segmenter(sam3_model_path: str):
    """Create a SAM3 video segmenter."""
    segmenter = Sam3VideoSegmenter(
        checkpoint_path=sam3_model_path,
    )
    return segmenter


def segment_with_sam3(
    image: np.ndarray,
    prompts: list,
    segmenter,
) -> np.ndarray:
    """
    Segment the prompted objects with SAM3.

    Args:
        image: RGB image [H, W, 3]
        prompts: Object names to segment, for example ["car", "person"]
        segmenter: Loaded SAM3 segmenter

    Returns:
        mask: [H, W] bool, True on segmented objects; all False when nothing is found or segmentation fails.
    """
    if not prompts:
        return np.zeros(image.shape[:2], dtype=bool)

    # SAM3 takes a list of PIL images.
    pil_image = Image.fromarray(image)

    try:
        masks = segmenter.segment(
            video_path=[pil_image],
            prompts=prompts,
            frame_index=0,
            expected_frames=1,
        )

        if masks.size == 0 or not masks.any():  # noqa: SIM108 - Branches carry per-case comments.
            mask = np.zeros(image.shape[:2], dtype=bool)
        else:
            mask = masks[0]  # [H, W]
    except Exception as e:
        print(f"  SAM3 segmentation error: {e}")
        mask = np.zeros(image.shape[:2], dtype=bool)

    return mask


def compute_scale_factor(
    depth1: np.ndarray,
    depth2: np.ndarray,
    background_mask: np.ndarray,
    conf1: np.ndarray = None,
    conf2: np.ndarray = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_valid_pixels: int = MIN_SCALE_VALID_PIXELS,
    verbose: bool = True,
) -> float:
    """
    Estimate the depth scale between two frames on the background.

    Depth is estimated per image, so the scales can differ; the static background
    gives the alignment factor.

    Args:
        depth1: Depth of the first image [H, W]
        depth2: Depth of the second image [H, W]
        background_mask: Background mask [H, W], True on background
        conf1: Confidence of the first image [H, W] (optional)
        conf2: Confidence of the second image [H, W] (optional)
        conf_threshold: Pixels below this confidence are excluded from the estimate
        min_valid_pixels: Minimum number of valid pixels

    Returns:
        scale: depth2 * scale ~= depth1 on the background
    """
    _log = print if verbose else (lambda *a, **k: None)
    conf_threshold = _normalize_conf_threshold(conf_threshold, verbose=verbose)

    # Valid background pixels before the confidence filter.
    base_valid = background_mask & (depth1 > 0.1) & (depth2 > 0.1) & np.isfinite(depth1) & np.isfinite(depth2)
    base_valid_pixels = int(base_valid.sum())
    if base_valid_pixels < min_valid_pixels:
        raise RuntimeError(
            f"Scale estimation failed: too few valid background pixels ({base_valid_pixels} < {min_valid_pixels})"
        )

    if (conf1 is None) != (conf2 is None):
        raise ValueError("conf1 and conf2 must be both provided or both None")

    use_conf = conf1 is not None and conf2 is not None
    if use_conf and (conf1.shape != depth1.shape or conf2.shape != depth2.shape):
        raise ValueError(
            "confidence shape mismatch: "
            f"conf1={None if conf1 is None else conf1.shape}, "
            f"conf2={None if conf2 is None else conf2.shape}, "
            f"depth1={depth1.shape}, depth2={depth2.shape}"
        )

    if use_conf:
        valid_mask = base_valid & (conf1 > conf_threshold) & (conf2 > conf_threshold)
        label = f"conf>{conf_threshold:.2f}"
    else:
        valid_mask = base_valid
        label = "no_conf"

    num_valid = int(valid_mask.sum())
    if num_valid < min_valid_pixels:
        raise RuntimeError(
            f"Scale estimation failed: too few pixels after filtering [{label}] ({num_valid} < {min_valid_pixels})"
        )

    d1 = depth1[valid_mask]
    d2 = depth2[valid_mask]
    ratios = d1 / d2
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < min_valid_pixels:
        raise RuntimeError(
            f"Scale estimation failed: too few finite ratios [{label}] ({ratios.size} < {min_valid_pixels})"
        )

    # Robustify: remove ratio tails to reduce reflective/glass outliers.
    p5, p95 = np.percentile(ratios, [5, 95])
    robust = ratios[(ratios >= p5) & (ratios <= p95)]
    ratios_used = robust if robust.size >= min_valid_pixels else ratios

    scale = float(np.median(ratios_used))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Scale estimation produced invalid scale={scale}")

    std = float(np.std(ratios_used))
    _log(f"  Background scale [{label}]: scale={scale:.4f}, std={std:.4f}, num_pixels={ratios_used.size}")
    return scale


def unproject_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray = None,
    conf: np.ndarray = None,
    conf_threshold: float = 0.1,
    filter_depth_outliers: bool = False,
    depth_outlier_std_scale: float = 3.0,
) -> tuple:
    """
    Unproject a depth map to a point cloud in camera coordinates.

    Args:
        depth: [H, W] Z-depth
        rgb: [H, W, 3] RGB colors
        intrinsics: [3, 3] camera intrinsics
        mask: [H, W] bool, keep only pixels where mask is True (optional)
        conf: [H, W] confidence map (optional)
        conf_threshold: Points at or below this confidence are dropped (default: 0.1)
        filter_depth_outliers: Drop depth outliers inside ``mask`` (default: False)
        depth_outlier_std_scale: Keep depths within mean +/- std_scale * std (default: 3.0)

    Returns:
        points: [N, 3] points in camera coordinates
        colors: [N, 3] point colors
    """
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)  # [H, W, 3]

    valid = (z > 0.1) & np.isfinite(z)
    if mask is not None:
        valid = valid & mask
    if conf is not None:
        valid = valid & (conf > conf_threshold)

    # Depth outlier filtering over the masked points; needs more than 10 points for statistics.
    if filter_depth_outliers and mask is not None:
        valid_depths = z[valid]
        if len(valid_depths) > 10:
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)
            lower_bound = mean_depth - depth_outlier_std_scale * std_depth
            upper_bound = mean_depth + depth_outlier_std_scale * std_depth
            depth_inlier = (z >= lower_bound) & (z <= upper_bound)
            valid = valid & depth_inlier

    points_valid = points[valid]  # [N, 3]
    colors_valid = rgb[valid]  # [N, 3]

    return points_valid, colors_valid


def _resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to ``size_wh`` (W, H) with nearest-neighbor sampling."""
    return cv2.resize(mask.astype(np.uint8), size_wh, interpolation=cv2.INTER_NEAREST).astype(bool)


def _union_instance_masks(instance_masks: dict, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    """Union the frame ``frame_idx`` masks of all instances ({obj_id: (T, H, W) bool}) into one (H, W) mask."""
    mask = np.zeros(shape, dtype=bool)
    for inst_masks in instance_masks.values():
        mask |= inst_masks[frame_idx]
    return mask


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Save a point cloud as an ASCII PLY file."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b) in zip(points, colors, strict=False):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_depth_visualization(depth: np.ndarray, path: str):
    """Save a VIRIDIS depth visualization; invalid pixels are black and an all-invalid map writes nothing."""
    valid = (depth > 0) & np.isfinite(depth)
    if not valid.any():
        return

    d_min, d_max = depth[valid].min(), depth[valid].max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    depth_vis[~valid] = 0
    cv2.imwrite(path, depth_vis)


def render_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple,
    camera_pose: np.ndarray = None,
) -> np.ndarray:
    """
    Render a point cloud to an image with a per-pixel z-buffer.

    Args:
        points: [N, 3] points in world coordinates
        colors: [N, 3] RGB colors
        intrinsics: [3, 3] camera intrinsics
        image_size: (H, W)
        camera_pose: [4, 4] camera-to-world transform; None means the points are already in camera coordinates

    Returns:
        image: [H, W, 3] rendered image
    """
    H, W = image_size

    if camera_pose is not None:
        # world-to-camera
        w2c = np.linalg.inv(camera_pose)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        points_cam = (R @ points.T).T + t
    else:
        points_cam = points

    # Drop points behind the camera.
    valid_depth = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_depth]
    colors_valid = colors[valid_depth]

    if len(points_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Project to the image plane.
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # Drop points outside the image.
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]

    if len(u) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    depth_buffer = np.full((H, W), np.inf)
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # Sort far to near.
    sort_idx = np.argsort(-z)
    u = u[sort_idx]
    v = v[sort_idx]
    z = z[sort_idx]
    colors_valid = colors_valid[sort_idx]

    # Nearer points overwrite farther ones.
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            image[v[i], u[i]] = colors_valid[i]

    return image


def update_dynamic_object_depth(
    image1_path: str,
    image2_path: str,
    dynamic_object_prompts: list,
    output_dir: str,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    device: str = "cuda",
):
    """
    Merge the moving-object points of the second image into the first-image point cloud.

    Args:
        image1_path: First image (original frame)
        image2_path: Second image (after the objects moved)
        dynamic_object_prompts: Names of the moving objects, for example ["car", "person"]
        output_dir: Output directory
        sam3_model_path: SAM3 checkpoint path
        device: Compute device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dynamic Object Depth Update")
    print("=" * 60)
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Dynamic objects: {dynamic_object_prompts}")
    print(f"Output: {output_dir}")
    print()

    # ========== Step 1: Load images and models ==========
    print("[Step 1/6] Loading images and models...")
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    assert image1.shape == image2.shape, f"Image shapes must match: {image1.shape} vs {image2.shape}"
    H, W = image1.shape[:2]
    print(f"  Image size: {W}x{H}")

    # Save the inputs.
    cv2.imwrite(str(output_dir / "input_image1.png"), cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "input_image2.png"), cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))

    print("  Loading STream3R model...")
    stream3r_model = load_stream3r_model(device)

    print("  Loading SAM3 segmenter...")
    sam3_segmenter = load_sam3_segmenter(sam3_model_path)

    # ========== Step 2: Estimate depth ==========
    print("\n[Step 2/6] Estimating depth with STream3R...")

    print("  Processing image 1...")
    depth1, intrinsics1, pts3d_1, c2w_1, conf1 = estimate_depth_and_intrinsics(image1, stream3r_model, device)
    print(f"    Depth range: [{depth1[depth1 > 0].min():.2f}, {depth1.max():.2f}]")

    print("  Processing image 2...")
    depth2, intrinsics2, pts3d_2, c2w_2, conf2 = estimate_depth_and_intrinsics(image2, stream3r_model, device)
    print(f"    Depth range: [{depth2[depth2 > 0].min():.2f}, {depth2.max():.2f}]")

    save_depth_visualization(depth1, str(output_dir / "depth1.png"))
    save_depth_visualization(depth2, str(output_dir / "depth2.png"))

    # Release STream3R.
    del stream3r_model
    torch.cuda.empty_cache()

    # ========== Step 3: Segment the sky ==========
    print("\n[Step 3/6] Segmenting sky with SAM3...")
    sky_mask1 = segment_with_sam3(image1, ["sky"], sam3_segmenter)
    sky_mask2 = segment_with_sam3(image2, ["sky"], sam3_segmenter)
    print(f"  Sky pixels in image 1: {sky_mask1.sum()} ({sky_mask1.sum() / sky_mask1.size * 100:.1f}%)")
    print(f"  Sky pixels in image 2: {sky_mask2.sum()} ({sky_mask2.sum() / sky_mask2.size * 100:.1f}%)")

    # ========== Step 4: Segment moving objects ==========
    print("\n[Step 4/6] Segmenting dynamic objects with SAM3...")
    foreground_mask1 = segment_with_sam3(image1, dynamic_object_prompts, sam3_segmenter)
    foreground_mask2 = segment_with_sam3(image2, dynamic_object_prompts, sam3_segmenter)
    print(
        f"  Foreground pixels in image 1: {foreground_mask1.sum()} ({foreground_mask1.sum() / foreground_mask1.size * 100:.1f}%)"
    )
    print(
        f"  Foreground pixels in image 2: {foreground_mask2.sum()} ({foreground_mask2.sum() / foreground_mask2.size * 100:.1f}%)"
    )

    # Release SAM3.
    del sam3_segmenter
    torch.cuda.empty_cache()

    def save_mask_vis(image, mask, path, color=[255, 0, 0]):  # noqa: B006 - Keeps the upstream default object and signature.
        vis = image.copy()
        overlay = image.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    save_mask_vis(image1, foreground_mask1, str(output_dir / "foreground_mask1.png"))
    save_mask_vis(image2, foreground_mask2, str(output_dir / "foreground_mask2.png"))
    save_mask_vis(image1, sky_mask1, str(output_dir / "sky_mask1.png"), color=[0, 0, 255])
    save_mask_vis(image2, sky_mask2, str(output_dir / "sky_mask2.png"), color=[0, 0, 255])

    # Background = neither foreground nor sky.
    background_mask1 = (~foreground_mask1) & (~sky_mask1)
    background_mask2 = (~foreground_mask2) & (~sky_mask2)

    # ========== Step 4.5: Resize masks to the depth resolution ==========
    # The depth map can be smaller than the input image.
    depth_H1, depth_W1 = depth1.shape
    depth_H2, depth_W2 = depth2.shape

    if (depth_H1, depth_W1) != (H, W):
        print(f"\n[Step 4.5] Resizing masks to match depth size ({depth_W1}x{depth_H1})...")
        sky_mask1 = _resize_mask(sky_mask1, (depth_W1, depth_H1))
        foreground_mask1 = _resize_mask(foreground_mask1, (depth_W1, depth_H1))
        background_mask1 = _resize_mask(background_mask1, (depth_W1, depth_H1))
        # Point colors come from the resized image.
        image1_resized = cv2.resize(image1, (depth_W1, depth_H1), interpolation=cv2.INTER_LINEAR)
    else:
        image1_resized = image1

    if (depth_H2, depth_W2) != (H, W):
        sky_mask2 = _resize_mask(sky_mask2, (depth_W2, depth_H2))
        foreground_mask2 = _resize_mask(foreground_mask2, (depth_W2, depth_H2))
        background_mask2 = _resize_mask(background_mask2, (depth_W2, depth_H2))
        image2_resized = cv2.resize(image2, (depth_W2, depth_H2), interpolation=cv2.INTER_LINEAR)
    else:
        image2_resized = image2

    # ========== Step 5: Estimate the depth scale ==========
    print("\n[Step 5/6] Computing depth scale factor from background...")

    # Use the background shared by both images.
    common_background = background_mask1 & background_mask2
    scale = compute_scale_factor(depth1, depth2, common_background)

    # Align the second depth map to the first.
    depth2_scaled = depth2 * scale
    print(f"  Scaled depth2 range: [{depth2_scaled[depth2_scaled > 0].min():.2f}, {depth2_scaled.max():.2f}]")

    # ========== Step 6: Build and merge point clouds ==========
    print("\n[Step 6/6] Building and merging point clouds...")

    # The first-image background (without sky and foreground) is the base cloud.
    points_bg1, colors_bg1 = unproject_to_pointcloud(depth1, image1_resized, intrinsics1, mask=background_mask1)
    print(f"  Image 1 background points: {len(points_bg1)}")

    points_fg1, colors_fg1 = unproject_to_pointcloud(depth1, image1_resized, intrinsics1, mask=foreground_mask1)
    print(f"  Image 1 foreground points: {len(points_fg1)}")

    # Second-image foreground from the scaled depth; both images share the camera, so intrinsics1 is used.
    points_fg2, colors_fg2 = unproject_to_pointcloud(
        depth2_scaled,
        image2_resized,
        intrinsics1,
        mask=foreground_mask2,
    )
    print(f"  Image 2 foreground points (scaled): {len(points_fg2)}")

    # First-image background + second-image foreground places the moved objects in the first scene.
    merged_points = np.concatenate([points_bg1, points_fg2], axis=0)
    merged_colors = np.concatenate([colors_bg1, colors_fg2], axis=0)
    print(f"  Merged point cloud: {len(merged_points)} points")

    # Full first-image cloud (background + original foreground) for comparison.
    complete_points1 = np.concatenate([points_bg1, points_fg1], axis=0)
    complete_colors1 = np.concatenate([colors_bg1, colors_fg1], axis=0)

    # ========== Save results ==========
    print("\n[Saving results...]")

    save_pointcloud_ply(str(output_dir / "pointcloud_image1_complete.ply"), complete_points1, complete_colors1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image1_background.ply"), points_bg1, colors_bg1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image1_foreground.ply"), points_fg1, colors_fg1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image2_foreground_scaled.ply"), points_fg2, colors_fg2)
    save_pointcloud_ply(str(output_dir / "pointcloud_merged.ply"), merged_points, merged_colors)

    metadata = {
        "image1_path": str(image1_path),
        "image2_path": str(image2_path),
        "dynamic_object_prompts": dynamic_object_prompts,
        "scale_factor": float(scale),
        "image_size": [W, H],
        "depth1_range": [float(depth1[depth1 > 0].min()), float(depth1.max())],
        "depth2_range": [float(depth2[depth2 > 0].min()), float(depth2.max())],
        "depth2_scaled_range": [float(depth2_scaled[depth2_scaled > 0].min()), float(depth2_scaled.max())],
        "num_points": {
            "image1_background": len(points_bg1),
            "image1_foreground": len(points_fg1),
            "image2_foreground_scaled": len(points_fg2),
            "merged": len(merged_points),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.savez(
        output_dir / "depth_data.npz",
        depth1=depth1,
        depth2=depth2,
        depth2_scaled=depth2_scaled,
        intrinsics1=intrinsics1,
        intrinsics2=intrinsics2,
        c2w_1=c2w_1,
        c2w_2=c2w_2,
        foreground_mask1=foreground_mask1,
        foreground_mask2=foreground_mask2,
        background_mask1=background_mask1,
        background_mask2=background_mask2,
        sky_mask1=sky_mask1,
        sky_mask2=sky_mask2,
        scale_factor=scale,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("  - pointcloud_image1_complete.ply: full first-image point cloud")
    print("  - pointcloud_image1_background.ply: first-image background point cloud")
    print("  - pointcloud_image1_foreground.ply: first-image foreground point cloud")
    print("  - pointcloud_image2_foreground_scaled.ply: second-image foreground point cloud (scaled)")
    print("  - pointcloud_merged.ply: merged point cloud (first-image background + second-image foreground)")
    print("  - depth1.png, depth2.png: depth visualizations")
    print("  - foreground_mask1.png, foreground_mask2.png: foreground segmentation visualizations")
    print("  - metadata.json: metadata")
    print("  - depth_data.npz: depth data")
    print(f"\nScale factor: {scale:.4f}")
    print(f"Merged points: {len(merged_points)}")

    return {
        "merged_points": merged_points,
        "merged_colors": merged_colors,
        "scale_factor": scale,
        "metadata": metadata,
    }


def merge_video_foregrounds(
    video_path: str,
    dynamic_object_prompts: list,
    output_dir: str,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    rotation_angle: float = 45.0,
    depth_outlier_std_scale: float = 3.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: str = "cuda",
    verbose: bool = True,
    stream3r_model=None,
    sam3_segmenter=None,
    stream3r_session=None,
    stream3r_frames_fed: int = 0,
    video_frames=None,
    video_fps: float | None = None,
    stride: int = 1,
    save_depth_maps: bool = True,
    save_rendered_frames: bool = True,
    save_rendered_video: bool = True,
    save_final_merged_pointcloud: bool = True,
    fg_mask_erode: int = 0,
    skip_background: bool = False,
):
    """
    Video mode: merge the foreground objects of every frame into the first-frame point cloud and render
    it from an offset viewpoint.

    Args:
        video_path: Input video path (required when video_frames is None)
        video_frames: Optional RGB frames that replace reading the video file
        video_fps: Optional frame rate of video_frames
        dynamic_object_prompts: Foreground object names
        output_dir: Output directory
        sam3_model_path: SAM3 checkpoint path
        rotation_angle: Camera offset angle in degrees
        depth_outlier_std_scale: Standard-deviation multiplier for depth outlier filtering (default: 3.0)
        conf_threshold: Confidence threshold; must lie in [MIN_CONF_THRESHOLD, MAX_CONF_THRESHOLD]
        device: Compute device
        save_depth_maps: Save per-frame depth visualizations
        save_rendered_frames: Save rendered frames as PNG files
        save_rendered_video: Save rendered_merged.mp4
        save_final_merged_pointcloud: Save final_merged_pointcloud.ply
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log = print if verbose else (lambda *a, **k: None)

    _log("=" * 60)
    _log("Merge Video Foregrounds")
    _log("=" * 60)
    _log(f"Dynamic objects: {dynamic_object_prompts}")
    _log(f"Rotation angle: {rotation_angle}°")
    _log(f"Output: {output_dir}")
    _log()

    effective_conf_threshold = _normalize_conf_threshold(conf_threshold, verbose=verbose)
    _log(f"  Effective confidence threshold: {effective_conf_threshold:.3f}")
    if fg_mask_erode > 0:
        _log("  Note: fg_mask_erode is ignored; foreground points are kept without erosion.")

    # ========== Step 1: Load frames and models ==========
    _log("[Step 1] Loading video and models...")
    if video_frames is not None:
        frames = list(video_frames)
        if len(frames) == 0:
            raise ValueError("video_frames is empty")
        fps = float(video_fps) if video_fps is not None and np.isfinite(video_fps) and float(video_fps) > 0 else 16.0
        source_desc = "in-memory frames"
    else:
        if not video_path:
            raise ValueError("video_path must be provided when video_frames is None")
        frames = load_video_frames(video_path)
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        fps = get_video_fps(video_path)
        source_desc = str(video_path)

    H, W = frames[0].shape[:2]

    # Subsample frames if stride > 1 (reduces STream3R + SAM3 cost).
    # Always keep the first and last frame.
    stride = max(1, int(stride))
    if stride > 1:
        n_orig = len(frames)
        indices = list(range(0, n_orig, stride))
        if indices[-1] != n_orig - 1:
            indices.append(n_orig - 1)
        frames = [frames[i] for i in indices]
        _log(f"  Stride={stride}: subsampled {len(frames)}/{n_orig} frames (first+last guaranteed)")

    num_frames = len(frames)
    _log(f"  Source: {source_desc}")
    _log(f"  Frames: {num_frames}, FPS: {fps:.2f}, Size: {W}x{H}")

    _owns_stream3r = stream3r_model is None
    if _owns_stream3r:
        _log("  Loading STream3R model...")
        stream3r_model = load_stream3r_model(device)
    else:
        _log("  Reusing shared STream3R model")

    _owns_sam3 = sam3_segmenter is None
    if _owns_sam3:
        _log("  Loading SAM3 segmenter...")
        sam3_segmenter = load_sam3_segmenter(sam3_model_path)
    else:
        _log("  Reusing shared SAM3 segmenter")

    # ========== Step 2: Estimate depth for all frames with a StreamSession ==========
    _log("\n[Step 2] Estimating depth for all frames (session mode)...")

    depth_results, stream3r_frames_fed = estimate_all_depths_with_session(
        frames,
        stream3r_model,
        device=device,
        verbose=verbose,
        session=stream3r_session,
        initial_frames_fed=stream3r_frames_fed,
    )
    _log(f"  Got depth for {len(depth_results)} frames via session (total fed: {stream3r_frames_fed})")

    # ========== Step 3: Per-frame segmentation and point extraction ==========
    _log("\n[Step 3] Segmenting and extracting point clouds...")

    depths_dir = output_dir / "depths"
    if save_depth_maps:
        depths_dir.mkdir(exist_ok=True)

    # -- Per-instance temporal segmentation --
    # Call segment_instances_video() once on the full video to get per-instance
    # masks across all frames, enabling per-instance point cloud extraction.
    instance_masks_dict = {}  # {obj_id: (T, H, W) bool}
    _has_instance_seg = hasattr(sam3_segmenter, "segment_instances_video")
    if not _has_instance_seg:
        # LiveWorld segmenter: use segment_instances() which has compatible return type.
        _has_instance_seg = hasattr(sam3_segmenter, "segment_instances")
    if _has_instance_seg and dynamic_object_prompts:
        _log("  Running per-instance temporal segmentation...")
        pil_frames = [Image.fromarray(f) for f in frames]
        if hasattr(sam3_segmenter, "segment_instances_video"):
            instance_masks_dict = sam3_segmenter.segment_instances_video(
                video_path=pil_frames,
                prompts=dynamic_object_prompts,
                frame_index=0,
                expected_frames=num_frames,
            )
        else:
            instance_masks_dict = sam3_segmenter.segment_instances(
                video_path=pil_frames,
                prompts=dynamic_object_prompts,
                frame_index=0,
                expected_frames=num_frames,
            )
        _log(f"  Per-instance segmentation: {len(instance_masks_dict)} instances detected")
        for obj_id, masks_arr in instance_masks_dict.items():
            _log(f"    obj_{obj_id}: shape={masks_arr.shape}, pixels_frame0={masks_arr[0].sum()}")

    # -- Rescale depth maps to video resolution and store per frame --
    depth1_raw, intrinsics1_raw, _, conf1_raw = depth_results[0]
    depth_H_raw, depth_W_raw = depth1_raw.shape
    _log(f"  Raw depth size: {depth_W_raw}x{depth_H_raw}")

    # Rescale depth, confidence and intrinsics to the video resolution.
    if (depth_H_raw, depth_W_raw) != (H, W):
        _log(f"  Rescaling depth and intrinsics to original size ({W}x{H})...")
        depth1 = cv2.resize(depth1_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        conf1 = cv2.resize(conf1_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        scale_x = W / depth_W_raw
        scale_y = H / depth_H_raw
        intrinsics1 = intrinsics1_raw.copy()
        intrinsics1[0, 0] *= scale_x
        intrinsics1[1, 1] *= scale_y
        intrinsics1[0, 2] *= scale_x
        intrinsics1[1, 2] *= scale_y
    else:
        depth1 = depth1_raw
        intrinsics1 = intrinsics1_raw
        conf1 = conf1_raw

    # Store per-frame depth for per-instance point cloud extraction later.
    depth_maps = [depth1]

    if save_depth_maps:
        save_depth_visualization(depth1, str(depths_dir / "depth_0000.png"))

    # Build foreground mask from per-instance masks or fallback to per-frame SAM3.
    if instance_masks_dict:
        foreground_mask1 = _union_instance_masks(instance_masks_dict, 0, (H, W))
        _log(f"  Foreground mask (from {len(instance_masks_dict)} instances): {foreground_mask1.sum()} pixels")
    else:
        _log("  Segmenting foreground...")
        foreground_mask1 = segment_with_sam3(frames[0], dynamic_object_prompts, sam3_segmenter)

    # Background point cloud (only needed for standalone rendering/merged ply).
    points_bg1 = None
    colors_bg1 = None
    if not skip_background:
        _log("  Segmenting sky...")
        sky_mask1 = segment_with_sam3(frames[0], ["sky"], sam3_segmenter)
        background_mask1 = (~foreground_mask1) & (~sky_mask1)
        points_bg1, colors_bg1 = unproject_to_pointcloud(
            depth1, frames[0], intrinsics1, mask=background_mask1, conf=conf1, conf_threshold=effective_conf_threshold
        )
        _log(f"  Background points: {len(points_bg1)}")

    points_fg1, colors_fg1 = unproject_to_pointcloud(
        # Foreground points keep all valid masked depths.
        # No confidence filter, no fg-mask erosion.
        depth1,
        frames[0],
        intrinsics1,
        mask=foreground_mask1,
        conf=None,
        conf_threshold=effective_conf_threshold,
        filter_depth_outliers=True,
        depth_outlier_std_scale=depth_outlier_std_scale,
    )
    _log(f"  Foreground points: {len(points_fg1)}")

    # -- Remaining frames --
    all_foreground_points = [points_fg1]
    all_foreground_colors = [colors_fg1]

    for frame_idx in tqdm(range(1, num_frames), desc="Processing frames", disable=not verbose):
        frame = frames[frame_idx]
        depth_i_raw, _, _, _ = depth_results[frame_idx]
        depth_H_i_raw, depth_W_i_raw = depth_i_raw.shape

        if (depth_H_i_raw, depth_W_i_raw) != (H, W):
            depth_i = cv2.resize(depth_i_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            depth_i = depth_i_raw

        depth_maps.append(depth_i)

        if save_depth_maps:
            save_depth_visualization(depth_i, str(depths_dir / f"depth_{frame_idx:04d}.png"))

        # Build fg mask from per-instance masks or fallback to per-frame SAM3.
        if instance_masks_dict:
            fg_mask_i = _union_instance_masks(instance_masks_dict, frame_idx, (H, W))
        else:
            fg_mask_i = segment_with_sam3(frame, dynamic_object_prompts, sam3_segmenter)

        depth_i_scaled = depth_i * 1

        points_fg_i, colors_fg_i = unproject_to_pointcloud(
            # Foreground points keep all valid masked depths.
            # No confidence filter, no fg-mask erosion.
            depth_i_scaled,
            frame,
            intrinsics1,
            mask=fg_mask_i,
            conf=None,
            conf_threshold=effective_conf_threshold,
            filter_depth_outliers=True,
            depth_outlier_std_scale=depth_outlier_std_scale,
        )

        all_foreground_points.append(points_fg_i)
        all_foreground_colors.append(colors_fg_i)

    _log(f"  Total foreground point sets: {len(all_foreground_points)}")

    # -- Build per-instance per-frame point clouds --
    per_instance_per_frame_points = {}
    per_instance_per_frame_colors = {}
    per_instance_anchor_masks = {}
    if instance_masks_dict:
        _log("  Building per-instance point clouds...")
        for obj_id, inst_masks_TxHxW in instance_masks_dict.items():
            inst_pts_list = []
            inst_cols_list = []
            for t in range(num_frames):
                inst_mask_t = inst_masks_TxHxW[t]
                if not inst_mask_t.any():
                    inst_pts_list.append(np.zeros((0, 3), dtype=np.float32))
                    inst_cols_list.append(np.zeros((0, 3), dtype=np.uint8))
                    continue
                depth_t = depth_maps[t] * 1  # same scale as merged
                pts, cols = unproject_to_pointcloud(
                    depth_t,
                    frames[t],
                    intrinsics1,
                    mask=inst_mask_t,
                    # Per-instance foreground points keep all valid masked depths.
                    # No confidence filter, no fg-mask erosion.
                    conf=None,
                    conf_threshold=effective_conf_threshold,
                    filter_depth_outliers=True,
                    depth_outlier_std_scale=depth_outlier_std_scale,
                )
                inst_pts_list.append(pts)
                inst_cols_list.append(cols)
            per_instance_per_frame_points[obj_id] = inst_pts_list
            per_instance_per_frame_colors[obj_id] = inst_cols_list
            # Store anchor mask (frame 0) for instance reference extraction.
            anchor_mask = inst_masks_TxHxW[0]
            if anchor_mask.any():
                per_instance_anchor_masks[obj_id] = anchor_mask
            total_pts = sum(len(p) for p in inst_pts_list)
            _log(f"    obj_{obj_id}: {total_pts} total points across {num_frames} frames")

    # Release models created here.
    if _owns_stream3r:
        del stream3r_model
    if _owns_sam3:
        del sam3_segmenter
    torch.cuda.empty_cache()

    # ========== Step 4: Render every frame (optional) ==========
    need_rendering = save_rendered_video or save_rendered_frames
    rendered_frames = []

    if need_rendering:
        _log("\n[Step 4] Rendering frames...")

        # rotation_angle == 0 keeps the original view; otherwise look down at the scene from above.
        if abs(rotation_angle) < 1e-6:
            # The points are already in the original camera frame.
            c2w_render = None
            _log("  Using original viewpoint (no rotation)")
        else:
            # The point cloud center is the look-at target.
            bg_parts = [points_bg1] if points_bg1 is not None else []
            all_pts = np.concatenate(bg_parts + all_foreground_points, axis=0)
            center = all_pts.mean(axis=0)
            _log(f"  Point cloud center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

            # rotation_angle is the elevation above the horizontal plane (60 = looking down from 60 degrees).
            elevation_angle = np.radians(rotation_angle)

            cam_distance = center[2] * 2

            # Place the camera above and behind the center (toward the original camera origin).
            # Camera y points down, so moving up subtracts from y.
            cam_pos = np.array(
                [
                    center[0],
                    center[1] - cam_distance * np.sin(elevation_angle),
                    center[2] - cam_distance * np.cos(elevation_angle),
                ]
            )

            # Build a look-at camera-to-world matrix toward the center.
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)

            # World up is -y because camera y points down.
            world_up = np.array([0, -1, 0])

            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:  # noqa: SIM108 - Branches carry per-case comments.
                # forward is parallel to world_up; use the x axis as right.
                right = np.array([1, 0, 0])
            else:
                right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # OpenCV camera axes: x = right, y = down, z = forward.
            c2w_render = np.eye(4)
            c2w_render[:3, 0] = right
            c2w_render[:3, 1] = -up
            c2w_render[:3, 2] = forward
            c2w_render[:3, 3] = cam_pos

            _log(f"  Camera position: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
            _log(f"  Elevation angle: {rotation_angle}°")

        for frame_idx in tqdm(range(num_frames), desc="Rendering", disable=not verbose):
            # Foreground of the current frame only (not accumulated).
            current_fg_points = all_foreground_points[frame_idx]
            current_fg_colors = all_foreground_colors[frame_idx]

            if points_bg1 is not None:
                merged_points = np.concatenate([points_bg1, current_fg_points], axis=0)
                merged_colors = np.concatenate([colors_bg1, current_fg_colors], axis=0)
            else:
                merged_points = current_fg_points
                merged_colors = current_fg_colors

            # Render at the video resolution.
            rendered = render_pointcloud(
                merged_points,
                merged_colors,
                intrinsics1,
                (H, W),
                camera_pose=c2w_render,
            )

            rendered_frames.append(rendered)
    else:
        _log("\n[Step 4] Skipping rendering (render outputs disabled)")

    # ========== Step 5: Save outputs ==========
    _log("\n[Step 5] Saving outputs...")

    if save_rendered_video:
        output_video_path = output_dir / "rendered_merged.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (W, H))

        for frame in rendered_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        _log(f"  Saved: {output_video_path}")

    # Final merged point cloud: background plus the foreground of every frame.
    final_fg_points = np.concatenate(all_foreground_points, axis=0)
    final_fg_colors = np.concatenate(all_foreground_colors, axis=0)
    if points_bg1 is not None:
        final_merged_points = np.concatenate([points_bg1, final_fg_points], axis=0)
        final_merged_colors = np.concatenate([colors_bg1, final_fg_colors], axis=0)
    else:
        final_merged_points = final_fg_points
        final_merged_colors = final_fg_colors

    if save_final_merged_pointcloud:
        save_pointcloud_ply(str(output_dir / "final_merged_pointcloud.ply"), final_merged_points, final_merged_colors)
        _log(f"  Saved: {output_dir / 'final_merged_pointcloud.ply'}")

    num_bg = len(points_bg1) if points_bg1 is not None else 0

    metadata = {
        "video_path": str(video_path) if video_path is not None else "<in_memory>",
        "video_source": source_desc,
        "num_frames": num_frames,
        "fps": fps,
        "image_size": [W, H],
        "raw_depth_size": [depth_W_raw, depth_H_raw],
        "rotation_angle": rotation_angle,
        "conf_threshold": float(effective_conf_threshold),
        "dynamic_object_prompts": dynamic_object_prompts,
        "num_background_points": num_bg,
        "num_foreground_points_per_frame": [len(pts) for pts in all_foreground_points],
        "total_merged_points": len(final_merged_points),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _log(f"  Saved: {output_dir / 'metadata.json'}")

    if save_rendered_frames:
        frames_dir = output_dir / "rendered_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(rendered_frames):
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        _log(f"  Saved frames to: {frames_dir}")

    if save_depth_maps:
        _log(f"  Saved depth maps to: {depths_dir}")

    _log("\n" + "=" * 60)
    _log("Done!")
    _log("=" * 60)

    return {
        "rendered_frames": rendered_frames,
        "final_merged_points": final_merged_points,
        "final_merged_colors": final_merged_colors,
        # Foreground-only (no background) for event projection.
        "final_fg_points": final_fg_points,
        "final_fg_colors": final_fg_colors,
        # Per-frame foreground points for finer control.
        "all_foreground_points": all_foreground_points,
        "all_foreground_colors": all_foreground_colors,
        "metadata": metadata,
        # Session state for caller to sync back.
        "stream3r_frames_fed": stream3r_frames_fed,
        # Per-instance per-frame point clouds (empty dicts when instance seg unavailable).
        "per_instance_per_frame_points": per_instance_per_frame_points,
        "per_instance_per_frame_colors": per_instance_per_frame_colors,
        "per_instance_anchor_masks": per_instance_anchor_masks,
    }


def main(
    mode: str = "video",
    # Shared settings
    dynamic_object_prompts: list | None = None,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    device: str = "cuda",
    # Image mode
    image1_path: str | None = None,
    image2_path: str | None = None,
    # Video mode
    video_path: str | None = None,
    output_dir: str | None = None,
    rotation_angle: float = 20.0,
    depth_outlier_std_scale: float = 3.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
):
    """
    Run image mode or video mode.

    Args:
        mode: "image" or "video"
        dynamic_object_prompts: Moving-object class names (default ["car"])
        sam3_model_path: SAM3 checkpoint path
        device: Compute device
        image1_path: Image mode - first image
        image2_path: Image mode - second image
        video_path: Video mode - video path
        output_dir: Output directory
        rotation_angle: Video mode - camera elevation angle in degrees
        depth_outlier_std_scale: Standard-deviation multiplier for depth outlier filtering
        conf_threshold: Confidence threshold (default: 0.10); out-of-range values raise ValueError
    """
    if dynamic_object_prompts is None:
        dynamic_object_prompts = ["car"]

    if mode == "image":
        if output_dir is None:
            output_dir = "outputs/dynamic_depth_update"

        result = update_dynamic_object_depth(
            image1_path=image1_path,
            image2_path=image2_path,
            dynamic_object_prompts=dynamic_object_prompts,
            output_dir=output_dir,
            sam3_model_path=sam3_model_path,
            device=device,
        )

    elif mode == "video":
        if output_dir is None:
            output_dir = "outputs/merge_video_foregrounds"

        result = merge_video_foregrounds(
            video_path=video_path,
            dynamic_object_prompts=dynamic_object_prompts,
            output_dir=output_dir,
            sam3_model_path=sam3_model_path,
            rotation_angle=rotation_angle,
            depth_outlier_std_scale=depth_outlier_std_scale,
            conf_threshold=conf_threshold,
            device=device,
        )

    return result


if __name__ == "__main__":
    # ========== Settings ==========

    # "image" or "video"
    mode = "video"

    # Moving-object class names for SAM3, for example ["car", "person", "dog"].
    dynamic_object_prompts = ["car"]

    sam3_model_path = "ckpts/facebook--sam3/sam3.pt"

    device = "cuda"

    # Keep depths within mean +/- std_scale * std.
    depth_outlier_std_scale = 3.0

    # Confidence threshold (0.0 disables filtering; larger values filter more).
    conf_threshold = 0.4

    # ========== Image mode ==========
    image1_path = "example_imgs/first_frame/car_wild_far.png"
    image2_path = "example_imgs/first_frame/car_wild_near.png"
    image_output_dir = "outputs/dynamic_depth_update"

    # ========== Video mode ==========
    video_path = "example_imgs/videos/name_car_wild_far_step_unknown_idx_0.mp4"
    video_output_dir = "outputs/event_video_proj"
    rotation_angle = 20  # Elevation angle in degrees.
    # rotation_angle = 0  # Original viewpoint.

    # ========== Run ==========
    if mode == "image":
        result = main(
            mode=mode,
            dynamic_object_prompts=dynamic_object_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            image1_path=image1_path,
            image2_path=image2_path,
            output_dir=image_output_dir,
        )
    else:
        result = main(
            mode=mode,
            dynamic_object_prompts=dynamic_object_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            video_path=video_path,
            output_dir=video_output_dir,
            rotation_angle=rotation_angle,
            depth_outlier_std_scale=depth_outlier_std_scale,
            conf_threshold=conf_threshold,
        )
