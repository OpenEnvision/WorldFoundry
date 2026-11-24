"""
Render interpolated video from Gaussian Splatting predictions.

Interpolates smooth camera trajectories using SLERP quaternions,
renders each frame via gsplat, and saves MP4 videos.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.models.models.rasterization import (
    GaussianSplatRenderer,
)
from worldfoundry.core.geometry.transforms import (
    quaternion_wxyz_to_rotation_matrix_normalized as quaternion_to_rotation_matrix,
)
from worldfoundry.core.geometry.transforms import (
    rotation_matrix_to_quaternion_wxyz_trace as rotation_matrix_to_quaternion,
)
from worldfoundry.core.geometry.transforms import slerp_quaternions as slerp_quaternions


def render_interpolated_video(
    gs_renderer: GaussianSplatRenderer,
    splats: dict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple,
    out_path: Path,
    interp_per_pair: int = 20,
    loop_reverse: bool = True,
    save_mode: str = "split",
    frame_times: list | None = None,
    render_depth: bool = False,
) -> None:
    """Render an interpolated fly-through video from Gaussian splat predictions.

    Args:
        gs_renderer: GaussianSplatRenderer instance (from the model).
        splats: Dict with keys 'means', 'scales', 'quats', 'opacities', 'sh'/'colors'.
        camtoworlds: Camera-to-world matrices [B, S, 4, 4].
        intrinsics: Camera intrinsic matrices [B, S, 3, 3].
        hw: Tuple of (height, width) for rendering.
        out_path: Output path (without extension).
        interp_per_pair: Number of interpolated frames per camera pair.
        loop_reverse: Append reversed video for smooth looping.
        save_mode: 'split' (separate rgb/depth) or 'both' (combined).
        frame_times: Optional list of timestamps for adaptive interpolation.
        render_depth: Whether to also render depth video.
    """
    b, s, _, _ = camtoworlds.shape
    h, w = hw

    def build_interpolated_traj(index, base_interp_per_pair: int):
        exts, ints = [], []
        tmp_camtoworlds = camtoworlds[:, index]
        tmp_intrinsics = intrinsics[:, index]

        use_time_based = frame_times is not None and len(frame_times) == len(index)
        if use_time_based and len(index) > 1:
            times = np.array([frame_times[i] for i in index], dtype=np.float32)
            gaps = np.diff(times)
            gaps[gaps < 0] = 0.0
            total_gap = float(gaps.sum())
            target_total_interp = max(1, (len(index) - 1) * base_interp_per_pair)
            gap_scale = target_total_interp / total_gap if total_gap > 1e-6 else 0.0
        else:
            gaps = None
            gap_scale = None

        for i in range(len(index) - 1):
            exts.append(tmp_camtoworlds[:, i : i + 1])
            ints.append(tmp_intrinsics[:, i : i + 1])
            R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
            R1, t1 = tmp_camtoworlds[:, i + 1, :3, :3], tmp_camtoworlds[:, i + 1, :3, 3]

            q0 = rotation_matrix_to_quaternion(R0)
            q1 = rotation_matrix_to_quaternion(R1)

            if use_time_based:
                gap = float(gaps[i]) if gaps is not None else 0.0
                num_interp = max(0, int(round(gap * gap_scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
            else:
                num_interp = base_interp_per_pair

            for j in range(1, num_interp + 1):
                alpha = j / (num_interp + 1)
                t_interp = (1 - alpha) * t0 + alpha * t1
                q_interp = slerp_quaternions(q0, q1, alpha)
                R_interp = quaternion_to_rotation_matrix(q_interp)

                ext = torch.eye(4, device=R_interp.device, dtype=R_interp.dtype)[None].repeat(b, 1, 1)
                ext[:, :3, :3] = R_interp
                ext[:, :3, 3] = t_interp

                K0 = tmp_intrinsics[:, i]
                K1 = tmp_intrinsics[:, i + 1]
                K = (1 - alpha) * K0 + alpha * K1

                exts.append(ext[:, None])
                ints.append(K[:, None])

        # Include the final keyframe exactly once.  Without this endpoint the
        # fly-through never reaches the last input camera.
        exts.append(tmp_camtoworlds[:, -1:])
        ints.append(tmp_intrinsics[:, -1:])

        exts = torch.cat(exts, dim=1)[:1]
        ints = torch.cat(ints, dim=1)[:1]
        return exts, ints

    def build_wobble_traj(nums, delta):
        if s != 1:
            raise ValueError("Wobble trajectory requires exactly 1 input view")
        t = torch.linspace(0, 1, nums, dtype=torch.float32, device=camtoworlds.device)
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        tf = torch.eye(4, dtype=torch.float32, device=camtoworlds.device)
        radius = delta * 0.15
        tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
        radius = radius[..., None]
        radius = radius * t
        tf[..., 0, 3] = torch.sin(2 * torch.pi * t) * radius
        tf[..., 1, 3] = -torch.cos(2 * torch.pi * t) * radius
        exts = camtoworlds @ tf
        ints = intrinsics.repeat(1, exts.shape[1], 1, 1)
        return exts, ints

    if s > 1:
        all_ext, all_int = build_interpolated_traj(list(range(s)), interp_per_pair)
    else:
        means = splats["means"][0]
        center = means.median(dim=0).values
        scene_extent = torch.linalg.vector_norm(means - center, dim=-1).quantile(0.9).clamp_min(1e-3)
        all_ext, all_int = build_wobble_traj(interp_per_pair * 12, scene_extent[None])

    rendered_rgbs, rendered_depths = [], []
    chunk = 40

    # Always prune splats to remove scale outliers
    try:
        pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
    except (AttributeError, RuntimeError):
        pruned_splats = splats

    for st in tqdm(range(0, all_ext.shape[1], chunk)):
        ed = min(st + chunk, all_ext.shape[1])
        colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
            pruned_splats["means"][:1],
            pruned_splats["quats"][:1],
            pruned_splats["scales"][:1],
            pruned_splats["opacities"][:1],
            pruned_splats["sh"][:1] if "sh" in pruned_splats else pruned_splats["colors"][:1],
            all_ext[:, st:ed].to(torch.float32),
            all_int[:, st:ed].to(torch.float32),
            width=w,
            height=h,
            sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
        )
        rendered_rgbs.append(colors)
        if render_depth:
            rendered_depths.append(depths)

    rgbs = torch.cat(rendered_rgbs, dim=1)[0]  # [N, H, W, 3]
    if render_depth:
        depths_all = torch.cat(rendered_depths, dim=1)[0, ..., 0]  # [N, H, W]
    del rendered_rgbs, rendered_depths

    def _depth_vis(d: torch.Tensor) -> torch.Tensor:
        """Simple turbo colormap depth visualization."""
        import matplotlib.pyplot as plt

        valid = d > 0
        near = d[valid].float().quantile(0.01).log() if valid.any() else torch.tensor(0.0, device=d.device)
        far = d.flatten().float().quantile(0.99).log()
        x = d.float().clamp(min=1e-9).log()
        x = 1.0 - (x - near) / (far - near + 1e-9)
        x_np = x.cpu().numpy()
        colored = torch.from_numpy(plt.cm.turbo(x_np)[..., :3]).permute(2, 0, 1).float()
        return colored

    rgb_frames = []
    depth_frames = []

    if render_depth:
        for rgb, dep in zip(rgbs, depths_all, strict=False):
            rgb_frames.append(rgb.permute(2, 0, 1))
            depth_frames.append(_depth_vis(dep))
    else:
        for rgb in rgbs:
            rgb_frames.append(rgb.permute(2, 0, 1))

    def _make_video(frames, path):
        video = torch.stack([f.cpu() for f in frames]).clamp(0, 1)
        video = video.permute(0, 2, 3, 1)
        video = (video * 255).to(torch.uint8).numpy()
        if loop_reverse and video.shape[0] > 1:
            video = np.concatenate([video, video[::-1][1:-1]], axis=0)
        imageio.mimwrite(str(path), list(video), fps=30, macro_block_size=1)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    if save_mode == "split":
        _make_video(rgb_frames, out_path / "rendered_rgb.mp4")
        if render_depth:
            _make_video(depth_frames, out_path / "rendered_depth.mp4")
    elif save_mode == "both" and render_depth:
        combined = [torch.cat([r, d], dim=1) for r, d in zip(rgb_frames, depth_frames, strict=False)]
        _make_video(combined, out_path / "rendered.mp4")

    print(f"Video saved to {out_path} (mode: {save_mode})")
    torch.cuda.empty_cache()
