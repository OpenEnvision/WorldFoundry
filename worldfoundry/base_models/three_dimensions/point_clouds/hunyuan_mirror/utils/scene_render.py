import contextlib
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.models.rasterization import (
    GaussianSplatRenderer,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.sh_utils import RGB2SH, SH2RGB
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.utils.gs_effects import GSEffects
from worldfoundry.core.geometry.transforms import (
    quaternion_wxyz_to_rotation_matrix_normalized as quaternion_to_rotation_matrix,
)
from worldfoundry.core.geometry.transforms import (
    rotation_matrix_to_quaternion_wxyz_trace as rotation_matrix_to_quaternion,
)
from worldfoundry.core.geometry.transforms import slerp_quaternions as slerp_quaternions
from worldfoundry.core.media.artifacts import apply_color_map_to_image


def render_interpolated_video(
    gs_renderer: GaussianSplatRenderer,
    splats: dict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple[int, int],
    out_path: Path,
    interp_per_pair: int = 20,
    loop_reverse: bool = True,
    effects: GSEffects = None,
    effect_type: int = 2,
    save_mode: str = "split",
) -> None:
    # camtoworlds: [B, S, 4, 4], intrinsics: [B, S, 3, 3]
    b, s, _, _ = camtoworlds.shape
    h, w = hw

    # Build interpolated trajectory
    def build_interpolated_traj(index, nums):
        exts, ints = [], []
        tmp_camtoworlds = camtoworlds[:, index]
        tmp_intrinsics = intrinsics[:, index]
        for i in range(len(index) - 1):
            exts.append(tmp_camtoworlds[:, i : i + 1])
            ints.append(tmp_intrinsics[:, i : i + 1])
            # Extract rotation and translation
            R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
            R1, t1 = tmp_camtoworlds[:, i + 1, :3, :3], tmp_camtoworlds[:, i + 1, :3, 3]

            # Convert rotations to quaternions
            q0 = rotation_matrix_to_quaternion(R0)
            q1 = rotation_matrix_to_quaternion(R1)

            # Interpolate using smooth quaternion slerp
            for j in range(1, nums + 1):
                alpha = j / (nums + 1)

                # Linear interpolation for translation
                t_interp = (1 - alpha) * t0 + alpha * t1

                # Spherical interpolation for rotation
                q_interp = slerp_quaternions(q0, q1, alpha)
                R_interp = quaternion_to_rotation_matrix(q_interp)

                # Create interpolated extrinsic matrix
                ext = torch.eye(4, device=R_interp.device, dtype=R_interp.dtype)[None].repeat(b, 1, 1)
                ext[:, :3, :3] = R_interp
                ext[:, :3, 3] = t_interp

                # Linear interpolation for intrinsics
                K0 = tmp_intrinsics[:, i]
                K1 = tmp_intrinsics[:, i + 1]
                K = (1 - alpha) * K0 + alpha * K1

                exts.append(ext[:, None])
                ints.append(K[:, None])

        # Preserve both trajectory endpoints; interpolation alone otherwise
        # stops before the final input camera.
        exts.append(tmp_camtoworlds[:, -1:])
        ints.append(tmp_intrinsics[:, -1:])

        exts = torch.cat(exts, dim=1)[:1]
        ints = torch.cat(ints, dim=1)[:1]
        return exts, ints

    # Build wobble trajectory
    def build_wobble_traj(nums, delta):
        assert s == 1
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
    chunk = 40 if effects is None else 1
    t = 0
    t_skip = 0
    if effects is not None:
        try:
            pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
        except BaseException:
            pruned_splats = splats
        # indices = [x for x in range(0, all_ext.shape[1], 2)][:4]
        # add_ext, add_int = build_interpolated_traj(indices, 150)
        # add_ext = torch.flip(add_ext, dims=[1])
        # add_int = torch.flip(add_int, dims=[1])
        add_ext = all_ext[:, :1, :, :].repeat(1, 320, 1, 1)
        add_int = all_int[:, :1, :, :].repeat(1, 320, 1, 1)
        shift = pruned_splats["means"][0].median(dim=0).values
        scale_factor = (pruned_splats["means"][0] - shift).abs().quantile(0.95, dim=0).max()
        all_ext[0, :, :3, -1] = (all_ext[0, :, :3, -1] - shift) / scale_factor
        add_ext[0, :, :3, -1] = (add_ext[0, :, :3, -1] - shift) / scale_factor
        flag = None
        with contextlib.suppress(BaseException):
            raw_splats = gs_renderer.rasterizer.runner.splats
        for st in range(add_ext.shape[1]):
            ed = min(st + 1, add_ext.shape[1])
            assert gs_renderer.sh_degree == 0
            if flag is not None and (flag < 0.99).any():
                break
            sample_gsplat = {
                "means": (pruned_splats["means"][0] - shift) / scale_factor,
                "quats": pruned_splats["quats"][0],
                "scales": pruned_splats["scales"][0] / scale_factor,
                "opacities": pruned_splats["opacities"][0],
                "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
            }
            effects_splats, flag = effects.apply_effect(sample_gsplat, t, effect_type=effect_type)
            t += 0.04
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(-1, 1, 3)
            try:
                gs_renderer.rasterizer.runner.splats  # noqa: B018 - Probes the renderer for its splat attributes.
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6))
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    None,
                    None,
                    None,
                    None,
                    None,
                    add_ext[:, st:ed].to(torch.float32),
                    add_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree,
                )
            except BaseException:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    effects_splats["means"][None],
                    effects_splats["quats"][None],
                    effects_splats["scales"][None],
                    effects_splats["opacities"][None],
                    effects_splats["sh"][None],
                    add_ext[:, st:ed].to(torch.float32),
                    add_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
                )

            if st > add_ext.shape[1] * 0.14:
                t_skip = t if t_skip == 0 else t_skip
                # break
                rendered_rgbs.append(colors)
                rendered_depths.append(depths)
            # if (flag == 0).all():
            #     break
    t_st = t
    t_ed = 0
    loop_dir = 1
    ignore_scale = False
    for st in tqdm(range(0, all_ext.shape[1], chunk)):
        ed = min(st + chunk, all_ext.shape[1])
        if effects is not None:
            try:
                sample_gsplat = {
                    "means": (pruned_splats["means"][0] - shift) / scale_factor,
                    "quats": pruned_splats["quats"][0],
                    "scales": pruned_splats["scales"][0] / scale_factor,
                    "opacities": pruned_splats["opacities"][0],
                    "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
                }
            except BaseException:
                sample_gsplat = {
                    "means": (pruned_splats["means"][0] - shift) / scale_factor,
                    "quats": pruned_splats["quats"][0],
                    "scales": pruned_splats["scales"][0] / scale_factor,
                    "opacities": pruned_splats["opacities"][0],
                    "colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3)),
                }
            effects_splats, flag = effects.apply_effect(
                sample_gsplat, t, effect_type=effect_type, ignore_scale=ignore_scale
            )
            if loop_dir < 0:
                t -= 0.04
            else:
                t += 0.04
            if flag.mean() < 0.01 and t_ed == 0:
                t_ed = t
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(-1, 1, 3)
            effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
            effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
            try:
                gs_renderer.rasterizer.runner.splats  # noqa: B018 - Probes the renderer for its splat attributes.
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6))
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    None,
                    None,
                    None,
                    None,
                    None,
                    all_ext[:, st:ed].to(torch.float32),
                    all_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree,
                )
            except BaseException:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    effects_splats["means"][None],
                    effects_splats["quats"][None],
                    effects_splats["scales"][None],
                    effects_splats["opacities"][None],
                    effects_splats["sh"][None],
                    all_ext[:, st:ed].to(torch.float32),
                    all_int[:, st:ed].to(torch.float32),
                    width=w,
                    height=h,
                    sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
                )

            if t > (all_ext.shape[1]) * 0.04 + t_st - (t_ed - t_st) * 2 - 15 * 0.04 or t < t_st:
                # ignore_scale = True
                loop_dir *= -1
                t = t_ed if loop_dir == -1 else t
        else:
            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1],
                splats["quats"][:1],
                splats["scales"][:1],
                splats["opacities"][:1],
                splats["sh"][:1] if "sh" in splats else splats["colors"][:1],
                all_ext[:, st:ed].to(torch.float32),
                all_int[:, st:ed].to(torch.float32),
                width=w,
                height=h,
                sh_degree=gs_renderer.sh_degree if "sh" in splats else None,
            )
        rendered_rgbs.append(colors)
        rendered_depths.append(depths)

    rgbs = torch.cat(rendered_rgbs, dim=1)[0]  # [N, H, W, 3]
    depths = torch.cat(rendered_depths, dim=1)[0, ..., 0]  # [N, H, W]

    def depth_vis(d: torch.Tensor) -> torch.Tensor:
        valid = d > 0
        near = d[valid].float().quantile(0.01).log() if valid.any() else torch.tensor(0.0, device=d.device)
        far = d.flatten().float().quantile(0.99).log()
        x = d.float().clamp(min=1e-9).log()
        x = 1.0 - (x - near) / (far - near + 1e-9)
        return apply_color_map_to_image(x, "turbo")

    frames = []
    rgb_frames = []
    depth_frames = []

    for rgb, dep in zip(rgbs, depths, strict=False):
        rgb_img = rgb.permute(2, 0, 1)  # [3, H, W]
        depth_img = depth_vis(dep)  # [3, H, W]

        if save_mode == "both":
            combined = torch.cat([rgb_img, depth_img], dim=1)  # [3, 2*H, W]
            frames.append(combined)
        elif save_mode == "split":
            rgb_frames.append(rgb_img)
            depth_frames.append(depth_img)
        else:
            raise ValueError("save_mode must be 'both' or 'split'")

    def _make_video(frames, path):
        video = torch.stack(frames).clamp(0, 1)  # [N, 3, H, W]
        video = video.permute(0, 2, 3, 1)  # [N, H, W, 3] for video encoding
        video = (video * 255).to(torch.uint8).cpu().numpy()
        if loop_reverse and video.shape[0] > 1:
            video = np.concatenate([video, video[::-1][1:-1]], axis=0)
        imageio.mimwrite(str(path), list(video), fps=30, macro_block_size=1)

    # Save videos
    if save_mode == "both":
        _make_video(frames, f"{out_path}.mp4")
    elif save_mode == "split":
        _make_video(rgb_frames, f"{out_path}_rgb.mp4")
        _make_video(depth_frames, f"{out_path}_depth.mp4")

    print(f"Video saved to {out_path} (mode: {save_mode})")

    if effects is not None:
        with contextlib.suppress(BaseException):
            gs_renderer.rasterizer.runner.splats = raw_splats
    torch.cuda.empty_cache()
