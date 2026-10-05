# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MegaSAM single-scene tracking, adapted from the upstream test_demo inference path."""

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def image_stream(
    image_list,
    mono_disp_list,
    scene_name,
    use_depth=False,
    aligns=None,
    K=None,
    stride=1,
):
    """image generator."""
    del scene_name, stride

    fx, fy, cx, cy = (
        K[0, 0],
        K[1, 1],
        K[0, 2],
        K[1, 2],
    )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

    for t, (image_file) in enumerate(image_list):
        image = cv2.imread(image_file)

        mono_disp = mono_disp_list[t]
        depth = np.clip(
            1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
            1e-4,
            1e4,
        )
        depth[depth < 1e-2] = 0.0

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]

        image = torch.as_tensor(image).permute(2, 0, 1)

        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()
        depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

        mask = torch.ones_like(depth)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        if use_depth:
            yield t, image[None], depth, intrinsics, mask
        else:
            yield t, image[None], intrinsics, mask


def run_tracking(frames_dir, mono_root, metric_root, scene_name, output_path, *, droid_factory, buffer_size=1024):
    """Track aligned frame sequences and write camera-to-world poses in input order."""
    from lietorch import SE3

    args = SimpleNamespace(
        datapath=str(frames_dir),
        mono_depth_path=str(mono_root),
        metric_depth_path=str(metric_root),
        scene_name=scene_name,
        weights=None,
        buffer=buffer_size,
        image_size=[240, 320],
        disable_vis=True,
        beta=0.3,
        filter_thresh=2.0,
        warmup=8,
        keyframe_thresh=2.0,
        frontend_thresh=12.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        stereo=False,
        depth=False,
        upsample=False,
        backend_thresh=16.0,
        backend_radius=2,
        backend_nms=3,
    )
    print("Running evaluation on {}".format(args.datapath))
    print(args)

    scene_name = args.scene_name.split("/")[-1]

    image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpg")))
    image_list += sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.png")))

    mono_disp_paths = sorted(glob.glob(os.path.join("%s/%s" % (args.mono_depth_path, scene_name), "*.npy")))
    metric_depth_paths = sorted(glob.glob(os.path.join("%s/%s" % (args.metric_depth_path, scene_name), "*.npz")))

    if len(image_list) < 2:
        raise ValueError("MegaSAM tracking requires at least two frames")
    stems = [Path(p).stem for p in image_list]
    if stems != [Path(p).stem for p in mono_disp_paths] or stems != [Path(p).stem for p in metric_depth_paths]:
        raise ValueError("MegaSAM RGB, mono-depth and metric-depth frame names must match")
    img_0 = cv2.imread(image_list[0])
    scales = []
    shifts = []
    mono_disp_list = []
    fovs = []
    for t, (mono_disp_file, metric_depth_file) in enumerate(zip(mono_disp_paths, metric_depth_paths)):
        da_disp = np.float32(np.load(mono_disp_file))  # / 300.0
        with np.load(metric_depth_file) as uni_data:
            metric_depth = uni_data["depth"]
            fovs.append(uni_data["fov"])

        da_disp = cv2.resize(
            da_disp,
            (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        mono_disp_list.append(da_disp)
        gt_disp = 1.0 / (metric_depth + 1e-8)

        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2

        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8

        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * da_disp)

        scales.append(scale)
        shifts.append(shift)

    print("************** UNIDEPTH FOV ", np.median(fovs))
    ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
    K = np.eye(3)
    K[0, 0] = ff * 1.0  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 1] = ff * 1.0  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
    K[0, 2] = img_0.shape[1] / 2.0  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 2] = img_0.shape[0] / 2.0  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

    align_scale = scales[med_idx]  # np.median(np.array(scales))
    align_shift = shifts[med_idx]  # np.median(np.array(shifts))
    normalize_scale = np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98) / 2.0

    aligns = (align_scale, align_shift, normalize_scale)

    for t, image, depth, intrinsics, mask in tqdm(
        image_stream(
            image_list,
            mono_disp_list,
            scene_name,
            use_depth=True,
            aligns=aligns,
            K=K,
        )
    ):
        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = droid_factory(args)

        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

    droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

    traj_est, depth_est, motion_prob = droid.terminate(
        image_stream(
            image_list,
            mono_disp_list,
            scene_name,
            use_depth=True,
            aligns=aligns,
            K=K,
        ),
        _opt_intr=True,
        full_ba=True,
        scene_name=scene_name,
    )

    intrinsics = droid.video.intrinsics[0].cpu().numpy() * 8.0
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intrinsics
    cam_c2w = SE3(torch.as_tensor(traj_est, device="cpu")).inv().matrix().numpy()
    if len(cam_c2w) != len(image_list) or not np.isfinite(cam_c2w).all():
        raise ValueError("MegaSAM returned invalid or incomplete camera poses")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, intrinsic=K, cam_c2w=cam_c2w)
    return output_path
