"""MiraBench official FVD/KVD protocol adapter.

Provenance: the Fréchet/covariance math, uint8-video preprocessing, the I3D
backbone, and the checkpoint resolution that used to be vendored in this file
are single-sourced in ``worldfoundry.evaluation.tasks.metrics.fvd`` (that code
was originally adapted from this MiraBench copy). The polynomial-kernel MMD
behind KVD is single-sourced in ``worldfoundry.evaluation.tasks.metrics.kid``.
This module keeps only the MiraBench-specific protocol: the numbered
``frames_<n>.png`` directory layout, 100-frame uniform subsampling, and the
batch-1 video loader.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from worldfoundry.evaluation.tasks.metrics.fvd.fvd_core import (
    frechet_distance,
    load_fvd_i3d,
    resolve_i3d_checkpoint,
)
from worldfoundry.evaluation.tasks.metrics.kid import polynomial_mmd

TARGET_RESOLUTION = (224, 224)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.target_resolution = TARGET_RESOLUTION

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i, select_frame=100):
        frame_path = self.files[i]
        frames_list = ["frames_" + str(f) + ".png" for f in range(1, 1 + len(os.listdir(frame_path)))]
        if len(frames_list) > select_frame:
            frames_list = [frames_list[i] for i in np.linspace(0, len(frames_list) - 1, select_frame).astype(int)]
        video = []
        for frame in frames_list:
            img = Image.open(os.path.join(frame_path, frame)).convert("RGB")
            img = torch.FloatTensor(np.array(img)).permute(2, 0, 1)
            img = F.interpolate(img[None], size=self.target_resolution, mode="bilinear", align_corners=False)
            video.append(img)
        video = torch.cat(video, dim=0).permute(1, 0, 2, 3)
        return (video / 255) * 2.0 - 1.0


def get_logits(i3d, frame_dir, device, batch_size, num_workers):
    frame_list = [os.path.join(frame_dir, video_path) for video_path in os.listdir(frame_dir)]
    dataloader = torch.utils.data.DataLoader(
        VideoDataset(frame_list),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    with torch.no_grad():
        logits = [i3d(batch.to(device)) for batch in tqdm(dataloader)]
        return torch.cat(logits, dim=0)


def EvaluateFVD(store_image_folder, store_gt_image_folder, ckpt_path, device):
    checkpoint = resolve_i3d_checkpoint(
        extra_candidates=(os.path.join(ckpt_path, "fvd/i3d_pretrained_400.pt"),),
    )
    i3d = load_fvd_i3d(torch.device(device), checkpoint)

    res_embed = get_logits(i3d, store_image_folder, device, 1, 1)
    gt_embed = get_logits(i3d, store_gt_image_folder, device, 1, 1)

    fvd = float(frechet_distance(res_embed, gt_embed).item())
    kvd = polynomial_mmd(res_embed.cpu().numpy(), gt_embed.cpu().numpy())
    return fvd, kvd
