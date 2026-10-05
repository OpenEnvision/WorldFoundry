"""MiraBench official FID/KID protocol adapter.

Provenance: the torchmetrics FID/KID update loops previously vendored here are
single-sourced in ``worldfoundry.evaluation.tasks.metrics.fid`` /
``worldfoundry.evaluation.tasks.metrics.kid``; the unused Inception-v3 FID
network copy (``evaluation/inception.py``) was deleted as a dead duplicate of
the in-tree FID backbone. This module keeps only the MiraBench-specific
protocol: 100-frame uniform subsampling per video directory and the official
Resize(299) + ImageNet-normalization transform.
"""

import os

import numpy as np
import torchvision.transforms as TF
from PIL import Image

from worldfoundry.evaluation.tasks.metrics.fid import compute_paired_fid_kid


def load_frame_path_from_dir(datadir, select_frame=100):
    dir_list = [os.path.join(datadir, video_path) for video_path in os.listdir(datadir)]
    all_files = []
    for dir in dir_list:
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        files.sort()
        if len(files) > select_frame:
            files = [files[i] for i in np.linspace(0, len(files) - 1, select_frame).astype(int)]
        all_files += files
    return all_files


def _load_image_batches(frame_dir, transform, device):
    for image_path in load_frame_path_from_dir(frame_dir):
        img = Image.open(image_path).convert("RGB")
        yield transform(img).unsqueeze(0).to(device)


def EvaluateFID(store_image_folder, store_gt_image_folder, ckpt_path, device):
    del ckpt_path  # torchmetrics stages the Inception weights itself
    fid_image_transforms = TF.Compose(
        [
            TF.Resize((299, 299)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return compute_paired_fid_kid(
        _load_image_batches(store_gt_image_folder, fid_image_transforms, device),
        _load_image_batches(store_image_folder, fid_image_transforms, device),
        device=device,
        kid_subset_size=100,
    )
