import glob
import os

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def cal_ssim(gt_img, pd_img):
    return structural_similarity(gt_img, pd_img, channel_axis=-1)


def compute_basic_metrics(gt_path, pd_path, metric_names=["psnr", "ssim"]):  # noqa: B006 - Read-only default; retain the public signature.
    metric_funcs = {
        "psnr": peak_signal_noise_ratio,
        "ssim": cal_ssim,
    }
    res = {}
    for _ in metric_names:
        assert _ in metric_funcs
        res.update({_: {}})

    for task_id in sorted(os.listdir(pd_path)):
        task_path = os.path.join(pd_path, task_id)

        for _ in metric_names:
            res[_][task_id] = {}

        for episode_id in tqdm(sorted(os.listdir(task_path))):
            if episode_id.endswith((".png", ".json")):
                continue

            for _ in metric_names:
                res[_][task_id][episode_id] = {}

            gt_image_list = glob.glob(os.path.join(gt_path, task_id, episode_id, "video", "frame_*.png"))
            gt_image_list.sort()
            n_frames = len(gt_image_list)

            gid_list = sorted(os.listdir(os.path.join(task_path, episode_id)))
            for gid in gid_list:
                pd_image_list = glob.glob(os.path.join(task_path, episode_id, gid, "video", "frame_*.png"))
                pd_image_list += glob.glob(os.path.join(task_path, episode_id, gid, "video", "frame_*.jpg"))
                pd_image_list.sort()
                try:
                    assert len(pd_image_list) == n_frames
                except BaseException:
                    print(len(pd_image_list), n_frames)
                cur_metrics = {}
                for pd_img, gt_img in zip(pd_image_list, gt_image_list, strict=False):
                    pd_img = np.asanyarray(Image.open(pd_img))
                    gt_img = np.asanyarray(Image.open(gt_img))
                    if pd_img.shape != gt_img.shape:
                        pd_img = cv2.resize(pd_img, dsize=tuple(gt_img.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
                    for _ in metric_names:
                        if _ not in cur_metrics:
                            cur_metrics.update({_: 0.0})
                        cur_metrics[_] += metric_funcs[_](gt_img, pd_img)
                for _ in metric_names:
                    res[_][task_id][episode_id].update({gid: cur_metrics[_] / n_frames})

    return res
