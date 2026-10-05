from typing import List
from pathlib import Path

import torch
import cv2
import numpy as np
from tqdm import tqdm

from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba import config as cfg
from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba.inference import Model, convert
from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba.benchmark.utils.padder import InputPadder
from worldfoundry.core.execution.device import get_current_torch_device

from .utils import load_dimension_info, read_video_frames_cv2
from worldfoundry.core.distributed.collectives.evaluation import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)

'''==========Model setting=========='''
TTA = True
cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
    F = 32,
    depth = [2, 2, 2, 3, 3]
)

class MotionSmoothnessMetric:
    """Motion smoothness via VFI model with SSIM weighting."""

    def __init__(self, ckpt_path=None, device=None) -> None:
        self._device = device or get_current_torch_device()

        model = Model(-1)
        ckpt_resolved = Path(ckpt_path) if ckpt_path else None
        if ckpt_resolved is not None:
            if not ckpt_resolved.exists():
                raise FileNotFoundError(f"WorldArena VFIMamba checkpoint is not staged: {ckpt_resolved}")
            state = torch.load(ckpt_resolved, map_location="cpu")
            model.net.load_state_dict(convert(state), strict=True)
        else:
            model.load_model()

        model.eval()
        if torch.cuda.is_available():
            model.device()
        else:
            model.net.to(self._device)

        self._model = model
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self._ssim_metric = StructuralSimilarityIndexMeasure().to(self._device)
        # self._psnr_metric = PeakSignalNoiseRatioMetric()
        
    def _compute_scores(self, rendered_images: List[np.ndarray]) -> float:
        even_images = rendered_images[::2]
        odd_images = rendered_images[1::2]

        scores_final = []

        for i, (image1, image2) in enumerate(zip(even_images[:-1], even_images[1:])):
            if i >= len(odd_images):
                break

            I0 = image1 if isinstance(image1, np.ndarray) else cv2.imread(image1)
            I2 = image2 if isinstance(image2, np.ndarray) else cv2.imread(image2)
            mid_gt = odd_images[i] if isinstance(odd_images[i], np.ndarray) else cv2.imread(odd_images[i])

            if I0 is None or I2 is None or mid_gt is None:
                continue

            raw_diff = np.mean(np.abs(I0.astype(np.float32) - I2.astype(np.float32)))

            if raw_diff < 1.0:
                continue

            I0_ = (torch.tensor(I0.transpose(2, 0, 1), device=self._device).float() / 255.).unsqueeze(0)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1), device=self._device).float() / 255.).unsqueeze(0)
            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)
            mid_pred_t = self._model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=0.0)
            mid_pred = (padder.unpad(mid_pred_t)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            ssim_pred = torch.tensor(mid_pred.transpose(2, 0, 1), device=self._device).unsqueeze(0).float() / 255.0
            ssim_gt = torch.tensor(mid_gt.transpose(2, 0, 1), device=self._device).unsqueeze(0).float() / 255.0
            ssim_val = float(self._ssim_metric(ssim_pred, ssim_gt).detach().cpu().item())
            final_metric = ssim_val * np.log1p(raw_diff)

            scores_final.append(final_metric)

        if not scores_final:
            return 0.0
        avg_score = float(np.mean(scores_final))

        return avg_score


def compute_motion_smoothness(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    model_path = submodules_list.get("model")

    metric = MotionSmoothnessMetric(model_path, device=device)

    video_list, _ = load_dimension_info(
        json_dir,
        dimension='motion_smoothness',
        lang='en'
    )

    video_list = distribute_list_to_rank(video_list)

    scores = []
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        frames = read_video_frames_cv2(video_path)
        if len(frames) < 3:
            video_results.append({
                'video_path': video_path,
                'video_results': 0.0,
                'error': 'insufficient_frames',
            })
            continue

        score = metric._compute_scores(frames)
        video_results.append({
            'video_path': video_path,
            'video_results': score,
        })
        scores.append(score)

    avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        avg_score = sum([d['video_results'] for d in video_results]) / len(video_results) if video_results else 0.0

    return avg_score, video_results
