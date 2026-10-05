from typing import List
import os
import cv2
import torch
import cv2
import numpy as np

from .base_metrics import BaseMetric
from .utils import load_dimension_info

from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba import config as cfg
from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba.inference import Model
from worldfoundry.base_models.perception_core.frame_interpolation.vfimamba.benchmark.utils.padder import InputPadder

from .torchmetrics.lpips_metrics import LearnedPerceptualImagePatchSimilarityMetric
# from .torchmetrics.psnr_metrics import PeakSignalNoiseRatioMetric
from .torchmetrics.ssim_metrics import StructuralSimilarityIndexMeasureMetric

import json
'''==========Model setting=========='''
TTA = True
cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
    F = 32,
    depth = [2, 2, 2, 3, 3]
)

class MotionSmoothnessMetric(BaseMetric):
    """
    add description here
    """
    
    def __init__(self) -> None:
        super().__init__()
        model = Model(-1)
        model.load_model()
        model.eval()
        model.device()  
        self._model = model
        self._ssim_metric = StructuralSimilarityIndexMeasureMetric()
        self._lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()
        # self._psnr_metric = PeakSignalNoiseRatioMetric()
        
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:
        scores_ssim = []
        scores_lpips = []
        scores_mse = []
        even_images = rendered_images[::2] #0,2,4,6.........
        odd_images = rendered_images[1::2] #1,3,5,7........
        
        print(f'=========================Start Interpolating=========================')
        for i, (image1, image2) in enumerate(zip(even_images[:-1], even_images[1:])): #(0,2),(2,4),(4,6).....
            odd_image = odd_images[i]  #1,3,5,7.........
            mid_pred = cv2.imread(odd_image) #1,3,5,7.........
            I0 = cv2.imread(image1) #0,2,4,6.........   (H, W, C)
            I2 = cv2.imread(image2) #2,4,6,8.........

            I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)  #(1,3,384,672)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)  #(1,3,384,672)
            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)  #(1, 3, 384, 672)

            # Model prediction: intermediate frame interpolation between adjacent frames (I0 and I2)
            mid = (padder.unpad(self._model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=0.0))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            
            ssim_score = self._ssim_metric._compute_scores(mid, mid_pred)
            lpips_score = self._lpips_metric._compute_scores(mid, mid_pred)
            # psnr_score = self._psnr_metric._compute_scores(mid, mid_pred)
            mse_score = np.mean((mid - mid_pred) ** 2)
            
            scores_ssim.append(ssim_score)
            scores_lpips.append(lpips_score)
            scores_mse.append(mse_score)
        print(f'=========================Done=========================')
        
        score_ssim = sum(scores_ssim) / len(scores_ssim)
        score_lpips = sum(scores_lpips) / len(scores_lpips)
        score_mse = sum(scores_mse) / len(scores_mse)
        #return (score_mse, score_ssim, score_lpips)   # Lower MSE is better, higher SSIM is better, lower LPIPS is better
        return (score_ssim,score_lpips)  # Higher SSIM is better, lower LPIPS is better


def _extract_frames_to_dir(video_path: str, out_root: str, max_frames: int = 120) -> List[str]:
    os.makedirs(out_root, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.png')])
    if existing:
        return existing

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    indices = list(range(total))
    if max_frames > 0 and total > max_frames:
        step = total / float(max_frames)
        indices = [int(i * step) for i in range(max_frames)]

    saved: List[str] = []
    idx = 0
    next_set = set(indices)
    cur = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur in next_set:
            out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            idx += 1
        cur += 1
    cap.release()
    return saved


def compute_consistency_motion_smoothness(json_dir, device, submodules_dict, **kwargs):
    dimension = os.path.splitext(os.path.basename(json_dir))[0]
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension, lang='en')

    metric = MotionSmoothnessMetric()
    details = []
    ssim_list: List[float] = []
    lpips_list: List[float] = []
    out_root = os.path.join(os.path.dirname(json_dir), 'frames_cache')

    for item in prompt_dict_ls:
        for video_path in item.get('video_list', []) or []:
            frames = _extract_frames_to_dir(video_path, out_root)
            if len(frames) < 3:
                continue
            ssim_score, lpips_score = metric._compute_scores(frames)
            ssim_list.append(float(ssim_score))
            lpips_list.append(float(lpips_score))
            details.append({
                'video_path': video_path,
                'num_frames': len(frames),
                'ssim': float(ssim_score),
                'lpips': float(lpips_score),
            })

    # Use SSIM as the primary scalar score
    final = float(sum(ssim_list) / len(ssim_list)) if ssim_list else 0.0
    # Include auxiliary aggregates
    details.append({
        'aggregate': {
            'avg_ssim': float(sum(ssim_list) / len(ssim_list)) if ssim_list else 0.0,
            'avg_lpips': float(sum(lpips_list) / len(lpips_list)) if lpips_list else 0.0,
        }
    })
    # Save detailed results JSON
    try:
        output_dir = os.path.dirname(json_dir)
        dim_name = os.path.splitext(os.path.basename(json_dir))[0]
        model = kwargs.get('model', '')
        dataset_json = kwargs.get('dataset_json', '')
        dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else 'dataset'
        suffix = f"{dim_name}__{model}__{dataset_base}_results.json" if model else f"{dim_name}_results.json"
        output_file = os.path.join(output_dir, suffix)
        detailed_output = {
            "evaluation_summary": {
                "total_videos": len([d for d in details if 'video_path' in d]),
                "average_score": final,
                "avg_ssim": float(sum(ssim_list) / len(ssim_list)) if ssim_list else 0.0,
                "avg_lpips": float(sum(lpips_list) / len(lpips_list)) if lpips_list else 0.0,
            },
            "video_details": [d for d in details if 'video_path' in d],
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

    return final, details
