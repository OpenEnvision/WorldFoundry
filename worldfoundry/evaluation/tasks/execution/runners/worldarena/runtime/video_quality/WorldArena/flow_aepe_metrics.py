from typing import List
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
from tqdm import tqdm

from worldfoundry.base_models.perception_core.optical_flow.sea_raft import (
    RAFT,
    checkpoint_path as sea_raft_checkpoint_path,
    config_path as sea_raft_config_path,
    load_ckpt,
    parse_args,
)
from worldfoundry.core.execution.device import get_current_torch_device

from .utils import load_dimension_info, read_video_frames_cv2
from worldfoundry.core.distributed.collectives.evaluation import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def compute_epe(flow1, flow2, crop=30, k=1.0, error_threshold=5.0):
    """
    Calculate the End-Point Error (EPE) between two flow fields.

    Parameters:
    - flow1: Flow from image1 to image2, shape [H, W, 2].
    - flow2: Flow from image2 to image1, shape [H, W, 2].
    - crop: The size of pixels to crop for H and W.
    - k: The coefficient for the activation function.

    Returns:
    - epe: End-Point Error, a scalar value.
    """
    
    H, W, _ = flow1.shape
    crop_size_H = H - crop
    crop_size_W = W - crop
    start_x = (W - crop_size_W) // 2
    start_y = (H - crop_size_H) // 2
    
    # Crop the flow fields to the central crop_size_H x crop_size_W region
    flow1_cropped = flow1[start_y:start_y + crop_size_H, start_x:start_x + crop_size_W, :]
    flow2_cropped = flow2[start_y:start_y + crop_size_H, start_x:start_x + crop_size_W, :]
    
    # Create a grid of coordinates (x, y)
    y_coords, x_coords = np.meshgrid(np.arange(crop_size_H), np.arange(crop_size_W), indexing='ij')

    # Coordinates after applying flow1 (warping points from image1 to image2)
    warped_x1 = x_coords + flow1_cropped[..., 0]
    warped_y1 = y_coords + flow1_cropped[..., 1]

    # Warp back using flow2 from the warped positions in image2
    # First, round the warped positions and clip to image boundaries
    warped_x1_rounded = np.clip(np.round(warped_x1).astype(int), 0, crop_size_W - 1)
    warped_y1_rounded = np.clip(np.round(warped_y1).astype(int), 0, crop_size_H - 1)

    # Get the corresponding flow2 values at the new positions in image2
    flow2_at_warped = flow2_cropped[warped_y1_rounded, warped_x1_rounded, :]
    
    # Get corresponding flow from flow2 (warping back from image2 to image1)
    warped_back_x1 = warped_x1 + flow2_at_warped[..., 0]
    warped_back_y1 = warped_y1 + flow2_at_warped[..., 1]

    # Compute the End-Point Error (EPE)
    epe = np.sqrt((warped_back_x1 - x_coords) ** 2 + (warped_back_y1 - y_coords) ** 2)

    failure_mask = (epe > error_threshold).astype(np.uint8)
    # Average EPE across all pixels
    avg_epe = np.mean(epe)

    return avg_epe, failure_mask
   
class OpticalFlowAverageEndPointErrorMetric:
    """
    
    Using estimated optical-flow between two consecutive frames to calculate verage end-point-error.
    
    Optical-flow estimation -- SEA-RAFT
    
    RANGE: [0, ~] lower the better
    """
    
    def __init__(self, cfg_path=None, checkpoint_path=None, device=None) -> None:
        self._device = device or get_current_torch_device()

        cfg_resolved = Path(cfg_path) if cfg_path else sea_raft_config_path("spring-M.json")
        ckpt_resolved = Path(checkpoint_path) if checkpoint_path else sea_raft_checkpoint_path()

        args = argparse.Namespace(cfg=str(cfg_resolved), path=str(ckpt_resolved))
        args = parse_args(args)

        # load model
        model = RAFT(args)
        load_ckpt(model, args.path)
        model.to(self._device)
        model.eval()
        self._model = model
        self._args = args    
    
    def load_image(self, imfile):
        if isinstance(imfile, str):
            image = cv2.imread(imfile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        else:
            image = imfile
        if image is None:
            return None
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image[None].to(self._device)
        return image
    
    def forward_flow(self, image1, image2):
        with torch.amp.autocast(device_type=self._device.type, enabled=self._device.type == "cuda"):
            output = self._model(image1, image2, iters=self._args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final
    def _dynamic_thres(self, image):
        scale = min(image.shape[-2:])
        return 6.0 * (scale / 256.0)


    def _compute_dynamic_raw(self, flow):
        u = flow[..., 0]
        v = flow[..., 1]
        rad = np.sqrt(u ** 2 + v ** 2)

        h, w = rad.shape
        rad_flat = rad.reshape(-1)
        cut_index = int(h * w * 0.05)

        if cut_index == 0:
            return 0.0

        top_rad = np.sort(rad_flat)[-cut_index:]
        return float(np.mean(np.abs(top_rad)))


    def _soft_motion_score(self, score, thres, alpha=5.0):
        x = score / thres - 1.0
        return 1.0 / (1.0 + np.exp(-alpha * x))

    def _compute_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self._args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self._args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self._args.scale, mode='bilinear', align_corners=False) * (0.5 ** self._args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self._args.scale, mode='area')
        
        flow = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
        return flow
            
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:

        scores = []
        dynamic_raw_scores = []   # ← NEW

        with torch.no_grad():
            images = rendered_images
            
            for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)
                if image1 is None or image2 is None:
                    continue
                
                flow1 = self._compute_flow(image1, image2)
                flow2 = self._compute_flow(image2, image1)

                epe_score, failure_mask = compute_epe(flow1, flow2)
                scores.append(epe_score)

                dynamic_raw = self._compute_dynamic_raw(flow1)
                dynamic_raw_scores.append(dynamic_raw)

        if len(scores) == 0:
            return 0.0

        score = sum(scores) / len(scores)
        empirical_max = 3.4536
        empirical_min = 0
        # score = np.clip(score, empirical_min, empirical_max)
        # score = 1 - (score - empirical_min) / (empirical_max - empirical_min)
        score = 1.0 / max(float(score), 1e-6)
        thres = self._dynamic_thres(image1)
        dynamic_soft = [
            self._soft_motion_score(s, thres) for s in dynamic_raw_scores
        ]
        dynamic_degree = float(np.mean(dynamic_soft)) if len(dynamic_soft) > 0 else 0.0

        if dynamic_degree <= 0.1213:
            score = score * dynamic_degree

        return float(score)

    def compute_video(self, frames: List[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.0
        return self._compute_scores(frames)


def compute_photometric_smoothness(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    cfg_path = submodules_list.get("config")
    model_path = submodules_list.get("model")

    metric = OpticalFlowAverageEndPointErrorMetric(cfg_path, model_path, device=device)

    video_list, _ = load_dimension_info(
        json_dir,
        dimension='photometric_smoothness',
        lang='en'
    )

    video_list = distribute_list_to_rank(video_list)

    scores = []
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        frames = read_video_frames_cv2(video_path)
        if len(frames) < 2:
            video_results.append({
                'video_path': video_path,
                'video_results': 0.0,
                'error': 'insufficient_frames',
            })
            continue

        score = metric.compute_video(frames)
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
