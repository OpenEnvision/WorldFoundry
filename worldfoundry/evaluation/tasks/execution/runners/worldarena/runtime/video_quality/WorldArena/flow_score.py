import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

from .utils import load_dimension_info
from worldfoundry.base_models.perception_core.optical_flow.raft import RAFT
from worldfoundry.base_models.perception_core.optical_flow.raft.utils.utils import InputPadder
from worldfoundry.core.execution.device import get_current_torch_device
from worldfoundry.core.distributed.collectives.evaluation import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)


class FlowScore:
    """Compute mean optical-flow magnitude per video using RAFT."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._load_model()

    def _load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()

    def _read_frames(self, video_path):
        """Load frames from mp4 or an image folder."""
        frame_list = []

        if os.path.isdir(video_path):
            exts = {
                "jpg", "jpeg", "png", "bmp", "tif", "tiff",
                "JPG", "JPEG", "PNG", "BMP", "TIF", "TIFF"
            }
            imgs = sorted([
                os.path.join(video_path, p)
                for p in os.listdir(video_path)
                if p.split(".")[-1] in exts
            ])
            for img_path in imgs:
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame_list.append(tensor)
            return frame_list

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame_list.append(tensor)
        cap.release()
        return frame_list

    def _pairwise_flow_mean(self, image1, image2):
        image1 = image1.unsqueeze(0).to(self.device)
        image2 = image2.unsqueeze(0).to(self.device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
        return float(flow_magnitude.mean().item())

    def infer(self, video_path):
        frames = self._read_frames(video_path)
        if len(frames) < 2:
            return 0.0

        pair_scores = []
        with torch.no_grad():
            for first, second in zip(frames[:-1], frames[1:]):
                pair_scores.append(self._pairwise_flow_mean(first, second))

        return float(np.mean(pair_scores)) if len(pair_scores) > 0 else 0.0


def flow_score(flow_scorer, video_list):
    sim = []
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        score_per_video = flow_scorer.infer(video_path)
        video_results.append({
            'video_path': video_path,
            'video_results': score_per_video,
        })
        sim.append(score_per_video)

    avg_score = float(np.mean(sim)) if len(sim) > 0 else 0.0
    return avg_score, video_results


def compute_flow_score(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    model_path = submodules_list.get("model")
    if model_path is None:
        raise ValueError("flow_score requires raft checkpoint from config")

    args_new = edict({
        "model": model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False,
    })

    scorer = FlowScore(args_new, device)

    video_list, _ = load_dimension_info(
        json_dir,
        dimension='flow_score',
        lang='en'
    )

    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = flow_score(scorer, video_list)

    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum(
            [d['video_results'] for d in video_results]
        ) / len(video_results)

    return all_results, video_results
