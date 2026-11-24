"""Dynamic degree — VBench-aligned RAFT optical flow with Farneback fallback."""

from types import SimpleNamespace

import cv2
import numpy as np
import torch

from worldfoundry.base_models.perception_core.optical_flow.raft import (
    RAFT,
    InputPadder,
)
from worldfoundry.base_models.perception_core.optical_flow.raft import (
    checkpoint_path as raft_checkpoint_path,
)

from ..base import BaseMetric


def _load_raft(device):
    """Attempt to load RAFT model; returns None on failure."""
    try:
        weight_path = raft_checkpoint_path()
        if not weight_path.is_file():
            return None, None

        args = SimpleNamespace(model=str(weight_path), small=False, mixed_precision=False, alternate_corr=False)
        model = RAFT(args)
        ckpt = torch.load(weight_path, map_location="cpu")
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt)
        model.to(device).eval()
        return model, InputPadder
    except Exception:
        return None, None


class DynamicDegreeMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.raft_model, self.InputPadder = _load_raft(device)

    @property
    def name(self):
        return "dynamic_degree"

    def _get_score_raft(self, flo):
        flo = flo[0].permute(1, 2, 0).cpu().numpy()
        rad = np.sqrt(flo[:, :, 0] ** 2 + flo[:, :, 1] ** 2)
        cut_index = int(rad.size * 0.05)
        return float(np.mean(np.sort(rad.flatten())[-cut_index:]))

    @staticmethod
    def _resize_frame(frame, max_edge=512):
        w, h = frame.size
        if min(w, h) <= max_edge:
            return frame
        if h < w:
            new_h, new_w = max_edge, int(w * max_edge / h)
        else:
            new_w, new_h = max_edge, int(h * max_edge / w)
        return frame.resize((new_w, new_h))

    def _compute_raft(self, frames):
        frames = [self._resize_frame(f, 512) for f in frames]
        tensors = []
        for f in frames:
            t = torch.from_numpy(np.array(f).astype(np.uint8)).permute(2, 0, 1).float()
            tensors.append(t[None].to(self.device))

        scale = min(tensors[0].shape[-2:])
        thres = 6.0 * (scale / 256.0)
        count_num = max(1, round(4 * (len(tensors) / 16.0)))
        move_count = 0

        with torch.no_grad():
            for i in range(len(tensors) - 1):
                padder = self.InputPadder(tensors[i].shape)
                img1, img2 = padder.pad(tensors[i], tensors[i + 1])
                _, flow_up = self.raft_model(img1, img2, iters=20, test_mode=True)
                if self._get_score_raft(flow_up) > thres:
                    move_count += 1
                if move_count >= count_num:
                    break

        return {f"{self.name}_score": 1.0 if move_count >= count_num else 0.0}

    def _compute_farneback(self, frames):
        frames = [self._resize_frame(f, 512) for f in frames]
        h, w = np.array(frames[0]).shape[:2]
        scale = min(h, w)
        thres = 6.0 * (scale / 256.0)
        count_num = max(1, round(4 * (len(frames) / 16.0)))
        move_count = 0
        prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            top_mag = np.mean(np.sort(mag.flatten())[-int(mag.size * 0.05) :])
            if top_mag > thres:
                move_count += 1
            if move_count >= count_num:
                break
            prev_gray = curr_gray

        return {f"{self.name}_score": 1.0 if move_count >= count_num else 0.0}

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        if len(frames) < 2:
            return {f"{self.name}_score": 0.0}
        if self.raft_model is not None:
            return self._compute_raft(frames)
        return self._compute_farneback(frames)
