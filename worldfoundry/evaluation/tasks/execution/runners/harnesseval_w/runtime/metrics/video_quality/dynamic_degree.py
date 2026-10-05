"""Dynamic degree — VBench-aligned RAFT optical flow."""
import numpy as np
import torch

from ..base import BaseMetric
from ..weight_utils import get_weight_file



def _load_raft(device):
    from worldfoundry.base_models.perception_core.optical_flow.raft.raft import RAFT
    from worldfoundry.base_models.perception_core.optical_flow.raft.utils.utils import InputPadder
    from easydict import EasyDict

    weight_path = get_weight_file("raft", "raft-things.pth")
    model = RAFT(EasyDict(small=False, mixed_precision=False, alternate_corr=False))
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    return model.to(device).eval(), InputPadder


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

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        if len(frames) < 2:
            return {f"{self.name}_score": 0.0}
        return self._compute_raft(frames)
