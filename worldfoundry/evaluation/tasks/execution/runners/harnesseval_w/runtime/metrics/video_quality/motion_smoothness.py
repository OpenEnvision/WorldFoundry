"""Motion smoothness — AMT-S frame interpolation error (VBench-aligned)."""
import cv2
import numpy as np
import torch

from ..base import BaseMetric
from ..weight_utils import get_weight_file



class MotionSmoothnessMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.model = self._load_amt_model()

    @property
    def name(self):
        return "motion_smoothness"

    def _load_amt_model(self):
        from importlib import import_module

        model_type = import_module(
            "worldfoundry.base_models.perception_core.frame_interpolation.amt.networks.AMT-S"
        ).Model
        ckpt_path = get_weight_file("amt", "amt-s.pth")
        model = model_type()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        return model.to(self.device).eval()

    def compute(self, frames, first_frame=None, prompt=None, batch_size=8, **kwargs):
        """VBench-aligned: predict odd frames from even frame pairs, compute MAE."""
        from worldfoundry.base_models.perception_core.frame_interpolation.amt.utils.utils import InputPadder, img2tensor, tensor2img
        from PIL import Image as _Image

        if self.model is None:
            raise ValueError("AMT model not loaded")
        if len(frames) < 3:
            raise ValueError("Not enough frames")

        resized = []
        for f in frames:
            w, h = f.size
            if min(w, h) > 512:
                if h < w:
                    new_h, new_w = 512, int(w * 512 / h)
                else:
                    new_w, new_h = 512, int(h * 512 / w)
                f = f.resize((new_w, new_h), _Image.LANCZOS)
            resized.append(f)
        frames = resized

        inputs = [img2tensor(np.array(f)).to(self.device) for f in frames]
        padder = InputPadder(inputs[0].shape)
        embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)
        pairs = [(i, i + 2, i + 1) for i in range(0, len(inputs) - 2, 2)]

        vfi_diffs = []
        with torch.no_grad():
            for batch_start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                bs = len(batch_pairs)
                img0_batch = torch.cat([inputs[p[0]] for p in batch_pairs], dim=0)
                img1_batch = torch.cat([inputs[p[1]] for p in batch_pairs], dim=0)
                gt_list = [np.array(frames[p[2]]) for p in batch_pairs]
                img0_pad, img1_pad = padder.pad(img0_batch, img1_batch)
                embt_batch = embt.expand(bs, -1, -1, -1)
                out = self.model(img0_pad, img1_pad, embt_batch, scale_factor=1.0, eval=True)
                pred_batch = padder.unpad(out['imgt_pred'])
                for j in range(bs):
                    pred_img = tensor2img(pred_batch[j:j + 1])
                    diff = np.mean(cv2.absdiff(pred_img, gt_list[j]))
                    vfi_diffs.append(diff)

        avg_mae = np.mean(vfi_diffs)
        score = float((255.0 - avg_mae) / 255.0)
        return {f"{self.name}_score": score, "vfi_mae": float(avg_mae)}
