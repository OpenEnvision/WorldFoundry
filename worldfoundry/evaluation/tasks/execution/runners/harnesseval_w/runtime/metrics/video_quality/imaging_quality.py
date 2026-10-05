"""Imaging quality metric — MUSIQ (via pyiqa)."""
import numpy as np
import torch
import torchvision.transforms as T
from contextlib import contextmanager

from ..base import BaseMetric
from ..weight_utils import setup_torch_hub_dir, get_weights_dir


_GPU_BATCH_SIZE = 16


def _vbench_transform(img_tensor, max_edge=512):
    """VBench-aligned: scale long edge to max_edge, normalize to [0,1]."""
    _, h, w = img_tensor.shape
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        new_h, new_w = int(scale * h), int(scale * w)
        img_tensor = T.Resize(size=(new_h, new_w), antialias=False)(img_tensor)
    return img_tensor.float() / 255.0


class ImagingQualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        setup_torch_hub_dir()
        import pyiqa
        import pyiqa.utils.download_util as _dl
        # Redirect pyiqa cache to local weights/pyiqa/
        _dl.DEFAULT_CACHE_DIR = get_weights_dir("pyiqa")
        self.model = pyiqa.create_metric('musiq', device=self.device)

    @property
    def name(self):
        return "imaging_quality"

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        with self.prepare_cpu(frames) as prepared:
            return self.compute_prepared(prepared)

    @staticmethod
    def _prepare_frame(frame, to_tensor):
        return _vbench_transform(to_tensor(frame))

    @contextmanager
    def prepare_cpu(self, frames, *, executor=None):
        to_tensor = T.PILToTensor()
        def prepare(frame):
            return self._prepare_frame(frame, to_tensor)

        prepared = (
            list(executor.map(prepare, frames))
            if executor is not None
            else [prepare(frame) for frame in frames]
        )
        yield prepared

    def compute_prepared(self, prepared):
        scores = []
        for start in range(0, len(prepared), _GPU_BATCH_SIZE):
            batch = torch.stack(prepared[start : start + _GPU_BATCH_SIZE]).to(
                self.device
            )
            with torch.no_grad():
                batch_scores = self.model(batch).reshape(-1)
            expected = len(batch)
            if batch_scores.numel() != expected:
                raise RuntimeError(
                    f"MUSIQ returned {batch_scores.numel()} scores for {expected} frames"
                )
            scores.extend((batch_scores / 100.0).cpu().tolist())
        return {f"{self.name}_score": float(np.mean(scores))}
