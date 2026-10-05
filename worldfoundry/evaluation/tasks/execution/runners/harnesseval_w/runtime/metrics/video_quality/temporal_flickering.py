"""Temporal flickering metric — VBench-aligned MAE between consecutive frames."""
import numpy as np
from ..base import BaseMetric


class TemporalFlickeringMetric(BaseMetric):
    cpu_only = True

    @property
    def name(self):
        return "temporal_flickering"

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        if len(frames) < 2:
            return {f"{self.name}_score": 1.0}

        diffs = []
        for i in range(len(frames) - 1):
            img1 = np.array(frames[i]).astype(np.float32)
            img2 = np.array(frames[i + 1]).astype(np.float32)
            mae = np.mean(np.abs(img1 - img2))
            diffs.append(mae)

        avg_mae = float(np.mean(diffs))
        score = float((255.0 - avg_mae) / 255.0)
        return {f"{self.name}_score": score, "avg_mae": avg_mae}
