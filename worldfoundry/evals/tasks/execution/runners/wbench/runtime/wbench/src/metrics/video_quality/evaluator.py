"""
Unified video quality evaluator.
Automatically selects appropriate frame sampling strategy per metric (VBench-aligned).
"""

import time
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image

from . import (
    get_aesthetic_quality_metric,
    get_dynamic_degree_metric,
    get_hpsv3_quality_metric,
    get_imaging_quality_metric,
    get_motion_smoothness_metric,
    get_temporal_flickering_metric,
)

_SHARED_RESIZE_CROP_CPU = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ]
)


def _shared_preprocess_gpu(frames, device="cuda"):
    """GPU-accelerated batch preprocessing: PIL → tensor (N, 3, 224, 224)."""
    import numpy as np

    arrays = np.stack([np.array(f) for f in frames])
    batch = torch.from_numpy(arrays).permute(0, 3, 1, 2).float().div_(255.0)
    batch = batch.to(device)
    batch = T.functional.resize(batch, 224, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    batch = T.functional.center_crop(batch, 224)
    return batch.cpu()


FRAME_SAMPLING_CONFIG = {
    "aesthetic_quality": {"fps": 2.0},
    "imaging_quality": {"fps": 2.0},
    "temporal_flickering": {"fps": 10.0},
    "dynamic_degree": {"fps": 8.0},
    "motion_smoothness": {"fps": 10.0},
    "hpsv3_quality": 20,
}

METRIC_GETTERS = {
    "aesthetic_quality": get_aesthetic_quality_metric,
    "imaging_quality": get_imaging_quality_metric,
    "temporal_flickering": get_temporal_flickering_metric,
    "dynamic_degree": get_dynamic_degree_metric,
    "motion_smoothness": get_motion_smoothness_metric,
    "hpsv3_quality": get_hpsv3_quality_metric,
}

DEFAULT_METRICS = list(METRIC_GETTERS)


class VideoQualityEvaluator:
    """Unified video quality evaluator."""

    def __init__(self, device: str = "cuda", metrics: list[str] | None = None):
        self.device = device
        self.metrics = metrics or list(DEFAULT_METRICS)
        self._metric_instances = {}
        self._all_frames = None
        self._video_fps = 24.0
        self._sampled_cache = {}

    def _get_metric(self, name: str):
        if name not in self._metric_instances:
            getter = METRIC_GETTERS[name]
            MetricClass = getter()
            self._metric_instances[name] = MetricClass(device=self.device)
        return self._metric_instances[name]

    def _decode_all_frames(self, video_path: str):
        if self._all_frames is not None:
            return
        import cv2

        cap = cv2.VideoCapture(video_path)
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        self._all_frames = frames

    def _load_frames(self, video_path: str, sampling_cfg) -> list[Image.Image]:
        self._decode_all_frames(video_path)
        all_frames = self._all_frames
        vlen = len(all_frames)

        cache_key = str(sampling_cfg)
        if cache_key in self._sampled_cache:
            return self._sampled_cache[cache_key]

        if isinstance(sampling_cfg, dict):
            fps = sampling_cfg["fps"]
            step = max(1, round(self._video_fps / fps))
            indices = list(range(0, vlen, step))
        elif sampling_cfg == -1 or sampling_cfg >= vlen:
            self._sampled_cache[cache_key] = all_frames
            return all_frames
        else:
            import numpy as np

            indices = np.linspace(0, vlen - 1, sampling_cfg, dtype=int).tolist()

        sampled = [all_frames[i] for i in indices]
        self._sampled_cache[cache_key] = sampled
        return sampled

    def evaluate(self, video_path: str, metrics: list[str] | None = None, verbose: bool = True) -> dict[str, Any]:
        """Evaluate video quality across all configured metrics."""
        metrics_to_eval = metrics or self.metrics
        results = {"video_path": video_path, "metrics": {}, "summary": {}}

        self._all_frames = None
        self._sampled_cache = {}
        self._preprocessed_cache = {}
        total_time = 0

        for name in metrics_to_eval:
            if name not in METRIC_GETTERS:
                continue
            try:
                start = time.time()
                sampling_cfg = FRAME_SAMPLING_CONFIG.get(name, {"fps": 2.0})
                frames = self._load_frames(video_path, sampling_cfg)

                metric = self._get_metric(name)
                result = metric.compute(frames)
                elapsed = time.time() - start
                total_time += elapsed

                score = result.get(f"{name}_score", None)
                results["metrics"][name] = {
                    "score": score,
                    "time": round(elapsed, 2),
                    "num_frames": len(frames),
                    "details": result,
                }
                results["summary"][name] = score

                if verbose:
                    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                    print(f"  {name:25} | {elapsed:6.2f}s | {len(frames):3} frames | {score_str}")
            except Exception as e:
                results["metrics"][name] = {"error": str(e)}
                if verbose:
                    print(f"  {name:25} | error: {e}")

        results["total_time"] = round(total_time, 2)
        self._all_frames = None
        self._preprocessed_cache = {}
        return results
