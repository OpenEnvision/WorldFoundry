"""Shared local runtime code for local metric-backed skills."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from numbers import Real
from pathlib import Path
from typing import Any

from ..io import file_fingerprint


def _configure_outer_cpu_parallelism() -> None:
    """Keep library thread pools from multiplying the request-level workers."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits changing this before the first inter-op task.
        # A second backend in the same process can safely reuse the first value.
        pass


def _decode_frames(video: Path) -> tuple[list[Any], float]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if source_fps <= 0 or not frames:
        raise RuntimeError(f"video has no valid frames: {video}")
    return frames, source_fps


def _sample_frames(
    frames: Sequence[Any], source_fps: float, sampling: Mapping[str, float | int]
) -> list[Any]:
    if "fps" in sampling:
        target_fps = float(sampling["fps"])
        if target_fps <= 0:
            raise ValueError("target fps must be positive")
        return list(frames[:: max(1, round(source_fps / target_fps))])
    maximum = int(sampling["max_frames"])
    if maximum <= 0:
        raise ValueError("max_frames must be positive")
    count = min(maximum, len(frames))
    if count == len(frames):
        return list(frames)
    if count == 1:
        return [frames[0]]
    indices = [
        int(index * (len(frames) - 1) / (count - 1))
        for index in range(count)
    ]
    return [frames[index] for index in indices]


def _validate_video(video_path: Path, expected: Mapping[str, Any]) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    actual = file_fingerprint(video_path, include_sha256=bool(expected.get("sha256")))
    for key in ("path", "size_bytes", "mtime_ns", "sha256"):
        if expected.get(key) is not None and expected[key] != actual.get(key):
            raise RuntimeError(f"backend video {key} does not match the request")


def _raw_score(metric_name: str, result: Mapping[str, Any]) -> float:
    value = result.get(f"{metric_name}_score")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"{metric_name} returned no finite numeric score")
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{metric_name} returned no finite numeric score")
    return score


class LocalMetricBackend:
    """Reusable local runtime; subclasses fix one skill's metrics and identity."""

    backend_id = ""
    version = ""
    config_digest = ""
    output_schema = ""
    metric_sampling: Mapping[str, Mapping[str, float | int]] = {}
    execution_mode = "local"

    def __init__(
        self,
        weights_root: Path,
        device: str = "cuda",
        *,
        cpu_workers: int = 8,
    ) -> None:
        if cpu_workers <= 0:
            raise ValueError("cpu_workers must be positive")
        self.weights_root = weights_root.resolve()
        self.device = device
        self.cpu_workers = cpu_workers
        self.cpu_intraop_threads = 1
        self.load_count = 0
        self._evaluator: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=cpu_workers,
            thread_name_prefix="metric-cpu",
        )

    def _create_evaluator(self) -> Any:
        _configure_outer_cpu_parallelism()
        if not self.weights_root.is_dir():
            raise FileNotFoundError(self.weights_root)
        os.environ["HARNESSEVAL_WEIGHTS_ROOT"] = str(self.weights_root)
        from ..metrics import video_quality

        return {
            name: getattr(video_quality, f"get_{name}_metric")()(device=self.device)
            for name in self.metric_sampling
        }

    def ensure_ready(self) -> None:
        if self._evaluator is not None:
            return
        with self._load_lock:
            if self._evaluator is None:
                self._evaluator = self._create_evaluator()
                self.load_count += 1

    def evaluate(
        self, video_path: Path, video_fingerprint: Mapping[str, Any]
    ) -> dict[str, Any]:
        _validate_video(video_path, video_fingerprint)
        self.ensure_ready()
        # Decode and prepare future requests concurrently. Only model forward passes
        # remain serialized because the resident evaluator shares one model instance.
        frames, source_fps = _decode_frames(video_path)
        prepared_metrics = []
        with ExitStack() as resources:
            for metric_name, sampling in self.metric_sampling.items():
                sampled = _sample_frames(frames, source_fps, sampling)
                metric = self._evaluator[metric_name]
                if getattr(metric, "cpu_only", False):
                    result = metric.compute(sampled)
                    prepared = None
                else:
                    prepare_cpu = getattr(metric, "prepare_cpu", None)
                    prepared = (
                        resources.enter_context(
                            prepare_cpu(sampled, executor=self._cpu_executor)
                        )
                        if prepare_cpu is not None
                        else sampled
                    )
                    result = None
                prepared_metrics.append(
                    (metric_name, metric, prepared, result, len(sampled))
                )

            with self._inference_lock:
                for index, (metric_name, metric, prepared, result, count) in enumerate(
                    prepared_metrics
                ):
                    if result is None:
                        compute_prepared = getattr(metric, "compute_prepared", None)
                        result = (
                            compute_prepared(prepared)
                            if compute_prepared is not None
                            else metric.compute(prepared)
                        )
                    prepared_metrics[index] = (
                        metric_name,
                        metric,
                        prepared,
                        result,
                        count,
                    )

            metrics = {
                metric_name: {
                    "raw_score": _raw_score(metric_name, result),
                    "sampled_frames": count,
                }
                for metric_name, _, _, result, count in prepared_metrics
            }
        return {
            "schema_version": self.output_schema,
            "metrics": metrics,
            "source_fps": source_fps,
            "decoded_frames": len(frames),
        }

    def close(self) -> None:
        self._cpu_executor.shutdown(wait=True, cancel_futures=True)
