"""Resident product session for the native SANA-WM diffusion recipe."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec

DEFAULT_REALTIME_WINDOW_FRAMES = 81
DEFAULT_SAMPLING_STEPS = 60
DEFAULT_CFG_SCALE = 5.0

_TOKEN_TO_KEY = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "j",
    "camera_r": "l",
}


def ensure_sana_import_paths() -> None:
    """Compatibility no-op: native Sana uses package imports only."""


def _snap_num_frames(value: int) -> int:
    value = max(int(value), 9)
    lower = value - ((value - 1) % 8)
    upper = lower + 8
    return lower if value - lower < upper - value else upper


def _frames_for_segments(
    segments: Sequence[Mapping[str, Any]] | None,
    interactions: Sequence[str],
    *,
    frame_count: int,
    fps: int,
) -> list[frozenset[str]]:
    fallback = frozenset(
        _TOKEN_TO_KEY[token]
        for item in interactions
        if (token := str(item).strip().lower()) in _TOKEN_TO_KEY
    )
    if not segments:
        return [fallback] * frame_count
    rows = [
        (
            max(float(segment.get("duration", 0.0) or 0.0), 0.0),
            frozenset(str(item).lower() for item in (segment.get("keys") or ()) if str(item).lower() in "wasdijkl"),
        )
        for segment in segments
    ]
    rows = [(duration, keys) for duration, keys in rows if duration]
    if not rows:
        return [fallback] * frame_count
    scale = (frame_count / float(fps)) / sum(duration for duration, _ in rows)
    boundaries = []
    elapsed = 0.0
    for duration, keys in rows:
        elapsed += duration * scale
        boundaries.append((elapsed, keys))
    output = []
    index = 0
    for frame_index in range(frame_count):
        timestamp = (frame_index + 1) / float(fps)
        while index + 1 < len(boundaries) and timestamp > boundaries[index][0] + 1e-8:
            index += 1
        output.append(boundaries[index][1])
    return output


def _compress_action_frames(frames: Sequence[frozenset[str]]) -> str:
    if not frames:
        return "none-1"
    segments: list[str] = []
    previous = frames[0]
    length = 1
    for keys in frames[1:]:
        if keys == previous:
            length += 1
        else:
            segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
            previous, length = keys, 1
    segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
    return ",".join(segments)


def _validate_output_video(value: Any, *, expected_frames: int) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().float()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4 or tensor.shape[0] not in {1, 3, 4}:
            raise RuntimeError(f"SANA-WM returned an invalid video tensor: {tuple(tensor.shape)}")
        video = tensor.permute(1, 2, 3, 0)
        video = ((video.clamp(-1, 1) + 1) * 127.5).byte().numpy()
    else:
        video = np.ascontiguousarray(value, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
        raise RuntimeError(f"SANA-WM returned an invalid video array: {video.shape}")
    if len(video) != expected_frames:
        raise RuntimeError(f"SANA-WM returned {len(video)} frames; expected {expected_frames}")
    return video


def _device_topology() -> dict[str, torch.device]:
    """Report the one policy-owned device used by unified native inference."""

    if not torch.cuda.is_available():
        raise RuntimeError("SANA-WM realtime requires CUDA")
    device = torch.device("cuda:0")
    return {"native_diffusion": device}


class SanaWMRealtimeSession:
    """Stateful interaction boundary; model execution stays in shared infra."""

    def __init__(
        self,
        checkpoint_source: str | Path | None = None,
        *,
        device: str = "cuda",
        fps: int = 16,
        num_frames: int = DEFAULT_REALTIME_WINDOW_FRAMES,
        **kwargs: Any,
    ) -> None:
        from worldfoundry.pipelines.sana_wm.pipeline_sana_wm import SanaWMPipeline

        self.pipeline = SanaWMPipeline.from_pretrained(checkpoint_source, device=device, **kwargs)
        self.fps = int(fps)
        self.num_frames = _snap_num_frames(num_frames)
        self.output_frames = self.num_frames
        self._prompt = ""
        self._configured = False
        self.last_metrics: dict[str, float] = {}
        self._state_lock = threading.RLock()

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.output_frames,
            steady_chunk_frames=self.output_frames,
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "execution": "worldfoundry-native-diffusion",
            "tensor_parallel": False,
            "device": str(self.pipeline.native_pipeline.device),
            "num_frames": self.num_frames,
            "sampling_steps": self.pipeline.DEFAULT_NUM_INFERENCE_STEPS,
            "prompt_updates": "chunk-boundary",
        }

    def configure(self, *, image: Any, prompt: str, **kwargs: Any) -> dict[str, Any]:
        with self._state_lock:
            self._prompt = str(prompt).strip()
            result = self.pipeline.configure_realtime(
                images=image,
                prompt=self._prompt,
                fps=self.fps,
                window_frames=self.num_frames,
                **kwargs,
            )
            self._configured = True
            return result

    def update_prompt(self, prompt: str) -> bool:
        normalized = str(prompt or "").strip()
        if not normalized or normalized == self._prompt:
            return False
        self._prompt = normalized
        if self.pipeline._realtime_config is not None:
            self.pipeline._realtime_config["prompt"] = normalized
        return True

    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            if not self._configured:
                raise RuntimeError("SANA-WM realtime session is not configured")
            if prompt is not None:
                self.update_prompt(prompt)
            frames = _frames_for_segments(
                control_segments,
                list(interactions or ()),
                frame_count=self.num_frames - 1,
                fps=self.fps,
            )
            result = self.pipeline.stream_realtime(
                interactions=["".join(sorted(keys)) or "none" for keys in frames],
                seed=seed,
                prompt=self._prompt,
            )
            result["frames"] = result.get("video", result.get("sample"))
            result["realtime_spec"] = self.realtime_spec().to_payload()
            result["realtime_metrics"] = dict(self.last_metrics)
            return result

    def reset(self) -> None:
        self.pipeline.reset_realtime()
        self._configured = False


__all__ = [
    "DEFAULT_CFG_SCALE",
    "DEFAULT_REALTIME_WINDOW_FRAMES",
    "DEFAULT_SAMPLING_STEPS",
    "SanaWMRealtimeSession",
    "ensure_sana_import_paths",
]
