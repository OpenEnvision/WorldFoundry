"""Lazy Zing rollout with the released keyboard and prompt timeline contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.zing.config import (
    load_config,
    with_cache_window,
)
from worldfoundry.core.io import resolve_data_path
from worldfoundry.runtime.official_inference import positive_int

from .processor import MessageProcessor


class ZingRuntime:
    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str = "cuda",
        config_path: str | None = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        fps: int = 24,
        seed: int = 0,
        controls: list[dict[str, Any]] | None = None,
        local_attn_size: int | None = None,
        sink_size: int | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        config_path = config_path or resolve_data_path("models/runtime/configs/zing/zing.yaml")
        self.config = with_cache_window(load_config(config_path), local_attn_size, sink_size)
        self.device = torch.device(device)
        self.num_frames = positive_int(num_frames, "num_frames")
        self.height = positive_int(height, "height")
        self.width = positive_int(width, "width")
        self.fps, self.seed = positive_int(fps, "fps"), int(seed)
        self.controls = controls or []
        self.pipeline = None
        if self.num_frames < 1 or (self.num_frames - 1) % 4:
            raise ValueError("Zing num_frames includes the reference frame and must equal 1 + 4*N.")
        if min(self.height, self.width) < 32 or self.height % 32 or self.width % 32:
            raise ValueError("Zing height and width must be positive multiples of 32.")
        if self.fps < 1:
            raise ValueError("fps must be positive")

    @torch.inference_mode()
    def generate_video(self, prompt: str, image_path: str | None = None):
        if not str(prompt).strip():
            raise ValueError("Zing requires a non-empty prompt.")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Zing generation requires CUDA and the released checkpoints.")
        if image_path is not None and self.num_frames == 1:
            raise ValueError("Image-conditioned Zing requires at least 5 total frames.")
        if self.pipeline is None:
            from .pipeline import InferencePipeline

            self.pipeline = InferencePipeline(
                self.config,
                self.checkpoint_dir / "pretrained",
                self.checkpoint_dir / "generator/model.pt",
                str(self.device),
            )
        reference_count = int(image_path is not None)
        target = {
            "role": "target",
            "type": "video",
            "reference_frame_count": reference_count,
            "output": {"frames": self.num_frames - reference_count, "height": self.height, "width": self.width},
            "controls": self.controls,
        }
        if image_path is not None:
            target["uri"] = image_path
        sample = {"messages": [{"role": "user", "type": "text", "content": prompt}, target]}
        request = MessageProcessor(self.config, self.pipeline.encode_reference).process(sample)
        device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]), torch.cuda.device(self.device):
            torch.random.default_generator.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            return self.pipeline.generate(request)[0]

    def close(self):
        self.pipeline = None
