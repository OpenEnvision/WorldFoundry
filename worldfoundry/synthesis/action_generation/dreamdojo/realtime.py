# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Resident DreamDojo student sessions with bounded observation/action history."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from worldfoundry.base_models.diffusion_model.contracts import Conditioning, DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import robot_actions
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.core.utils import load_pil_image


def student_sampling_options(
    *,
    height: int = 480,
    width: int = 640,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    fps: float = 10,
) -> None:
    """Validate released student geometry, fixed guidance, and playback cadence."""
    for size in (height, width):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 16:
            raise ValueError("DreamDojo student height and width must be positive integer multiples of 16")
    if (
        isinstance(num_inference_steps, bool)
        or not isinstance(num_inference_steps, int)
        or not 1 <= num_inference_steps <= 4
    ):
        raise ValueError("DreamDojo student num_inference_steps must be an integer from 1 to 4")
    if guidance_scale != 1.0:
        raise ValueError("DreamDojo student uses distilled guidance_scale=1")
    if fps != 10:
        raise ValueError("DreamDojo student playback cadence is 10 fps")


class DreamDojoStudentSession:
    """Own one rollout; weights are shared, mutable frames/actions are not.

    First input: twelve normalized actions → thirteen RGB frames. Subsequent
    inputs: four actions → four new RGB frames. Nine preceding RGB frames and
    their eight transition actions are re-encoded at each window boundary.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.condition_frames: torch.Tensor | None = None
        self.action_history: torch.Tensor | None = None
        self.text_conditioning: Conditioning | None = None
        self.chunk_index = 0
        self.prompt = ""
        self.height, self.width, self.steps, self.seed = 480, 640, 4, 42

    @staticmethod
    def realtime_spec() -> RealtimeSpec:
        """Model-owned 10-fps playback and 12/4 normalized-action cadence."""
        return RealtimeSpec(
            fps=10,
            first_chunk_frames=13,
            steady_chunk_frames=4,
            input_fps=10,
            first_input_frames=12,
            steady_input_frames=4,
            controls=("normalized_unified_384d_action",),
        )

    def next_output_frames(self) -> int:
        """Report the next output budget without consuming state."""
        return 13 if self.chunk_index == 0 else 4

    @torch.inference_mode()
    def configure(
        self,
        images: Any,
        prompt: str = "",
        *,
        height: int = 480,
        width: int = 640,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        fps: float = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Encode text once and retain a seed image without running diffusion."""
        student_sampling_options(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
        )
        image = load_pil_image(images).convert("RGB")
        frames = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0).float()
        frames = (
            functional.interpolate(frames, (height, width), mode="bilinear", align_corners=False).byte().unsqueeze(2)
        )
        self.reset()
        try:
            request = DiffusionRequest(prompt=prompt, num_frames=13, inputs={"actions": torch.zeros(1, 12, 384)})
            self.text_conditioning = self.pipeline.components.conditioner.encode(
                request,
                device=self.pipeline.device,
                dtype=self.pipeline.dtype,
            )
            self.condition_frames = frames
            self.height, self.width, self.steps, self.seed = height, width, num_inference_steps, int(seed)
            self.prompt = prompt
        except BaseException:
            self.reset()
            raise
        return {"status": "configured", "realtime_spec": self.realtime_spec().to_payload()}

    @torch.inference_mode()
    def generate(self, actions: Any) -> dict[str, Any]:
        """Advance one window atomically; failures invalidate the rollout, not weights."""
        if self.condition_frames is None:
            raise RuntimeError("DreamDojo student must be configured before streaming")
        actions = robot_actions(actions)
        expected = 12 if self.chunk_index == 0 else 4
        if actions.shape[1] != expected:
            raise ValueError(f"DreamDojo student expects {expected} new actions for this chunk")
        window_actions = (
            actions if self.action_history is None else torch.cat((self.action_history.to(actions), actions), dim=1)
        )
        request = DiffusionRequest(
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            num_frames=13,
            sampling=SamplingConfig(
                num_inference_steps=self.steps, guidance_scale=1.0, seed=self.seed + self.chunk_index
            ),
            inputs={
                "actions": window_actions,
                "conditioning_frames": self.condition_frames,
                "text_conditioning": self.text_conditioning,
            },
        )
        try:
            output = self.pipeline(request)
            pixels = output.sample
            if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (1, 3, 13, self.height, self.width):
                raise RuntimeError("DreamDojo student decoder must return [1,3,13,H,W] generated pixels")
            start = 0 if self.chunk_index == 0 else 9
            new_pixels = pixels[:, :, start:]
            self.condition_frames = pixels[:, :, -9:].detach().clamp(-1, 1).add(1).mul(127.5).byte()
            self.action_history = window_actions[:, -8:].detach().clone()
            self.chunk_index += 1
        except BaseException:
            self.reset()
            raise
        return {
            "status": "success",
            "model_id": "dreamdojo-student",
            "runtime": "worldfoundry-native-diffusion",
            "backend_quality": "in_tree_runtime",
            "variant": "student",
            "num_frames": new_pixels.shape[2],
            "fps": 10,
            "video_chunk": VideoChunk(frames=new_pixels, layout="bcthw", value_range="-1,1", metadata={"fps": 10}),
            "realtime_spec": self.realtime_spec().to_payload(),
            "chunk_index": self.chunk_index - 1,
            "seed": request.sampling.seed,
        }

    def reset(self) -> None:
        """Release rollout references; leave the shared DiT, text encoder, and VAE resident."""
        self.condition_frames = None
        self.action_history = None
        self.text_conditioning = None
        self.chunk_index = 0
