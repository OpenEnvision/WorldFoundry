# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Public offline low-poly video renderer for Matrix-Game 4 Ancient Egypt."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
from worldfoundry.operators.matrix_game4_operator import MatrixGame4Operator
from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline


class MatrixGame4EgyptPipeline(NativeVisualDiffusionPipeline):
    """Render one low-poly clip with the pinned Egypt model, independently of a game."""

    MODEL_ID = "matrix-game-4-egypt"
    OWNER = "Matrix-Game 4 Ancient Egypt"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae", "sink")
    PRIMARY_CHECKPOINT_ROLE = "dit"
    GENERATION_TYPE = "v2v"
    ACCEPTS_VIDEO = True
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 864
    DEFAULT_NUM_INFERENCE_STEPS = 2
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 20
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = MatrixGame4Operator()

    def _prepare_call(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        image_path: str | Path | None = None,
        video: Any = None,
        video_path: str | Path | None = None,
        input_path: str | Path | None = None,
        ref_image_path: str | Path | None = None,
        interactions: Any = None,
        negative_prompt: str | list[str] | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        fps: float | None = None,
        return_dict: bool = False,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 0,
        output_type: str = "auto",
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[DiffusionRequest, dict[str, Any]]:
        from worldfoundry.base_models.diffusion_model.recipes.matrix_game import MATRIX_GAME4_EGYPT_PROMPT

        for name, value, expected in (
            ("height", height, 480),
            ("width", width, 864),
            ("guidance_scale", guidance_scale, 1.0),
            ("num_inference_steps", num_inference_steps, 2),
        ):
            if value is not None and value != expected:
                raise ValueError(f"{self.MODEL_ID} requires {name}={expected}")
        source = video if video is not None else video_path if video_path is not None else input_path
        pixels, source_fps = self.operator.prepare_video(source)
        frames = pixels.shape[2]
        if num_frames not in (None, frames) or any(
            kwargs.get(name) not in (None, frames) for name in self.NUM_FRAMES_ALIASES
        ):
            raise ValueError("Matrix-Game 4 num_frames must match the low-poly input; no resampling is performed")
        prompt = prompt or MATRIX_GAME4_EGYPT_PROMPT
        if not isinstance(prompt, str):
            raise ValueError("Matrix-Game 4 accepts one fixed prompt per video")
        frame_rate = float(fps if fps is not None else source_fps if source_fps is not None else self.DEFAULT_FPS)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("Matrix-Game 4 fps must be finite and positive")
        request, output = super()._prepare_call(
            prompt=prompt,
            images=images,
            image_path=image_path,
            video=pixels,
            video_path=video_path,
            input_path=input_path,
            ref_image_path=ref_image_path,
            interactions=interactions,
            negative_prompt=negative_prompt,
            output_path=output_path,
            output_dir=output_dir,
            return_dict=return_dict,
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_type=output_type,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        if request.sampling.num_inference_steps != 2 or request.sampling.guidance_scale != 1.0:
            raise ValueError("Matrix-Game 4 Egypt requires two steps and guidance_scale=1")
        if request.sampling.scheduler_options["shift"] != 5.0:
            raise ValueError("Matrix-Game 4 Egypt requires scheduler shift=5")
        request = replace(request, inputs={**request.inputs, "fps": frame_rate})
        output["actual_fps"] = frame_rate
        return request, output

    def _format_output(self, output: DiffusionOutput, **kwargs: Any) -> Any:
        output = output.with_metadata(
            scene="ancient-egypt",
            fps=kwargs["actual_fps"],
            padded_frames=cast(int, output.artifacts.get("padded_frames", 0)),
        )
        output_path = kwargs.get("output_path")
        if output_path is not None and kwargs["output_type"] != "latent" and output.sample.shape[2] == 1:
            from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames, write_video

            path = str(output_path)
            if not Path(path).suffix:
                path += ".mp4"
            frames = video_tensor_to_uint8_frames(output.sample, value_range=self.OUTPUT_VALUE_RANGE, layout="bcthw")
            write_video(frames, path, fps=kwargs["actual_fps"])
            format_options: dict[str, Any] = {**kwargs, "output_path": None, "return_dict": True}
            result = super()._format_output(output, **format_options)
            result.update(artifact_path=path, generated_video_path=path)
            return result if kwargs["return_dict"] else path
        return super()._format_output(output, **kwargs)
