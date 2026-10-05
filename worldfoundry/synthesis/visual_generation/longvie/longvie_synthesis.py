"""
WorldFoundry synthesis wrapper for LongVie, a control-video generation model.

This module provides a specialized synthesis class for interacting with the
LongVie model, allowing users to generate videos conditioned on text prompts,
initial images, and control videos (e.g., depth maps, sparse tracking).
It integrates with the WorldFoundry framework, offering a standardized
interface for model execution and artifact handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io import file_sha256
from worldfoundry.synthesis.visual_generation.longvie.worldfoundry_runtime import (
    LONGVIE_NEGATIVE_PROMPT,
    TARGET_SIZE,
    LongVieOfficialRuntime,
    first_present,
    pick,
    pop_first,
    to_rgb_image,
    video_to_frames,
)

from ...base_synthesis import BaseSynthesis

# Keys that are typically handled by the WorldFoundry framework and should be removed
# from model-specific options to prevent unexpected arguments.
FRAMEWORK_KEYS = {
    "acquisition_root",
    "hf_models_root",
    "manifest_path",
    "model_id",
    "pipeline_target",
    "profile_id",
    "profile_path",
}

LONGVIE_SEGMENT_FRAMES = 81
LONGVIE_SEGMENT_FPS = 16
LONGVIE_HISTORY_FRAMES = 8


class LongVieSynthesis(BaseSynthesis):
    """WorldFoundry synthesis wrapper for LongVie control-video generation.

    This class provides an interface to the LongVie model for video generation,
    inheriting from `BaseSynthesis` to fit into the WorldFoundry evaluation framework.
    It manages the LongVie runtime and handles input/output processing for video
    synthesis tasks.
    """

    MODEL_ID = "longvie-1"

    def __init__(
        self,
        *,
        model_id: str = "longvie-1",
        runtime: LongVieOfficialRuntime,
        execute_by_default: bool = False,
    ) -> None:
        """Initializes the LongVieSynthesis wrapper.

        Args:
            model_id: The identifier for the LongVie model variant. Defaults to "longvie-1".
            runtime: An initialized `LongVieOfficialRuntime` instance to perform the actual generation.
            execute_by_default: If True, the `predict` method will execute the generation by default.
                                If False, it would typically return a plan, but LongVie requires immediate
                                execution.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = model_id
        self.generation_type = "i2v"  # Indicates Image-to-Video generation capability
        self.runtime = runtime
        self.execute_by_default = bool(execute_by_default)
        self.history: list[Any] = []  # Stores a list of generated frames for memory/continuity
        self.noise: Any = None  # Stores the noise tensor for memory/continuity
        self.last_frame: Any = None
        self.segment_index = 0

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LongVieSynthesis":
        """Loads a LongVie synthesis model from a pretrained path or options.

        This factory method configures and initializes a `LongVieOfficialRuntime`
        based on the provided paths and keyword arguments, then wraps it in a
        `LongVieSynthesis` instance.

        Args:
            pretrained_model_path: Path to the pretrained model weights or a mapping
                                   of options.
            args: Placeholder for additional arguments (currently ignored).
            device: The device to load the model onto (e.g., "cuda", "cpu").
            model_id: The specific model variant ID to load (e.g., "longvie-1", "longvie-2").
            **kwargs: Additional options for configuring the LongVie runtime, such as
                      `longvie_weight_dir`, `control_weight_path`, `torch_dtype`, etc.

        Returns:
            An initialized `LongVieSynthesis` instance ready for video generation.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        del args  # `args` is not used in this factory method.
        # Initialize options from `pretrained_model_path` if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If `pretrained_model_path` is a simple path, use it as the `longvie_weight_dir`.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["longvie_weight_dir"] = str(pretrained_model_path)
        options.update(kwargs)

        requested_model_id = options.pop("model_id", None)

        # Remove framework-specific keys from the options to avoid passing them to the runtime.
        for key in FRAMEWORK_KEYS:
            options.pop(key, None)

        # Determine the model ID, prioritizing 'variant', then 'model_id', then the class default.
        resolved_model_id = str(
            options.pop("variant", None) or requested_model_id or model_id or cls.MODEL_ID
        )
        # Normalize model ID aliases to standard forms.
        if resolved_model_id in {"longvie2", "longvie-v2"}:
            resolved_model_id = "longvie-2"
        if resolved_model_id in {"longvie", "longvie1", "longvie-v1"}:
            resolved_model_id = "longvie-1"

        # Extract weight directories, handling common aliases.
        weight_dir = options.pop("longvie_weight_dir", options.pop("weight_dir", None))
        # Initialize the official LongVie runtime with all resolved and provided options.
        use_usp = bool(options.pop("use_usp", False))
        ring_degree = int(options.pop("ring_degree", 1))
        ulysses_degree = int(options.pop("ulysses_degree", 1))
        if resolved_model_id == "longvie-2":
            try:
                world_size = max(int(os.getenv("WORLD_SIZE", "1") or "1"), 1)
            except ValueError as exc:
                raise ValueError("LongVie 2 requires an integer WORLD_SIZE.") from exc
            if world_size not in {1, 4}:
                raise ValueError(
                    "LongVie 2 supports either one GPU or its official four-rank USP topology; "
                    f"got WORLD_SIZE={world_size}."
                )
            use_usp = world_size == 4
            ring_degree = 1
            ulysses_degree = world_size

        runtime = LongVieOfficialRuntime(
            control_weight_path=options.pop("control_weight_path", None),
            dit_weight_path=options.pop("dit_weight_path", None),
            weight_dir=weight_dir,
            wan_base_dir=options.pop("wan_base_dir", options.pop("base_model_path", None)),
            tokenizer_dir=options.pop("tokenizer_dir", options.pop("tokenizer_path", None)),
            device=str(device or options.pop("device", "cuda")),
            torch_dtype=str(options.pop("torch_dtype", "bfloat16")),
            use_usp=use_usp,
            ring_degree=ring_degree,
            ulysses_degree=ulysses_degree,
            enable_vram_management=bool(options.pop("enable_vram_management", True)),
            control_layers=int(options.pop("control_layers", 12)),
            variant=resolved_model_id,
        )
        # Return a new LongVieSynthesis instance with the configured runtime.
        return cls(
            model_id=resolved_model_id,
            runtime=runtime,
            execute_by_default=bool(options.pop("execute_by_default", False)),
        )

    @staticmethod
    def _normalize_num_frames(num_frames: int) -> int:
        """Normalizes the requested number of frames to meet LongVie's requirements.

        LongVie typically expects a specific structure for the number of frames.
        This method ensures the `num_frames` is positive, at least 5, and if
        greater than 5, it's adjusted to be 1 plus a multiple of 4 (i.e., `4n + 1`).

        Args:
            num_frames: The desired number of frames.

        Returns:
            The normalized number of frames.

        Raises:
            ValueError: If `num_frames` is not positive.
        """
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError(f"LongVie num_frames must be positive, got {num_frames}.")
        if num_frames < 5:
            return 5
        # If num_frames is already of the form 4n + 1, return as is.
        if num_frames % 4 == 1:
            return num_frames
        # Otherwise, adjust to the nearest 4n + 1 value, ensuring it's at least 5.
        return max(5, ((num_frames - 1) // 4) * 4 + 1)

    @staticmethod
    def _fit_control_frames(
        frames: list[Any],
        num_frames: int,
        *,
        control_name: str = "control video",
        allow_padding: bool = False,
    ) -> list[Any]:
        """Adjusts the list of control frames to match the target `num_frames`.

        Full-quality LongVie segments need one control frame per generated
        frame. Short controls therefore fail by default instead of silently
        freezing the last depth/track frame for the remainder of the segment.
        Padding remains available as an explicit compatibility opt-in.

        Args:
            frames: A list of control frames (e.g., depth images, sparse tracks).
            num_frames: The desired number of frames for the output video.

        Returns:
            A new list of control frames adjusted to `num_frames`.

        Raises:
            ValueError: If the input `frames` list is empty.
        """
        if not frames:
            raise ValueError(f"LongVie {control_name} must contain at least one frame.")
        if len(frames) < num_frames and not allow_padding:
            raise ValueError(
                f"LongVie {control_name} has {len(frames)} frame(s), but the requested "
                f"segment needs {num_frames}. Provide a complete control-video segment "
                "or explicitly set allow_control_padding=True."
            )
        # Take up to `num_frames` from the input list.
        fitted = list(frames[:num_frames])
        # If the fitted list is shorter than required, pad by repeating the last frame.
        if len(fitted) < num_frames:
            fitted.extend([fitted[-1]] * (num_frames - len(fitted)))
        return fitted

    @staticmethod
    def _segment_spec(*, fps: int, num_frames: int, segment_index: int) -> dict[str, Any]:
        """Describe LongVie as queued segment generation, not live key control."""

        return {
            "mode": "queued_control_video_segments",
            "realtime": False,
            "keyboard_controls": False,
            "fps": int(fps),
            "segment_frames": int(num_frames),
            # The first frame is the segment boundary/anchor, so 81 frames at
            # 16 FPS span the official five-second (80 interval) timeline.
            "segment_seconds": float(max(num_frames - 1, 0)) / float(fps),
            "encoded_duration_seconds": float(num_frames) / float(fps),
            "history_frames": LONGVIE_HISTORY_FRAMES,
            "segment_index": int(segment_index),
            "required_inputs": [
                "prompt",
                "initial_image_for_first_segment",
                "dense_depth_video_for_each_segment",
                "sparse_pointmap_or_track_video_for_each_segment",
            ],
            "continuation": "previous_final_frame_plus_last_8_frames_plus_noise",
        }

    def predict(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = 16,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generates a video using the LongVie model based on a prompt and control signals.

        Args:
            prompt: The text prompt describing the desired video content.
            images: An initial image (first frame) for conditioning. Can be a path, URL, or image object.
            video: A video source for control signals (e.g., depth video, sparse track video).
                   Can be a path, URL, video object, or a mapping containing 'dense_video'/'sparse_video'.
            interactions: An alternative source for control signals, similar to `video`.
            output_path: Optional path to save the generated video artifact.
            fps: Frames per second for the output video. Defaults to 16.
            return_dict: If True, returns a dictionary containing generation details and artifacts.
                         If False, returns the generated video frames directly.
            **kwargs: Additional parameters for the generation process, such as:
                      `seed`, `tiled`, `height`, `width`, `num_frames`, `negative_prompt`,
                      `history`, `noise`, `update_memory`, `target_size`, etc.

        Returns:
            The generated video (list of frames) or a dictionary with generation details
            and artifact path, depending on `return_dict`.

        Raises:
            RuntimeError: If `execute` is False, as LongVie requires immediate execution.
            ValueError: If control frames are missing when expected.
        """
        # Extract dense and sparse video controls from various input sources,
        # prioritizing kwargs, then `video` mapping, then `interactions` mapping.
        dense_video = pop_first(kwargs, "dense_video", "depth_video", "depth")
        sparse_video = pop_first(kwargs, "sparse_video", "track_video", "track", "pointmap_video")
        if isinstance(video, Mapping):
            dense_video = first_present(dense_video, pick(video, "dense_video", "depth", "depth_video"))
            sparse_video = first_present(
                sparse_video,
                pick(video, "sparse_video", "track", "track_video", "pointmap_video"),
            )
        if isinstance(interactions, Mapping):
            dense_video = first_present(dense_video, pick(interactions, "dense_video", "depth", "depth_video"))
            sparse_video = first_present(
                sparse_video,
                pick(interactions, "sparse_video", "track", "track_video", "pointmap_video"),
            )

        # Check if execution is explicitly requested; LongVie requires immediate execution.
        execute = bool(kwargs.pop("execute", self.execute_by_default))
        continue_from_memory = bool(kwargs.pop("continue_from_memory", False))
        # Determine the input image, prioritizing kwargs then the 'images' argument.
        supplied_image = first_present(pop_first(kwargs, "input_image", "image", "first_frame"), images)
        input_image = self.last_frame if continue_from_memory and self.last_frame is not None else supplied_image
        if not execute:
            raise RuntimeError("LongVie requires execute=True; request-plan artifacts are no longer emitted.")

        missing: list[str] = []
        if not str(prompt or "").strip():
            missing.append("a non-empty prompt")
        if input_image is None:
            missing.append("an initial image")
        if dense_video is None:
            missing.append("a dense depth control video")
        if sparse_video is None:
            missing.append("a sparse pointmap/track control video")
        if missing:
            raise ValueError(
                "LongVie segment generation requires user-provided conditioning. Missing: "
                + ", ".join(missing)
                + ". Every continuation segment needs new dense and sparse control videos; "
                "the initial image is reused automatically from the previous segment's final frame."
            )

        # Determine target size and convert input image and control videos to frames.
        target_size = tuple(kwargs.pop("target_size", TARGET_SIZE))
        input_image = to_rgb_image(input_image, target_size=target_size)
        dense_frames = video_to_frames(dense_video, target_size=target_size)
        sparse_frames = video_to_frames(sparse_video, target_size=target_size)

        # Normalize the number of frames for generation and fit control frames to this count.
        requested_frames = kwargs.pop("num_frames", None)
        num_frames = self._normalize_num_frames(
            requested_frames if requested_frames is not None else LONGVIE_SEGMENT_FRAMES
        )
        allow_control_padding = bool(kwargs.pop("allow_control_padding", False))
        dense_frames = self._fit_control_frames(
            dense_frames,
            num_frames,
            control_name="dense depth control video",
            allow_padding=allow_control_padding,
        )
        sparse_frames = self._fit_control_frames(
            sparse_frames,
            num_frames,
            control_name="sparse pointmap/track control video",
            allow_padding=allow_control_padding,
        )

        # Retrieve history and noise from kwargs or instance memory.
        memory_history = self.history if continue_from_memory else []
        memory_noise = self.noise if continue_from_memory else None
        history = list(kwargs.pop("history", memory_history) or [])[-LONGVIE_HISTORY_FRAMES:]
        noise = kwargs.pop("noise", memory_noise)
        negative_prompt = kwargs.pop("negative_prompt", LONGVIE_NEGATIVE_PROMPT)
        update_memory = bool(kwargs.pop("update_memory", True))

        # Remove framework-specific path keys that are not directly used by the runtime.
        for framework_key in ("image_path", "video_path", "input_path", "output_path"):
            kwargs.pop(framework_key, None)

        # Perform the actual video generation using the LongVie runtime.
        generated, next_noise = self.runtime.generate_segment(
            input_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(kwargs.pop("seed", 0)),
            tiled=bool(kwargs.pop("tiled", False)),
            height=int(kwargs.pop("height", target_size[1])),
            width=int(kwargs.pop("width", target_size[0])),
            num_frames=num_frames,
            dense_video=dense_frames,
            sparse_video=sparse_frames,
            history=history,
            noise=noise,
            **kwargs,
        )

        if generated is None or len(generated) == 0:
            raise RuntimeError("LongVie official runtime returned an empty video segment.")

        # Update the model's internal memory (history and noise) if `update_memory` is True.
        if update_memory:
            self.history = list(generated[-LONGVIE_HISTORY_FRAMES:])
            self.noise = next_noise
            self.last_frame = generated[-1]
            self.segment_index += 1

        artifact_path = None
        # Save the generated video if an output path is provided.
        if output_path is not None and self.runtime.is_output_rank():
            artifact = self.runtime.save_video(generated, output_path, fps=int(fps or 16), quality=10)
            artifact_path = str(artifact)

        effective_fps = int(fps or LONGVIE_SEGMENT_FPS)
        segment_index = self.segment_index if update_memory else self.segment_index + 1
        segment_spec = self._segment_spec(
            fps=effective_fps,
            num_frames=len(generated),
            segment_index=segment_index,
        )

        # Prepare the result dictionary with metadata and artifact information.
        result = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": artifact_path,
            # Calculate SHA256 hash of the saved artifact if available.
            "artifact_sha256": file_sha256(artifact_path) if artifact_path else None,
            "runtime": "worldfoundry.native_diffusion.wan.longvie",
            "backend_quality": "native_inference",
            "metadata": {
                "fps": effective_fps,
                "frames": len(generated),
                "target_size": list(target_size),
                "use_usp": self.runtime.use_usp,
                "ring_degree": self.runtime.ring_degree,
                "ulysses_degree": self.runtime.ulysses_degree,
                "interaction_mode": "queued_control_video_segments",
                "resident_weights": self.runtime.loaded,
                "history_frames": len(self.history),
                "continued_from_previous_final_frame": bool(
                    continue_from_memory and supplied_image is not input_image
                ),
            },
            "video": generated,
            "noise": next_noise,
            "last_frame": generated[-1],
            "segment_spec": segment_spec,
            # RealtimeSpec-shaped metadata lets Studio display the correct
            # 81-frame cadence while the explicit transport/mode prevents this
            # offline diffusion workflow from being presented as WASD realtime.
            "realtime_spec": {
                "fps": effective_fps,
                "first_chunk_frames": len(generated),
                "steady_chunk_frames": len(generated),
                "controls": ["dense_depth_video", "sparse_pointmap_or_track_video"],
                "transport": "queued-segment-rgb",
                "stateful": True,
            },
        }
        if return_dict:
            return result
        return generated

    def reset_memory(self) -> None:
        """Resets the model's internal memory (history and noise).

        This clears the history of previously generated frames and the noise tensor,
        effectively starting a new generation segment without continuity from previous calls.
        """
        self.history = []
        self.noise = None
        self.last_frame = None
        self.segment_index = 0
