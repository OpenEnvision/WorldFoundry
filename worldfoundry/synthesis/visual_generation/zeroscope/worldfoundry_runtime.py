"""A runtime module for the ZeroScope text-to-video model using Hugging Face Diffusers.

This module provides a class, `ZeroScopeRuntime`, to encapsulate the loading,
configuration, and execution of the ZeroScope text-to-video stable diffusion
pipeline. It handles model initialization, device management, and video generation
based on text prompts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldfoundry.core.io import file_sha256


class ZeroScopeRuntime:
    """Manages the ZeroScope diffusers pipeline for text-to-video generation.

    This class provides methods to initialize the ZeroScope model, ensure it's loaded
    onto the correct device, and generate video frames from a text prompt. It also
    includes utility methods for handling video frames and their checksums.
    """

    def __init__(
        self,
        *,
        profile: Any,
        model_id: str,
        device: str,
        model_path: str,
    ) -> None:
        """Initializes a new instance of the ZeroScopeRuntime.

        Args:
            profile: An object containing profile-specific information, such as artifact kind.
            model_id: The identifier for the ZeroScope model.
            device: The device to run the model on (e.g., 'cuda:0', 'cpu').
            model_path: The local path to the pretrained ZeroScope model.
        """
        self.profile = profile
        self.model_id = model_id
        self.device = device
        self.model_path = model_path
        self._pipe = None  # Stores the loaded diffusers pipeline instance.

    @classmethod
    def from_synthesis(cls, synthesis: Any) -> "ZeroScopeRuntime":
        """Creates a ZeroScopeRuntime instance from a synthesis object.

        This class method is a convenient constructor that extracts required parameters
        directly from a given synthesis object.

        Args:
            synthesis: An object containing `profile`, `model_id`, `device`, and `model_path` attributes.

        Returns:
            A new ZeroScopeRuntime instance.
        """
        return cls(
            profile=synthesis.profile,
            model_id=synthesis.model_id,
            device=synthesis.device,
            model_path=synthesis.model_path,
        )

    @staticmethod
    def frames_sha256(frames: Sequence[Any]) -> str:
        """Calculates the SHA256 hash of a sequence of video frames.

        This method normalizes frame data to `uint8` and then computes a hash
        based on the contiguous byte representation of the frames and their shapes.

        Args:
            frames: A sequence of video frames, typically numpy arrays or PIL Images.

        Returns:
            The SHA256 hexadecimal digest of the processed frames.
        """
        arrays = []
        for frame in frames:
            array = np.asarray(frame)
            # Ensure all frame data is of type uint8 for consistent hashing.
            if array.dtype != np.uint8:
                # If frame is floating point, scale to 0-255 if max is <= 1.0, otherwise clip.
                if np.issubdtype(array.dtype, np.floating):
                    if array.max(initial=0) <= 1.0:
                        array = np.clip(array * 255.0, 0, 255)
                    else:
                        array = np.clip(array, 0, 255)
                # Convert array to uint8.
                array = array.astype(np.uint8)
            # Ensure the array is contiguous for consistent tobytes() conversion.
            arrays.append(np.ascontiguousarray(array))

        digest = hashlib.sha256()
        for array in arrays:
            # Include shape in the hash to differentiate between different frame dimensions.
            digest.update(str(array.shape).encode("ascii"))
            # Update hash with the raw bytes of the array.
            digest.update(array.tobytes())
        return digest.hexdigest()

    def ensure_pipe(self):
        """Ensures the Diffusers pipeline is loaded and returns it.

        If the pipeline is not yet loaded, this method initializes it from the
        `model_path`, moves it to the appropriate device, and stores it for
        future use. It also handles compatibility workarounds for older PyTorch
        versions with Hugging Face Transformers.

        Raises:
            ValueError: If `model_path` is not set.

        Returns:
            The loaded TextToVideoSDPipeline instance.
        """
        if self._pipe is not None:
            return self._pipe
        if not self.model_path:
            raise ValueError("ZeroScopeRuntime requires a local model_path/pretrained_model_path.")

        import torch
        from diffusers import TextToVideoSDPipeline

        # Determine the actual device and data type based on availability.
        device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # Load the pipeline from the local path.
        pipe = TextToVideoSDPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            local_files_only=True,  # Ensure model is loaded only from local files.
            use_safetensors=True,
        )
        pipe = pipe.to(device)
        self.device = device  # Update the device in case it changed (e.g., from cuda to cpu).
        self._pipe = pipe
        return pipe

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generates a video from a text prompt using the ZeroScope model.

        Args:
            prompt: The text prompt to generate the video from.
            images: Not supported by ZeroScope. Raises ValueError if provided.
            video: Not supported by ZeroScope. Raises ValueError if provided.
            interactions: A sequence of interaction strings (unused for ZeroScope).
            output_path: The path to save the generated video. If None, uses a default path.
            fps: Frames per second for the output video. If None, uses a default value.
            **kwargs: Additional parameters passed to the diffusers pipeline,
                      e.g., `num_frames`, `infer_steps`, `cfg_scale`, `height`, `width`, `seed`.

        Raises:
            ValueError: If `images` or `video` inputs are provided.

        Returns:
            A dictionary containing generation status, model metadata, artifact path,
            and checksums for frames and the final video.
        """
        del interactions  # ZeroScope is text-to-video, interactions are not applicable here.
        if images is not None or video is not None:
            raise ValueError("ZeroScope is text-to-video and does not accept image/video inputs.")

        import torch
        from diffusers.utils import export_to_video

        pipe = self.ensure_pipe()
        seed = int(kwargs.get("seed", 1234))
        # Initialize generator with the specified seed for reproducibility.
        generator_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        # Run the text-to-video generation pipeline.
        frames = pipe(
            prompt,
            num_frames=int(kwargs.get("num_frames", 8)),
            num_inference_steps=int(kwargs.get("infer_steps", kwargs.get("num_inference_steps", 25))),
            guidance_scale=float(kwargs.get("cfg_scale", kwargs.get("guidance_scale", 9.0))),
            height=int(kwargs.get("height", 320)),
            width=int(kwargs.get("width", 576)),
            generator=generator,
        ).frames[0]

        # Determine the output path, defaulting to current working directory if not specified.
        target = Path(output_path) if output_path is not None else Path.cwd() / self.profile.artifact_filename
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist.

        # Export the generated frames to a video file.
        export_to_video(frames, str(target), fps=fps or int(kwargs.get("fps", 8)))

        # Calculate SHA256 hash of the generated video file.
        video_sha = file_sha256(target)

        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": self.profile.artifact_kind,
            "artifact_path": str(target),
            "frames_sha256": self.frames_sha256(frames),  # Hash of individual frames (normalized).
            "video_sha256": video_sha,  # Hash of the final video file.
            "profile": self.profile.to_dict(),
            "runtime": "diffusers.TextToVideoSDPipeline",
            "device": self.device,
            "seed": seed,
        }
