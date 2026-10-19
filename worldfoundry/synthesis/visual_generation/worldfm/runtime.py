"""
Module for interacting with the WorldFM tri-condition image generation model.

Provides an in-process Python runtime interface for WorldFM, allowing users to
load a pre-trained model from local assets and perform image generation based
on provided conditions.
"""

from __future__ import annotations

import random
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

from worldfoundry.core.media.processing.postprocess import VideoChunk

from .worldfm_runtime import DEFAULT_WORLDFM_REPO, resolve_worldfm_assets


def missing_worldfm_runtime_dependencies() -> list[str]:
    """
    Checks for the availability of essential Python packages required for WorldFM inference.

    Returns:
        A list of strings, where each string is the name of a missing package.
        Returns an empty list if all required packages are available.
    """

    required = ("diffusers", "torchvision", "PIL")
    # Identify which of the required packages are not importable
    return [name for name in required if find_spec(name) is None]


def load_worldfm_runtime(
    *,
    pretrained_model_path: str = DEFAULT_WORLDFM_REPO,
    device: str | None = None,
    vae_path: str | Path | None = None,
    checkpoint_filename: str | None = None,
    step: int = 2,
    image_size: int = 512,
    version: str = "sigma",
    cfg_scale: float = 4.5,
    weight_dtype: torch.dtype | None = None,
) -> "WorldFMRuntime":
    """
    Loads an in-process WorldFM runtime instance directly from local assets.

    This function simplifies the process of initializing a WorldFM model by
    handling the resolution of model assets and setting up the runtime
    configuration.

    Args:
        pretrained_model_path: The path to the directory containing the pretrained WorldFM model.
                               Defaults to the standard WorldFM repository path.
        device: The device to run the model on (e.g., "cuda" or "cpu"). If None,
                it will try to use CUDA if available, otherwise CPU.
        vae_path: The path to the VAE model. If None, it will be resolved from
                  `pretrained_model_path`.
        checkpoint_filename: The specific filename of the checkpoint to load
                             from `pretrained_model_path`. If None, a default
                             checkpoint will be used.
        step: The number of inference steps to perform.
        image_size: The target image resolution for generation.
        version: The version string of the WorldFM model.
        cfg_scale: Classifier-free guidance scale for inference.
        weight_dtype: The data type to use for model weights (e.g., torch.float16, torch.float32).

    Returns:
        An initialized `WorldFMRuntime` instance ready for predictions.
    """

    # Resolve paths for the model checkpoint and VAE from the provided sources
    assets = resolve_worldfm_assets(
        pretrained_model_path,
        vae_path,
        step=step,
        checkpoint_filename=checkpoint_filename,
        allow_missing=False,
    )

    # Instantiate WorldFMRuntime using the resolved asset paths and specified configuration
    return WorldFMRuntime.from_resolved_assets(
        service_assets={
            "checkpoint_path": str(assets.checkpoint_path),
            "vae_path": str(assets.vae_path),
        },
        device=device,
        step=step,
        image_size=image_size,
        version=version,
        cfg_scale=cfg_scale,
        weight_dtype=weight_dtype,
    )


class WorldFMRuntime:
    """
    Represents an in-process runtime for the WorldFM tri-condition image generation model.

    This class provides an interface to load the WorldFM model, manage its configuration,
    and perform predictions based on provided conditions. It encapsulates the underlying
    inference service and asset management.
    """

    def __init__(
        self,
        service: Any,
        checkpoint_path: str,
        vae_path: str,
        *,
        step: int = 2,
        image_size: int = 512,
        version: str = "sigma",
        cfg_scale: float = 4.5,
        device: str = "cuda",
    ) -> None:
        """
        Initializes a new instance of the WorldFMRuntime.

        Args:
            service: The underlying WorldFM inference service object (e.g., WorldFMTriConditionInprocess).
            checkpoint_path: The file path to the main WorldFM model checkpoint.
            vae_path: The file path to the VAE model used by WorldFM.
            step: The number of inference steps to perform.
            image_size: The target image resolution for generation.
            version: The version string of the WorldFM model.
            cfg_scale: Classifier-free guidance scale for inference.
            device: The device on which the model is loaded and run (e.g., "cuda", "cpu").
        """
        self.service = service
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path
        self.step = int(step)
        self.image_size = int(image_size)
        self.version = str(version)
        self.cfg_scale = float(cfg_scale)
        self.device = device

    @staticmethod
    def resolve_assets(
        checkpoint_source: str | Path | None,
        vae_source: str | Path | None,
        *,
        step: int,
        checkpoint_filename: str | None,
    ) -> tuple[str, str]:
        """
        Resolves local WorldFM model assets (checkpoint and VAE paths).

        This static method is used internally and externally to determine the
        absolute paths for the WorldFM checkpoint and VAE from various input sources.

        Args:
            checkpoint_source: The source path for the WorldFM model checkpoint,
                               which can be a directory or a specific file.
            vae_source: The source path for the VAE model.
            step: The inference step count, used in asset resolution logic.
            checkpoint_filename: A specific filename for the checkpoint to use,
                                 if `checkpoint_source` is a directory.

        Returns:
            A tuple containing the absolute path to the WorldFM checkpoint and
            the absolute path to the VAE model, both as strings.
        """

        # Delegate asset resolution to the external utility function
        assets = resolve_worldfm_assets(
            checkpoint_source,
            vae_source,
            step=step,
            checkpoint_filename=checkpoint_filename,
        )
        return str(assets.checkpoint_path), str(assets.vae_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_WORLDFM_REPO,
        *,
        device: str | None = None,
        vae_path: str | Path | None = None,
        checkpoint_filename: str | None = None,
        step: int = 2,
        image_size: int = 512,
        version: str = "sigma",
        cfg_scale: float = 4.5,
        weight_dtype: torch.dtype | None = None,
    ) -> "WorldFMRuntime":
        """
        Factory method to load WorldFM from local assets, resolving paths as needed.

        This method handles the complete process of resolving model and VAE paths
        and then instantiating the `WorldFMRuntime` using these resolved paths.

        Args:
            pretrained_model_path: The path to the directory containing the pretrained WorldFM model.
                                   Defaults to the standard WorldFM repository path.
            device: The device to run the model on (e.g., "cuda" or "cpu"). If None,
                    it will try to use CUDA if available, otherwise CPU.
            vae_path: The path to the VAE model. If None, it will be resolved from
                      `pretrained_model_path`.
            checkpoint_filename: The specific filename of the checkpoint to load
                                 from `pretrained_model_path`. If None, a default
                                 checkpoint will be used.
            step: The number of inference steps to perform.
            image_size: The target image resolution for generation.
            version: The version string of the WorldFM model.
            cfg_scale: Classifier-free guidance scale for inference.
            weight_dtype: The data type to use for model weights (e.g., torch.float16, torch.float32).

        Returns:
            An initialized `WorldFMRuntime` instance.
        """

        # Resolve checkpoint and VAE paths using the static asset resolution method
        checkpoint_path, resolved_vae_path = cls.resolve_assets(
            pretrained_model_path,
            vae_path,
            step=step,
            checkpoint_filename=checkpoint_filename,
        )
        # Delegate to from_resolved_assets for actual service instantiation
        return cls.from_resolved_assets(
            service_assets={
                "checkpoint_path": checkpoint_path,
                "vae_path": resolved_vae_path,
            },
            device=device,
            step=step,
            image_size=image_size,
            version=version,
            cfg_scale=cfg_scale,
            weight_dtype=weight_dtype,
        )

    @classmethod
    def from_resolved_assets(
        cls,
        *,
        service_assets: Mapping[str, str],
        device: str | None = None,
        step: int = 2,
        image_size: int = 512,
        version: str = "sigma",
        cfg_scale: float = 4.5,
        weight_dtype: torch.dtype | None = None,
    ) -> "WorldFMRuntime":
        """
        Factory method to load the in-process WorldFM service from already resolved checkpoint paths.

        This method is responsible for instantiating the core WorldFM inference
        service after all asset paths have been definitively determined.

        Args:
            service_assets: A mapping containing "checkpoint_path" and "vae_path" with their string values.
            device: The device to run the model on (e.g., "cuda" or "cpu"). If None,
                    it will try to use CUDA if available, otherwise CPU.
            step: The number of inference steps to perform.
            image_size: The target image resolution for generation.
            version: The version string of the WorldFM model.
            cfg_scale: Classifier-free guidance scale for inference.
            weight_dtype: The data type to use for model weights (e.g., torch.float16, torch.float32).

        Returns:
            An initialized `WorldFMRuntime` instance.

        Raises:
            RuntimeError: If essential WorldFM runtime dependencies are missing.
        """

        # Check for mandatory dependencies before attempting to load the service
        missing_dependencies = missing_worldfm_runtime_dependencies()
        if missing_dependencies:
            raise RuntimeError("Missing WorldFM runtime dependencies: " + ", ".join(missing_dependencies))

        checkpoint_path = str(service_assets["checkpoint_path"])
        resolved_vae_path = str(service_assets["vae_path"])

        # Determine the effective device for model execution
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Fallback to CPU if CUDA is explicitly requested but not available
        if not str(resolved_device).startswith("cuda") or not torch.cuda.is_available():
            resolved_device = "cpu"

        # Determine the appropriate weight data type based on the resolved device
        if weight_dtype is None:
            weight_dtype = torch.float16 if str(resolved_device).startswith("cuda") else torch.float32

        # Import the WorldFM inference service classes dynamically
        from .worldfm_runtime.inference import WorldFMInprocessConfig, WorldFMTriConditionInprocess

        # Configure and instantiate the WorldFM tri-condition in-process service
        service = WorldFMTriConditionInprocess(
            WorldFMInprocessConfig(
                model_path=checkpoint_path,
                vae_path=resolved_vae_path,
                image_size=image_size,
                version=version,
                disable_cross_attn=True,  # This seems to be a fixed parameter for this setup
                step=int(step),
                mid_t=200,  # This seems to be a fixed parameter for this setup
                cfg_scale=float(cfg_scale),
                device=str(resolved_device),
                weight_dtype=weight_dtype,
            )
        )

        # Return a new WorldFMRuntime instance with the configured service
        return cls(
            service=service,
            checkpoint_path=checkpoint_path,
            vae_path=resolved_vae_path,
            step=step,
            image_size=image_size,
            version=version,
            cfg_scale=cfg_scale,
            device=str(resolved_device),
        )

    def _decode_render(self, render_rgb_u8: Any, cond_nearest_rgb: np.ndarray) -> np.ndarray:
        """
        Decodes a rendered RGB image and applies a nearest neighbor condition
        to generate a new image using the WorldFM service.

        Args:
            render_rgb_u8: The initial rendered RGB image (uint8), which can be a
                           torch.Tensor or a numpy array.
            cond_nearest_rgb: The nearest neighbor RGB image condition as a numpy array (uint8).

        Returns:
            A numpy array representing the generated RGB image (uint8).
        """
        device = self.service.device
        # Convert input render image to a torch.uint8 tensor on the specified device
        if isinstance(render_rgb_u8, torch.Tensor):
            render_u8 = render_rgb_u8.to(device=device, dtype=torch.uint8)
        else:
            render_u8 = torch.from_numpy(np.asarray(render_rgb_u8, dtype=np.uint8)).to(
                device=device,
                dtype=torch.uint8,
            )

        # Ensure the condition image is a numpy array (uint8) and set it in the service
        cond_nearest_rgb = np.asarray(cond_nearest_rgb, dtype=np.uint8)
        self.service.set_cond2_from_array(cond_nearest_rgb)

        # Perform inference based on the configured step count
        if self.step in (1, 2):
            decoded = self.service.infer_from_render_u8(render_u8)
        else:
            decoded = self.service.infer_from_render_u8_multistep(
                render_u8,
                sample_steps=self.step,
                cfg_scale=self.cfg_scale,
            )

        # Post-process the decoded tensor: clamp, scale, permute dimensions, convert to uint8, and move to CPU
        return (
            torch.clamp(127.5 * decoded[0] + 128.0, 0, 255)  # Scale from [-1, 1] to [0, 255] and clamp
            .permute(1, 2, 0)  # Change from CHW to HWC format
            .to(torch.uint8)  # Convert to unsigned 8-bit integer
            .cpu()
            .numpy()  # Move to CPU and convert to NumPy array
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Sets the random seed for various libraries to ensure reproducibility.

        This affects `random`, `numpy`, and `torch` (both CPU and CUDA if available).

        Args:
            seed: The integer seed value to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Set CUDA specific seed if a GPU is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _save_outputs(
        frames: Sequence[np.ndarray],
        *,
        output_dir: str | Path | None,
        scene_name: str,
        save_mode: str,
        fps: int,
    ) -> Dict[str, Any]:
        """
        Saves a sequence of generated frames either as a video or individual images.

        Args:
            frames: A sequence of NumPy arrays, where each array represents an image frame.
            output_dir: The base directory where the output should be saved. If None,
                        no files will be saved.
            scene_name: A sub-directory name within `output_dir` to organize outputs for the current scene.
            save_mode: The desired save mode, either "image" (for individual PNGs) or "video" (for an MP4).
            fps: Frames per second for video saving mode.

        Returns:
            A dictionary containing paths to the generated video and/or images.
        """
        # Return empty paths if no output directory is specified
        if output_dir is None:
            return {
                "generated_video_path": None,
                "generated_image_paths": [],
            }

        # Resolve and create the scene-specific output directory
        scene_dir = Path(output_dir).expanduser().resolve() / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        if save_mode == "image":
            image_paths = []
            # Save each frame as a separate PNG image
            for index, frame in enumerate(frames):
                file_name = "output.png" if len(frames) == 1 else f"output_{index:04d}.png"
                target = scene_dir / file_name
                imageio.imwrite(target, frame)
                image_paths.append(str(target))
            return {
                "generated_video_path": None,
                "generated_image_paths": image_paths,
            }

        # Save all frames as a single MP4 video
        video_path = scene_dir / "output.mp4"
        imageio.mimsave(video_path, frames, fps=int(fps))
        return {
            "generated_video_path": str(video_path),
            "generated_image_paths": [],
        }

    @torch.inference_mode()
    def predict(
        self,
        frame_conditions: Sequence[Dict[str, Any]],
        output_dir: str | Path | None = None,
        scene_name: str = "worldfm_scene",
        save_mode: str = "video",
        fps: int = 30,
        return_dict: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[np.ndarray]:
        """
        Generates a sequence of images or a video based on provided frame conditions.

        This is the main inference method of the WorldFMRuntime. It processes each
        condition pair (render_rgb_u8, cond_nearest_rgb) to generate an output frame.

        Args:
            frame_conditions: A sequence of dictionaries, where each dictionary
                              represents conditions for a single frame. Each dict
                              must contain 'render_rgb_u8' (initial render image)
                              and 'cond_nearest_rgb' (nearest neighbor condition).
            output_dir: The base directory to save the generated output. If None,
                        outputs will not be saved to disk.
            scene_name: The name of the scene, used to create a dedicated subdirectory
                        within `output_dir` for organizing outputs.
            save_mode: The method to save outputs: "video" (default) for an MP4,
                       or "image" for individual PNG frames.
            fps: Frames per second, used when `save_mode` is "video".
            return_dict: If True, returns a dictionary containing frames and paths.
                         If False, returns only the list of generated NumPy frames.
            seed: An optional integer seed for reproducibility. If provided,
                  random states for `random`, `numpy`, and `torch` will be set.
            **kwargs: Additional keyword arguments, currently ignored (`del kwargs`).

        Returns:
            A dictionary containing generated frames, video path, and image paths
            if `return_dict` is True, otherwise a list of generated NumPy image arrays.

        Raises:
            ValueError: If `frame_conditions` is empty.
        """
        del kwargs  # Discard any unexpected keyword arguments

        # Validate input: ensure at least one condition is provided
        if not frame_conditions:
            raise ValueError("WorldFM synthesis requires at least one rendered condition pair.")

        # Set the random seed for reproducibility if provided
        if seed is not None:
            self._set_seed(int(seed))

        # Generate each frame by decoding the render with its corresponding condition
        frames = [
            self._decode_render(item["render_rgb_u8"], item["cond_nearest_rgb"])
            for item in frame_conditions
        ]

        # Save the generated frames to disk based on the specified output options
        saved_outputs = self._save_outputs(
            frames,
            output_dir=output_dir,
            scene_name=scene_name,
            save_mode=save_mode,
            fps=fps,
        )

        # Construct the result dictionary including generated frames and file paths
        result = {
            "frames": frames,  # All generated frames as a list of NumPy arrays
            "video": frames,  # Alias for frames, for compatibility
            "video_chunk": VideoChunk(frames=frames, layout="frame-list", value_range="0,255"),
            "scene_name": scene_name,
            "save_mode": save_mode,
            **saved_outputs,  # Include paths from _save_outputs
        }
        # Return either the full result dictionary or just the list of frames
        if return_dict:
            return result
        return frames