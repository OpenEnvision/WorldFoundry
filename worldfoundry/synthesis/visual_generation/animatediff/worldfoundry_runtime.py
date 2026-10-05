"""
This module defines the AnimateDiff in-process runtime, providing an interface for generating
animated videos from text prompts using the AnimateDiff model.

It includes functionality for managing model weights, integrating with the Diffusers
pipeline, and handling input/output for video generation.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldfoundry.core.io import file_sha256
from worldfoundry.evaluation.utils import REPO_ROOT, worldfoundry_data_path
from worldfoundry.runtime.env import resolve_hfd_root

# Default paths for AnimateDiff repository, integrated assets, configurations, and models.
DEFAULT_ANIMATEDIFF_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ANIMATEDIFF_INTEGRATED_ROOT = REPO_ROOT / "cache" / "generative_taxonomy" / "integrated" / "animatediff"
DEFAULT_ANIMATEDIFF_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "animatediff")
DEFAULT_SHARED_HFD_ROOT = resolve_hfd_root()
DEFAULT_SD15_ROOT = Path(
    os.environ.get(
        "WORLDFOUNDRY_SD15_ROOT",
        str(DEFAULT_SHARED_HFD_ROOT / "stable-diffusion-v1-5--stable-diffusion-v1-5"),
    )
)
DEFAULT_ANIMATEDIFF_MOTION_MODULE = DEFAULT_SHARED_HFD_ROOT / "guoyww--animatediff" / "mm_sd_v15_v2.ckpt"
DEFAULT_ANIMATEDIFF_V3_MOTION_MODULE = DEFAULT_SHARED_HFD_ROOT / "guoyww--animatediff" / "v3_sd15_mm.ckpt"
DEFAULT_ANIMATEDIFF_REALISTIC_VISION = (
    DEFAULT_SHARED_HFD_ROOT / "guoyww--animatediff_t2i_backups" / "realisticVisionV60B1_v51VAE.safetensors"
)
DEFAULT_ANIMATEDIFF_INFERENCE_CONFIG = DEFAULT_ANIMATEDIFF_CONFIG_ROOT / "inference" / "inference-v2.yaml"
DEFAULT_ANIMATEDIFF_HF_HUB_CACHE = DEFAULT_SHARED_HFD_ROOT / ".hf_home" / "hub"


class AnimateDiffRuntime:
    """
    AnimateDiff in-process runtime owned by synthesis.

    This class provides an interface to run the AnimateDiff model directly within
    the current process, leveraging the official AnimateDiff repository's
    `AnimationPipeline` for text-to-video generation.
    """

    def __init__(
        self,
        *,
        profile: Any,
        model_id: str,
        device: str,
        motion_module_path: str,
        base_model_path: str,
        dreambooth_model_path: str,
        official_python: str,
        hf_hub_cache: str,
        integrated_runtime_root: str,
        inference_config: str,
        negative_prompt: str = "",
    ) -> None:
        """
        Initializes the AnimateDiffRuntime with specified configurations and paths.

        Args:
            profile: A profile object containing runtime configuration details.
            model_id: Identifier for the AnimateDiff model.
            device: The compute device to use (e.g., "cuda:0", "cpu").
            motion_module_path: Path to the AnimateDiff motion module checkpoint.
            base_model_path: Path to the base Stable Diffusion model (e.g., SD 1.5).
            dreambooth_model_path: Path to the DreamBooth model or LoRA for fine-tuning.
            official_python: Path to the Python executable used by the official repo (if applicable).
            hf_hub_cache: Path to the Hugging Face Hub cache directory.
            integrated_runtime_root: Root directory for integrated runtime assets.
            inference_config: Path to the YAML configuration file for inference settings.
            negative_prompt: Default negative prompt to use during generation.
        """
        self.profile = profile
        self.model_id = model_id
        self.device = device
        self.motion_module_path = motion_module_path
        self.base_model_path = base_model_path
        self.dreambooth_model_path = dreambooth_model_path
        self.official_python = official_python
        self.hf_hub_cache = hf_hub_cache
        self.integrated_runtime_root = integrated_runtime_root
        self.inference_config = inference_config
        self.negative_prompt = negative_prompt
        self._pipe = None  # Internal variable to hold the initialized AnimateDiff pipeline

    @classmethod
    def from_synthesis(cls, synthesis: Any) -> "AnimateDiffRuntime":
        """
        Builds an AnimateDiffRuntime instance from a RuntimeProfileSynthesis adapter.

        This class method extracts configuration parameters from a synthesis object
        to construct a new AnimateDiffRuntime.

        Args:
            synthesis: An object containing all necessary parameters, typically from
                       a runtime profile synthesis process.

        Returns:
            An instance of AnimateDiffRuntime.
        """
        return cls(
            profile=synthesis.profile,
            model_id=synthesis.model_id,
            device=synthesis.device,
            motion_module_path=synthesis.motion_module_path,
            base_model_path=synthesis.base_model_path,
            dreambooth_model_path=synthesis.dreambooth_model_path,
            official_python=synthesis.official_python,
            hf_hub_cache=synthesis.hf_hub_cache,
            integrated_runtime_root=synthesis.integrated_runtime_root,
            inference_config=synthesis.inference_config,
            negative_prompt=synthesis.negative_prompt,
        )

    @staticmethod
    def file_sha256(path: Path) -> str:
        """
        Calculates the SHA256 hash of a file's contents.

        Args:
            path: The path to the file.

        Returns:
            A hexadecimal string representing the SHA256 hash.
        """
        return file_sha256(path)

    @staticmethod
    def frames_sha256(frames: Sequence[Any] | np.ndarray) -> str:
        """
        Calculates the SHA256 hash of a sequence of video frames.

        The hash is computed based on the shape and byte content of each frame,
        ensuring consistent hashing across different frame representations.

        Args:
            frames: A sequence of image frames (e.g., list of NumPy arrays or a single 4D NumPy array).

        Returns:
            A hexadecimal string representing the SHA256 hash of the frames.
        """
        digest = hashlib.sha256()
        for frame in frames:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                # Normalize float arrays to [0, 255] range and convert to uint8
                if np.issubdtype(array.dtype, np.floating):
                    if array.max(initial=0) <= 1.0:
                        array = np.clip(array * 255.0, 0, 255)
                    else:
                        array = np.clip(array, 0, 255)
                array = array.astype(np.uint8)
            # Ensure array is contiguous for tobytes() and update digest with shape and data
            array = np.ascontiguousarray(array)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def video_tensor_to_frames(video: Any) -> list[np.ndarray]:
        """
        Converts an AnimateDiff video tensor into a list of NumPy array frames.

        Expected tensor format is (batch, channels, time, height, width).

        Args:
            video: The video tensor from the AnimateDiff pipeline output.

        Returns:
            A list of NumPy arrays, where each array represents a frame in `uint8` format.

        Raises:
            ValueError: If the input tensor does not match the expected 5 dimensions
                        or has an incorrect channel count.
        """
        # Convert tensor to NumPy array, handling potential detach/cpu operations
        array = video.detach().cpu().float().numpy() if hasattr(video, "detach") else np.asarray(video)
        if array.ndim != 5:
            raise ValueError(f"expected official AnimateDiff video tensor with 5 dims, got {array.shape}")
        if array.shape[1] != 3:  # Check for (B, C, T, H, W) where C should be 3 (RGB)
            raise ValueError(f"expected b,c,t,h,w tensor with c=3, got {array.shape}")

        frames = []
        # Transpose to (time, batch, channels, height, width) and iterate over frames
        # Then, extract the first batch item, transpose to (H, W, C), and normalize to [0, 255] uint8
        for frame in np.transpose(array, (2, 0, 1, 3, 4)):
            frame = frame[0].transpose(1, 2, 0)  # Extract batch 0 and convert from (C, H, W) to (H, W, C)
            frames.append((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))
        return frames

    def ensure_pipe(self):
        """
        Ensures the AnimateDiff `AnimationPipeline` is initialized and ready.

        If the pipeline (`_pipe`) is not already created, this method
        initializes it by loading models, tokenizers, and schedulers,
        and then loads the specified motion module and DreamBooth weights.
        It also ensures the `src` directory of the repository is in `sys.path`
        for module imports.

        Returns:
            The initialized `AnimationPipeline` instance.

        Raises:
            ValueError: If `motion_module_path` is not provided.
            RuntimeError: If the device is set to 'cpu' as the official AnimateDiff
                          modules require CUDA.
        """
        if self._pipe is not None:
            return self._pipe
        if not self.motion_module_path:
            raise ValueError("AnimateDiffRuntime requires a local motion_module_path.")

        # Add the repository's 'src' directory to Python path to enable local module imports
        src_root = REPO_ROOT
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf
        from transformers import CLIPTextModel, CLIPTokenizer

        # Local imports from the AnimateDiff repository structure
        from .runtime.models.unet import UNet3DConditionModel
        from .runtime.pipelines.pipeline_animation import AnimationPipeline
        from .runtime.utils.util import load_weights

        # Determine the device; force CPU if CUDA is not available, but then raise error
        device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError("Official AnimateDiff repo path requires CUDA because upstream modules call .cuda().")

        # Load inference configuration from YAML
        inference_config = OmegaConf.load(self.inference_config)
        # Load tokenizer and text encoder from the base model path
        tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.base_model_path, subfolder="text_encoder").cuda()
        # Load VAE from the base model path
        vae = AutoencoderKL.from_pretrained(self.base_model_path, subfolder="vae").cuda()
        # Load 3D UNet model, configuring it with additional kwargs from inference config
        unet = UNet3DConditionModel.from_pretrained_2d(
            self.base_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),
        ).cuda()

        # Initialize the AnimateDiff AnimationPipeline
        pipe = AnimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        # Load motion module and DreamBooth/LoRA weights into the pipeline
        pipe = load_weights(
            pipe,
            motion_module_path=self.motion_module_path,
            dreambooth_model_path=self.dreambooth_model_path,
        ).to("cuda") # Ensure pipeline is moved to cuda after weights are loaded
        pipe = pipe.to(device) # Final move to the specified device
        self.device = device
        self._pipe = pipe
        return pipe

    @staticmethod
    def symlink_or_keep(link_path: Path, target_path: Path) -> None:
        """
        Creates a symbolic link if it doesn't exist.

        If `link_path` already exists (as a file, directory, or symlink),
        no action is taken. Otherwise, a symlink is created from `link_path`
        to `target_path`. Parent directories for `link_path` are created if needed.

        Args:
            link_path: The desired path for the symbolic link.
            target_path: The actual file or directory the link should point to.

        Raises:
            FileNotFoundError: If `target_path` does not exist.
        """
        if link_path.exists() or link_path.is_symlink():
            return
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target_path)

    def ensure_integrated_assets(self, runtime_root: Path) -> None:
        """
        Ensures that necessary AnimateDiff assets are symbolically linked
        into the specified runtime root directory.

        This helps organize assets for the official AnimateDiff repository structure.

        Args:
            runtime_root: The root directory where assets should be linked.
        """
        self.symlink_or_keep(runtime_root / "configs", DEFAULT_ANIMATEDIFF_CONFIG_ROOT)
        self.symlink_or_keep(
            runtime_root / "models" / "Motion_Module" / "mm_sd_v15_v2.ckpt",
            Path(self.motion_module_path).expanduser().resolve(),
        )
        self.symlink_or_keep(
            runtime_root / "models" / "Motion_Module" / "v3_sd15_mm.ckpt",
            DEFAULT_ANIMATEDIFF_V3_MOTION_MODULE,
        )
        self.symlink_or_keep(
            runtime_root / "models" / "DreamBooth_LoRA" / "realisticVisionV60B1_v51VAE.safetensors",
            DEFAULT_ANIMATEDIFF_REALISTIC_VISION,
        )

    def official_runtime_env(self) -> dict[str, str]:
        """
        Prepares the environment variables required for running the official
        AnimateDiff repository's scripts (e.g., via subprocess).

        This includes setting `PYTHONPATH` and Hugging Face Hub cache paths,
        and setting offline flags.

        Returns:
            A dictionary of environment variables.
        """
        env = os.environ.copy()
        pythonpath = [str(REPO_ROOT)]
        # Append existing PYTHONPATH if it exists
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        # Set Hugging Face cache directories, resolving and expanding paths
        env["HF_HOME"] = str(Path(self.hf_hub_cache).expanduser().resolve().parent)
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(self.hf_hub_cache).expanduser().resolve())
        # Set offline flags for Hugging Face components to prevent accidental network calls
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("DIFFUSERS_OFFLINE", "1")
        return env

    def run_official_config(
        self,
        config_path: str | Path,
        output_path: str | Path | None,
        *,
        timeout_seconds: int = 21600,
    ) -> dict[str, Any]:
        """
        Placeholder method for executing AnimateDiff via an official config file.

        This method is blocked in the current integration because direct subprocess
        execution of the official repository's scripts is deprecated in favor of
        the in-process `AnimationPipeline` execution.

        Args:
            config_path: Path to the official AnimateDiff configuration file.
            output_path: Desired output path for generated videos.
            timeout_seconds: Maximum time allowed for the execution.

        Raises:
            RuntimeError: Always, as this method is explicitly blocked.
        """
        del config_path, output_path, timeout_seconds # Acknowledge unused parameters
        raise RuntimeError(
            "AnimateDiff config execution is blocked because this path previously used a subprocess. "
            "Use the in-process AnimationPipeline path after staging the required checkpoints."
        )

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
        """
        Generates an animated video from a text prompt using the AnimateDiff pipeline.

        Args:
            prompt: The text prompt for video generation.
            images: Not supported in this integration; will raise ValueError if provided.
            video: Not supported in this integration; will raise ValueError if provided.
            interactions: A sequence of interaction strings (currently unused).
            output_path: Optional path to save the generated video. If None,
                         defaults to "animatediff.mp4" in the current working directory.
            fps: Frames per second for the output video. If None, defaults to 8.
            **kwargs: Additional parameters for the AnimateDiff pipeline, including:
                - `official_config_path`, `config_path`, `config`: (Deprecated) for legacy config execution.
                - `timeout_seconds`: (Deprecated) timeout for legacy config execution.
                - `seed`: Random seed for reproducibility (default: 1234).
                - `negative_prompt`: Negative prompt (overrides instance default).
                - `infer_steps`, `num_inference_steps`: Number of inference steps (default: 25).
                - `cfg_scale`, `guidance_scale`: Classifier-free guidance scale (default: 7.5).
                - `height`: Height of the generated video frames (default: 256).
                - `width`: Width of the generated video frames (default: 256).
                - `num_frames`, `video_length`: Number of frames in the generated video (default: 16).

        Returns:
            A dictionary containing details about the generated artifact, including
            status, model ID, artifact paths, hashes, and runtime parameters.

        Raises:
            ValueError: If `images` or `video` inputs are provided (as this is text-to-video).
            RuntimeError: If `official_config_path` is used, triggering the blocked legacy path.
        """
        del interactions # Acknowledge unused parameter
        if images is not None or video is not None:
            raise ValueError("AnimateDiff text-to-video does not accept image/video inputs in this integration.")

        # Check for legacy official config path and raise error if used
        official_config_path = (
            kwargs.pop("official_config_path", None)
            or kwargs.pop("config_path", None)
            or kwargs.pop("config", None)
        )
        if official_config_path:
            return self.run_official_config(
                official_config_path,
                output_path,
                timeout_seconds=int(kwargs.pop("timeout_seconds", 21600)),
            )

        import torch

        # Set manual seed for reproducibility of generation
        seed = int(kwargs.get("seed", 1234))
        torch.manual_seed(seed)
        pipe = self.ensure_pipe()

        # Call the AnimateDiff pipeline with parsed parameters
        sample = pipe(
            prompt,
            negative_prompt=str(kwargs.get("negative_prompt", self.negative_prompt)),
            num_inference_steps=int(kwargs.get("infer_steps", kwargs.get("num_inference_steps", 25))),
            guidance_scale=float(kwargs.get("cfg_scale", kwargs.get("guidance_scale", 7.5))),
            height=int(kwargs.get("height", 256)),
            width=int(kwargs.get("width", 256)),
            video_length=int(kwargs.get("num_frames", kwargs.get("video_length", 16))),
        ).videos

        # Determine and resolve the output path, ensuring parent directories exist
        target = Path(output_path) if output_path is not None else Path.cwd() / "animatediff.mp4"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # The evaluation runner historically proposed a GIF name for this
        # profile.  Normalize every direct inference artifact to MP4 so it is
        # a regular, frame-accurate video demo.
        if target.suffix.lower() != ".mp4":
            target = target.with_suffix(".mp4")

        from worldfoundry.core.media.codecs.video import save_videos_grid

        # Save the generated video tensor to a file
        save_videos_grid(sample, str(target), fps=fps or int(kwargs.get("fps", 8)))
        # Convert the video tensor to a list of frames for SHA256 hashing
        frames = self.video_tensor_to_frames(sample)

        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": self.profile.artifact_kind,
            "artifact_path": str(target),
            "artifact_size": target.stat().st_size,
            "frames_sha256": self.frames_sha256(frames),
            "video_sha256": file_sha256(target),
            "profile": self.profile.to_dict(),
            "runtime": "official_repo.AnimateDiff.AnimationPipeline",
            "motion_module_path": self.motion_module_path,
            "dreambooth_model_path": self.dreambooth_model_path,
            "device": self.device,
            "seed": seed,
        }
