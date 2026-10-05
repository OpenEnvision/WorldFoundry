"""
This module provides a runtime interface for MotionCtrl, a model designed for video generation with
camera and/or object motion conditioning. It includes functionality for loading the MotionCtrl model,
handling its configuration, applying necessary patches for dependencies like OpenCLIP, and
performing video generation based on text prompts and motion conditions.

The module supports both official MotionCtrl repository integration (if available) and
vendored configurations and checkpoints provided within the WorldFoundry framework.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.evaluation.models.runtime.profiles import DEFAULT_SHARED_HFD_ROOT
from worldfoundry.evaluation.utils import worldfoundry_data_path


def _find_official_motionctrl_root() -> Path | None:
    """
    Placeholder function to find the root directory of an official MotionCtrl repository.
    Currently, this implementation does not locate an external repository and always returns None.
    """
    return None


# Path to the root of the official MotionCtrl repository, if found.
OFFICIAL_MOTIONCTRL_ROOT = _find_official_motionctrl_root()
# Path to the official MotionCtrl inference configuration file, if the root is found.
OFFICIAL_MOTIONCTRL_CONFIG = (
    OFFICIAL_MOTIONCTRL_ROOT / "configs" / "inference" / "config_both.yaml"
    if OFFICIAL_MOTIONCTRL_ROOT is not None
    else None
)
# Path to the MotionCtrl configuration file vendored within WorldFoundry data.
WORLDFOUNDRY_MOTIONCTRL_CONFIG = worldfoundry_data_path(
    "models",
    "runtime", "configs",
    "camera_control",
    "motionctrl_config_both.yaml",
)
# The default configuration path for MotionCtrl, preferring the official config if it exists.
DEFAULT_MOTIONCTRL_CONFIG = (
    OFFICIAL_MOTIONCTRL_CONFIG
    if OFFICIAL_MOTIONCTRL_CONFIG is not None and OFFICIAL_MOTIONCTRL_CONFIG.exists()
    else WORLDFOUNDRY_MOTIONCTRL_CONFIG
)
# Default directory for MotionCtrl condition files (e.g., camera poses, object trajectories).
DEFAULT_MOTIONCTRL_COND_DIR = worldfoundry_data_path("test_cases", "motionctrl_conditions")
# Default path to the MotionCtrl model checkpoint.
DEFAULT_MOTIONCTRL_CKPT = DEFAULT_SHARED_HFD_ROOT / "TencentARC--MotionCtrl" / "motionctrl.pth"
# Hugging Face ID for the OpenCLIP model used by MotionCtrl.
MOTIONCTRL_OPENCLIP_HF_ID = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# Architecture string for the OpenCLIP model.
MOTIONCTRL_OPENCLIP_ARCH = "ViT-H-14"
# Internal marker used to prevent re-patching of the OpenCLIP download function.
_OPENCLIP_PATCH_MARKER = "_worldfoundry_motionctrl_openclip_no_hf_download"


def _prepare_official_motionctrl_imports() -> None:
    """
    Placeholder function for preparing imports specific to an official MotionCtrl setup.
    Currently, this implementation performs no actions.
    """
    return


def _patch_motionctrl_openclip_download() -> None:
    """
    Patches the `open_clip.create_model_and_transforms` function to prevent
    downloading the CLIP-ViT-H-14 model from Hugging Face when MotionCtrl initializes
    its conditional stage model.

    MotionCtrl's `motionctrl.pth` checkpoint already includes the necessary weights
    for the CLIP model, making a separate download redundant and potentially problematic.
    This patch ensures that OpenCLIP is initialized with `pretrained=None` and other
    download-related flags set to False when the specific MotionCtrl CLIP model ID is requested.
    """
    try:
        import open_clip
    except ImportError:
        # If open_clip is not installed, no patching is needed or possible.
        return

    original_create = open_clip.create_model_and_transforms
    # Check if the patch has already been applied using a custom marker.
    if getattr(original_create, _OPENCLIP_PATCH_MARKER, False):
        return

    def create_model_and_transforms(model_name: str, *args: Any, **kwargs: Any):
        # Intercept calls for the specific MotionCtrl OpenCLIP model ID.
        if str(model_name) == MOTIONCTRL_OPENCLIP_HF_ID:
            # Force pretrained and load_weights to None/False to prevent HF download.
            kwargs.setdefault("pretrained", None)
            kwargs.setdefault("load_weights", False)
            kwargs.setdefault("pretrained_text", False)
            kwargs.setdefault("pretrained_image", False)
            print(
                "WorldFoundry MotionCtrl: initializing OpenCLIP ViT-H-14 without HF download; "
                "motionctrl.pth supplies cond_stage_model weights.",
                flush=True,
            )
            # Call the original function with the modified arguments and the specific architecture string.
            return original_create(MOTIONCTRL_OPENCLIP_ARCH, *args, **kwargs)
        # For any other model, call the original function unmodified.
        return original_create(model_name, *args, **kwargs)

    # Apply the marker to the new patched function.
    setattr(create_model_and_transforms, _OPENCLIP_PATCH_MARKER, True)
    # Replace the original function with the patched version.
    open_clip.create_model_and_transforms = create_model_and_transforms


def _uses_official_motionctrl_config(config_path: str) -> bool:
    """
    Placeholder function to check if a given configuration path belongs to an
    official MotionCtrl repository.
    Currently, this implementation always returns False.
    """
    return False


def _motionctrl_seed(value: Any, default: int = 20230211) -> int:
    """
    Converts an arbitrary value into a non-negative integer seed.
    If the value cannot be converted to an integer, or if the resulting integer is negative,
    a default seed is returned.

    Args:
        value: The value to attempt to convert into a seed.
        default: The default integer seed to use if conversion fails or is invalid.

    Returns:
        A non-negative integer seed.
    """
    try:
        seed = int(value)
    except (TypeError, ValueError):
        # If conversion to int fails, return the default seed.
        return default
    # If the converted seed is negative, return the default; otherwise, return the seed.
    return seed if seed >= 0 else default


class MotionCtrlRuntime:
    """
    Provides a runtime interface for MotionCtrl, enabling video generation with
    camera and object motion control. It manages model loading, device placement,
    and handling of various configuration and condition inputs.

    This runtime uses vendored LVDM and control modules.
    """

    MODEL_ID = "motionctrl"
    DISPLAY_NAME = "MotionCtrl"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        ckpt_path: str | Path = DEFAULT_MOTIONCTRL_CKPT,
        config_path: str | Path = DEFAULT_MOTIONCTRL_CONFIG,
        cond_dir: str | Path = DEFAULT_MOTIONCTRL_COND_DIR,
        adapter_ckpt: str | Path | None = None,
        condtype: str = "both",
    ) -> None:
        """
        Initializes the MotionCtrl runtime.

        Args:
            model_id: Identifier for the model profile. Defaults to "motionctrl".
            device: The compute device to use (e.g., "cuda", "cuda:0").
            ckpt_path: Path to the MotionCtrl model checkpoint.
            config_path: Path to the MotionCtrl configuration file (YAML).
            cond_dir: Directory containing default motion condition files.
            adapter_ckpt: Optional path to an adapter checkpoint to be loaded with the main model.
            condtype: The type of motion conditioning to use by default ("camera_motion", "object_motion", "both").
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "motion_control_video"
        self.device = device
        # Resolve and store all path arguments.
        self.ckpt_path = str(Path(ckpt_path).expanduser())
        self.config_path = str(Path(config_path).expanduser())
        self.cond_dir = str(Path(cond_dir).expanduser())
        self.adapter_ckpt = None if adapter_ckpt is None else str(Path(adapter_ckpt).expanduser())
        self.condtype = condtype
        self._model = None  # Model is loaded lazily on first access.

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "MotionCtrlRuntime":
        """
        Factory method to build a lazy MotionCtrl runtime instance.
        It parses various ways of specifying model paths and options.

        Args:
            pretrained_model_path: Optional checkpoint path (str/Path) or a mapping (dict)
                                   containing runtime options like 'ckpt_path', 'config_path', etc.
            args: Unused compatibility argument for pipeline factories.
            device: Target torch device to use for generation (e.g., "cuda").
            model_id: Runtime profile id, defaults to `motionctrl`.
            **kwargs: Additional runtime options, such as `ckpt_path`, `config_path`, `cond_dir`, etc.,
                      which override values from `pretrained_model_path`.

        Returns:
            An initialized `MotionCtrlRuntime` instance.
        """
        del args  # This argument is not used in this runtime.
        # If pretrained_model_path is a mapping, use it as initial options. Otherwise, it might be a checkpoint path.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["ckpt_path"] = str(pretrained_model_path)
        # Update options with any direct kwargs, which take precedence.
        options.update(kwargs)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            ckpt_path=options.get("ckpt_path") or options.get("checkpoint_path") or DEFAULT_MOTIONCTRL_CKPT,
            config_path=options.get("config_path") or options.get("config") or options.get("base") or DEFAULT_MOTIONCTRL_CONFIG,
            cond_dir=options.get("cond_dir") or DEFAULT_MOTIONCTRL_COND_DIR,
            adapter_ckpt=options.get("adapter_ckpt"),
            condtype=str(options.get("condtype") or "both"),
        )

    def _ensure_model(self):
        """
        Ensures the MotionCtrl model is loaded and initialized.
        The model is loaded once from the vendored LVDM configuration upon its first access.
        It handles device placement, OpenCLIP patching, and checkpoint loading.

        Raises:
            RuntimeError: If MotionCtrl is attempted without CUDA being available.

        Returns:
            The loaded MotionCtrl model instance.
        """
        if self._model is not None:
            return self._model

        import torch

        from worldfoundry.core.model_loading.factory import instantiate_from_config
        from worldfoundry.synthesis.visual_generation.motionctrl.motionctrl_runtime.main.inference.motionctrl_inference import (
            load_model_checkpoint,
        )

        # Prepare any official MotionCtrl-specific imports, if necessary.
        _prepare_official_motionctrl_imports()

        # Dynamically import OmegaConf to avoid a top-level dependency if not strictly needed.
        OmegaConf = import_module("omegaconf").OmegaConf

        # MotionCtrl's in-tree inference path strictly requires CUDA.
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("MotionCtrl requires CUDA for the in-tree official inference path.")

        # Extract GPU number from device string (e.g., "cuda:0" -> 0).
        gpu_no = int(str(self.device).split(":", maxsplit=1)[1]) if ":" in str(self.device) else 0

        # Apply the OpenCLIP patching to prevent redundant downloads.
        _patch_motionctrl_openclip_download()

        # Load the model configuration and instantiate the model.
        config = OmegaConf.load(self.config_path)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)

        # Move the model to the specified CUDA device.
        model = model.cuda(gpu_no)

        # Load the main model checkpoint and any optional adapter checkpoint.
        model = load_model_checkpoint(model, self.ckpt_path, self.adapter_ckpt)

        # Set the model to evaluation mode.
        model.eval()
        self._model = model
        return model

    def _default_conditions(self, condtype: str) -> tuple[list[str], Any, Any]:
        """
        Resolves and returns the default MotionCtrl condition fixtures (prompts, camera poses, trajectories)
        based on the specified condition type.

        Args:
            condtype: The type of motion conditioning requested. Valid values are
                      "camera_motion", "object_motion", or "both".

        Returns:
            A tuple containing:
            - A list of default prompts (str).
            - A tensor of default camera poses (torch.Tensor or None).
            - A tensor of default object trajectories (torch.Tensor or None).

        Raises:
            ValueError: If an unsupported `condtype` is provided.
        """
        import torch

        from worldfoundry.synthesis.visual_generation.motionctrl.motionctrl_runtime.main.inference.motionctrl_inference import (
            load_camera_pose,
            load_trajs,
        )
        from worldfoundry.synthesis.visual_generation.motionctrl.motionctrl_runtime.main.inference.motionctrl_prompts_camerapose_trajs import (
            both_prompt_camerapose_traj,
            cmcm_prompt_camerapose,
            omom_prompt_traj,
        )

        if condtype == "camera_motion":
            prompts = [cmcm_prompt_camerapose["prompts"][0]]
            camera_poses, _ = load_camera_pose(self.cond_dir, [cmcm_prompt_camerapose["camera_poses"][0]])
            return prompts, torch.stack(camera_poses, dim=0), None
        if condtype == "object_motion":
            prompts = [omom_prompt_traj["prompts"][0]]
            trajs, _ = load_trajs(self.cond_dir, [omom_prompt_traj["trajs"][0]])
            return prompts, None, torch.stack(trajs, dim=0)
        if condtype == "both":
            prompts = [both_prompt_camerapose_traj["prompts"][0]]
            camera_poses, _ = load_camera_pose(self.cond_dir, [both_prompt_camerapose_traj["camera_poses"][0]])
            trajs, _ = load_trajs(self.cond_dir, [both_prompt_camerapose_traj["trajs"][0]])
            return prompts, torch.stack(camera_poses, dim=0), torch.stack(trajs, dim=0)
        raise ValueError(f"Unsupported MotionCtrl condtype: {condtype}")

    @staticmethod
    def _load_direct_conditions(kwargs: Mapping[str, Any], interactions: Sequence[str]) -> tuple[str | None, Any, Any]:
        """
        Loads explicit camera pose or object trajectory conditions from files specified
        in `kwargs` or `interactions`.

        Args:
            kwargs: Keyword arguments that may contain paths like "camera_pose_file"
                    or "trajectory_file".
            interactions: A sequence of strings, typically file paths for conditions,
                          that can also specify a condition type (e.g., "camera_motion").

        Returns:
            A tuple containing:
            - `condtype` (str or None): The detected condition type ("camera_motion", "object_motion", "both").
            - `camera_poses` (torch.Tensor or None): Loaded camera pose tensor.
            - `trajs` (torch.Tensor or None): Loaded object trajectory tensor.
        """
        import json

        import numpy as np
        import torch

        camera_pose_file = kwargs.get("camera_pose_file") or kwargs.get("camera_pose_path")
        trajectory_file = kwargs.get("trajectory_file") or kwargs.get("object_trajectory_file")

        cond_tokens = {"camera_motion", "object_motion", "both"}
        interaction_condtype = None
        if trajectory_file is None and interactions:
            first = str(interactions[0]).strip()
            normalized_first = first.replace("-", "_")
            if normalized_first in cond_tokens:
                interaction_condtype = normalized_first
            else:
                trajectory_file = first

        camera_poses = None
        trajs = None
        condtype = interaction_condtype

        if camera_pose_file:
            # Load camera pose from a JSON file.
            pose = np.asarray(json.loads(Path(camera_pose_file).read_text(encoding="utf-8")), dtype=np.float32)
            camera_poses = torch.as_tensor(pose).float()[None]  # Add batch dimension.
            condtype = "camera_motion"

        if trajectory_file:
            path = Path(str(trajectory_file))
            if path.suffix == ".json":
                # If a trajectory file is JSON, it might actually be a camera pose.
                pose = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
                camera_poses = torch.as_tensor(pose).float()[None]
                condtype = "camera_motion"
            else:
                # Load object trajectory from a NumPy file (e.g., .npy).
                traj = torch.tensor(np.load(path)).permute(3, 0, 1, 2).float()  # Adjust dimensions.
                trajs = traj[None]  # Add batch dimension.
                # Determine condtype based on whether camera poses were also loaded.
                condtype = "object_motion" if camera_poses is None else "both"

        return condtype, camera_poses, trajs

    @staticmethod
    def _save_video(samples: Any, target: Path, fps: int) -> None:
        """
        Saves a batch of MotionCtrl generated video samples as an MP4 video file.
        The samples are arranged into a grid before saving.

        Args:
            samples: A tensor of video samples, expected shape `[n_samples, c, t, h, w]`.
            target: The output file path for the MP4 video.
            fps: The frames per second for the output video.
        """
        import torch
        import torchvision

        from worldfoundry.core.io import write_video

        raw = samples.detach().cpu().float()

        # Check for finite values to detect generation issues.
        finite = torch.isfinite(raw)
        finite_count = int(finite.sum().item())
        total_count = raw.numel()
        if finite_count == 0:
            raise ValueError(f"MotionCtrl generated no finite sample values; shape={tuple(raw.shape)}")

        finite_values = raw[finite]
        print(
            "MotionCtrl sample stats before save: "
            f"shape={tuple(raw.shape)} finite={finite_count}/{total_count} "
            f"min={float(finite_values.min().item()):.6g} "
            f"max={float(finite_values.max().item()):.6g} "
            f"mean={float(finite_values.mean().item()):.6g} "
            f"std={float(finite_values.std(unbiased=False).item()):.6g}",
            flush=True,
        )

        # Handle NaNs/Infs and clamp values to [-1, 1] range.
        video = torch.nan_to_num(raw, nan=-1.0, posinf=1.0, neginf=-1.0)
        video = torch.clamp(video, -1.0, 1.0)

        # Reshape for torchvision.utils.make_grid: [t, n_samples, c, h, w] -> [t, c, h*N, w*N]
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(frame_sheet, nrow=int(samples.shape[0])) for frame_sheet in video]
        grid = torch.stack(frame_grids, dim=0)

        # Scale pixel values from [-1, 1] to [0, 255] and convert to uint8.
        grid = ((grid + 1.0) / 2.0 * 255).to(torch.uint8).permute(0, 2, 3, 1)  # [t, h, w, c] for imageio.

        write_video(grid.numpy(), target, fps=fps, quality=8)

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
        Generates a MotionCtrl video based on a text prompt and motion conditions.
        The output is an MP4 video saved to `output_path`.

        Args:
            prompt: The text prompt for video generation. If empty, a default fixture prompt is used.
            images: Reserved for pipeline compatibility; MotionCtrl does not consume input images.
            video: Reserved for pipeline compatibility; must be `None` as MotionCtrl consumes motion conditions, not input video.
            interactions: An optional sequence of strings, typically paths to condition files,
                          provided by an operator or pipeline.
            output_path: The target file path to save the generated MP4 video. If None, it defaults
                         to 'motionctrl.mp4' in the current working directory.
            fps: The frames per second for the output video. If None, defaults to 10.
            **kwargs: Additional generation and condition options, including:
                      - `seed`: Random seed for reproducibility.
                      - `camera_pose_file`, `trajectory_file`: Paths to explicit condition files.
                      - `condtype`: Explicit condition type ("camera_motion", "object_motion", "both").
                      - `height`, `width`: Dimensions of the generated video frames.
                      - `batch_size`, `n_samples`: Number of samples to generate.
                      - `cfg_scale`, `unconditional_guidance_scale`: Classifier-free guidance scale.
                      - `unconditional_guidance_scale_temporal`: Temporal guidance scale.
                      - `infer_steps`, `ddim_steps`: Number of DDIM inference steps.
                      - `ddim_eta`: DDIM eta parameter.
                      - `cond_T`: Conditional time step.

        Returns:
            A dictionary containing generation metadata, including status, model_id,
            artifact path, SHA256 hash of the video, and used parameters.

        Raises:
            ValueError: If `video` is not None, or if height/width are not multiples of 16.
        """

        from worldfoundry.synthesis.visual_generation.motionctrl.motionctrl_runtime.main.inference.motionctrl_inference import (
            motionctrl_sample,
        )

        # Dynamically import seed_everything to avoid a top-level dependency if not strictly needed.
        seed_everything = import_module("pytorch_lightning").seed_everything

        del images  # MotionCtrl does not consume input images.
        if video is not None:
            raise ValueError("MotionCtrl inference consumes motion conditions, not input video.")

        model = self._ensure_model()

        # Resolve the generation seed.
        seed = _motionctrl_seed(kwargs.get("seed"), default=20230211)
        seed_everything(seed)

        # Load any explicit condition files provided.
        direct_condtype, camera_poses, trajs = self._load_direct_conditions(kwargs, interactions)
        # Determine the final condition type, prioritizing direct_condtype, then instance condtype.
        condtype = str(kwargs.get("condtype") or direct_condtype or self.condtype)

        # Get default conditions based on the determined condtype.
        prompts, default_camera_poses, default_trajs = self._default_conditions(condtype)

        # Use default conditions if direct conditions were not provided.
        if camera_poses is None:
            camera_poses = default_camera_poses
        if trajs is None:
            trajs = default_trajs

        # Use the provided prompt or the default fixture prompt.
        prompts = [prompt or prompts[0]]

        # Move conditions to the correct device if they exist.
        if camera_poses is not None:
            camera_poses = camera_poses.to("cuda")
        if trajs is not None:
            trajs = trajs.to("cuda")

        # Validate height and width constraints.
        height = int(kwargs.get("height", 256))
        width = int(kwargs.get("width", 256))
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("MotionCtrl height and width must be multiples of 16.")

        # Construct the shape for the initial noise tensor.
        noise_shape = [
            int(kwargs.get("batch_size", kwargs.get("bs", 1))),
            model.channels,
            model.temporal_length,
            height // 8,
            width // 8,
        ]

        # Perform the video generation sample.
        batch_samples = motionctrl_sample(
            model,
            prompts,
            noise_shape,
            camera_poses=camera_poses,
            trajs=trajs,
            n_samples=int(kwargs.get("n_samples", 1)),
            unconditional_guidance_scale=float(
                kwargs.get("cfg_scale", kwargs.get("unconditional_guidance_scale", 7.5))
            ),
            unconditional_guidance_scale_temporal=kwargs.get("unconditional_guidance_scale_temporal"),
            ddim_steps=int(kwargs.get("infer_steps", kwargs.get("ddim_steps", 50))),
            ddim_eta=float(kwargs.get("ddim_eta", 1.0)),
            cond_T=int(kwargs.get("cond_T", 800)),
        )

        # Resolve and create the output directory for the video.
        target = Path(output_path) if output_path is not None else Path.cwd() / "motionctrl.mp4"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Save the generated video.
        self._save_video(batch_samples[0], target, fps=fps or int(kwargs.get("fps", 10)))

        # Return generation metadata.
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "video_sha256": file_sha256(target),
            "runtime": "worldfoundry.motionctrl.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "condtype": condtype,
            "seed": seed,
        }


__all__ = [
    "DEFAULT_MOTIONCTRL_CKPT",
    "DEFAULT_MOTIONCTRL_COND_DIR",
    "DEFAULT_MOTIONCTRL_CONFIG",
    "MotionCtrlRuntime",
]
