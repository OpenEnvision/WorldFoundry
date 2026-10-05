"""Yume 1P5 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from worldfoundry.runtime.env import resolve_ckpt_dir

from ..pipeline_utils import PipelineABC


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CKPT_ROOT = resolve_ckpt_dir()
_DEFAULT_YUME_1P5_CKPT = _DEFAULT_CKPT_ROOT / "Yume-5B-720P"
_INTERNAL_RUNTIME_ROOT = _REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "yume"
_SUPPORTED_YUME_1P5_SIZES = ("704*1280", "1280*704")
_REQUIRED_YUME_1P5_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "google/umt5-xxl/tokenizer_config.json",
)


def _resolve_local_checkpoint(value: Any, default_path: Path) -> str:
    """Resolve an explicit local checkpoint directory."""
    path = Path(str(value or default_path)).expanduser()
    if path.is_dir():
        missing = [name for name in _REQUIRED_YUME_1P5_FILES if not (path / name).is_file()]
        if not missing:
            return str(path.resolve())
        raise FileNotFoundError(
            f"Incomplete Yume 1.5 checkpoint directory: {path}. "
            f"Missing required files: {', '.join(missing)}"
        )
    raise FileNotFoundError(
        "Yume 1.5 requires a local checkpoint directory. "
        f"Expected {path}. Download weights with huggingface-cli or "
        "scripts/model_zoo/download_checkpoints.py, then set WORLDFOUNDRY_CKPT_DIR "
        "or pass an explicit local path."
    )


def _normalise_request(
    interactions: Optional[Union[str, List[str]]],
    interaction_speeds: Optional[Union[float, List[float]]],
    interaction_distances: Optional[Union[float, List[float]]],
    *,
    task_type: Optional[str],
    size: Optional[str],
    num_euler_timesteps: Optional[int],
    images: Optional[Any],
    videos: Optional[List[Any]],
    require_conditioning_input: bool = True,
) -> tuple[List[str], List[float], List[Optional[float]], str, str, int]:
    """Validate lightweight request geometry before entering the YUME runtime."""
    if isinstance(interactions, str):
        normalised_interactions = [interactions]
    elif interactions is None:
        normalised_interactions = []
    else:
        normalised_interactions = list(interactions)
    if not normalised_interactions:
        raise ValueError("Yume 1.5 requires at least one interaction.")

    if isinstance(interaction_speeds, (float, int)):
        normalised_speeds = [float(interaction_speeds)] * len(normalised_interactions)
    elif interaction_speeds is None:
        normalised_speeds = [4.0 if "camera_" in action else 100.0 for action in normalised_interactions]
    else:
        normalised_speeds = [float(value) for value in interaction_speeds]

    if isinstance(interaction_distances, (float, int)):
        normalised_distances: List[Optional[float]] = [float(interaction_distances)] * len(normalised_interactions)
    elif interaction_distances is None:
        normalised_distances = [None if "camera_" in action else 4.0 for action in normalised_interactions]
    else:
        normalised_distances = [None if value is None else float(value) for value in interaction_distances]

    if not (
        len(normalised_interactions) == len(normalised_speeds) == len(normalised_distances)
    ):
        raise ValueError(
            "interactions, interaction_speeds, and interaction_distances must have the same length."
        )
    for action, distance in zip(normalised_interactions, normalised_distances):
        if "camera_" not in action and distance is None:
            raise ValueError(f"Movement interaction {action!r} requires a numeric distance.")

    normalised_task_type = str(task_type or "i2v").lower()
    if normalised_task_type not in {"t2v", "i2v", "v2v"}:
        raise ValueError("task_type must be one of: t2v, i2v, v2v.")
    if require_conditioning_input and normalised_task_type == "i2v" and images is None:
        raise ValueError("Yume 1.5 i2v requires an image input.")
    if require_conditioning_input and normalised_task_type == "v2v" and not videos:
        raise ValueError("Yume 1.5 v2v requires a non-empty video input.")

    normalised_size = str(size or "704*1280")
    if normalised_size not in _SUPPORTED_YUME_1P5_SIZES:
        raise ValueError(
            f"Unsupported Yume 1.5 size {normalised_size!r}; "
            f"choose one of {_SUPPORTED_YUME_1P5_SIZES}."
        )
    if isinstance(num_euler_timesteps, bool) or not isinstance(num_euler_timesteps, int):
        raise ValueError("num_euler_timesteps must be a positive integer.")
    if num_euler_timesteps < 1:
        raise ValueError("num_euler_timesteps must be a positive integer.")

    return (
        normalised_interactions,
        normalised_speeds,
        normalised_distances,
        normalised_task_type,
        normalised_size,
        num_euler_timesteps,
    )


class Yume1p5Pipeline(PipelineABC):

    """Pipeline implementation for Yume1p5 visual generation."""
    def __init__(
        self,
        synthesis_model: Optional[Any] = None,
        operators: Optional[Any] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        checkpoint_root: str | Path = _DEFAULT_CKPT_ROOT,
    ) -> None:
        
        """Initialize the pipeline and configure runtime components."""
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.runtime_root = _INTERNAL_RUNTIME_ROOT
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[dict[str, Any]] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        fsdp: bool = False,
        **kwargs: Any,
    ) -> "Yume1p5Pipeline":
        """
        Load Yume 1.5 from local official repo metadata and staged checkpoints.

        Args:
            model_path: Local checkpoint path, repo id, or options mapping.
            required_components: Optional repo/checkpoint path mapping.
            device: Runtime device label.
            weight_dtype: Torch dtype used by the runtime.
            fsdp: Whether to enable the official FSDP path.
            **kwargs: Additional adapter/runtime options.
        """
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["checkpoint_path"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        synthesis_model_path = _resolve_local_checkpoint(
            options.get("checkpoint_path") or options.get("pretrained_model_path") or options.get("model_path"),
            _DEFAULT_YUME_1P5_CKPT,
        )

        import torch

        from ...synthesis.visual_generation.memory.stream import VisualContextMemory
        from ...operators.yume_1p5_operator import Yume1p5Operator
        from ...synthesis.visual_generation.yume.yume_1p5_synthesis import Yume1p5Synthesis

        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype = weight_dtype or torch.bfloat16
        synthesis_model = Yume1p5Synthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            weight_dtype=weight_dtype,
            fsdp=fsdp
        )
        operators = Yume1p5Operator()
        memory_module = VisualContextMemory(model_id="yume-1.5")

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_root=options.get("checkpoint_root") or options.get("ckpt_root") or _DEFAULT_CKPT_ROOT,
        )
        return pipeline

    def _dist_rank(self) -> int:
        """Dist rank for Yume1p5Pipeline."""
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _to_pil_frames(self, video: Any) -> List[Any]:
        """To pil frames for Yume1p5Pipeline."""
        import numpy as np
        import torch
        from PIL import Image

        def _frame_to_pil(frame: Any) -> Any:
            """Frame to pil for Yume1p5Pipeline."""
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.ndim != 3:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0
                    if frame.max() <= 1.0:
                        frame = np.clip(frame, 0.0, 1.0)
                        frame = (frame * 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            return Image.fromarray(frame)

        if isinstance(video, list):
            if len(video) == 0:
                return []
            if isinstance(video[0], Image.Image):
                return video
            if isinstance(video[0], np.ndarray):
                return [_frame_to_pil(frame) for frame in video]
            raise TypeError(f"Unsupported frame type in list: {type(video[0])}")

        if isinstance(video, np.ndarray):
            if video.ndim != 4:
                raise ValueError(f"Expected video ndarray shape (T,H,W,C), got {video.shape}")
            return [_frame_to_pil(frame) for frame in video]

        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
            if v.ndim != 4:
                raise ValueError(f"Expected video tensor ndim=4, got shape {tuple(v.shape)}")

            if v.shape[-1] in (1, 3, 4):
                arr = v.numpy()  # (T,H,W,C)
            elif v.shape[1] in (1, 3, 4):
                arr = v.permute(0, 2, 3, 1).numpy()  # (T,C,H,W) -> (T,H,W,C)
            elif v.shape[0] in (1, 3, 4):
                arr = v.permute(1, 2, 3, 0).numpy()  # (C,T,H,W) -> (T,H,W,C)
            else:
                raise ValueError(f"Cannot infer tensor video layout from shape {tuple(v.shape)}")
            return [_frame_to_pil(frame) for frame in arr]

        raise TypeError(f"Unsupported video type: {type(video)}")

    def _materialise_video_frames(self, video: Any) -> Optional[List[Any]]:
        """Convert a runner video input into the 33-frame YUME context contract."""
        if video is None:
            return None
        if isinstance(video, list):
            frames = video
        elif isinstance(video, (str, Path)):
            from decord import VideoReader
            from PIL import Image

            reader = VideoReader(str(video))
            if len(reader) == 0:
                raise ValueError(f"Yume 1.5 video input is empty: {video}")
            indices = list(range(min(len(reader), 33)))
            indices.extend([indices[-1]] * (33 - len(indices)))
            frames = [Image.fromarray(reader[index].asnumpy()).convert("RGB") for index in indices]
        else:
            frames = self._to_pil_frames(video)
        if not frames:
            raise ValueError("Yume 1.5 video input is empty.")
        if len(frames) < 33:
            frames = [*frames, *([frames[-1]] * (33 - len(frames)))]
        return list(frames[:33])

    def run_pipeline_invocation(self, invocation: Any) -> Mapping[str, Any]:
        """Adapt the unified runner contract and materialize its video artifact."""
        from diffusers.utils import export_to_video

        kwargs = dict(invocation.pipeline_kwargs)
        fps = int(kwargs.pop("fps", 16))
        task_type = str(kwargs.get("task_type") or "i2v").lower()
        images = invocation.image if task_type == "i2v" else None
        videos = self._materialise_video_frames(invocation.video) if task_type == "v2v" else None
        output_video = self(
            prompt=invocation.prompt,
            interactions=invocation.interactions,
            images=images,
            videos=videos,
            **kwargs,
        )
        output_path = Path(invocation.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(output_video, str(output_path), fps=fps)
        return {
            "status": "succeeded",
            "runtime": "yume-1p5",
            "artifact_kind": "generated_world",
            "artifact_path": str(output_path),
            "metadata": {
                "fps": fps,
                "frame_count": len(output_video),
                "task_type": task_type,
            },
        }

    def cleanup(self) -> None:
        """Release YUME-owned distributed state after runner execution."""
        cleanup = getattr(self.synthesis_model, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def process(
        self,
        interactions: Union[str, List[str]],
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[str] = None
    ) -> Dict[str, Any]:
        
        """Process and normalize input arguments and conditions for inference."""
        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()
        
        visual_context = self.operators.process_perception(images=images, videos=videos, size=size)

        return {
            "operator_condition": operator_condition,
            "visual_context": visual_context
        }

    def __call__(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[str] = "704*1280",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
    ) -> Any:

        """Execute the complete pipeline generation flow."""
        (
            interactions,
            interaction_speeds,
            interaction_distances,
            task_type,
            size,
            num_euler_timesteps,
        ) = _normalise_request(
            interactions,
            interaction_speeds,
            interaction_distances,
            task_type=task_type,
            size=size,
            num_euler_timesteps=num_euler_timesteps,
            images=images,
            videos=videos,
        )
        output_dict = self.process(
            interactions=interactions,
            images=images,
            videos=videos,
            size=size
        )
        seed = -1 if seed is None else int(seed)

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=output_dict["visual_context"]["ref_images"],
            video=output_dict["visual_context"]["ref_videos"], # pixel values of the video frames
            interactions=interactions, # list of interaction strings, e.g., ["forward", "camera_l"]
            interaction_captions=output_dict["operator_condition"], # list of interaction captions, e.g., ["The camera pushes forward (W).", "The camera pans to the left (←)."]
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps
        )

        return output_video

    def stream(
        self,
        prompt: Optional[str] = "",
        interactions: Optional[Union[str, List[str]]] = None,
        interaction_speeds: Optional[Union[float, List[float]]] = None,
        interaction_distances: Optional[Union[float, List[float]]] = None,
        images: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        size: Optional[str] = "704*1280",
        seed: Optional[int] = None,
        task_type: Optional[str] = "i2v",
        num_euler_timesteps: Optional[int] = 100,
    ) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        from PIL import Image

        if self.memory_module is None:
            raise ValueError("memory_module is None")

        rank = self._dist_rank()

        (
            interactions,
            interaction_speeds,
            interaction_distances,
            task_type,
            size,
            num_euler_timesteps,
        ) = _normalise_request(
            interactions,
            interaction_speeds,
            interaction_distances,
            task_type=task_type,
            size=size,
            num_euler_timesteps=num_euler_timesteps,
            images=images,
            videos=videos,
            require_conditioning_input=False,
        )

        if videos is not None and not (isinstance(videos, list) and (len(videos) == 0 or isinstance(videos[0], Image.Image))):
            videos = self._to_pil_frames(videos)

        if images is not None or videos is not None:
            visual_context = self.operators.process_perception(images=images, videos=videos, size=size)
            input_data = images if images is not None else videos
            self.memory_module.record(
                input_data,
                visual_context=visual_context,
                as_context=True,
                record_frames=False,
            )

        ctx = self.memory_module.select_context()
        if task_type != "t2v" and ctx is None:
            raise ValueError("No context in memory. Provide 'images' or 'videos' in the first stream() call.")

        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        effective_task_type = task_type
        predict_image = None if ctx is None else ctx.get("ref_images")
        predict_video = None if ctx is None else ctx.get("ref_videos")

        # In streaming continuation, prefer i2v with the latest frame as anchor.
        if task_type == "i2v" and images is None and videos is None and predict_image is None:
            if predict_video is not None and getattr(self.memory_module, "n_generated_segments", 0) > 0:
                effective_task_type = "v2v"
            else:
                raise ValueError("No valid image/video context for i2v stream continuation.")

        if effective_task_type == "i2v":
            predict_video = None
        seed = -1 if seed is None else int(seed)

        output_video = self.synthesis_model.predict(
            prompt=prompt,
            image=predict_image,
            video=predict_video,
            interactions=interactions,
            interaction_captions=operator_condition,
            interaction_speeds=interaction_speeds,
            interaction_distances=interaction_distances,
            task_type=effective_task_type,
            size=size,
            seed=seed,
            num_euler_timesteps=num_euler_timesteps,
        )

        output_video_frames = self._to_pil_frames(output_video)
        if len(output_video_frames) == 0:
            raise RuntimeError("Synthesis returned an empty video in stream().")
        output_visual_context = self.operators.process_perception(
            images=output_video_frames[-1],
            videos=output_video_frames,
            size=size
        )
        self.memory_module.record(
            output_video_frames,
            visual_context=output_visual_context,
            record_frames=(rank == 0),
        )

        return output_video
