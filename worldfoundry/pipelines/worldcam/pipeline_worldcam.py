"""Worldcam visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from pathlib import Path
from typing import Any, Optional, Sequence

from worldfoundry.synthesis.visual_generation.worldcam import runtime_root as worldcam_runtime_root
from worldfoundry.runtime.env import resolve_ckpt_dir, resolve_hfd_root

_DEFAULT_CKPT_ROOT = resolve_ckpt_dir()
_DEFAULT_HFD_ROOT = resolve_hfd_root()
_DEFAULT_WORLDCAM_CKPT = _DEFAULT_HFD_ROOT / "worldcam--worldcam"
_DEFAULT_WAN_CKPT = _DEFAULT_HFD_ROOT / "Wan-AI--Wan2.1-T2V-1.3B"


def _resolve_local_path(value: Any, repo_id: str, default_path: Path, label: str) -> str:
    """Resolve local path helper function."""
    text = str(value or default_path)
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    if text == repo_id and default_path.exists():
        return str(default_path.resolve())
    if "/" in text and not path.is_absolute():
        cached = _DEFAULT_HFD_ROOT / text.replace("/", "--")
        if cached.exists():
            return str(cached.resolve())
    raise FileNotFoundError(
        f"WorldCam requires local {label}. Expected {path} or cached copy under {_DEFAULT_HFD_ROOT}. "
        "Set WORLDFOUNDRY_CKPT_DIR/WORLDFOUNDRY_HFD_ROOT or pass an explicit local path."
    )


class WorldCamPipeline(PipelineABC):
    """WorldCam pipeline following the official video+camera demo interface."""

    def __init__(
        self,
        operator: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        checkpoint_root: str | Path = _DEFAULT_CKPT_ROOT,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.runtime_root = worldcam_runtime_root()
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        **kwargs,
    ) -> "WorldCamPipeline":
        """
        Load WorldCam from local checkpoint roots while keeping module import lightweight.

        Args:
            model_path: Local checkpoint path, repo id, or options mapping.
            required_components: Optional dependency paths such as ``wan_model_path``.
            device: Runtime device label.
            weight_dtype: Torch dtype used by the runtime.
            **kwargs: Additional WorldCam synthesis options.
        """
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["worldcam_ckpt_path"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        import torch

        from ...synthesis.visual_generation.worldcam.worldcam_synthesis import WorldCamSynthesis

        wan_model_path = _resolve_local_path(
            options.get("wan_model_path") or "Wan-AI/Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-1.3B",
            _DEFAULT_WAN_CKPT,
            "Wan2.1 base model",
        )
        worldcam_ckpt_path = _resolve_local_path(
            options.get("worldcam_ckpt_path") or options.get("checkpoint_path") or options.get("ckpt_path") or "worldcam/worldcam",
            "worldcam/worldcam",
            _DEFAULT_WORLDCAM_CKPT,
            "WorldCam checkpoint",
        )
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype = weight_dtype or torch.bfloat16
        synthesis_options = {
            key: value
            for key, value in options.items()
            if key
            not in {
                "checkpoint_root",
                "ckpt_root",
                "wan_model_path",
                "worldcam_ckpt_path",
                "checkpoint_path",
                "ckpt_path",
            }
        }
        synthesis_model = WorldCamSynthesis.from_pretrained(
            pretrained_model_path=wan_model_path,
            worldcam_ckpt_path=worldcam_ckpt_path,
            device=device,
            weight_dtype=weight_dtype,
            **synthesis_options,
        )
        return cls(
            operator=None,
            synthesis_model=synthesis_model,
            memory_module=None,
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_root=options.get("checkpoint_root") or options.get("ckpt_root") or _DEFAULT_CKPT_ROOT,
        )

    def process(
        self,
        images,
        interactions: Sequence[str],
        prompt: str = "",
    ):
        """Process and normalize input arguments and conditions for inference."""
        del images, interactions, prompt
        raise NotImplementedError(
            "WorldCam official integration uses conditioning video plus intrinsics/extrinsics files; "
            "action-string interactions are not part of the official demo."
        )

    @staticmethod
    def _first_present(*values: Any) -> Any:
        """First present for WorldCamPipeline."""
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    @staticmethod
    def _load_camera_tensor(value: Any = None, path: str | Path | None = None, *, name: str):
        """Load camera tensor for WorldCamPipeline."""
        import numpy as np
        import torch

        source = value if value is not None else path
        if source is None:
            raise ValueError(f"WorldCam official demo requires {name} or {name}_path.")
        if isinstance(source, (str, Path)):
            source_path = Path(source).expanduser()
            if not source_path.is_file():
                raise FileNotFoundError(f"WorldCam {name} file not found: {source_path}")
            array = np.load(source_path)
            tensor = torch.from_numpy(array)
        elif isinstance(source, np.ndarray):
            tensor = torch.from_numpy(source)
        elif isinstance(source, torch.Tensor):
            tensor = source
        else:
            tensor = torch.as_tensor(source)
        if tensor.ndim == 2 and tensor.shape[-1] == 4:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tuple(tensor.shape[-2:]) == (4, 4):
            tensor = tensor.unsqueeze(0)
        return tensor

    def _condition_video_from_source(
        self,
        source: Any,
        *,
        height: int | None = None,
        width: int | None = None,
        conditioning_frames: int | None = None,
    ) -> Any:
        """Condition video from source for WorldCamPipeline."""
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"WorldCam conditioning video not found: {path}")
            from worldfoundry.core.io import VideoData

            video = VideoData(
                str(path),
                height=int(height or self.synthesis_model.height),
                width=int(width or self.synthesis_model.width),
            )
            video.length = int(conditioning_frames or 65)
            return video
        if hasattr(source, "set_length") and conditioning_frames is not None:
            source.set_length(int(conditioning_frames))
        elif hasattr(source, "length") and conditioning_frames is not None:
            source.length = int(conditioning_frames)
        return source

    def __call__(
        self,
        prompt: str = "",
        video: Any = None,
        input_path: str | Path | None = None,
        video_path: str | Path | None = None,
        intrinsics: Any = None,
        extrinsics: Any = None,
        intrinsics_path: str | Path | None = None,
        extrinsics_path: str | Path | None = None,
        num_ar_steps: int = 50,
        conditioning_frames: int = 65,
        cfg_scale: float = 4.0,
        seed: int = 0,
        negative_prompt: str | None = None,
        long_term_memory_start_step: int = 30,
        long_term_memory_num_clips: int = 4,
        long_term_memory_ref_indices: list[int] | None = None,
        attention_sink_inference: bool = False,
        trim_conditioning: bool = False,
        num_inference_steps: int = 50,
        tiled: bool = True,
        output_path: str | Path | None = None,
        fps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        return_dict: bool = False,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")
        source_video = self._first_present(video, video_path, input_path)
        if source_video is None:
            raise ValueError("WorldCam official inference requires video_path or input_path.")
        if intrinsics is None and intrinsics_path is None:
            raise ValueError("WorldCam official inference requires intrinsics_path.")
        if extrinsics is None and extrinsics_path is None:
            raise ValueError("WorldCam official inference requires extrinsics_path.")

        condition_video = self._condition_video_from_source(
            source_video,
            height=height,
            width=width,
            conditioning_frames=int(conditioning_frames),
        )
        intrinsics_tensor = self._load_camera_tensor(intrinsics, intrinsics_path, name="intrinsics")
        extrinsics_tensor = self._load_camera_tensor(extrinsics, extrinsics_path, name="extrinsics")
        result = self.synthesis_model.predict(
            prompt=prompt,
            condition_video=condition_video,
            intrinsics=intrinsics_tensor,
            extrinsics=extrinsics_tensor,
            num_ar_steps=max(1, int(num_ar_steps)),
            negative_prompt=negative_prompt,
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            long_term_memory_start_step=int(long_term_memory_start_step),
            long_term_memory_num_clips=int(long_term_memory_num_clips),
            long_term_memory_ref_indices=long_term_memory_ref_indices,
            attention_sink_inference=bool(attention_sink_inference),
            trim_conditioning=bool(trim_conditioning),
            num_inference_steps=max(1, int(num_inference_steps)),
            tiled=bool(tiled),
            return_dict=True,
        )
        result.update(
            {
                "num_output_frames": len(result["video"]),
                "conditioning_frames": int(conditioning_frames),
                "official_demo": True,
            }
        )
        if output_path is not None:
            from worldfoundry.core.io import write_video

            write_video(result["video"], output_path, fps=int(fps) if fps is not None else 30)
            result["artifact_path"] = str(output_path)
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        images: Optional[Any],
        interactions: Sequence[str],
        prompt: str = "",
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        del images, interactions, prompt, reset_memory, return_dict, kwargs
        raise NotImplementedError("WorldCam official demo does not define a stream/interactions mode.")
