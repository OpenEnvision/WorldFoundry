"""Module for base_models -> three_dimensions -> depth -> dvlt -> runtime.py functionality."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.base_models.three_dimensions.depth.dvlt import runtime_root as dvlt_runtime_root
from worldfoundry.core.io import jsonable, write_json



DEFAULT_DVLT_CHECKPOINT = "nvidia/dvlt"
DEFAULT_DVLT_IMAGE_SIZE = 504
DEFAULT_DVLT_PATCH_SIZE = 14
DEFAULT_DVLT_INFERENCE_STEPS = 12


class DVLTRuntime:
    """Lazy WorldFoundry wrapper for the vendored DVLT reconstruction runtime."""

    MODEL_ID = "dvlt"
    DISPLAY_NAME = "Déjà View Looping Transformer"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path = DEFAULT_DVLT_CHECKPOINT,
        img_size: int = DEFAULT_DVLT_IMAGE_SIZE,
        patch_size: int = DEFAULT_DVLT_PATCH_SIZE,
        inference_steps: int = DEFAULT_DVLT_INFERENCE_STEPS,
        depth_head_type: str = "conv",
        mixed_precision: str = "bf16",
        load_patch_embed_weights: bool = False,
        max_points: int = 500_000,
        confidence_percentile: float = 50.0,
    ) -> None:
        """Init.

        Returns:
            The return value.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "3d_reconstruction"
        self.device = device
        self.checkpoint_path = str(checkpoint_path)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.inference_steps = int(inference_steps)
        self.depth_head_type = depth_head_type
        self.mixed_precision = mixed_precision
        self.load_patch_embed_weights = bool(load_patch_embed_weights)
        self.max_points = int(max_points)
        self.confidence_percentile = float(confidence_percentile)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "DVLTRuntime":
        """
        Build a lazy DVLT runtime wrapper.

        Args:
            pretrained_model_path: Optional checkpoint repo/path or mapping with runtime options.
            args: Unused compatibility argument for pipeline factories.
            device: Target device label.
            model_id: Runtime profile id, defaults to ``dvlt``.
            **kwargs: Runtime options such as ``img_size`` or checkpoint settings.
        """
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        if "plan_only" in options:
            raise ValueError("DVLT no longer supports plan_only; provide inputs, dependencies, and run the real runtime.")
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoint_path=(
                options.get("checkpoint_path")
                or options.get("ckpt_path")
                or options.get("pretrained_model_path")
                or options.get("model_path")
                or DEFAULT_DVLT_CHECKPOINT
            ),
            img_size=int(options.get("img_size") or options.get("image_size") or DEFAULT_DVLT_IMAGE_SIZE),
            patch_size=int(options.get("patch_size") or DEFAULT_DVLT_PATCH_SIZE),
            inference_steps=int(options.get("inference_steps") or DEFAULT_DVLT_INFERENCE_STEPS),
            depth_head_type=str(options.get("depth_head_type") or "conv"),
            mixed_precision=str(options.get("mixed_precision") or "bf16"),
            load_patch_embed_weights=bool(options.get("load_patch_embed_weights", False)),
            max_points=int(options.get("max_points") or 500_000),
            confidence_percentile=float(options.get("confidence_percentile") or options.get("conf_threshold") or 50.0),
        )

    @staticmethod
    def runtime_root() -> Path:
        """Runtime root.

        Returns:
            The return value.
        """
        return dvlt_runtime_root()

    @classmethod
    def runtime_src(cls) -> Path:
        """Runtime src.

        Returns:
            The return value.
        """
        return cls.runtime_root() / "src"

    @classmethod
    def _activate_runtime_path(cls) -> None:
        """Helper function to activate runtime path.

        Returns:
            The return value.
        """
        runtime_src = str(cls.runtime_src())
        if runtime_src not in sys.path:
            sys.path.insert(0, runtime_src)

    @staticmethod
    def _is_local_reference(value: str) -> bool:
        """Helper function to is local reference.

        Args:
            value: The value.

        Returns:
            The return value.
        """
        path = Path(value).expanduser()
        return value.startswith(("/", ".", "~")) or path.suffix in {".pt", ".pth", ".safetensors"}

    @staticmethod
    def _missing_dependencies() -> list[str]:
        """Helper function to missing dependencies.

        Returns:
            The return value.
        """
        modules = (
            "accelerate",
            "cv2",
            "huggingface_hub",
            "matplotlib",
            "numpy",
            "open3d",
            "PIL",
            "safetensors",
            "torch",
            "torchvision",
            "trimesh",
        )
        return [module for module in modules if importlib.util.find_spec(module) is None]

    def _blocked_reason(self, images: Any, video: Any, allow_cpu: bool) -> str | None:
        """Helper function to blocked reason.

        Args:
            images: The images.
            video: The video.
            allow_cpu: The allow cpu.

        Returns:
            The return value.
        """
        if images is None and video is None:
            return "DVLT requires image paths, an image directory, PIL images, or a video path."
        if not allow_cpu and not str(self.device).startswith("cuda"):
            return "DVLT official-quality inference requires a CUDA GPU; pass allow_cpu=True to force a local CPU run."
        checkpoint = str(self.checkpoint_path)
        offline = any(
            os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WORLDFOUNDRY_OFFLINE")
        )
        if offline and ("://" in checkpoint or not self._is_local_reference(checkpoint)):
            return "DVLT offline mode is enabled; provide an existing local checkpoint_path."
        if self._is_local_reference(checkpoint) and not Path(checkpoint).expanduser().exists():
            return f"DVLT checkpoint_path is missing: {checkpoint}"
        missing = self._missing_dependencies()
        if missing:
            return "DVLT runtime dependencies are missing: " + ", ".join(missing)
        if not allow_cpu:
            import torch

            if str(self.device).startswith("cuda") and not torch.cuda.is_available():
                return "DVLT official-quality inference requires a CUDA GPU, but torch.cuda.is_available() is false."
        if not (self.runtime_src() / "dvlt" / "model" / "dvlt" / "model.py").is_file():
            return f"Vendored DVLT runtime source is incomplete under {self.runtime_root()}"
        return None

    @staticmethod
    def _load_frames(source: Any, load_sequence: Any, image_cls: Any) -> tuple[str, list[Any]]:
        """Helper function to load frames.

        Args:
            source: The source.
            load_sequence: The load sequence.
            image_cls: The image cls.

        Returns:
            The return value.
        """
        if isinstance(source, (str, Path)):
            return load_sequence(source)
        if hasattr(source, "convert"):
            return "input", [source]
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            frames: list[Any] = []
            name = "sequence"
            for index, item in enumerate(source):
                if hasattr(item, "convert"):
                    frames.append(item)
                    continue
                path = Path(item).expanduser()
                if index == 0:
                    name = path.parent.name if path.parent.name else path.stem
                frames.append(image_cls.open(path))
            if not frames:
                raise ValueError("DVLT image sequence is empty.")
            return name, frames
        raise TypeError(f"Unsupported DVLT input type: {type(source).__name__}")

    @staticmethod
    def _artifact_path(output_path: str | Path | None, sequence_name: str) -> Path:
        """Helper function to artifact path.

        Args:
            output_path: The output path.
            sequence_name: The sequence name.

        Returns:
            The return value.
        """
        if output_path is None:
            return (Path.cwd() / f"{sequence_name}_dvlt.glb").resolve()
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".glb":
            return path.resolve()
        if path.suffix:
            return path.with_suffix(".glb").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return (path / f"{sequence_name}_dvlt.glb").resolve()

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run DVLT reconstruction and export a GLB point cloud when prerequisites are available.

        Args:
            prompt: Metadata-only prompt field.
            images: Image path, directory, sequence of image paths, or PIL images.
            video: Optional video path used when ``images`` is absent.
            interactions: Optional model controls; stored in metadata.
            output_path: Target ``.glb`` path or output directory.
            fps: Reserved compatibility field.
            **kwargs: Runtime options such as ``allow_cpu`` and ``max_points``.
        """
        del fps
        actions = tuple(interactions or ())
        allow_cpu = bool(kwargs.get("allow_cpu", False))
        if "plan_only" in kwargs:
            raise ValueError("DVLT no longer supports plan_only; provide inputs, dependencies, and run the real runtime.")
        reason = self._blocked_reason(images, video, allow_cpu=allow_cpu)
        if reason is not None:
            raise RuntimeError(f"DVLT cannot run: {reason}")

        self._activate_runtime_path()
        import torch
        from accelerate import Accelerator
        from PIL import Image

        from dvlt.common.constants import DataField, PredictionField
        from dvlt.common.pose import to4x4
        from dvlt.model.dvlt.model import DVLT
        from dvlt.util.preprocess import load_sequence, preprocess_images
        from worldfoundry.base_models.three_dimensions.depth.dvlt.viz.glb import pointcloud_to_glb
        from worldfoundry.base_models.three_dimensions.depth.dvlt.viz.pointcloud import (
            SPATIAL_PERCENTILE_DEFAULT,
            build_pointcloud,
        )

        source = images if images is not None else video
        sequence_name, frames = self._load_frames(source, load_sequence, Image)
        accelerator = Accelerator(mixed_precision=self.mixed_precision)
        model = DVLT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            inference_steps=self.inference_steps,
            depth_head_type=self.depth_head_type,
            load_patch_embed_weights=self.load_patch_embed_weights,
        )
        model.load_pretrained(self.checkpoint_path, strict=True)
        model.setup_test(accelerator)

        batch = preprocess_images(frames, self.img_size, self.patch_size, accelerator.device)
        with torch.no_grad(), accelerator.autocast():
            predictions = model.predict(batch, accelerator)

        max_points = int(kwargs.get("max_points") or self.max_points)
        confidence_percentile = float(kwargs.get("confidence_percentile") or self.confidence_percentile)
        pts, rgb = build_pointcloud(predictions, batch, max_points, confidence_percentile, SPATIAL_PERCENTILE_DEFAULT)
        cameras = predictions[PredictionField.CAMERAS][0]
        cameras_c2w = to4x4(cameras.camera_to_worlds).detach().float().cpu().numpy()
        intrinsics = cameras.get_intrinsics_matrices().detach().float().cpu().numpy()
        image_hw = tuple(batch[DataField.IMAGES].shape[-2:])
        artifact_path = self._artifact_path(output_path, sequence_name)
        pointcloud_to_glb(
            pts,
            rgb,
            output_path=artifact_path,
            name=f"{sequence_name}_dvlt",
            cameras_c2w=cameras_c2w,
            intrinsics=intrinsics,
            image_hw=image_hw,
            frustum_frac=float(kwargs.get("frustum_scale", 0.01)),
        )

        metadata_path = artifact_path.with_suffix(".json")
        metadata = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_3d_asset",
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "runtime": "worldfoundry.dvlt.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "checkpoint_path": self.checkpoint_path,
            "source_runtime": str(self.runtime_root()),
            "sequence_name": sequence_name,
            "frame_count": len(frames),
            "point_count": int(len(pts)),
            "actions": jsonable(list(actions)),
        }
        write_json(metadata_path, metadata)
        return metadata


def load_runtime(*args: Any, **kwargs: Any) -> DVLTRuntime:
    """Load runtime.

    Returns:
        The return value.
    """
    return DVLTRuntime.from_pretrained(*args, **kwargs)


__all__ = [
    "DEFAULT_DVLT_CHECKPOINT",
    "DEFAULT_DVLT_IMAGE_SIZE",
    "DEFAULT_DVLT_INFERENCE_STEPS",
    "DEFAULT_DVLT_PATCH_SIZE",
    "DVLTRuntime",
    "load_runtime",
]
