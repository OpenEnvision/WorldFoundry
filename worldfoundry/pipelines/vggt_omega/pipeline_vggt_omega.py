"""Vggt Omega visual generation pipeline module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.runtime import (
    VGGTOmegaRepresentation,
)
from worldfoundry.core.io import artifact_root_path

from ...operators.vggt_omega_operator import VGGTOmegaOperator
from ...synthesis.visual_generation.memory.runtime import RuntimeMemory
from ..pipeline_utils import PipelineABC


def _default_output_dir(name: str = "vggt_omega_output") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VGGT_OMEGA_CHECKPOINT = Path(
    os.environ.get(
        "WORLDFOUNDRY_VGGT_OMEGA_CHECKPOINT_DIR",
        str(_REPO_ROOT / "assets" / "checkpoints" / "facebook__VGGT-Omega"),
    )
)


class VGGTOmegaPipeline(PipelineABC):
    """VGGT-Omega pipeline using the official Omega model architecture."""

    MODEL_ID = "vggt-omega"
    OPERATOR_CLS = VGGTOmegaOperator
    MEMORY_CLS = RuntimeMemory

    def __init__(
        self,
        representation_model: Optional[Any] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        base_pipeline: Optional[Any] = None,
        operator: Optional[VGGTOmegaOperator] = None,
        memory_module: Optional[RuntimeMemory] = None,
        model_id: str | None = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.base_pipeline = base_pipeline
        self.operator = operator or self.OPERATOR_CLS()
        self.model_id = model_id or self.MODEL_ID
        self.memory_module = memory_module or self.MEMORY_CLS(model_id=self.model_id)

    @staticmethod
    def _normalize_image_paths(input_: Union[str, Path, list[str], list[Path]]) -> list[str]:
        """
        Normalize VGGT-Omega inputs to image file paths.

        Args:
            input_: Image file, image directory, text list, or explicit image path list.
        """
        if isinstance(input_, (str, Path)):
            path = Path(input_)
            if path.is_dir():
                return sorted(
                    str(item)
                    for item in path.iterdir()
                    if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                )
            if path.suffix.lower() == ".txt":
                return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [str(path)]
        if isinstance(input_, list) and all(isinstance(item, (str, Path)) for item in input_):
            return [str(item) for item in input_]
        raise ValueError("VGGT-Omega official runtime requires image paths.")

    @classmethod
    def _representation_path(cls, model_path: Any = None, **kwargs: Any) -> str:
        """
        Resolve the VGGT-Omega checkpoint path.

        Args:
            model_path: Optional path or mapping from adapter/runtime config.
            **kwargs: Optional explicit checkpoint keys.
        """
        options: Dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["representation_path"] = str(model_path)
        options.update(kwargs)
        value = (
            options.get("representation_path")
            or options.get("pretrained_model_path")
            or options.get("model_path")
            or options.get("checkpoint_dir")
        )
        if value:
            return str(value)
        if DEFAULT_VGGT_OMEGA_CHECKPOINT.exists():
            return str(DEFAULT_VGGT_OMEGA_CHECKPOINT)
        return "facebook/VGGT-Omega"

    @classmethod
    def from_pretrained(
        cls,
        representation_path: str | None = None,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "VGGTOmegaPipeline":
        """
        Load VGGT-Omega through the official Omega representation loader.

        Args:
            representation_path: Optional direct checkpoint path.
            reasoning_path: Reserved compatibility path.
            synthesis_path: Reserved compatibility path.
            model_path: Unified runner path or mapping.
            required_components: Optional adapter component mapping.
            device: Target torch device.
            model_id: Optional runtime model id.
            **kwargs: VGGT preprocessing/load options.
        """
        options = dict(required_components or {})
        options.update(kwargs)
        resolved_path = representation_path or cls._representation_path(model_path, **options)
        if device is not None:
            options["device"] = device
        del reasoning_path, synthesis_path

        from ..vggt.pipeline_vggt import VGGTPipeline

        representation_model = VGGTOmegaRepresentation.from_pretrained(
            pretrained_model_path=resolved_path,
            **options,
        )
        base = VGGTPipeline(representation_model=representation_model)
        return cls(
            representation_model=representation_model,
            reasoning_model=base.reasoning_model,
            synthesis_model=base.synthesis_model,
            base_pipeline=base,
            operator=cls.OPERATOR_CLS(),
            memory_module=cls.MEMORY_CLS(model_id=model_id or cls.MODEL_ID),
            model_id=model_id or cls.MODEL_ID,
        )

    def _delegate(self) -> Any:
        """
        Return a loaded VGGT delegate pipeline.

        Args:
            None. The delegate is created by ``from_pretrained``.
        """
        if self.base_pipeline is not None:
            return self.base_pipeline
        if self.representation_model is None:
            raise RuntimeError("VGGT-Omega delegate is not loaded. Use from_pretrained() first.")
        from ..vggt.pipeline_vggt import VGGTPipeline

        self.base_pipeline = VGGTPipeline(
            representation_model=self.representation_model,
            reasoning_model=self.reasoning_model,
            synthesis_model=self.synthesis_model,
            operator=self.operator,
        )
        return self.base_pipeline

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run VGGT-Omega base reconstruction while preserving official image preprocessing.

        Args:
            *args: Positional input image path or path list.
            **kwargs: Prediction and preprocessing options.
        """
        input_ = args[0] if args else kwargs.pop("input_", None)
        if input_ is None:
            input_ = kwargs.pop("images", None)
        if input_ is None:
            input_ = kwargs.pop("image_path", None)
        if input_ is None:
            raise ValueError("Provide image_path, images, or input_.")
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        interaction = kwargs.pop("interaction", None)
        image_paths = self._normalize_image_paths(input_)
        if interaction is None:
            interaction_dict = {
                "predict_cameras": True,
                "predict_depth": True,
                "predict_points": True,
                "predict_tracks": False,
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction

        data = {
            "images": image_paths,
            "predict_cameras": interaction_dict.get("predict_cameras", True),
            "predict_depth": interaction_dict.get("predict_depth", True),
            "predict_points": interaction_dict.get("predict_points", True),
            "predict_tracks": False,
            "preprocess_mode": kwargs.get("preprocess_mode", getattr(self.representation_model, "preprocess_mode", "balanced")),
            "resolution": kwargs.get("resolution", getattr(self.representation_model, "resolution", 512)),
            "patch_size": kwargs.get("patch_size", getattr(self.representation_model, "patch_size", 16)),
        }
        results = self.representation_model.get_representation(data)

        numpy_data = {}
        for key in [
            "extrinsic",
            "intrinsic",
            "depth_map",
            "depth_conf",
            "point_map",
            "point_conf",
            "point_map_from_depth",
            "camera_and_register_tokens",
            "text_alignment_embedding",
        ]:
            if key in results:
                numpy_data[key] = results[key]

        camera_params = []
        if "extrinsic" in results and "intrinsic" in results:
            for idx in range(results["extrinsic"].shape[0]):
                camera_params.append(
                    {
                        "extrinsic": results["extrinsic"][idx].tolist(),
                        "intrinsic": results["intrinsic"][idx].tolist(),
                    }
                )

        images = []
        if kwargs.get("return_visualization", True) and "depth_map" in results:
            from worldfoundry.core.media.artifacts import depths_to_pil_images

            depth_maps = results["depth_map"]
            if depth_maps.ndim == 2:
                depth_maps = depth_maps[np.newaxis, ...]
            images.extend(depths_to_pil_images(depth_maps, mode="grayscale"))

        from ..vggt.pipeline_vggt import VGGTResult

        return VGGTResult(
            images=images,
            numpy_data=numpy_data,
            camera_params=camera_params,
            data_type="image",
        )

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        return self._delegate().stream(*args, **kwargs)

    def reconstruct_ply(self, *args: Any, **kwargs: Any) -> Any:
        """Reconstruct ply for VGGTOmegaPipeline."""
        return self._delegate().reconstruct_ply(*args, **kwargs)

    def render_interaction_video_with_3dgs(self, *args: Any, **kwargs: Any) -> Any:
        """Render interaction video with 3dgs for VGGTOmegaPipeline."""
        return self._delegate().render_interaction_video_with_3dgs(*args, **kwargs)

    def render_orbit_video_with_3dgs(self, *args: Any, **kwargs: Any) -> Any:
        """Render orbit video with 3dgs for VGGTOmegaPipeline."""
        return self._delegate().render_orbit_video_with_3dgs(*args, **kwargs)

    def run_official_scene_export(
        self,
        image_path: Union[str, list[str]],
        output_dir: Optional[str] = None,
        image_resolution: Optional[int] = None,
        preprocess_mode: str = "balanced",
        patch_size: Optional[int] = None,
        video_sample_fps: float = 1.0,
        conf_thres: float = 20.0,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        show_cam: bool = True,
        max_points_k: int = 1000,
        output_name: Optional[str] = None,
        **_unused: Any,
    ) -> Dict[str, str]:
        """Run official scene export for VGGTOmegaPipeline."""
        if self.representation_model is None or self.representation_model.model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        if output_name is not None and not output_name.lower().endswith(".glb"):
            output_name = f"{Path(output_name).stem}.glb"

        from .official_runtime import (
            run_official_scene_export as _run_official_scene_export,
        )

        return _run_official_scene_export(
            input_source=image_path,
            model=self.representation_model.model,
            output_dir=output_dir or _default_output_dir(),
            device=self.representation_model.device,
            image_resolution=image_resolution or getattr(self.representation_model, "resolution", 512),
            preprocess_mode=preprocess_mode,
            patch_size=patch_size or getattr(self.representation_model, "patch_size", 16),
            video_sample_fps=video_sample_fps,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            max_points_k=max_points_k,
            output_name=output_name,
        )

    def __call__(
        self,
        image_path: Optional[Union[str, list[str]]] = None,
        images: Any = None,
        interactions: Optional[list[str]] = None,
        camera_view: Optional[list[float]] = None,
        task_type: Optional[str] = None,
        output_dir: Optional[str] = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any], Any]:
        """Execute the complete pipeline generation flow."""
        del camera_view
        data = images if images is not None else image_path
        if data is None:
            data = kwargs.pop("input_path", None) or kwargs.pop("video", None) or kwargs.pop("videos", None)
        if task_type in {"vggt_omega_official_scene_export", "vggt_omega_official_glb", "official"}:
            if output_dir is not None:
                kwargs.setdefault("output_dir", output_dir)
            result = self.run_official_scene_export(
                image_path=data,
                **kwargs,
            )
        elif task_type == "vggt_two_stage_3dgs":
            result = self._delegate()(
                image_path=image_path,
                images=images,
                interactions=interactions,
                task_type=task_type,
                **kwargs,
            )
        else:
            result = self.process(input_=data, interaction=interactions, **kwargs)
        self.memory_module.record(result, metadata={"type": "runtime_result", "model_id": self.model_id})
        if not return_dict:
            return result
        if isinstance(result, str):
            return {
                "status": "success",
                "model_id": self.model_id,
                "artifact_kind": "generated_video",
                "artifact_path": result,
                "runtime": "worldfoundry.pipelines.vggt_omega",
                "backend_quality": "in_tree_vggt_runtime",
            }
        artifact_path = result.get("glb_path") if isinstance(result, dict) else None
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_3d_asset",
            "artifact_path": artifact_path,
            "preview_model": artifact_path,
            "runtime": "worldfoundry.pipelines.vggt_omega",
            "backend_quality": "in_tree_vggt_runtime",
            "result_type": result.__class__.__name__,
            "result": result,
        }


__all__ = ["VGGTOmegaPipeline"]
