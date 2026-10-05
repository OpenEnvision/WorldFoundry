"""HY-World 2.0 pipeline wrapper.

This WorldFoundry wrapper currently exposes WorldMirror 2.0 reconstruction and
HY-Pano 2.0 panorama generation. The official HY-World 2.0 repository also
ships full world-generation stages, but those WorldNav / WorldStereo / 3DGS
stage runners are not yet vendored or wrapped here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...synthesis.visual_generation.hunyuan_world.hy_world_2p0_worldgen_runtime import (
    HYWorld2WorldgenNotIntegratedError,
)
from ..pipeline_utils import PipelineABC

_WORLDRECON_TASKS = {
    "worldrecon",
    "world-recon",
    "reconstruction",
    "worldmirror",
    "world-mirror",
    "mirror",
}
_PANO_TASKS = {
    "pano",
    "panorama",
    "panogen",
    "pano-gen",
    "hy-pano",
    "hy-pano-2.0",
}
_WORLDGEN_TASKS = {
    "worldgen",
    "world-gen",
    "world-generation",
    "generation",
    "text-to-3d-world",
    "image-to-3d-world",
}
_HUNYUAN_PANO_BACKENDS = {
    "hunyuan",
    "hunyuan-image",
    "hunyuan-image-3",
    "hunyuan_image",
    "hunyuan_image_3",
}
_QWEN_PANO_BACKENDS = {
    "qwen",
    "qwen-image",
    "qwen-image-edit",
    "qwen_image",
    "qwen_image_edit",
}
_WORLDRECON_PASSTHROUGH_KWARGS = {
    "target_size",
    "fps",
    "video_strategy",
    "video_min_frames",
    "video_max_frames",
    "save_depth",
    "save_normal",
    "save_gs",
    "save_camera",
    "save_points",
    "save_colmap",
    "save_conf",
    "apply_sky_mask",
    "apply_edge_mask",
    "apply_confidence_mask",
    "save_sky_mask",
    "sky_mask_source",
    "model_sky_threshold",
    "confidence_percentile",
    "edge_normal_threshold",
    "edge_depth_threshold",
    "compress_pts",
    "compress_pts_max_points",
    "compress_pts_voxel_size",
    "max_resolution",
    "compress_gs_max_points",
    "prior_cam_path",
    "prior_depth_path",
    "save_rendered",
    "render_interp_per_pair",
    "render_depth",
    "log_time",
    "strict_output_path",
}


def _canonical_key(value: Any) -> str:
    """Canonical key helper function."""
    return str(value).strip().lower().replace("_", "-")


def _pop_first(mapping: Dict[str, Any], keys, default=None):
    """Pop first helper function."""
    for key in keys:
        if key in mapping:
            return mapping.pop(key)
    return default


class HYWorld2Pipeline(PipelineABC):
    """Wrapper around the vendored HY-World 2.0 runtimes."""

    def __init__(
        self,
        runtime_pipeline: Any,
        *,
        task: str = "worldrecon",
        backend: Optional[str] = None,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.runtime_pipeline = runtime_pipeline
        self.task = task
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "tencent/HY-World-2.0",
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        task: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ) -> "HYWorld2Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        runtime_kwargs: Dict[str, Any] = {}
        if required_components:
            runtime_kwargs.update(required_components)
        runtime_kwargs.update(kwargs)

        task = task or _pop_first(
            runtime_kwargs,
            ("task", "mode", "pipeline_type"),
            "worldrecon",
        )
        task_key = _canonical_key(task)
        backend = backend or _pop_first(runtime_kwargs, ("backend", "pano_backend"))

        if task_key in _PANO_TASKS:
            runtime, resolved_backend = cls._load_pano_runtime(
                model_path=model_path,
                runtime_kwargs=runtime_kwargs,
                backend=backend,
                device=device,
            )
            return cls(
                runtime_pipeline=runtime,
                task="panorama",
                backend=resolved_backend,
            )

        if task_key in _WORLDGEN_TASKS:
            from ...synthesis.visual_generation.hunyuan_world.hy_world_2p0_worldgen_runtime import (
                raise_hy_world_2p0_worldgen_not_integrated,
            )

            raise_hy_world_2p0_worldgen_not_integrated(str(task))

        if task_key not in _WORLDRECON_TASKS:
            raise ValueError(
                f"Unsupported HY-World 2.0 task: {task!r}. "
                "Use 'worldrecon' or 'panorama'. Full 'worldgen' is not integrated in-tree."
            )

        from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.runtime import (
            WorldMirrorPipeline,
        )

        runtime = WorldMirrorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device=device,
            **runtime_kwargs,
        )
        return cls(runtime_pipeline=runtime, task="worldrecon", backend="worldmirror")

    @staticmethod
    def _load_pano_runtime(
        *,
        model_path: str,
        runtime_kwargs: Dict[str, Any],
        backend: Optional[str],
        device: str,
    ):
        """Load pano runtime for HYWorld2Pipeline."""
        backend_key = _canonical_key(backend or "hunyuan-image-3")

        if backend_key in _QWEN_PANO_BACKENDS:
            from ...synthesis.visual_generation.hunyuan_world.hy_world_2p0_panogen_runtime.pipeline_with_qwen_image import (
                HunyuanPanoPipeline,
            )

            init_kwargs = dict(runtime_kwargs)
            base_model = _pop_first(
                init_kwargs,
                ("pretrained_model_name_or_path", "base_model_path", "base_model"),
                HunyuanPanoPipeline.DEFAULT_MODEL_ID,
            )
            init_kwargs.setdefault("lora_path", model_path)
            init_kwargs.setdefault("device", device)
            return (
                HunyuanPanoPipeline.from_pretrained(base_model, **init_kwargs),
                "qwen_image",
            )

        if backend_key in _HUNYUAN_PANO_BACKENDS:
            from ...synthesis.visual_generation.hunyuan_world.hy_world_2p0_panogen_runtime.pipeline import (
                HunyuanPanoPipeline,
            )

            init_kwargs = dict(runtime_kwargs)
            init_kwargs.pop("device", None)
            pano_model_path = _pop_first(
                init_kwargs,
                ("pretrained_model_name_or_path", "pano_model_path"),
                model_path,
            )
            return (
                HunyuanPanoPipeline.from_pretrained(pano_model_path, **init_kwargs),
                "hunyuan_image_3",
            )

        raise ValueError(
            f"Unsupported HY-Pano 2.0 backend: {backend!r}. "
            "Use 'hunyuan_image_3' or 'qwen_image'."
        )

    def __call__(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.task == "panorama":
            image = kwargs.pop("image", None) or kwargs.pop("image_path", None) or input_path
            if image is None:
                raise ValueError("HY-Pano 2.0 requires an input image path.")

            save_path = kwargs.pop("save_path", None) or output_path
            output = self.runtime_pipeline(image, **kwargs)
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                output.save(save_path)
                return str(save_path)
            return output

        input_path = input_path or kwargs.pop("image_path", None)
        images = kwargs.pop("images", None)
        if input_path is None and images:
            if isinstance(images, (list, tuple)):
                input_path = images[0] if images else None
            else:
                input_path = images
        if input_path is None:
            input_path = kwargs.pop("image", None)
        runtime_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in _WORLDRECON_PASSTHROUGH_KWARGS
        }
        return self.runtime_pipeline(
            input_path=input_path,
            output_path=output_path or "inference_output",
            **runtime_kwargs,
        )

    def process(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        """Process and normalize input arguments and conditions for inference."""
        return self(
            input_path=input_path,
            output_path=output_path,
            **kwargs,
        )

    @staticmethod
    def worldgen_plan(repo_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the current HY-World 2.0 worldgen migration plan."""
        from ...synthesis.visual_generation.hunyuan_world.hy_world_2p0_worldgen_runtime import (
            get_hy_world_2p0_worldgen_plan,
        )

        return get_hy_world_2p0_worldgen_plan(repo_path)

    def __getattr__(self, name: str):
        """Getattr for HYWorld2Pipeline."""
        return getattr(self.runtime_pipeline, name)


HYWorldMirror2Pipeline = HYWorld2Pipeline


class HYWorld2PanoPipeline(HYWorld2Pipeline):
    """Convenience alias for loading HY-Pano 2.0 through HYWorld2Pipeline."""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "tencent/HY-World-2.0",
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        backend: Optional[str] = None,
        **kwargs,
    ) -> "HYWorld2PanoPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        return super().from_pretrained(
            model_path=model_path,
            required_components=required_components,
            device=device,
            task="panorama",
            backend=backend,
            **kwargs,
        )


__all__ = [
    "HYWorld2Pipeline",
    "HYWorldMirror2Pipeline",
    "HYWorld2PanoPipeline",
    "HYWorld2WorldgenNotIntegratedError",
]
