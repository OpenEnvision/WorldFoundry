"""HY-World 2.0 pipeline wrapper for WorldMirror 2.0 reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..pipeline_utils import PipelineABC

if TYPE_CHECKING:
    from typing_extensions import Self

_WORLDRECON_TASKS = {
    "worldrecon",
    "world-recon",
    "reconstruction",
    "worldmirror",
    "world-mirror",
    "mirror",
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
    return str(value).strip().lower().replace("_", "-")


def _pop_first(mapping: dict[str, Any], keys, default=None):
    for key in keys:
        if key in mapping:
            return mapping.pop(key)
    return default


class HYWorld2Pipeline(PipelineABC):
    """Wrapper around the in-tree WorldMirror 2.0 reconstruction runtime."""

    def __init__(self, runtime_pipeline: Any, *, task: str = "worldrecon"):
        """Initialize the pipeline and configure runtime components."""
        self.runtime_pipeline = runtime_pipeline
        self.task = task

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str = "tencent/HY-World-2.0",
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        task: str | None = None,
        **kwargs,
    ) -> Self:
        """Load the pipeline from pretrained checkpoints and configurations."""
        runtime_kwargs: dict[str, Any] = {}
        if required_components:
            runtime_kwargs.update(required_components)
        runtime_kwargs.update(kwargs)

        task = task or _pop_first(
            runtime_kwargs,
            ("task", "mode", "pipeline_type"),
            "worldrecon",
        )
        if _canonical_key(task) not in _WORLDRECON_TASKS:
            raise ValueError(
                f"Unsupported HY-World 2.0 task: {task!r}. WorldFoundry runs WorldMirror 2.0 "
                "reconstruction (task='worldrecon') only."
            )

        from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.runtime import (
            WorldMirrorPipeline,
        )

        runtime = WorldMirrorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device=device,
            **runtime_kwargs,
        )
        return cls(runtime_pipeline=runtime, task="worldrecon")

    def __call__(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        input_path = input_path or kwargs.pop("image_path", None)
        images = kwargs.pop("images", None)
        if input_path is None and images:
            input_path = images[0] if isinstance(images, (list, tuple)) else images
        if input_path is None:
            input_path = kwargs.pop("image", None)
        runtime_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in _WORLDRECON_PASSTHROUGH_KWARGS}
        return self.runtime_pipeline(
            input_path=input_path,
            output_path=output_path or "inference_output",
            **runtime_kwargs,
        )

    def process(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        **kwargs,
    ):
        """Process and normalize input arguments and conditions for inference."""
        return self(
            input_path=input_path,
            output_path=output_path,
            **kwargs,
        )

    def __getattr__(self, name: str):
        """Getattr for HYWorld2Pipeline."""
        return getattr(self.runtime_pipeline, name)


HYWorldMirror2Pipeline = HYWorld2Pipeline


__all__ = [
    "HYWorld2Pipeline",
    "HYWorldMirror2Pipeline",
]
