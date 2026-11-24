"""Depth Anything V3 visual generation pipeline module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v3 import (
    DEFAULT_DEPTH_ANYTHING3_REPO,
    DepthAnything3Representation,
)

from ...operators.depth_anything_v3_operator import DepthAnything3Operator
from ..pipeline_utils import PipelineABC


class DepthAnything3Pipeline(PipelineABC):
    """Pipeline wrapper for Depth Anything 3 depth, pose and point-cloud inference."""

    def __init__(
        self,
        representation: DepthAnything3Representation | None = None,
        operator: DepthAnything3Operator | None = None,
        device: str | None = None,
        default_process_res: int = 504,
        default_process_res_method: str = "upper_bound_resize",
        default_ref_view_strategy: str = "saddle_balanced",
        default_align_to_input_ext_scale: bool = True,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.representation = representation
        self.operator = operator or DepthAnything3Operator()
        self.default_process_res = int(default_process_res)
        self.default_process_res_method = str(default_process_res_method)
        self.default_ref_view_strategy = str(default_ref_view_strategy)
        self.default_align_to_input_ext_scale = bool(default_align_to_input_ext_scale)

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str | dict[str, Any] | None = None,
        required_components: dict[str, Any] | None = None,
        pretrained_model_path: str = DEFAULT_DEPTH_ANYTHING3_REPO,
        device: str | None = None,
        default_process_res: int = 504,
        default_process_res_method: str = "upper_bound_resize",
        default_ref_view_strategy: str = "saddle_balanced",
        default_align_to_input_ext_scale: bool = True,
        **kwargs,
    ) -> DepthAnything3Pipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop("model_path", None)
        pretrained_model_path = (
            model_path or component_options.pop("pretrained_model_path", None) or pretrained_model_path
        )
        default_process_res = component_options.pop("default_process_res", default_process_res)
        default_process_res_method = component_options.pop(
            "default_process_res_method",
            default_process_res_method,
        )
        default_ref_view_strategy = component_options.pop(
            "default_ref_view_strategy",
            default_ref_view_strategy,
        )
        default_align_to_input_ext_scale = component_options.pop(
            "default_align_to_input_ext_scale",
            default_align_to_input_ext_scale,
        )
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})

        representation = DepthAnything3Representation.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            default_process_res=default_process_res,
            default_process_res_method=default_process_res_method,
            default_ref_view_strategy=default_ref_view_strategy,
            default_align_to_input_ext_scale=default_align_to_input_ext_scale,
            **kwargs,
        )
        return cls(
            representation=representation,
            operator=DepthAnything3Operator(),
            device=device,
            default_process_res=default_process_res,
            default_process_res_method=default_process_res_method,
            default_ref_view_strategy=default_ref_view_strategy,
            default_align_to_input_ext_scale=default_align_to_input_ext_scale,
        )

    def _resolve_input_images(
        self,
        input_data=None,
        images=None,
        videos=None,
        data_type: str | None = None,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> dict[str, Any]:
        """Resolve input images for DepthAnything3Pipeline."""
        if videos is not None:
            video_info = self.operator.load_video_frames(
                videos,
                max_frames=max_frames,
                frame_stride=frame_stride,
            )
            return {
                "images": video_info["frames"],
                "input_type": "video",
                "video_info": video_info,
            }

        candidate = images if images is not None else input_data
        if candidate is None:
            raise ValueError("DepthAnything3Pipeline requires `input_data`, `images`, or `videos`.")

        if data_type == "video" or (isinstance(candidate, (str, Path)) and self.operator.is_video_path(candidate)):
            video_info = self.operator.load_video_frames(
                candidate,
                max_frames=max_frames,
                frame_stride=frame_stride,
            )
            return {
                "images": video_info["frames"],
                "input_type": "video",
                "video_info": video_info,
            }

        return {
            "images": self.operator.process_perception(candidate),
            "input_type": "image_sequence",
            "video_info": None,
        }

    def _resolve_interaction_flags(
        self,
        interactions: Sequence[str] | str | None,
        infer_gs: bool,
    ) -> dict[str, Any]:
        """Resolve interaction flags for DepthAnything3Pipeline."""
        flags = {
            "infer_gs": infer_gs,
            "expects_camera_inputs": False,
        }
        for signal in self.operator.normalize_interaction_sequence(interactions):
            self.operator.get_interaction(signal)
            try:
                processed = self.operator.process_interaction()
            finally:
                self.operator.delete_last_interaction()
            flags["infer_gs"] = flags["infer_gs"] or processed.get("infer_gs", False)
            flags["expects_camera_inputs"] = flags["expects_camera_inputs"] or processed.get(
                "expects_camera_inputs",
                False,
            )
        return flags

    def process(
        self,
        input_data=None,
        images=None,
        videos=None,
        interactions: Sequence[str] | str | None = None,
        extrinsics=None,
        intrinsics=None,
        output_dir: str | None = None,
        export_format: str = "depth_vis",
        export_feat_layers=None,
        process_res: int | None = None,
        process_res_method: str | None = None,
        use_ray_pose: bool = False,
        infer_gs: bool = False,
        ref_view_strategy: str | None = None,
        align_to_input_ext_scale: bool | None = None,
        render_exts=None,
        render_ixts=None,
        render_hw=None,
        data_type: str | None = None,
        max_frames: int | None = None,
        frame_stride: int = 1,
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
        export_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.representation is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        inputs = self._resolve_input_images(
            input_data=input_data,
            images=images,
            videos=videos,
            data_type=data_type,
            max_frames=max_frames,
            frame_stride=frame_stride,
        )
        interaction_flags = self._resolve_interaction_flags(
            interactions=interactions,
            infer_gs=infer_gs,
        )

        if interaction_flags["expects_camera_inputs"] and (extrinsics is None or intrinsics is None):
            raise ValueError(
                "DepthAnything3Pipeline received `pose_conditioned_depth` interaction but "
                "no `extrinsics`/`intrinsics` were provided."
            )

        result = self.representation.get_representation(
            {
                "images": inputs["images"],
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "export_dir": output_dir,
                "export_format": export_format,
                "export_feat_layers": export_feat_layers,
                "process_res": process_res or self.default_process_res,
                "process_res_method": process_res_method or self.default_process_res_method,
                "use_ray_pose": use_ray_pose,
                "infer_gs": interaction_flags["infer_gs"],
                "ref_view_strategy": ref_view_strategy or self.default_ref_view_strategy,
                "align_to_input_ext_scale": (
                    self.default_align_to_input_ext_scale
                    if align_to_input_ext_scale is None
                    else align_to_input_ext_scale
                ),
                "render_exts": render_exts,
                "render_ixts": render_ixts,
                "render_hw": render_hw,
                "conf_thresh_percentile": conf_thresh_percentile,
                "num_max_points": num_max_points,
                "show_cameras": show_cameras,
                "feat_vis_fps": feat_vis_fps,
                "export_kwargs": export_kwargs,
            }
        )
        result["input_type"] = inputs["input_type"]
        result["video_info"] = inputs["video_info"]
        return result

    def __call__(
        self,
        input_data=None,
        images=None,
        videos=None,
        video=None,
        interactions: Sequence[str] | str | None = None,
        output_dir: str | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if videos is None and video is not None:
            videos = video
        kwargs.pop("prompt", None)
        result = self.process(
            input_data=input_data,
            images=images,
            videos=videos,
            interactions=interactions,
            output_dir=output_dir,
            **kwargs,
        )
        if return_dict:
            return result
        return result

    def stream(self, *args, **kwargs):
        """Stream visual generation outputs chunk by chunk."""
        raise NotImplementedError(
            "DepthAnything3 streaming is provided by the upstream DA3-Streaming runtime and "
            "is not exposed by this WorldBench-X wrapper."
        )


__all__ = ["DepthAnything3Pipeline"]
