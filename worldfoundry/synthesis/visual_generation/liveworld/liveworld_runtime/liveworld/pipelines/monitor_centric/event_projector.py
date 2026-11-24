"""Event projection utilities.

This module converts event videos into event point clouds and supports
projection of those point clouds into arbitrary camera views.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# isort: split
from .video_projection import merge_video_foregrounds

# isort: split

from .event_types import EventPointCloud, EventVideo


class EventProjector:
    """Project event videos into point clouds and view-space projections."""

    def __init__(
        self,
        sam3_model_path: str,
        device: str,
        conf_threshold: float,
        rotation_angle: float = 0.0,
        depth_outlier_std_scale: float = 3.0,
        stride: int = 1,
        fg_mask_erode: int = 0,
    ) -> None:
        self.sam3_model_path = sam3_model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.rotation_angle = rotation_angle
        self.depth_outlier_std_scale = depth_outlier_std_scale
        self.stride = stride
        self.fg_mask_erode = fg_mask_erode

    def project(
        self,
        event_video: EventVideo,
        dynamic_prompts: list[str],
        output_dir: str,
        stream3r_model=None,
        sam3_segmenter=None,
        stream3r_session=None,
        stream3r_frames_fed: int = 0,
    ) -> tuple:
        """Generate an event point cloud from a fixed-camera event video.

        Args:
            stream3r_model: Optional pre-loaded STream3R model to reuse.
            sam3_segmenter: Optional pre-loaded SAM3 segmenter to reuse.
            stream3r_session: Optional shared StreamSession.
            stream3r_frames_fed: Frames already fed to the session.

        Returns:
            Tuple of (EventPointCloud, updated_frames_fed).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = merge_video_foregrounds(
            video_path=event_video.video_path,
            video_frames=event_video.frames,
            video_fps=event_video.fps,
            dynamic_object_prompts=dynamic_prompts,
            output_dir=str(output_path),
            sam3_model_path=self.sam3_model_path,
            rotation_angle=self.rotation_angle,
            depth_outlier_std_scale=self.depth_outlier_std_scale,
            conf_threshold=self.conf_threshold,
            device=self.device,
            # Keep projection internals quiet; outer pipeline logger already reports stage progress.
            verbose=False,
            stream3r_model=stream3r_model,
            sam3_segmenter=sam3_segmenter,
            stream3r_session=stream3r_session,
            stream3r_frames_fed=stream3r_frames_fed,
            stride=self.stride,
            save_depth_maps=False,
            save_rendered_frames=False,
            save_rendered_video=False,
            save_final_merged_pointcloud=False,
            fg_mask_erode=self.fg_mask_erode,
            skip_background=True,
        )

        points = result.get("final_fg_points")
        colors = result.get("final_fg_colors")
        if points is None or colors is None:
            raise KeyError("merge_video_foregrounds must return final_fg_points/colors")

        per_frame_points = result.get("all_foreground_points")
        per_frame_colors = result.get("all_foreground_colors")
        updated_frames_fed = result.get("stream3r_frames_fed", stream3r_frames_fed)

        # Per-instance data (empty dicts when instance segmentation is unavailable).
        pi_points = result.get("per_instance_per_frame_points") or None
        pi_colors = result.get("per_instance_per_frame_colors") or None
        pi_anchor = result.get("per_instance_anchor_masks") or None

        # Anchor frame (frame 0 of the I2V video) that masks were computed on.
        anchor_frame = None
        if event_video.frames is not None and len(event_video.frames) > 0:
            anchor_frame = np.asarray(event_video.frames[0], dtype=np.uint8)

        event_pc = EventPointCloud(
            event_id=event_video.event_id,
            points=points,
            colors=colors,
            anchor_pose_c2w=event_video.anchor_pose_c2w,
            frame_range=(0, event_video.num_frames),
            per_frame_points=per_frame_points,
            per_frame_colors=per_frame_colors,
            per_instance_per_frame_points=pi_points,
            per_instance_per_frame_colors=pi_colors,
            per_instance_anchor_masks=pi_anchor,
            anchor_frame=anchor_frame,
        )
        return event_pc, updated_frames_fed
