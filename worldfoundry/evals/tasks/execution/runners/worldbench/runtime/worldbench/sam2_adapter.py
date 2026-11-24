"""Compatibility imports for the shared in-tree SAM2 video tracker."""

from worldfoundry.base_models.perception_core.segment.sam2.video_tracker import (
    SAM2MaskTracker,
    stage_video_frames,
)

__all__ = ["SAM2MaskTracker", "stage_video_frames"]
