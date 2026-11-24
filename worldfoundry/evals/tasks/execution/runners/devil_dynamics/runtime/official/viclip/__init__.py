"""Compatibility shim for DEVIL's upstream ``viclip`` imports."""

from worldfoundry.base_models.perception_core.video_text.viclip import (
    _frame_from_video,
    frames2tensor,
    get_viclip,
    get_vid_feat,
    retrieve_text,
)

__all__ = [
    "_frame_from_video",
    "frames2tensor",
    "get_viclip",
    "get_vid_feat",
    "retrieve_text",
]
