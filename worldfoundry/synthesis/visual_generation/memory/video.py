"""Memory for a model's generated image and video artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from worldfoundry.core.memory.media import extract_last_frame, infer_content_type, to_pil_image

if TYPE_CHECKING:
    from worldfoundry.core.media.processing.postprocess import VideoChunk

from ._model_scoped import ModelScopedMemory


class VideoArtifactMemory(ModelScopedMemory):
    """Rolling memory for one visual-generation model's image/video artifacts.

    Recorded videos also carry their last frame in metadata, so a caller asking
    for an image can be served from a video without decoding it again.
    """

    def record(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        video_chunk: VideoChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """Store an artifact, inferring its content type and keyframe.

        Args:
            data: Artifact content, e.g. image bytes or a video object.
            metadata: Extra metadata; its ``type`` overrides the inferred kind.
            video_chunk: Declared in-memory pixels for layout/range-aware keyframe extraction.
            kwargs: Further metadata entries, overridden by ``metadata``.
        """
        entry_metadata = {**kwargs, **dict(metadata or {})}
        if self.model_id is not None:
            entry_metadata = {"model_id": self.model_id, **entry_metadata}

        content_type = "video" if video_chunk is not None else infer_content_type(data)
        if content_type == "video":
            if video_chunk is None:
                last_frame = extract_last_frame(data)
            elif video_chunk.frame_count:
                from dataclasses import replace

                from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
                from worldfoundry.core.media.processing.postprocess import video_chunk_as_thwc

                tail = replace(
                    video_chunk,
                    frames=video_chunk_as_thwc(video_chunk)[-1:],
                    layout="frame-list" if video_chunk.layout == "frame-list" else "thwc",
                )
                last_frame = to_pil_image(video_chunk_to_uint8_frames(tail)[0])
            else:
                last_frame = None
            if last_frame is not None:
                entry_metadata["last_frame"] = last_frame

        self.append_record(
            data,
            kind=str(entry_metadata.get("type") or content_type),
            metadata=entry_metadata,
        )

    def select(self, context_query: Optional[Any] = None, prefer_type: str = "image", **kwargs: Any):
        """Return the newest artifact matching ``prefer_type``, or None.

        When an image is wanted, a stored video's last frame counts as a match.

        Args:
            context_query: Ignored.
            prefer_type: Artifact kind to look for, e.g. ``"image"``.
            kwargs: Ignored.
        """
        del context_query, kwargs
        for item in reversed(self.storage):
            if prefer_type == "image":
                if item["type"] == "image":
                    return item["content"]
                if item["metadata"].get("last_frame") is not None:
                    return item["metadata"]["last_frame"]
            if item["type"] == prefer_type:
                return item["content"]
        return None


__all__ = ["VideoArtifactMemory"]
