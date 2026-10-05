"""Memory for a model's generated image and video artifacts."""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldfoundry.core.memory.media import extract_last_frame, infer_content_type

from ._model_scoped import ModelScopedMemory


class VideoArtifactMemory(ModelScopedMemory):
    """Rolling memory for one visual-generation model's image/video artifacts.

    Recorded videos also carry their last frame in metadata, so a caller asking
    for an image can be served from a video without decoding it again.
    """

    def record(self, data: Any, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Store an artifact, inferring its content type and keyframe.

        Args:
            data: Artifact content, e.g. image bytes or a video object.
            metadata: Extra metadata; its ``type`` overrides the inferred kind.
            kwargs: Further metadata entries, overridden by ``metadata``.
        """
        entry_metadata = {**kwargs, **dict(metadata or {})}
        if self.model_id is not None:
            entry_metadata = {"model_id": self.model_id, **entry_metadata}

        content_type = infer_content_type(data)
        if content_type == "video":
            last_frame = extract_last_frame(data)
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
