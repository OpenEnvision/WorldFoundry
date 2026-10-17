"""
Module for defining various memory structures tailored for visual data streams.

These memory classes extend a base memory system to specifically handle images, videos,
frames, and related visual context, enabling intelligent retrieval and state management
for models interacting with visual environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from worldfoundry.core.memory import BaseMemory
from worldfoundry.core.memory.media import (
    VIDEO_EXTENSIONS,
    extract_last_frame,
    extract_video_frames,
    infer_content_type,
    release_accelerator_cache,
    to_pil_image,
)


class VisualFrameMemory(BaseMemory):
    """
    Reusable image/video stream memory with latest-frame selection.

    This memory type stores and manages visual data, primarily focusing on
    keeping track of the most recent image frame, video path, and a general state.
    It can record various forms of visual input and allows selecting specific
    parts of the stored memory.
    """

    MODEL_ID: str | None = None

    def __init__(self, capacity: int | None = None, model_id: str | None = None, **kwargs: Any):
        """
        Initializes the VisualFrameMemory.

        Args:
            capacity: The maximum number of records to store. If None, capacity is unlimited.
            model_id: An optional ID for the model using this memory.
            **kwargs: Additional keyword arguments passed to the BaseMemory constructor.
        """
        super().__init__(capacity=capacity, **kwargs)
        self.model_id = model_id or self.MODEL_ID
        self.current_image: Image.Image | None = None  # Stores the latest PIL Image frame.
        self.current_video: Any = None  # Stores the path or object representing the latest video.
        self.current_state: Any = None  # Stores the latest general state or context.
        self.latest_result: Mapping[str, Any] | None = None  # Stores the raw latest result, often a mapping.
        self.all_frames: list[Any] = []  # Stores all recorded frames if `record_frames` is true.
        self.pose_history: list[Any] = []  # Stores a history of poses, if provided.

    def record(self, data: Any, metadata: Mapping[str, Any] | None = None, *, record_frames: bool = True, **kwargs: Any):
        """
        Records new data into the memory.

        Data can be an image, a video, a path, a list of frames, or a general state object.
        The method normalizes the input and updates the internal `current_image`,
        `current_video`, and `current_state` accordingly.

        Args:
            data: The data to record (e.g., PIL Image, video path, list of frames, dictionary state).
            metadata: Optional dictionary of metadata to associate with the record.
            record_frames: If True, all individual frames from a video/frame list will be stored
                           in `self.all_frames`.
            **kwargs: Additional metadata to merge with the `metadata` dictionary.
        """
        # Merge provided metadata with any additional kwargs.
        metadata = {**dict(metadata or {}), **kwargs}
        # Normalize the input data and determine its content type.
        content, kind = self._normalize_record(data, metadata=metadata, record_frames=record_frames)
        # If a model_id is set, ensure it's part of the metadata.
        if self.model_id is not None:
            metadata = {"model_id": self.model_id, **metadata}
        # Append the normalized content to the underlying BaseMemory store.
        self.append_record(content, kind=str(metadata.get("type") or kind), metadata=metadata)

    def select(self, context_query: Any = None, prefer_type: str | None = None, mode: str | None = None, **kwargs: Any):
        """
        Selects and retrieves specific content from the memory based on the query.

        Args:
            context_query: A specific query string (e.g., "latest_result", "state", "context").
            prefer_type: A content type to prefer when selecting the latest record.
            mode: A specific mode for selection (e.g., "dynamic").
            **kwargs: Additional keyword arguments (currently unused, deleted for clarity).

        Returns:
            The selected content (e.g., an image, video path, state object), or None if not found.
        """
        del kwargs  # Unused argument.
        if context_query == "latest_result":
            return self.latest_result
        if context_query in {"state", "context"}:
            return self.current_state
        if prefer_type is not None:
            # Retrieve the latest record of a specific type.
            record = self.latest_record(prefer_type=prefer_type)
            return record["content"] if record is not None else None
        if mode is not None and str(mode).lower() == "dynamic":
            # In "dynamic" mode, prioritize video over image.
            return self.current_video if self.current_video is not None else self.current_image
        # Default selection: prioritize image, then general state, then video.
        return self.current_image or self.current_state or self.current_video

    def manage(self, action: str = "reset", **kwargs: Any):
        """
        Manages the memory, supporting actions like "reset" and "evict".

        Args:
            action: The action to perform (e.g., "reset" to clear memory, "evict" to prune records).
            **kwargs: Additional keyword arguments (currently unused, deleted for clarity).
        """
        del kwargs  # Unused argument.
        if action == "reset":
            # Reset all internal state variables and clear records from the base memory.
            self.reset_records()
            self.current_image = None
            self.current_video = None
            self.current_state = None
            self.latest_result = None
            self.all_frames = []
            self.pose_history = []
            return
        if action == "evict" and self.capacity is not None:
            # Evict records if capacity is set and action is "evict".
            self._store.evict()

    def _normalize_record(self, data: Any, *, metadata: Mapping[str, Any], record_frames: bool) -> tuple[Any, str]:
        """
        Normalizes various input data types into a consistent format for storage.

        This method updates `current_image`, `current_video`, and `current_state`
        based on the type of data provided.

        Args:
            data: The raw input data to normalize.
            metadata: Metadata associated with the record.
            record_frames: Whether to store all individual frames if the data is a video/frame list.

        Returns:
            A tuple containing the normalized content and its inferred kind (e.g., "image", "video", "other").
        """
        if isinstance(data, Mapping):
            # If data is a dictionary, use a dedicated method to parse its content.
            return self._normalize_mapping(data, metadata=metadata, record_frames=record_frames)
        if isinstance(data, Image.Image):
            # If data is a PIL Image, convert to RGB, store as current image and state.
            image = data.convert("RGB")
            self.current_image = image
            self.current_state = image
            return image, "image"
        if isinstance(data, (list, tuple)):
            # If data is a list/tuple (assumed to be frames), process them.
            return self._record_frames(list(data), record_frames=record_frames)
        if isinstance(data, (str, Path)):
            # If data is a string or Path, check if it's a video file.
            path = str(Path(data).expanduser())
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                # If it's a video path, store it and extract the last frame as current image.
                self.current_video = path
                last_frame = extract_last_frame(path)
                if last_frame is not None:
                    self.current_image = last_frame
                return path, "video"
            # Otherwise, infer content type from the path (e.g., "image" for image files).
            return data, infer_content_type(data)

        # For other data types, try to infer content type and process accordingly.
        kind = infer_content_type(data)
        if kind == "video":
            # If inferred as a video (e.g., raw video object), extract frames.
            frames = extract_video_frames(data)
            return self._record_frames(frames, content=data, record_frames=record_frames)
        if kind == "image":
            # If inferred as an image (e.g., raw image bytes), convert to PIL Image.
            image = to_pil_image(data)
            self.current_image = image
            self.current_state = image
            return image, "image"
        # For any other type, store as current state and return as "other".
        self.current_state = data
        return data, "other"

    def _normalize_mapping(self, data: Mapping[str, Any], *, metadata: Mapping[str, Any], record_frames: bool) -> tuple[Any, str]:
        """
        Normalizes dictionary-based input data, looking for specific keys.

        This method handles structured data (e.g., model outputs, scene contexts)
        to update the memory's `current_state`, `current_image`, `current_video`,
        `latest_result`, and `pose_history`.

        Args:
            data: The dictionary data to normalize.
            metadata: Metadata associated with the record.
            record_frames: Whether to store all individual frames if the data contains a video/frame list.

        Returns:
            A tuple containing the normalized content and its inferred kind.
        """
        if data.get("kind") == "request_state" and "state" in data:
            # Handle specific "request_state" kind, update current_state.
            self.current_state = data["state"]
            scene_name = self.current_state.get("scene_name") if isinstance(self.current_state, Mapping) else None
            return {"scene_name": scene_name}, "other"

        if "scene_context" in data:
            # Handle "scene_context", update current_state and current_scene_context.
            self.current_state = data["scene_context"]
            self.current_scene_context = data["scene_context"]
            # If no frames/video are present, the scene context itself is the primary content.
            if not any(key in data for key in ("frames", "video", "video_tensor")):
                return data["scene_context"], str(metadata.get("type") or "scene_context")

        if "pose_history" in data:
            # Update pose history if present.
            self.pose_history = list(data.get("pose_history") or [])

        from worldfoundry.core.media.processing.postprocess import VideoChunk

        chunk = data.get("video_chunk")
        if isinstance(chunk, VideoChunk):
            from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames

            self.latest_result = data
            content = next(
                (data[key] for key in ("frames", "video", "video_tensor") if data.get(key) is not None),
                chunk.frames,
            )
            return self._record_frames(
                list(video_chunk_to_uint8_frames(chunk)), content=content, record_frames=record_frames
            )

        for key in ("frames", "video"):
            value = data.get(key)
            if value is not None:
                # If frames or video are present, store the whole dict as latest result,
                # then process frames/video.
                self.latest_result = data
                frames = value if isinstance(value, (list, tuple)) else extract_video_frames(value)
                content, kind = self._record_frames(frames, content=value, record_frames=record_frames)
                return content, kind

        if "video_tensor" in data:
            # If a video tensor is present, store as latest result and extract frames.
            self.latest_result = data
            value = data["video_tensor"]
            frames = extract_video_frames(value)
            return self._record_frames(frames, content=value, record_frames=record_frames)

        for key in ("generated_video_path", "video_path"):
            value = data.get(key)
            if isinstance(value, str):
                # If a video path is provided, store as latest result and current video.
                self.latest_result = data
                self.current_video = value
                return value, "video"

        if isinstance(data.get("last_frame"), Image.Image):
            # If a "last_frame" PIL Image is provided, store as current image and state.
            image = data["last_frame"].convert("RGB")
            self.current_image = image
            self.current_state = image
            return image, "image"

        if isinstance(data.get("input_image"), Image.Image):
            # If an "input_image" PIL Image is provided, store as current image and state.
            image = data["input_image"].convert("RGB")
            self.current_image = image
            self.current_state = image
            return image, "image"

        # If none of the specific keys are found, store the entire data dictionary
        # as the latest result and current state.
        self.latest_result = data
        self.current_state = data
        return data, str(metadata.get("type") or "other")

    def _record_frames(self, frames: list[Any], *, content: Any | None = None, record_frames: bool) -> tuple[Any, str]:
        """
        Processes a list of frames, updating memory state.

        Args:
            frames: A list of image frames (e.g., PIL Images, numpy arrays).
            content: The original content that generated these frames (e.g., video path).
                     Used as the primary content if available, otherwise frames list.
            record_frames: If True, extends `self.all_frames` with the provided frames.

        Returns:
            A tuple containing the primary content (original content or frames list)
            and its inferred kind ("video" if multiple frames, "image" if one, or "video" if empty but originally video).
        """
        if not frames:
            # If no frames are provided, use the original content if available, otherwise an empty list.
            return content if content is not None else [], "video"
        if record_frames:
            # Optionally store copies of all frames.
            self.all_frames.extend(frame.copy() if hasattr(frame, "copy") else frame for frame in frames)
        # Convert the last frame to a PIL Image and set it as the current image and state.
        image = to_pil_image(frames[-1])
        self.current_image = image
        # Set current_video to the original content or the list of frames.
        self.current_video = content if content is not None else frames
        self.current_state = image
        # Return the original content or frames list, inferring kind based on frame count.
        return content if content is not None else frames, "video" if len(frames) > 1 else "image"


class VisualContextMemory(VisualFrameMemory):
    """
    Visual stream memory with explicit reference-image/video/latent context.

    This class extends `VisualFrameMemory` by adding dedicated attributes
    for storing various types of visual context (e.g., reference images,
    reference videos, latent representations) that models might need.
    """

    CONTEXT_KEYS: tuple[str, ...] = ("ref_images", "ref_videos", "last_latents", "ref_latents")
    REQUIRED_CONTEXT_KEYS: tuple[str, ...] = ()

    def __init__(self, capacity: int | None = None, model_id: str | None = None, **kwargs: Any):
        """
        Initializes the VisualContextMemory.

        Args:
            capacity: The maximum number of records to store.
            model_id: An optional ID for the model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(capacity=capacity, model_id=model_id, **kwargs)
        self.n_generated_segments = 0  # Counter for generated segments/outputs.
        # Initialize all context attributes to None.
        for key in self.CONTEXT_KEYS:
            setattr(self, key, None)

    def has_context(self) -> bool:
        """
        Checks if the memory currently holds any (or all required) visual context.

        Returns:
            True if context is present according to `REQUIRED_CONTEXT_KEYS` or `CONTEXT_KEYS`, False otherwise.
        """
        keys = self.REQUIRED_CONTEXT_KEYS or self.CONTEXT_KEYS
        # If REQUIRED_CONTEXT_KEYS are specified, all of them must be present.
        # Otherwise, any of CONTEXT_KEYS being present is sufficient.
        return all(getattr(self, key, None) is not None for key in keys) if self.REQUIRED_CONTEXT_KEYS else any(
            getattr(self, key, None) is not None for key in keys
        )

    def record(
        self,
        data: Any,
        metadata: Mapping[str, Any] | None = None,
        *,
        record_frames: bool = True,
        as_context: bool = False,
        **kwargs: Any,
    ):
        """
        Records new data, optionally updating visual context and tracking generated segments.

        Args:
            data: The data to record.
            metadata: Optional metadata.
            record_frames: If True, individual frames are recorded.
            as_context: If True, the data is treated purely as context and not counted as a generated segment.
            **kwargs: Additional arguments, potentially containing context keys.
        """
        # Update context attributes from 'visual_context' key or direct context keys in kwargs.
        self._update_context(kwargs.get("visual_context"))
        self._update_context(kwargs)
        # Increment generated segments counter if data is a list/tuple and not explicitly marked as context.
        if isinstance(data, (list, tuple)) and data and not as_context:
            self.n_generated_segments += 1
        # Add generated segments count to metadata.
        metadata = {"n_generated_segments": self.n_generated_segments, **dict(metadata or {})}
        # Filter out context-related kwargs before passing to super().record.
        record_kwargs = {key: value for key, value in kwargs.items() if key != "visual_context" and key not in self.CONTEXT_KEYS}
        super().record(data, metadata=metadata, record_frames=record_frames, **record_kwargs)

    def select_context(self) -> dict[str, Any] | None:
        """
        Retrieves a dictionary of all active visual context items.

        Returns:
            A dictionary where keys are context types and values are the stored context data,
            or None if no context is present.
        """
        if not self.has_context():
            return None
        return {key: getattr(self, key, None) for key in self.CONTEXT_KEYS}

    def manage(self, action: str = "reset", **kwargs: Any):
        """
        Manages the memory, performing reset action and also clearing context attributes.

        Args:
            action: The action to perform (e.g., "reset").
            **kwargs: Additional keyword arguments.
        """
        super().manage(action=action, **kwargs)
        if action == "reset":
            # Reset generated segments counter and all context attributes.
            self.n_generated_segments = 0
            for key in self.CONTEXT_KEYS:
                setattr(self, key, None)
            # Release any GPU memory cached by accelerators.
            release_accelerator_cache()

    def _update_context(self, context: Any) -> None:
        """
        Helper method to update internal context attributes from a mapping.

        Args:
            context: A dictionary-like object containing context keys and values.
        """
        if not isinstance(context, Mapping):
            return
        for key in self.CONTEXT_KEYS:
            if key in context and context[key] is not None:
                setattr(self, key, context[key])


class LatentContextMemory(VisualContextMemory):
    """
    Visual context memory that requires latent continuation state.

    This memory type is specialized for models that rely on latent
    representations (e.g., from VAEs or diffusion models) for
    context continuation. It explicitly requires `last_latents` and
    `ref_latents` to be present for `has_context()` to return True.
    """

    REQUIRED_CONTEXT_KEYS = ("last_latents", "ref_latents")


class SceneStateMemory(VisualFrameMemory):
    """
    Memory for models whose stream state is a scene/context object plus frames.

    This class extends `VisualFrameMemory` to specifically handle a distinct
    `current_scene_context` in addition to general `current_state` and visual frames.
    It ensures that a persistent scene context is maintained even when new frames are recorded.
    """

    def __init__(self, capacity: int | None = None, model_id: str | None = None, **kwargs: Any):
        """
        Initializes the SceneStateMemory.

        Args:
            capacity: The maximum number of records to store.
            model_id: An optional ID for the model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(capacity=capacity, model_id=model_id, **kwargs)
        self.current_scene_context: Any = None  # Stores the latest scene context object.

    def record(self, data: Any, metadata: Mapping[str, Any] | None = None, *, record_frames: bool = True, **kwargs: Any):
        """
        Records new data, preserving `current_scene_context` if new data primarily contains frames.

        If the new data contains frames or video but no new explicit scene context,
        the previously stored `current_scene_context` and `current_state` are preserved
        to prevent them from being overwritten by just the visual frame.

        Args:
            data: The data to record.
            metadata: Optional metadata.
            record_frames: If True, individual frames are recorded.
            **kwargs: Additional keyword arguments.
        """
        previous_state = self.current_state
        previous_scene_context = self.current_scene_context
        super().record(data, metadata=metadata, record_frames=record_frames, **kwargs)
        # If the incoming data was primarily frames/video and didn't update the scene context,
        # restore the previous scene context and state. This ensures that a scene context
        # isn't implicitly overridden by just a new image/video frame.
        if isinstance(data, Mapping) and any(key in data for key in ("frames", "video", "video_tensor")):
            if previous_scene_context is not None:
                self.current_scene_context = previous_scene_context
                self.current_state = previous_scene_context  # Also restore current_state to scene context.
            elif previous_state is not None:
                self.current_state = previous_state  # If no scene context, restore general state.

    def select(self, context_query: Any = None, prefer_type: str | None = None, mode: str | None = None, **kwargs: Any):
        """
        Selects content, prioritizing `current_scene_context` if no specific query is given.

        Args:
            context_query: A specific query string.
            prefer_type: A content type to prefer.
            mode: A specific mode for selection.
            **kwargs: Additional keyword arguments.

        Returns:
            The selected content.
        """
        if context_query == "latest_result":
            return self.latest_result
        # If no specific query is made, prioritize the current_scene_context.
        if self.current_scene_context is not None and context_query is None and prefer_type is None and mode is None:
            return self.current_scene_context
        # Otherwise, fall back to the default selection logic of the parent class.
        return super().select(context_query=context_query, prefer_type=prefer_type, mode=mode, **kwargs)

    def manage(self, action: str = "reset", **kwargs: Any):
        """
        Manages the memory, resetting `current_scene_context` and accelerator cache on "reset".

        Args:
            action: The action to perform.
            **kwargs: Additional keyword arguments.
        """
        super().manage(action=action, **kwargs)
        if action == "reset":
            # Reset current scene context.
            self.current_scene_context = None
            # Release any GPU memory cached by accelerators.
            release_accelerator_cache()


__all__ = [
    "LatentContextMemory",
    "SceneStateMemory",
    "VisualContextMemory",
    "VisualFrameMemory",
]