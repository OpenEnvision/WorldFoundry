"""Event-centric data structures and helpers.

This module keeps event-related types minimal and explicit to avoid
implicit behaviors in the pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Event identifier type alias (string hash).
EventID = str


def _sha1_hex(payload: bytes) -> str:
    """Compute a stable SHA1 hex digest for ID generation."""
    return hashlib.sha1(payload).hexdigest()


def hash_pose(pose_c2w: np.ndarray) -> str:
    """Hash a 4x4 pose matrix for stable IDs.

    The pose is rounded to reduce floating-point jitter before hashing.
    """
    if pose_c2w.shape != (4, 4):
        raise ValueError(f"pose_c2w must be 4x4, got {pose_c2w.shape}")
    rounded = np.round(pose_c2w.astype(np.float64), 6)
    return _sha1_hex(rounded.tobytes())


def make_event_id(entities: list[str], anchor_pose_c2w: np.ndarray, seed: str = "") -> EventID:
    """Create a deterministic event ID based on entities and anchor pose."""
    if not entities:
        raise ValueError("entities must be non-empty to create event_id")
    # Keep entity order stable to avoid ID drift across runs.
    ent_key = ",".join([e.strip().lower() for e in entities])
    pose_key = hash_pose(anchor_pose_c2w)
    payload = f"{ent_key}|{pose_key}|{seed}".encode()
    return _sha1_hex(payload)


@dataclass
class EntityDetectionResult:
    """Detection result for a single entity in a frame."""

    # Entity name (e.g., "car", "person").
    name: str
    # SAM3 segmentation mask for this entity (H, W), bool or uint8.
    mask: np.ndarray
    # Cropped entity image on black square background (H, W, 3), uint8.
    # Used for deduplication (DINOv3) and instance reference (observer conditioning).
    cropped_image: np.ndarray
    # Bounding box in original frame [x1, y1, x2, y2].
    bbox: tuple[int, int, int, int]
    # Cropped SAM instance mask aligned with cropped_image (H, W), uint8/bool.
    # Used by deduplication to select foreground patches for embedding.
    cropped_mask: np.ndarray | None = None


@dataclass
class EventObservation:
    """Observation of a potential event at a specific frame."""

    event_id: EventID
    frame_path: str
    frame_index: int
    pose_c2w: np.ndarray
    entities: list[str]
    timestamp: float
    # Frame as numpy array (H, W, 3), uint8 RGB. Used for detection.
    frame: np.ndarray | None = None
    # Detection result with mask and cropped image.
    detection: EntityDetectionResult | None = None
    # Optional full detection list for scene-level event registration.
    detections: list[EntityDetectionResult] | None = None


@dataclass
class EventScript:
    """Script used to evolve an event in time."""

    event_id: EventID
    text: str
    horizon: int
    fps: int


@dataclass
class EventVideo:
    """Generated event video (fixed camera)."""

    event_id: EventID
    video_path: str
    fps: int
    num_frames: int
    anchor_pose_c2w: np.ndarray
    # Optional in-memory frames [T, H, W, 3] uint8.
    # When provided, downstream projection can avoid re-reading encoded MP4.
    frames: np.ndarray | None = None


@dataclass
class EventPointCloud:
    """Event point cloud in the anchor camera coordinate system."""

    event_id: EventID
    points: np.ndarray
    colors: np.ndarray
    anchor_pose_c2w: np.ndarray
    frame_range: tuple[int, int]
    # Per-frame foreground point clouds (list of N arrays, one per event frame).
    # Each entry is (M_t, 3) points for frame t. Used for temporally-aligned
    # fg projection where frame t shows the entity at its position at time t.
    per_frame_points: list[np.ndarray] | None = None
    per_frame_colors: list[np.ndarray] | None = None
    # Per-instance per-frame point clouds: {obj_id: [pts_t0, pts_t1, ...]}.
    per_instance_per_frame_points: dict[int, list[np.ndarray]] | None = None
    per_instance_per_frame_colors: dict[int, list[np.ndarray]] | None = None
    # Per-instance anchor masks (frame 0): {obj_id: (H,W) bool}.
    per_instance_anchor_masks: dict[int, np.ndarray] | None = None
    # Anchor frame (frame 0 of the I2V video) that per_instance_anchor_masks
    # were computed on.  Used by _get_instance_reference_frames so the mask
    # always matches the image regardless of agent state updates.
    anchor_frame: np.ndarray | None = None  # (H, W, 3) uint8


@dataclass
class EventState:
    """Persistent state for a single event agent."""

    status: Literal["active", "ended", "paused"]
    anchor_pose_c2w: np.ndarray
    entities: list[str]
    # Current anchor frame for I2V generation.
    # For first iteration: the full scene frame (original input image).
    # For subsequent iterations: last frame of previous event video.
    current_anchor_frame: np.ndarray | None = None
    # Path to current anchor frame (if saved to disk).
    current_anchor_frame_path: str | None = None
    # Last generated video path.
    last_video_path: str | None = None
    # Number of evolution iterations completed.
    iteration_count: int = 0
    # DINOv3 embedding for entity deduplication (torch.Tensor stored as Any to avoid import).
    entity_embedding: object | None = None
    # Scene-level entity embeddings used for scene matching.
    scene_entity_embeddings: list[object] | None = None
    # Cropped RGB instances aligned with scene_entity_embeddings.
    scene_entity_crops: list[np.ndarray] | None = None
    # Cropped SAM masks aligned with scene_entity_embeddings.
    scene_entity_masks: list[np.ndarray] | None = None
    # Entity labels aligned with scene_entity_embeddings.
    scene_entity_labels: list[str] | None = None
    # Cached event target scene projection latent (fixed for this event camera).
    cached_target_scene_proj: object | None = None
    # Cached decoded RGB frames from cached_target_scene_proj.
    cached_scene_proj_pixels: list[np.ndarray] | None = None
    # Accumulated scene projection pixel frames across event iterations.
    # List of [H, W, 3] uint8 numpy arrays, decoded from target_scene_proj each iter.
    all_scene_proj_frames: list[np.ndarray] | None = None
    # Accumulated fg projection pixel frames across event iterations.
    # List of [H, W, 3] uint8 numpy arrays, decoded from target_fg_proj each iter.
    all_fg_proj_frames: list[np.ndarray] | None = None
    # All generated event frames across iterations (for preceding frame selection).
    # List of [H, W, 3] uint8 numpy arrays.
    all_generated_frames: list[np.ndarray] | None = None
    # Permanent scene reference frames (fg masked out): set once after first iteration.
    # [7, H, W, 3] uint8 numpy array.
    reference_frames_scene: np.ndarray | None = None
    # Permanent per-instance reference frames (fg crop on black bg): set once after first iteration.
    # [N_inst, H, W, 3] uint8 numpy array, one frame per detected instance.
    reference_frames_instance: np.ndarray | None = None
    # Whether this event was created via intermediate detection (mid-round).
    is_intermediate: bool = False
    # Local frame index within the originating round where entity first appeared.
    intermediate_anchor_local_frame: int | None = None
