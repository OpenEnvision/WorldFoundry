"""Global world state for event-centric pipeline.

Stores static scene point clouds and dynamic event point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.geometry_utils import (
    transform_points,
)

from .event_types import EventID, EventPointCloud


@dataclass
class WorldState:
    """Container for global static and dynamic geometry."""

    static_points: np.ndarray | None = None
    static_colors: np.ndarray | None = None
    event_pointclouds: dict[EventID, EventPointCloud] = field(default_factory=dict)

    def update_static(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Replace the global static point cloud."""
        if points is None or colors is None:
            raise ValueError("static point cloud update requires points and colors")
        self.static_points = points
        self.static_colors = colors

    def update_event(self, event_pc: EventPointCloud) -> None:
        """Insert or replace an event point cloud."""
        if event_pc is None:
            raise ValueError("event_pc cannot be None")
        self.event_pointclouds[event_pc.event_id] = event_pc

    def list_event_pointclouds(self) -> list[EventPointCloud]:
        """Return all event point clouds."""
        return list(self.event_pointclouds.values())

    def get_static(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the current static point cloud (points, colors)."""
        return self.static_points, self.static_colors

    def get_union_event_points_world(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the union of all event point clouds transformed to world space.

        Each event's points are stored in anchor-camera space. This method
        transforms them to world space and concatenates them.

        Returns:
            (union_points, union_colors) or (None, None) if no events.
        """
        all_pts: list[np.ndarray] = []
        all_cols: list[np.ndarray] = []
        for epc in self.event_pointclouds.values():
            if epc.points is None or len(epc.points) == 0:
                continue
            # Use last per-frame points (most recent entity position) if available,
            # otherwise fall back to merged points.
            if epc.per_frame_points and len(epc.per_frame_points) > 0:
                pts_local = epc.per_frame_points[-1]
                cols_local = epc.per_frame_colors[-1] if epc.per_frame_colors else epc.colors
            else:
                pts_local = epc.points
                cols_local = epc.colors
            if len(pts_local) == 0:
                continue
            pts_world = transform_points(pts_local, epc.anchor_pose_c2w)
            all_pts.append(pts_world)
            all_cols.append(cols_local[: len(pts_world)])

        if not all_pts:
            return None, None
        return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0)
