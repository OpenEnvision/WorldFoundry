"""Synchronized model-owned sensor chunks on the existing RGB transport."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.studio.inference.execution import _normalize_frame_list


def sensor_layout_from_result(result):
    """Normalized atlas cells in model-declared order, independent of mapping order."""
    if not isinstance(result, Mapping) or "sensor_frames" not in result:
        return []
    spec = RealtimeSpec.from_payload(result)
    if not spec.sensors:
        raise ValueError("sensor_frames requires a realtime_spec with declared sensors")
    if set(result["sensor_frames"]) != {sensor.key for sensor in spec.sensors}:
        raise ValueError("sensor_frames must contain exactly the model-declared sensor keys")
    columns = math.ceil(math.sqrt(len(spec.sensors)))
    rows = math.ceil(len(spec.sensors) / columns)
    return [
        {
            "key": sensor.key,
            "kind": sensor.kind,
            "label": sensor.label or sensor.key,
            "channel_index": sensor.channel_index,
            "x": (index % columns) / columns,
            "y": (index // columns) / rows,
            "width": 1.0 / columns,
            "height": 1.0 / rows,
        }
        for index, sensor in enumerate(spec.sensors)
    ]


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _lidar_frame(points, range_m):
    points = _numpy(points)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("LiDAR frames require [N,3] XYZ or [N,4] XYZ/intensity arrays")
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    points = points[(np.abs(points[:, :2]) < range_m).all(axis=1)]
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    if len(points):
        xy = (points[:, :2] / (2.0 * range_m) + 0.5) * 511
        x, y = xy.astype(np.int64).T
        height = np.clip((points[:, 2] + 3.0) / 6.0, 0.0, 1.0)
        colors = np.stack((height * 255, (1.0 - height) * 255, np.full_like(height, 160)), axis=1)
        frame[511 - y, x] = colors.astype(np.uint8)
    return frame


def _depth_frame(depth, range_m):
    depth = _numpy(depth)
    if depth.ndim != 2:
        raise ValueError("Depth frames require metric [H,W] arrays")
    valid = np.isfinite(depth) & (depth > 0)
    shade = np.clip(np.where(valid, depth, 0) / range_m, 0.0, 1.0)
    rgb = np.stack((shade, 1.0 - shade, np.zeros_like(shade)), axis=-1)
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)


def compose_sensor_frames(result):
    """One atlas per timestep; reject unsynchronized or missing sensor channels."""
    layout = sensor_layout_from_result(result)
    spec = RealtimeSpec.from_payload(result)
    chunks = []
    for sensor in spec.sensors:
        values = result["sensor_frames"][sensor.key]
        if sensor.kind == "rgb":
            frames = _normalize_frame_list(values)
        elif sensor.kind == "lidar":
            frames = [_lidar_frame(points, sensor.range_m) for points in values]
        else:
            frames = [_depth_frame(depth, sensor.range_m) for depth in values]
        if not frames:
            raise ValueError(f"Sensor {sensor.key!r} returned no frames")
        chunks.append(frames)
    count = len(chunks[0])
    if any(len(frames) != count for frames in chunks):
        raise ValueError("Sensor chunks must have identical frame counts and cadence")
    tile_width = max(frame.shape[1] for frames in chunks for frame in frames)
    tile_height = max(frame.shape[0] for frames in chunks for frame in frames)
    columns, rows = round(1.0 / layout[0]["width"]), round(1.0 / layout[0]["height"])
    output = []
    for time in range(count):
        atlas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for index, frames in enumerate(chunks):
            image = Image.fromarray(np.ascontiguousarray(frames[time]))
            image.thumbnail((tile_width, tile_height), Image.Resampling.BILINEAR)
            x = (index % columns) * tile_width + (tile_width - image.width) // 2
            y = (index // columns) * tile_height + (tile_height - image.height) // 2
            atlas[y : y + image.height, x : x + image.width] = np.asarray(image)
        output.append(atlas)
    return output


__all__ = ["sensor_layout_from_result", "compose_sensor_frames"]
