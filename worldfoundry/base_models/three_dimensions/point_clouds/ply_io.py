"""Minimal scalar-property PLY I/O used by Gaussian inference runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

_PLY_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}
_DTYPE_TO_PLY = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}


class PointCloud(NamedTuple):
    """XYZ points, RGB colors in [0, 1], and optional surface normals."""

    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def write_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """Save XYZ positions and RGB colors in the uint8 [0, 255] range."""

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("Point cloud positions and colors must both have shape (N, 3)")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("Point cloud positions and colors must be finite")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("Point cloud RGB colors must be in the range [0, 255]")
    vertices = np.zeros(
        len(points),
        dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
        + [(name, "u1") for name in ("red", "green", "blue")],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    return write_ply(path, [("vertex", vertices)])


def read_point_cloud(path: str | Path) -> PointCloud:
    """Load a colored PLY point cloud; absent normals default to zero."""

    vertices = read_ply_vertex(path)
    required = ("x", "y", "z", "red", "green", "blue")
    if any(name not in vertices.dtype.names for name in required):
        raise ValueError("Point cloud PLY requires XYZ and RGB vertex properties")
    points = np.column_stack([vertices[name] for name in required[:3]]).astype(np.float32)
    colors = np.column_stack([vertices[name] for name in required[3:]]).astype(np.float32) / 255.0
    normals = (
        np.column_stack([vertices[name] for name in ("nx", "ny", "nz")]).astype(np.float32)
        if all(name in vertices.dtype.names for name in ("nx", "ny", "nz"))
        else np.zeros_like(points)
    )
    return PointCloud(points, colors, normals)


def write_ply(
    path: str | Path,
    elements: Sequence[tuple[str, np.ndarray]],
) -> Path:
    """Write structured NumPy arrays as binary little-endian PLY elements."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0"]
    prepared: list[np.ndarray] = []
    for element_name, values in elements:
        if values.dtype.names is None:
            raise TypeError(f"PLY element {element_name!r} must use a structured dtype.")
        header.append(f"element {element_name} {len(values)}")
        fields = []
        for property_name in values.dtype.names:
            dtype = values.dtype.fields[property_name][0]
            key = f"{dtype.kind}{dtype.itemsize}"
            if key not in _DTYPE_TO_PLY:
                raise TypeError(f"Unsupported PLY dtype for {property_name!r}: {dtype}.")
            header.append(f"property {_DTYPE_TO_PLY[key]} {property_name}")
            fields.append((property_name, "<" + key))
        prepared.append(values.astype(np.dtype(fields), copy=False))
    header.append("end_header")
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for values in prepared:
            handle.write(values.tobytes(order="C"))
    return path


def read_ply_vertex(path: str | Path) -> np.ndarray:
    """Read scalar vertex properties from ASCII or binary PLY files."""

    path = Path(path)
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}.")
        format_name = ""
        elements: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}.")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                format_name = parts[1]
            elif parts[0] == "element":
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                elements.append(current)
            elif parts[0] == "property" and current is not None:
                if parts[1] == "list":
                    current["properties"].append((None, None))
                else:
                    current["properties"].append((parts[2], parts[1]))

        if not format_name:
            raise ValueError(f"PLY format is missing in {path}.")
        for element in elements:
            count = int(element["count"])
            properties = element["properties"]
            if any(name is None for name, _ in properties):
                if element["name"] == "vertex":
                    raise ValueError("List-valued PLY vertex properties are unsupported.")
                raise ValueError("List-valued elements before vertex are unsupported.")
            endian = ">" if format_name == "binary_big_endian" else "<"
            dtype = np.dtype(
                [
                    (name, endian + _PLY_TO_DTYPE[type_name])
                    for name, type_name in properties
                ]
            )
            if format_name == "ascii":
                rows = [handle.readline().decode("ascii").split() for _ in range(count)]
                values = np.empty(count, dtype=dtype)
                for index, (name, _type_name) in enumerate(properties):
                    values[name] = np.asarray([row[index] for row in rows], dtype=dtype[name])
            elif format_name in {"binary_little_endian", "binary_big_endian"}:
                raw_values = handle.read(dtype.itemsize * count)
                values = np.frombuffer(raw_values, dtype=dtype, count=count).copy()
            else:
                raise ValueError(f"Unsupported PLY format: {format_name}.")
            if element["name"] == "vertex":
                return values
    raise ValueError(f"PLY file has no vertex element: {path}.")
