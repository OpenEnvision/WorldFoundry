from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


SPLAT_UTILS = (
    Path(__file__).parents[2]
    / "worldfoundry/synthesis/visual_generation/worldgen/worldgen_runtime"
    / "src/worldgen/utils/splat_utils.py"
)
INFERENCE = SPLAT_UTILS.parents[3] / "inference.py"


def test_splat_export_clamps_degenerate_scales_before_log() -> None:
    tree = ast.parse(SPLAT_UTILS.read_text())
    save = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "save"
    )
    scale_assignment = next(
        node
        for node in ast.walk(save)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "scale" for target in node.targets)
    )
    expression = ast.Expression(scale_assignment.value)
    scales = np.array([[0.0, -1e-8, 1.0]], dtype=np.float32)
    result = eval(compile(expression, str(SPLAT_UTILS), "eval"), {"np": np, "self": type("S", (), {"scales": scales})()})

    assert np.isfinite(result).all()
    assert result[0, 2] == 0.0


def test_worldgen_mesh_export_round_trips_as_binary_glb(tmp_path: Path) -> None:
    tree = ast.parse(INFERENCE.read_text())
    save_scene = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_scene"
    )
    namespace = {"Path": Path, "os": os}
    exec(compile(ast.Module(body=[save_scene], type_ignores=[]), str(INFERENCE), "exec"), namespace)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float64)
        ),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    )

    output_path = namespace["save_scene"](mesh, tmp_path, return_mesh=True)
    open3d_mesh = o3d.io.read_triangle_mesh(str(output_path))
    trimesh_scene = trimesh.load(output_path, force="scene")

    assert len(open3d_mesh.vertices) == 3
    assert len(open3d_mesh.triangles) == 1
    assert sum(len(geometry.faces) for geometry in trimesh_scene.geometry.values()) == 1
