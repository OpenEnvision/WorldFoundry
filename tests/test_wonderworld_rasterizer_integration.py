from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WONDERWORLD_RENDERER = (
    REPO_ROOT
    / "worldfoundry"
    / "synthesis"
    / "visual_generation"
    / "wonderworld"
    / "wonderworld_runtime"
    / "gaussian_renderer"
    / "__init__.py"
)


def test_wonderworld_uses_canonical_rasterizer_with_aux_outputs() -> None:
    source = WONDERWORLD_RENDERER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "depth_diff_gaussian_rasterization_min" not in source
    assert "WORLDFOUNDRY_DEPTH_DIFF_GAUSSIAN_RASTERIZATION_EXTENSION_DIR" not in source
    assert "WORLDFOUNDRY_DIFF_GAUSSIAN_RASTERIZATION_EXTENSION_DIR" in source

    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "diff_gaussian_rasterization"
        for alias in node.names
    }
    assert imports == {"GaussianRasterizationSettings", "GaussianRasterizer"}

    rasterizer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GaussianRasterizer"
    ]
    assert len(rasterizer_calls) == 1
    keyword_values = {keyword.arg: keyword.value for keyword in rasterizer_calls[0].keywords}
    assert isinstance(keyword_values["return_extra_outputs"], ast.Constant)
    assert keyword_values["return_extra_outputs"].value is True


def test_wonderworld_unpacks_all_canonical_aux_outputs() -> None:
    tree = ast.parse(WONDERWORLD_RENDERER.read_text(encoding="utf-8"))

    rasterizer_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "rasterizer"
    ]
    assert rasterizer_assignments
    output_names = [
        element.id
        for element in rasterizer_assignments[0].targets[0].elts
        if isinstance(element, ast.Name)
    ]
    assert output_names == [
        "rendered_image",
        "radii",
        "depth",
        "median_depth",
        "final_opacity",
    ]
