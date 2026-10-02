from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RASTERIZER_ROOT = REPO_ROOT / "thirdparty" / "diff-gaussian-rasterization"
PACKAGE_ROOT = RASTERIZER_ROOT / "diff_gaussian_rasterization"


def _load_rasterizer_with_fake_extension(monkeypatch):
    package_name = "_test_diff_gaussian_rasterization"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_ROOT)]
    fake_extension = types.ModuleType(f"{package_name}._C")
    backward_calls = []

    def rasterize_gaussians(*args):
        means3d = args[1]
        height = args[12]
        width = args[13]
        float_options = {"dtype": means3d.dtype, "device": means3d.device}
        buffer = torch.empty(0, dtype=torch.uint8, device=means3d.device)
        return (
            means3d.shape[0],
            torch.full((3, height, width), 1.0, **float_options),
            torch.full((1, height, width), 2.0, **float_options),
            torch.full((3, height, width), 3.0, **float_options),
            torch.full((1, height, width), 4.0, **float_options),
            torch.zeros(means3d.shape[0], dtype=torch.int32, device=means3d.device),
            buffer,
            buffer,
            buffer,
        )

    def rasterize_gaussians_backward(*args):
        means3d = args[1]
        colors = args[3]
        scales = args[4]
        rotations = args[5]
        covariances = args[7]
        grad_color = args[12]
        grad_depth = args[13]
        sh = args[14]
        backward_calls.append((grad_color.clone(), grad_depth.clone()))
        factor = grad_color.mean() + grad_depth.mean()
        return (
            torch.full_like(means3d, factor),
            torch.full_like(colors, factor),
            torch.full(
                (means3d.shape[0], 1),
                factor,
                dtype=means3d.dtype,
                device=means3d.device,
            ),
            torch.full_like(means3d, factor),
            torch.full_like(covariances, factor),
            torch.full_like(sh, factor),
            torch.full_like(scales, factor),
            torch.full_like(rotations, factor),
        )

    fake_extension.rasterize_gaussians = rasterize_gaussians
    fake_extension.rasterize_gaussians_backward = rasterize_gaussians_backward
    fake_extension.mark_visible = lambda positions, *_: torch.ones(
        positions.shape[0], dtype=torch.bool, device=positions.device
    )

    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}._C", fake_extension)
    spec = importlib.util.spec_from_file_location(
        package_name,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    return module, backward_calls


def _rasterizer_inputs(module):
    means3d = torch.zeros((2, 3), requires_grad=True)
    means2d = torch.zeros((2, 3), requires_grad=True)
    colors = torch.zeros((2, 3), requires_grad=True)
    opacities = torch.zeros((2, 1), requires_grad=True)
    scales = torch.ones((2, 3), requires_grad=True)
    rotations = torch.zeros((2, 4), requires_grad=True)
    settings = module.GaussianRasterizationSettings(
        image_height=2,
        image_width=3,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4),
        projmatrix=torch.eye(4),
        sh_degree=0,
        campos=torch.zeros(3),
        prefiltered=False,
        debug=False,
    )
    inputs = {
        "means3D": means3d,
        "means2D": means2d,
        "colors_precomp": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }
    return settings, inputs


def test_default_api_stays_three_outputs_and_aux_is_explicit(monkeypatch):
    module, backward_calls = _load_rasterizer_with_fake_extension(monkeypatch)
    settings, inputs = _rasterizer_inputs(module)

    default_outputs = module.GaussianRasterizer(settings)(**inputs)
    auxiliary_outputs = module.GaussianRasterizer(
        settings, return_extra_outputs=True
    )(**inputs)

    assert len(default_outputs) == 3
    assert len(auxiliary_outputs) == 5
    assert torch.equal(default_outputs[0], torch.full((3, 2, 3), 1.0))
    assert torch.equal(default_outputs[1], torch.zeros(2, dtype=torch.int32))
    assert torch.equal(default_outputs[2], torch.full((1, 2, 3), 2.0))
    assert torch.equal(auxiliary_outputs[0], default_outputs[0])
    assert torch.equal(auxiliary_outputs[1], default_outputs[1])
    assert torch.equal(auxiliary_outputs[2], default_outputs[2])
    assert torch.equal(auxiliary_outputs[3], torch.full((3, 2, 3), 3.0))
    assert torch.equal(auxiliary_outputs[4], torch.full((1, 2, 3), 4.0))

    loss = (
        default_outputs[0].sum()
        + default_outputs[2].sum()
        + auxiliary_outputs[0].sum()
        + auxiliary_outputs[2].sum()
        + 100.0 * auxiliary_outputs[3].sum()
        + 100.0 * auxiliary_outputs[4].sum()
    )
    loss.backward()

    assert len(backward_calls) == 2
    for grad_color, grad_depth in backward_calls:
        assert torch.equal(grad_color, torch.ones_like(grad_color))
        assert torch.equal(grad_depth, torch.ones_like(grad_depth))
    # Each canonical backward receives only color + depth gradients (factor 2).
    # Large auxiliary losses must not leak into the supported gradient path.
    assert torch.equal(inputs["means3D"].grad, torch.full((2, 3), 4.0))


def test_python_and_cuda_forward_contracts_preserve_canonical_behavior():
    python_source = (PACKAGE_ROOT / "__init__.py").read_text()
    tree = ast.parse(python_source)
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "_RasterizeGaussians" in class_names
    assert "_RasterizeGaussiansWithAux" in class_names

    forward_source = (RASTERIZER_ROOT / "cuda_rasterizer" / "forward.cu").read_text()
    assert "\tcov[0][0] += 0.3f;" in forward_source
    assert "\tcov[1][1] += 0.3f;" in forward_source
    assert "out_median_depth[H * W + pix_id] = median_weight;" in forward_source
    assert "out_opacity[pix_id] = 1.0f - T;" in forward_source

    binding_source = (RASTERIZER_ROOT / "rasterize_points.cu").read_text()
    assert "out_median_depth, out_opacity, radii" in binding_source
    assert (
        "return std::make_tuple(rendered, out_color, out_depth, "
        "out_median_depth, out_opacity, radii, geomBuffer, binningBuffer, "
        "imgBuffer);"
    ) in binding_source

    binding_header = (RASTERIZER_ROOT / "rasterize_points.h").read_text()
    forward_declaration = binding_header.split("RasterizeGaussiansCUDA", 1)[0]
    assert forward_declaration.count("torch::Tensor") == 8
    # Depth backward is a canonical feature and must remain wired through.
    assert "dL_ddepths.contiguous().data<float>()" in binding_source
