"""Artifact visualization without Studio dependencies.

Depth colormaps, optical-flow RGB, point-cloud renders, and WASD / IJKL
overlays for eval and debug. Nothing here sits on a model forward path:
callers import these helpers only when writing human-readable evidence.

Boundaries:
- OpenCV / matplotlib / ONNX sky-seg are pulled at import or call time;
  this is not a codec or storage layer.
- Action overlays mutate or convert frames; they do not persist video.
- :func:`download_file_from_url` is a legacy helper; new code should use
  :func:`worldfoundry.core.io.assets.download.download_to_cache`.

Public surface: colormap constants, depth / flow / normal visualizers,
game-control overlays, and open-loop action comparison plots.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from worldfoundry.core.geometry.transforms import rotation_matrix_to_euler_angles_opencv

# ──────────────────────────────────────────────────────────────────────────
# Depth colormaps — per-frame min/max stretch so NaN / empty maps stay blank
# ──────────────────────────────────────────────────────────────────────────

COLORMAP_VIRIDIS = "viridis"
COLORMAP_INFERNO = "inferno"

_CV2_COLORMAPS = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
}


def _resolve_colormap(colormap: str) -> int:
    """Map a case-insensitive name to a cv2 LUT; unknown names become viridis."""

    return _CV2_COLORMAPS.get(str(colormap).lower(), cv2.COLORMAP_VIRIDIS)


def squeeze_depth_to_2d(depth: np.ndarray) -> Optional[np.ndarray]:
    """Return ``(H, W)`` depth or ``None`` when leftover axes remain after squeeze."""

    arr = np.asarray(depth)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    return arr


def depth_to_uint8(depth: np.ndarray) -> Optional[np.ndarray]:
    """Stretch finite depths onto ``[0, 255]``; all-NaN maps become zeros, not noise.

    Non-2D inputs return ``None`` so a batch visualizer can skip a frame.
    Constant maps get a tiny vmax epsilon so OpenCV colormaps still receive
    a defined uint8 plane.
    """

    arr = np.asarray(depth, dtype=np.float64)
    squeezed = squeeze_depth_to_2d(arr)
    if squeezed is None:
        return None
    arr = squeezed
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    vmin = float(arr[valid].min())
    vmax = float(arr[valid].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = np.zeros_like(arr, dtype=np.float32)
    normalized[valid] = (arr[valid] - vmin) / (vmax - vmin)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def depth_to_colormap_rgb(
    depth_uint8: np.ndarray,
    colormap: str = COLORMAP_INFERNO,
) -> np.ndarray:
    """Apply a cv2 LUT and convert BGR → RGB for matplotlib / PIL consumers."""

    arr = np.asarray(depth_uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    colored_bgr = cv2.applyColorMap(arr.astype(np.uint8), _resolve_colormap(colormap))
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def depth_to_colormap_pil(
    depth: np.ndarray,
    colormap: str = COLORMAP_VIRIDIS,
    grayscale: bool = False,
) -> Optional[Image.Image]:
    """Build a PIL RGB image; ``None`` when the depth plane cannot be squeezed."""

    depth_uint8 = depth_to_uint8(depth)
    if depth_uint8 is None:
        return None
    if grayscale:
        rgb = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
    else:
        rgb = depth_to_colormap_rgb(depth_uint8, colormap=colormap)
    return Image.fromarray(rgb)


def save_depth_colormap(
    depth: np.ndarray,
    output_path: str,
    colormap: str = COLORMAP_VIRIDIS,
) -> Optional[str]:
    """Write a colormap image; return ``None`` instead of leaving an empty file."""

    img = depth_to_colormap_pil(depth, colormap=colormap)
    if img is None:
        return None
    img.save(output_path)
    return output_path


def depths_to_pil_images(
    depth_maps: np.ndarray,
    mode: Literal["grayscale", "colormap"] = "grayscale",
    colormap: str = COLORMAP_VIRIDIS,
) -> List[Image.Image]:
    """Convert a ``(H, W)`` or ``(N, H, W)`` stack, dropping unsqueezable frames."""

    arr = np.asarray(depth_maps)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    images: List[Image.Image] = []
    for depth in arr:
        if mode == "grayscale":
            depth_uint8 = depth_to_uint8(depth)
            if depth_uint8 is None:
                continue
            rgb = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
            images.append(Image.fromarray(rgb))
        else:
            img = depth_to_colormap_pil(depth, colormap=colormap)
            if img is not None:
                images.append(img)
    return images


def prepare_depth_visualization(
    depth: np.ndarray,
    grayscale: bool = False,
    colormap: str = COLORMAP_INFERNO,
) -> np.ndarray:
    """Always return an HxWx3 plane so overlay code can assume RGB/BGR.

    Already-uint8 maps skip the min/max stretch (they are display-ready).
    Unsqueezeable float maps fall back to a raw cast rather than raising,
    because debug viewers prefer a garbage frame over a crashed job.
    """

    if depth.dtype == np.uint8:
        depth_uint8 = squeeze_depth_to_2d(depth)
    else:
        depth_uint8 = depth_to_uint8(depth)
    if depth_uint8 is None:
        depth_uint8 = np.asarray(depth, dtype=np.uint8)
    if grayscale:
        return np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
    return cv2.applyColorMap(depth_uint8.astype(np.uint8), _resolve_colormap(colormap))


def build_depth_visualizations(
    depth: np.ndarray,
    colormap: str = COLORMAP_INFERNO,
) -> np.ndarray:
    """Stack per-frame RGB visualizations; unsqueezable maps become black."""

    arr = np.asarray(depth)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    visualizations = []
    for depth_map in arr:
        depth_uint8 = depth_to_uint8(depth_map)
        if depth_uint8 is None:
            vis = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
        else:
            vis = depth_to_colormap_rgb(depth_uint8, colormap=colormap)
        visualizations.append(vis)
    return np.stack(visualizations)


def create_depth_visualization(
    depth: np.ndarray,
    colormap: str = COLORMAP_VIRIDIS,
) -> Optional[np.ndarray]:
    """Single-map RGB colormap, or ``None`` when *depth* is missing / not 2D."""

    if depth is None:
        return None
    depth_uint8 = depth_to_uint8(depth)
    if depth_uint8 is None:
        return None
    return depth_to_colormap_rgb(depth_uint8, colormap=colormap)


def colorize_depth_map(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    near: float | None = None,
    far: float | None = None,
    cmap: str = "Spectral",
) -> np.ndarray:
    """Colorize metric depth via inverse-depth so nearby surfaces keep contrast.

    Zero / masked pixels become NaN before the 0.001 / 0.999 quantiles so
    sky and holes do not pin the near/far range. The Spectral LUT is
    inverted (``1 - disp``) to match the common "warm = near" convention.
    """

    import matplotlib

    assert depth.ndim == 2, "depth should be of shape (H, W)"
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    if near is None:
        near = np.nanquantile(depth, 0.001)
    if far is None:
        far = np.nanquantile(depth, 0.999)

    # Inverse-depth stretch: linear depth compresses the near field.
    disp = (1 / depth - 1 / far) / (1 / near - 1 / far)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    return np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────
# Surface normals and ONNX sky masks (debug overlays, not model inputs)
# ──────────────────────────────────────────────────────────────────────────


def colorize_normal_map(
    normal: np.ndarray,
    mask: np.ndarray | None = None,
    flip_yz: bool = False,
) -> np.ndarray:
    """Map unit normals from ``[-1, 1]`` to RGB; optional YZ flip for OpenGL vs OpenCV."""

    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    if flip_yz:
        # OpenGL-style +Y-up / +Z-forward → OpenCV display axes.
        normal = normal * [0.5, -0.5, -0.5] + 0.5
    else:
        normal = normal * 0.5 + 0.5
    return (normal.clip(0, 1) * 255).astype(np.uint8)


def run_skyseg(onnx_session, input_size: Sequence[int], image: np.ndarray) -> np.ndarray:
    """Run a SkySeg ONNX session at *input_size* and return a min-max uint8 logit map.

    ImageNet mean/std and BGR→RGB match the published SkySeg preprocess.
    The caller still owns resize-back and thresholding via :func:`segment_sky`.
    """

    import copy

    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(int(input_size[0]), int(input_size[1])))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, int(input_size[0]), int(input_size[1])).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    return onnx_result.astype("uint8")


def segment_sky(image_or_path, onnx_session, threshold: int = 32) -> np.ndarray:
    """Return a 255-on-sky mask; raise :class:`FileNotFoundError` if a path cannot be read.

    SkySeg is invoked at 320×320 (the published input) then resized to the
    source resolution. Values *below* ``threshold`` are treated as sky because
    the network's low logit band is the sky class after min-max stretch.
    """

    import os

    if isinstance(image_or_path, (str, os.PathLike)):
        image = cv2.imread(os.fspath(image_or_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image for sky segmentation: {image_or_path}")
    else:
        image = np.asarray(image_or_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < threshold] = 255
    return output_mask


# ──────────────────────────────────────────────────────────────────────────
# Legacy URL fetch — prefer download_to_cache (atomic rename + validation)
# ──────────────────────────────────────────────────────────────────────────


def download_file_from_url(url: str, filename: str, timeout: float = 60.0) -> str | None:
    """Download ``url`` to ``filename``; return the path, or None on failure.

    A failed or interrupted download never leaves a partial file behind:
    callers gate re-downloads on ``os.path.exists``, so a truncated artifact
    would otherwise be treated as valid forever. Prefer
    ``worldfoundry.core.io.assets.download.download_to_cache`` for new code (atomic
    rename plus validation).
    """
    import logging
    import os

    import requests

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(filename, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return filename
    except requests.exceptions.RequestException as exc:
        try:
            os.remove(filename)
        except OSError:
            pass
        logging.getLogger(__name__).warning("Error downloading %s to %s: %s", url, filename, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────
# Matplotlib / Baker–Scharstein flow — LUT helpers for tensors and UV fields
# ──────────────────────────────────────────────────────────────────────────


def apply_color_map(x, color_map: str = "inferno"):
    """Apply a matplotlib LUT to a ``[0, 1]`` tensor, keeping device and dropping alpha."""

    import matplotlib
    import torch

    try:
        cmap = matplotlib.colormaps.get_cmap(color_map)
    except AttributeError:
        from matplotlib import cm

        cmap = cm.get_cmap(color_map)
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(image, color_map: str = "inferno"):
    """Colorize an HWC-last tensor and permute to channel-first for torchvision grids."""

    from einops import rearrange

    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def make_colorwheel() -> np.ndarray:
    """Baker–Scharstein colorwheel (Middlebury flow viz); segment lengths are fixed by convention."""

    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6

    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3))
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    """Map unit-normalized UV flow to RGB; magnitude > 1 is desaturated, not clipped.

    ``arctan2(-v, -u)`` matches the Middlebury hue convention (0° = leftward).
    """

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    # Negated axes: Middlebury 0° points left, not right.
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        channel_index = 2 - i if convert_to_bgr else i
        flow_image[:, :, channel_index] = np.floor(255 * col)
    return flow_image


def flow_to_image(
    flow_uv: np.ndarray,
    clip_flow: float | None = None,
    convert_to_bgr: bool = False,
) -> np.ndarray:
    """Normalize ``[H, W, 2]`` flow by its peak radius, then colorize.

    Raises :class:`AssertionError` when the layout is not ``[H, W, 2]``.
    """

    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def _reshape_viz_batch_img(img_data, shape: int | str = 7) -> tuple:
    """Pack a batch into an HWC grid; ``shape`` is ``N`` or ``\"RxC\"``.

    Raises:
        RuntimeError: *shape* is neither an int nor an ``RxC`` string.
        TypeError: *img_data* is neither a tensor nor an ndarray.
    """

    import torch
    from einops import rearrange
    from torchvision.utils import make_grid

    if isinstance(shape, int):
        nrow, ncol = shape, shape
    elif isinstance(shape, str):
        if "x" not in shape:
            nrow, ncol = int(shape), int(shape)
        else:
            nrow_str, ncol_str = shape.split("x")
            nrow, ncol = int(nrow_str), int(ncol_str)
    else:
        raise RuntimeError(f"shape {shape} not support")

    if isinstance(img_data, torch.Tensor):
        assert img_data.shape[1] in [1, 3]
        grid_img = make_grid(img_data[: nrow * ncol].detach().cpu(), ncol)
        img = grid_img.permute(1, 2, 0)
    elif isinstance(img_data, np.ndarray):
        if img_data.shape[1] in [1, 3]:
            img = rearrange(img_data[: nrow * ncol], "(b t) c h w -> (b h) (t w) c", b=nrow)
        else:
            img = rearrange(img_data[: nrow * ncol], "(b t) h w c -> (b h) (t w) c", b=nrow)
    else:
        raise TypeError(f"Unsupported image data type: {type(img_data)}")
    return img, nrow, ncol


def show_batch_img(
    img_data,
    shape: int | str = 7,
    grid: int = 3,
    is_n1p1: bool = False,
    auto_n1p1: bool = True,
) -> None:
    """Show a batch grid; auto-detect ``[-1, 1]`` when the min is below ``-0.5``."""

    import matplotlib.pyplot as plt
    import torch

    if is_n1p1:
        img_data = (img_data + 1) / 2
    elif auto_n1p1:
        if isinstance(img_data, torch.Tensor):
            if img_data.min().item() < -0.5:
                img_data = (img_data + 1) / 2
        elif isinstance(img_data, np.ndarray) and np.min(img_data) < -0.5:
            img_data = (img_data + 1) / 2
    img, nrow, ncol = _reshape_viz_batch_img(img_data, shape)
    plt.figure(figsize=(ncol * grid, nrow * grid))
    plt.axis("off")
    plt.imshow(img)


def save_batch_img(fpath: str, img_data, shape: int | str = 7) -> None:
    """Write a batch grid as uint8; values are assumed to be in ``[0, 1]``."""

    import os

    import torch
    from PIL import Image

    img, _, _ = _reshape_viz_batch_img(img_data, shape)
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    im.save(fpath)


def visualize_latent_tensor_bcthw(tensor, nrow: int = 1, show_norm: bool = False, save_fig_path: str | None = None):
    """Plot channel-mean (and optional L2) of a ``B C (T N) H W`` latent for debug dumps."""

    import os

    import einops
    import matplotlib.pyplot as plt

    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c (t n) h w -> (b t h) (n w) c", n=nrow)
    tensor_mean = tensor.mean(-1)
    tensor_norm = tensor.norm(dim=-1)
    plt.figure(figsize=(20, 20))
    plt.imshow(tensor_mean)
    plt.title(f"mean {tensor_mean.mean()}, std {tensor_mean.std()}")
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    if show_norm:
        plt.figure(figsize=(20, 20))
        plt.imshow(tensor_norm)
        plt.show()


def visualize_tensor_bcthw(tensor, nrow: int = 4, save_fig_path: str | None = None):
    """Grid-plot a ``BCTHW`` video tensor; abort if max > 200 (almost certainly uint8)."""

    import os

    import einops
    import matplotlib.pyplot as plt
    import torchvision

    assert tensor.max() < 200, f"tensor max {tensor.max()} > 200, the data range is likely wrong"
    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c t h w -> (b t) c h w")
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)
    if save_fig_path is not None:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        torchvision.utils.save_image(tensor, save_fig_path)
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


# ──────────────────────────────────────────────────────────────────────────
# Point-cloud splat (z-buffer) — eval evidence, not a renderer used at train time
# ──────────────────────────────────────────────────────────────────────────


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    camera_to_world: np.ndarray,
    height: int,
    width: int,
    focal_scale: float = 1.0,
    splat_radius: int = 3,
) -> Image.Image:
    """Render a point cloud with strict z-buffer and front-to-back splatting."""

    c2w = camera_to_world.astype(np.float64)
    w2c = np.linalg.inv(c2w)
    rotation, translation = w2c[:3, :3], w2c[:3, 3]

    pts_cam = (rotation @ points.T).T + translation
    valid = pts_cam[:, 2] > 1e-4
    pts_cam = pts_cam[valid]
    cols = colors[valid]
    if cols.dtype in (np.float64, np.float32):
        if cols.max() <= 1.0:
            cols = (cols * 255).clip(0, 255).astype(np.uint8)
        else:
            cols = cols.clip(0, 255).astype(np.uint8)

    fx = fy = focal_scale * max(height, width)
    cx_img, cy_img = width / 2.0, height / 2.0

    u = np.round(fx * pts_cam[:, 0] / pts_cam[:, 2] + cx_img).astype(np.int32)
    v = np.round(fy * pts_cam[:, 1] / pts_cam[:, 2] + cy_img).astype(np.int32)
    z = pts_cam[:, 2].astype(np.float32)

    sort_idx = np.argsort(z)
    u, v, z, cols = u[sort_idx], v[sort_idx], z[sort_idx], cols[sort_idx]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    z_buf = np.full((height, width), np.inf, dtype=np.float32)

    radius = splat_radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            py = v + dy
            px = u + dx
            mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px_m, py_m, z_m, cols_m = px[mask], py[mask], z[mask], cols[mask]
            closer = z_m < z_buf[py_m, px_m]
            px_c, py_c = px_m[closer], py_m[closer]
            z_buf[py_c, px_c] = z_m[closer]
            canvas[py_c, px_c] = cols_m[closer]

    return Image.fromarray(canvas)


# ──────────────────────────────────────────────────────────────────────────
# Game-control overlays — Matrix-Game / GTA / Temple Run HUD on eval frames
# ──────────────────────────────────────────────────────────────────────────


def parse_game_control_config(
    config, mode: str = "universal"
) -> tuple[dict[int, dict[str, bool]], dict[int, tuple[float, float]]]:
    """Parse Matrix-Game style keyboard and mouse controls for overlay rendering."""

    if mode not in {"universal", "gta_drive", "templerun"}:
        raise AssertionError("mode must be one of ['universal', 'gta_drive', 'templerun']")

    key_data: dict[int, dict[str, bool]] = {}
    mouse_data: dict[int, tuple[float, float]] = {}
    if mode != "templerun":
        key, mouse = config
    else:
        key = config
        mouse = None

    for i in range(len(key)):
        if mode == "templerun":
            _still, w, s, left, right, a, d = key[i]
        elif mode == "universal":
            w, s, a, d = key[i]
        else:
            w, s, a, d = key[i][0], key[i][1], mouse[i][1] < 0, mouse[i][1] > 0

        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
        }
        if mode == "templerun":
            key_data[i].update({"left": bool(left), "right": bool(right)})

        if mode == "universal":
            mouse_y, mouse_x = mouse[i]
            mouse_y = -1 * mouse_y  # recipe Y is inverted relative to image +Y
            if i == 0:
                # First frame pins the cursor to the 640×352 letterbox center.
                mouse_data[i] = (320, 352 // 2)
            else:
                global_scale_factor = 0.1
                mouse_scale_x = 15 * global_scale_factor
                mouse_scale_y = 15 * 4 * global_scale_factor
                mouse_data[i] = (
                    mouse_data[i - 1][0] + mouse_x * mouse_scale_x,
                    mouse_data[i - 1][1] + mouse_y * mouse_scale_y,
                )
    return key_data, mouse_data


def draw_game_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 10,
    alpha: float = 0.5,
) -> None:
    """Blend a rounded rectangle onto *image* in place (OpenCV BGR)."""

    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_game_keys_on_frame(
    frame: np.ndarray,
    keys: dict[str, bool],
    key_size: tuple[int, int] = (80, 50),
    spacing: int = 20,
    bottom_margin: int = 30,
    mode: str = "universal",
) -> None:
    """Draw WASD (and Temple Run left/right) keys; green means pressed."""

    h, w, _ = frame.shape
    horizon_shift = 90
    vertical_shift = -20
    horizon_shift_all = 50
    key_positions = {
        "W": (
            w // 2 - key_size[0] // 2 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] * 2 + vertical_shift - 20,
        ),
        "A": (
            w // 2 - key_size[0] * 2 + 5 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "S": (
            w // 2 - key_size[0] // 2 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "D": (
            w // 2 + key_size[0] - 5 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
    }
    key_icon = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "left", "right": "right"}
    if mode == "templerun":
        key_positions.update(
            {
                "left": (
                    w // 2 + key_size[0] * 2 + spacing * 2 - horizon_shift - horizon_shift_all,
                    h - bottom_margin - key_size[1] + vertical_shift,
                ),
                "right": (
                    w // 2 + key_size[0] * 3 + spacing * 7 - horizon_shift - horizon_shift_all,
                    h - bottom_margin - key_size[1] + vertical_shift,
                ),
            }
        )

    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        if key in {"left", "right"}:
            bottom_right = (x + key_size[0] + 40, y + key_size[1])
        else:
            bottom_right = (x + key_size[0], y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5
        draw_game_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)

        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        if key in {"left", "right"}:
            text_x = x + (key_size[0] + 40 - text_size[0]) // 2
        else:
            text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key_icon[key], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def overlay_rgba_icon(
    frame: np.ndarray,
    icon: np.ndarray,
    position: tuple[float, float],
    scale: float = 1.0,
    rotation: float = 0,
) -> None:
    """Alpha-composite a BGRA icon at *position*, clipped to the frame bounds."""

    x, y = position
    h, w, _ = icon.shape
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(
        icon_resized,
        rotation_matrix,
        (scaled_width, scaled_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape
    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]
    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    for channel in range(3):
        frame_region[:, :, channel] = (1 - alpha) * frame_region[:, :, channel] + alpha * icon_rgb[:, :, channel]
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region


def process_game_control_video(
    input_video: Sequence[np.ndarray],
    config,
    mouse_icon_path: str | None,
    mouse_scale: float = 1.0,
    mouse_rotation: float = 0,
    process_icon: bool = True,
    mode: str = "universal",
) -> list[np.ndarray]:
    """Overlay keys/cursor on each frame and return float RGB in ``[0, 1]``."""

    key_data, mouse_data = parse_game_control_config(config, mode=mode)
    frame_width = input_video[0].shape[1]
    frame_height = input_video[0].shape[0]
    frame_count = len(input_video)

    mouse_icon = None
    if mouse_icon_path is not None:
        mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)

    out_video = []
    for frame_idx, frame in enumerate(input_video):
        if process_icon:
            keys = key_data.get(
                frame_idx, {"W": False, "A": False, "S": False, "D": False, "left": False, "right": False}
            )
            draw_game_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20, mode=mode)
            if mode == "universal":
                mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
                if mouse_icon is not None:
                    overlay_rgba_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_video.append(frame / 255)
        print(f"Processing frame {frame_idx + 1}/{frame_count}", end="\r")
    print("\nProcessing complete!")
    return out_video


def draw_navigation_rounded_rectangle(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    r: int,
    alpha: float = 0.6,
) -> np.ndarray:
    """Blend a rounded rectangle and return the mutated *img* (in-place)."""

    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_chevron(
    img: np.ndarray,
    center: tuple[int, int],
    size: int,
    direction: str,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    """Stroke a three-point chevron; unknown *direction* leaves *img* unchanged."""

    x, y = center
    offset = size // 2
    if direction == "up":
        pts = np.array([[x - offset, y + offset // 2], [x, y - offset // 2], [x + offset, y + offset // 2]], np.int32)
    elif direction == "down":
        pts = np.array([[x - offset, y - offset // 2], [x, y + offset // 2], [x + offset, y - offset // 2]], np.int32)
    elif direction == "left":
        pts = np.array([[x + offset // 2, y - offset], [x - offset // 2, y], [x + offset // 2, y + offset]], np.int32)
    elif direction == "right":
        pts = np.array([[x - offset // 2, y - offset], [x + offset // 2, y], [x - offset // 2, y + offset]], np.int32)
    else:
        return img
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def draw_wasd_ui(frame: np.ndarray, wasd_onehot: np.ndarray, position: tuple[int, int]) -> np.ndarray:
    """Highlight WASD keys from a length-4 one-hot (W, A, S, D)."""

    key_size = 40
    spacing = 5
    x, y = position
    keys = {
        "W": (x + key_size + spacing, y, 0),
        "A": (x, y + key_size + spacing, 1),
        "S": (x + key_size + spacing, y + key_size + spacing, 2),
        "D": (x + 2 * (key_size + spacing), y + key_size + spacing, 3),
    }

    for key, (kx, ky, idx) in keys.items():
        bg_color = (0, 100, 200) if wasd_onehot[idx] == 1 else (50, 50, 50)
        frame = draw_navigation_rounded_rectangle(
            frame, (kx, ky), (kx + key_size, ky + key_size), bg_color, -1, r=5, alpha=0.7
        )
        text_color = (255, 255, 255)
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = kx + (key_size - text_size[0]) // 2
        text_y = ky + (key_size + text_size[1]) // 2
        cv2.putText(
            frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
        )

    return frame


def draw_ijkl_ui(frame: np.ndarray, rotation_direction: str, width: int, height: int) -> np.ndarray:
    """Draw IJKL look keys in the lower-right; *rotation_direction* is up/left/down/right."""

    key_size = 40
    spacing = 5
    margin_right = 50
    margin_bottom = 50

    x = width - margin_right - 3 * (key_size + spacing) + spacing
    y = height - margin_bottom - 2 * (key_size + spacing) + spacing
    keys = {
        "I": (x + key_size + spacing, y, "up"),
        "J": (x, y + key_size + spacing, "left"),
        "K": (x + key_size + spacing, y + key_size + spacing, "down"),
        "L": (x + 2 * (key_size + spacing), y + key_size + spacing, "right"),
    }

    for key, (kx, ky, direction) in keys.items():
        bg_color = (200, 100, 0) if rotation_direction == direction else (50, 50, 50)
        frame = draw_navigation_rounded_rectangle(
            frame, (kx, ky), (kx + key_size, ky + key_size), bg_color, -1, r=5, alpha=0.7
        )
        text_color = (255, 255, 255)
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = kx + (key_size - text_size[0]) // 2
        text_y = ky + (key_size + text_size[1]) // 2
        cv2.putText(
            frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
        )

    return frame


def draw_rotation_ui(
    frame: np.ndarray, rotation_direction: str, width: int, height: int,     mode: str = "arrow"
) -> np.ndarray:
    """Draw an edge chevron or IJKL keys; ``\"no\"`` / unknown directions are no-ops."""

    if mode == "arrow":
        if rotation_direction == "no":
            return frame
        box_size = 60
        margin = 30
        bg_color = (0, 0, 0)
        arrow_color = (255, 255, 255)
        alpha = 0.5

        if rotation_direction == "left":
            center_x = margin + box_size // 2
            center_y = height // 2
        elif rotation_direction == "right":
            center_x = width - margin - box_size // 2
            center_y = height // 2
        elif rotation_direction == "up":
            center_x = width // 2
            center_y = margin + box_size // 2
        elif rotation_direction == "down":
            center_x = width // 2
            center_y = height - margin - box_size // 2
        else:
            return frame

        pt1 = (center_x - box_size // 2, center_y - box_size // 2)
        pt2 = (center_x + box_size // 2, center_y + box_size // 2)
        frame = draw_navigation_rounded_rectangle(frame, pt1, pt2, bg_color, -1, r=10, alpha=alpha)
        frame = draw_chevron(
            frame,
            (center_x, center_y),
            size=int(box_size * 0.5),
            direction=rotation_direction,
            color=arrow_color,
            thickness=4,
        )
    elif mode == "keys":
        frame = draw_ijkl_ui(frame, rotation_direction, width, height)
    return frame


def compute_rotation_angles_batch_opencv(c2w_a_batch: np.ndarray, c2w_b_batch: np.ndarray) -> np.ndarray:
    """Relative ZYX Euler (OpenCV) between paired camera-to-world matrices."""

    c2w_a_inv_batch = np.linalg.inv(c2w_a_batch)
    t_rel_batch = np.matmul(c2w_a_inv_batch, c2w_b_batch)
    r_rel_batch = t_rel_batch[:, :3, :3]
    rotation_angles = []
    for r_rel in r_rel_batch:
        z_angle, y_angle, x_angle = rotation_matrix_to_euler_angles_opencv(r_rel)
        rotation_angles.append([z_angle, y_angle, x_angle])
    return np.array(rotation_angles)


# ──────────────────────────────────────────────────────────────────────────
# Pose → discrete WASD / look labels (pair-wise; last frame padded idle)
# ──────────────────────────────────────────────────────────────────────────


def extract_rotation_directions(c2w_poses: np.ndarray, threshold: float = 0.005) -> list[str]:
    """Classify consecutive pose deltas as look left/right/up/down; last frame is ``\"no\"``.

    A terminal ``\"no\"`` pads the pair-wise deltas to pose length so overlays
    stay index-aligned with frames.
    """

    rotation_actions = []
    rotates = compute_rotation_angles_batch_opencv(c2w_poses[:-1], c2w_poses[1:])
    for rotate in rotates:
        z, right, up = rotate
        if max(abs(z), abs(right), abs(up)) < threshold:
            rotation_actions.append("no")
        elif max(abs(z), abs(right), abs(up)) == abs(right):
            rotation_actions.append("right" if right > 0 else "left")
        elif max(abs(z), abs(right), abs(up)) == abs(up):
            rotation_actions.append("up" if up > 0 else "down")
        else:
            rotation_actions.append("no")
    rotation_actions.append("no")  # pair-wise deltas are one shorter than poses
    return rotation_actions


def extract_translation_wasd(c2ws: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """One-hot WASD from camera-space translation; last row is idle to match frame count."""

    c2w_a_inv = np.linalg.inv(c2ws[:-1])
    t_rels = np.matmul(c2w_a_inv, c2ws[1:])

    wasd_actions = []
    for t_rel in t_rels:
        right, _down, forward = t_rel[:3, -1]
        if max(abs(right), abs(forward)) < threshold:
            wasd_actions.append([0, 0, 0, 0])
        elif abs(forward) > abs(right):
            wasd_actions.append([1, 0, 0, 0] if forward > 0 else [0, 0, 1, 0])
        else:
            wasd_actions.append([0, 0, 0, 1] if right > 0 else [0, 1, 0, 0])

    wasd_actions.append([0, 0, 0, 0])
    return np.array(wasd_actions)


def ijkl_onehot_to_direction(ijkl_onehot: np.ndarray) -> str:
    """Decode IJKL one-hot (up/left/down/right); first hit above 0.5 wins, else ``\"no\"``."""

    idx_to_direction = ["up", "left", "down", "right"]
    for i in range(4):
        if ijkl_onehot[i] > 0.5:
            return idx_to_direction[i]
    return "no"


def visualize_wasd_and_rotation_ui(
    frames: np.ndarray,
    c2ws: np.ndarray | None = None,
    wasd_actions: np.ndarray | None = None,
    ijkl_actions: np.ndarray | None = None,
    translation_threshold: float = 0.01,
    rotation_threshold: float = 0.005,
    rotation_ui_mode: str = "keys",
) -> np.ndarray:
    """Stamp WASD / look HUD onto ``[0, 1]`` RGB frames; require actions or poses.

    Raises:
        ValueError: neither *wasd_actions* nor *c2ws* is provided.
    """

    frames = (frames * 255).astype(np.uint8)
    frames = frames[..., ::-1]  # RGB → BGR for OpenCV drawing, reversed on return

    if wasd_actions is None and c2ws is None:
        raise ValueError("Either wasd_actions or c2ws must be provided.")
    if wasd_actions is None:
        wasd_actions = extract_translation_wasd(c2ws, translation_threshold)

    if ijkl_actions is not None:
        rotation_directions = [ijkl_onehot_to_direction(ijkl) for ijkl in ijkl_actions]
    elif c2ws is not None:
        rotation_directions = extract_rotation_directions(c2ws, rotation_threshold)
    else:
        rotation_directions = None

    num_frames, height, width, _ = frames.shape
    output_frames = []
    for frame_idx in range(min(num_frames, len(wasd_actions))):
        frame = frames[frame_idx].copy()
        frame = draw_wasd_ui(frame, wasd_actions[frame_idx], (50, height - 150))
        if rotation_directions is not None:
            frame = draw_rotation_ui(frame, rotation_directions[frame_idx], width, height, mode=rotation_ui_mode)
        output_frames.append(frame)

    output_frames = np.array(output_frames)[..., ::-1]
    return output_frames.astype(np.float32) / 255.0


def rotation_directions_to_onehot(rotation_directions: Sequence[str]) -> np.ndarray:
    """Encode look labels as IJKL one-hots; unknown labels become idle."""

    direction_to_onehot = {
        "up": [1, 0, 0, 0],
        "left": [0, 1, 0, 0],
        "down": [0, 0, 1, 0],
        "right": [0, 0, 0, 1],
        "no": [0, 0, 0, 0],
    }
    return np.array([direction_to_onehot.get(direction, [0, 0, 0, 0]) for direction in rotation_directions])


def save_openloop_action_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    *,
    action_chunk: int,
    n_chunk_action: int,
    save_path: str,
) -> None:
    """Plot GT vs inferred action dims; chunk starts are marked for open-loop eval."""

    import matplotlib.pyplot as plt

    num_dims = gt_actions.shape[-1]
    n_rows = min(10, num_dims)
    n_cols = 2 if num_dims > 10 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 28), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    x_axis = np.arange(gt_actions.shape[0])
    start_indices = np.arange(0, gt_actions.shape[0], action_chunk)

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        ax.plot(x_axis, gt_actions[:, dim_idx], label="Ground Truth", color="cornflowerblue", alpha=0.9)
        ax.plot(x_axis, pred_actions[:, dim_idx], label="Inferred", color="tomato", linestyle="--", alpha=0.9)
        ax.scatter(
            start_indices, gt_actions[start_indices, dim_idx], c="blue", marker="o", s=40, zorder=5, label="GT Start"
        )
        ax.scatter(
            start_indices,
            pred_actions[start_indices, dim_idx],
            c="darkred",
            marker="x",
            s=40,
            zorder=5,
            label="Inferred Start",
        )
        ax.set_title(f"Dimension- {dim_idx}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    fig.supxlabel(f"Continuous Timestep (across {n_chunk_action} inferences)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle("Comparison of Ground Truth and Inferred Actions", fontsize=18)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.clf()


__all__ = [
    "COLORMAP_INFERNO",
    "COLORMAP_VIRIDIS",
    "apply_color_map",
    "apply_color_map_to_image",
    "build_depth_visualizations",
    "colorize_depth_map",
    "colorize_normal_map",
    "compute_rotation_angles_batch_opencv",
    "create_depth_visualization",
    "depth_to_colormap_pil",
    "depth_to_colormap_rgb",
    "depth_to_uint8",
    "depths_to_pil_images",
    "download_file_from_url",
    "draw_chevron",
    "draw_game_keys_on_frame",
    "draw_game_rounded_rectangle",
    "draw_ijkl_ui",
    "draw_navigation_rounded_rectangle",
    "draw_rotation_ui",
    "draw_wasd_ui",
    "extract_rotation_directions",
    "extract_translation_wasd",
    "flow_to_image",
    "flow_uv_to_colors",
    "ijkl_onehot_to_direction",
    "make_colorwheel",
    "overlay_rgba_icon",
    "parse_game_control_config",
    "prepare_depth_visualization",
    "process_game_control_video",
    "render_point_cloud",
    "rotation_directions_to_onehot",
    "rotation_matrix_to_euler_angles_opencv",
    "run_skyseg",
    "save_batch_img",
    "save_openloop_action_comparison",
    "save_depth_colormap",
    "segment_sky",
    "show_batch_img",
    "squeeze_depth_to_2d",
    "visualize_latent_tensor_bcthw",
    "visualize_tensor_bcthw",
    "visualize_wasd_and_rotation_ui",
]
