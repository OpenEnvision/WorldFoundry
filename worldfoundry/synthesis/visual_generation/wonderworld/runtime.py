# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""In-tree single-image, layered Gaussian world generation.

Implements the WonderWorld inference stages described by Yu et al.
(arXiv:2406.09394): guided depth diffusion, region-refined disparity,
occlusion-layer completion and normal-oriented Gaussian surfels. Native
recipes own diffusion execution; canonical perception owns segmentation;
gsplat supplies differentiable rasterization, not an upstream runtime.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline, RuntimePolicy
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.extensions.depth_guidance import DecodedDepthGuidance
from worldfoundry.base_models.perception_core.segment.scene_segmentation import SceneSegmenter
from worldfoundry.core.geometry.surfels import GaussianSurfels
from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.runtime.wonderworld_inference_spec import WONDERWORLD_NEGATIVE_PROMPT
from worldfoundry.synthesis.base_synthesis import BaseSynthesis


def image_tensor(image: Image.Image, device: str) -> torch.Tensor:
    """Convert RGB to unit-range FP32 BCHW without changing geometry."""
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().float() / 255
    return pixels[None].to(device)


def image_pil(pixels: torch.Tensor) -> Image.Image:
    """Use the released truncating unit-RGB to uint8 conversion."""
    array = (pixels[0].detach().float().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def refine_disparity(
    depth: torch.Tensor,
    regions: list[dict[str, Any]],
    ground: torch.Tensor,
    *,
    existing_mask: torch.Tensor | None = None,
    existing_depth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge SAM regions small-to-large, retaining known background anchors."""
    disparity = (1 / depth[0, 0]).detach().float().cpu().numpy()
    refined = disparity.copy()
    ground_array = ground.detach().cpu().numpy().astype(bool)
    known = existing_mask.squeeze().detach().cpu().numpy().astype(bool) if existing_mask is not None else None
    known_disparity = (
        existing_depth.squeeze().detach().float().clamp_min(1e-10).reciprocal().cpu().numpy()
        if existing_depth is not None
        else None
    )
    ordered = sorted(regions, key=lambda region: int(np.count_nonzero(region["segmentation"])))
    for region in ordered:
        mask = np.asarray(region["segmentation"], dtype=bool)
        if np.count_nonzero(mask) <= 100 or np.count_nonzero(mask & ground_array) > 20:
            continue
        values = disparity[mask]
        lower, upper = np.percentile(values, [20, 80])
        interior = values[(values >= lower) & (values <= upper)]
        if interior.std() > 10:
            continue
        median = np.percentile(interior, 50)
        if known is not None:
            if np.count_nonzero(mask & ~known) < 5:
                continue
            anchors = mask & known
            if known_disparity is not None and np.count_nonzero(anchors) > 20:
                median = np.percentile(known_disparity[anchors], 50)
        refined[mask] = median
    return torch.from_numpy(1 / refined).to(depth)[None, None]


def ground_mask(semantic: torch.Tensor) -> torch.Tensor:
    """Select released ADE20K ground categories and erode by one pixel."""
    ground = torch.zeros_like(semantic, dtype=torch.bool)
    for category in (3, 6, 9, 11, 13, 26, 29, 46, 128):
        ground |= semantic == category
    return functional.max_pool2d((~ground)[None, None].float(), 3, 1, 1)[0, 0] == 0


def occluder_mask(semantic: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Select dilated occluder support before separate RGB/depth erosion."""
    import cv2
    from matplotlib import colormaps
    from scipy.ndimage import label

    labels = semantic.detach().cpu().numpy()
    disparity = (1 / depth[0, 0]).detach().float().cpu().numpy()
    span = float(disparity.max() - disparity.min())
    normalized = (disparity - disparity.min()) / span if span else np.zeros_like(disparity)
    green = (colormaps["viridis"](normalized)[..., 1] * 255).astype(np.uint8)
    dx = cv2.Sobel(green, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(green, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    gradient = cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = gradient > 10
    mask = np.zeros_like(labels, dtype=bool)
    kernel = np.ones((9, 9), np.uint8)
    ground_categories = {3, 6, 9, 11, 13, 26, 29, 46, 52, 128}
    for category in np.unique(labels):
        if int(category) in ground_categories:
            continue
        expanded = cv2.dilate((labels == category).astype(np.uint8), kernel).astype(bool)
        if int(category) in {4, 76, 83, 87}:
            mask |= expanded
            continue
        connected, count = label(expanded)
        for index in range(1, count + 1):
            region = connected == index
            values = disparity[region]
            if values.mean() < np.percentile(disparity, 60) or edges[region].mean() < 0.02:
                continue
            if disparity[region & edges].mean() > np.percentile(values, 70) or region.mean() < 0.001:
                continue
            interior = cv2.erode(region.astype(np.uint8), kernel).astype(bool)
            lower, upper = np.percentile(disparity[interior], [20, 80])
            if 1 / lower - 1 / upper > 0.0003 and region.mean() > 0.05:
                continue
            mask |= region
    return torch.from_numpy(mask).to(depth.device)[None, None]


class WonderWorldSynthesis(BaseSynthesis):
    """Resident component weights with isolated, finite offline scene requests."""

    def __init__(self, paths: Mapping[str, str], *, device: str, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.paths = dict(paths)
        self.device = device
        self.dtype = dtype
        self.models: dict[str, NativeDiffusionPipeline] = {}
        self.segmenter: SceneSegmenter | None = None

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: Any = None, args: Any = None, device: str = "cuda", **kwargs: Any
    ) -> WonderWorldSynthesis:
        """Record explicit component paths; defer checkpoint and CUDA loading."""
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_root"] = str(pretrained_model_path)
        options.update(kwargs)
        options.pop("model_id", None)
        root = options.pop("checkpoint_root", None)
        names = {
            "inpainting_path": "stable-diffusion-2-inpainting",
            "depth_path": "prs-eth--marigold-depth-v1-0",
            "normals_path": "prs-eth--marigold-normals-v0-1",
            "oneformer_path": "oneformer_ade20k_swin_large",
            "repvit_checkpoint": "RepViT/repvit_sam.pt",
        }
        paths = {
            key: str(options.pop(key, Path(str(root)) / name if root else checkpoint_root_path(name)))
            for key, name in names.items()
        }
        dtype_name = str(options.pop("dtype", "bfloat16"))
        if dtype_name not in {"bfloat16", "float32"}:
            raise ValueError("WonderWorld dtype must be bfloat16 or float32")
        if options:
            raise TypeError(f"unsupported WonderWorld load options: {sorted(options)}")
        return cls(paths, device=device, dtype=getattr(torch, dtype_name))

    def _ensure_models(self) -> None:
        """Assemble native recipes only after input and asset preflight."""
        if self.models and self.segmenter is not None:
            return
        required = {
            "inpainting_path": "unet/diffusion_pytorch_model.fp16.safetensors",
            "depth_path": "unet/diffusion_pytorch_model.fp16.safetensors",
            "normals_path": "unet/diffusion_pytorch_model.fp16.safetensors",
            "oneformer_path": "model.safetensors",
            "repvit_checkpoint": "",
        }
        for name, relative in required.items():
            path = Path(self.paths[name]).expanduser() / relative
            if not path.is_file():
                raise FileNotFoundError(f"WonderWorld {name} is missing: {path}")
        policy = RuntimePolicy(device=self.device, dtype=self.dtype)
        try:
            for mode, model_id, path_name in (
                ("inpaint", "sd2-inpaint", "inpainting_path"),
                ("depth", "wonderworld-marigold-depth", "depth_path"),
                ("normals", "marigold-normals-v0-1", "normals_path"),
            ):
                root = self.paths[path_name]
                self.models[mode] = NativeDiffusionPipeline.from_pretrained(
                    model_id,
                    policy=policy,
                    checkpoint_overrides={
                        key: root for key in ("unet", "text_encoder", "tokenizer", "vae", "scheduler")
                    },
                    extensions=(DecodedDepthGuidance(),) if mode == "depth" else (),
                )
            self.segmenter = SceneSegmenter.from_pretrained(
                self.paths["oneformer_path"], self.paths["repvit_checkpoint"], device=self.device
            )
        except BaseException:
            self.models.clear()
            self.segmenter = None
            raise

    def _depth(
        self,
        pixels: torch.Tensor,
        *,
        steps: int,
        seed: int,
        target: torch.Tensor | None = None,
        align: torch.Tensor | None = None,
        farther: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signed = image_tensor(image_pil(pixels), self.device) * 2 - 1
        return self.models["depth"](
            DiffusionRequest(
                prompt="",
                height=512,
                width=512,
                sampling=SamplingConfig(num_inference_steps=steps, seed=seed),
                inputs={
                    "image": signed,
                    "target_depth": target,
                    "align_mask": align,
                    "farther_mask": farther,
                    "guidance_steps": min(8, steps),
                },
            )
        ).sample.float()

    def _normals(self, pixels: torch.Tensor, *, steps: int, seed: int) -> torch.Tensor:
        normals = self.models["normals"](
            DiffusionRequest(
                prompt="",
                height=512,
                width=512,
                sampling=SamplingConfig(num_inference_steps=steps, seed=seed),
                inputs={"image": pixels * 2 - 1, "processing_resolution": 768},
            )
        ).sample.float()
        return normals * normals.new_tensor([1, -1, -1])[None, :, None, None]

    def _inpaint(
        self,
        pixels: torch.Tensor,
        mask: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str,
        steps: int,
        seed: int,
        fill_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import cv2

        if not mask.any():
            return pixels
        raw = np.asarray(image_pil(pixels))
        fill_region = mask if fill_mask is None else fill_mask
        fill = cv2.inpaint(raw, fill_region[0, 0].detach().cpu().numpy().astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        # Any hole in an 8x8 latent block promotes that entire block.
        block_mask = functional.max_pool2d(mask.float(), 8, 8)
        mask = functional.interpolate(block_mask, (512, 512), mode="nearest")
        generated = self.models["inpaint"](
            DiffusionRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=512,
                width=512,
                sampling=SamplingConfig(num_inference_steps=steps, guidance_scale=7.5, seed=seed),
                inputs={"image": image_tensor(Image.fromarray(fill), self.device) * 2 - 1, "mask": mask},
            )
        ).sample.float()
        return generated

    @torch.no_grad()
    def predict(
        self,
        *,
        image: Image.Image,
        camera_path: np.ndarray,
        prompt: str = "",
        output_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        seed: int = 42,
        depth_steps: int = 30,
        inpaint_steps: int = 50,
        normal_steps: int = 10,
        refinement_steps: int = 100,
        focal_length: float = 500,
        gen_layer: bool = True,
        negative_prompt: str = WONDERWORLD_NEGATIVE_PROMPT,
    ) -> dict[str, Any]:
        """Generate a seed scene, extend it at every requested view, and export.

        Camera translations use exported scene units: depth is scaled by 1000,
        matching the upstream viewer. Every call starts a new world; component
        weights are the only resident state. No HTTP server or subprocess runs.
        """
        self._ensure_models()
        if self.segmenter is None:
            raise RuntimeError("WonderWorld scene segmenter is not initialized")
        directory = Path(run_dir) if run_dir is not None else Path(tempfile.mkdtemp(prefix="wonderworld_"))
        directory.mkdir(parents=True, exist_ok=True)
        destination = Path(output_path) if output_path is not None else directory / "wonderworld.splat"
        intrinsic = torch.tensor(
            [[focal_length, 0, 256], [0, focal_length, 256], [0, 0, 1]], device=self.device, dtype=torch.float32
        )
        cameras = [torch.eye(4, device=self.device), *torch.from_numpy(camera_path.copy()).to(self.device).unbind()]
        pixels = image_tensor(image, self.device)
        scene: GaussianSurfels | None = None
        views: list[dict[str, Any]] = []
        for index, camera in enumerate(cameras):
            current_seed = seed + index * 4
            if scene is None:
                new_region = torch.ones((1, 1, 512, 512), device=self.device, dtype=torch.bool)
                existing_depth = torch.zeros_like(new_region, dtype=torch.float32)
                existing_mask = torch.zeros_like(new_region)
            else:
                rendered, alpha, rendered_depth = scene.render(camera, intrinsic, height=512, width=512)
                fill_mask = alpha < 0.6
                holes = alpha < 0.01
                condition = rendered
                if (scene.layers == 0).any():
                    no_sky, no_sky_alpha, _ = scene.render(camera, intrinsic, height=512, width=512, exclude_layer=0)
                    full_render = torch.zeros_like(alpha, dtype=torch.bool)
                    columns = torch.nonzero((no_sky_alpha[0, 0] >= 0.6).sum(dim=0) > 150).flatten()
                    if columns.numel():
                        full_render[:, :, :, columns.min() : columns.max() + 1] = True
                    full_render[:, :, :40] = True
                    full_render[:, :, :128, :40] = True
                    full_render[:, :, :128, -40:] = True
                    condition = torch.where(full_render, rendered, no_sky)
                    fill_mask = torch.where(full_render, fill_mask, no_sky_alpha < 0.6)
                    holes = torch.where(full_render, holes, no_sky_alpha < 0.01)
                new_region = functional.max_pool2d(holes.float(), 7, 1, 3) > 0
                existing_mask = (rendered_depth > 1) & (rendered_depth < 4.8)
                existing_depth = rendered_depth / 1000
                if not new_region.any():
                    image_pil(rendered).save(directory / f"view_{index:03d}.png")
                    views.append(
                        {"index": index, "new_pixels": 0, "new_surfels": 0, "world_to_camera": camera.cpu().tolist()}
                    )
                    continue
                pixels = self._inpaint(
                    condition,
                    new_region,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=inpaint_steps,
                    seed=current_seed,
                    fill_mask=fill_mask,
                )
            semantic, regions = self.segmenter(image_pil(pixels))
            sky = (semantic == 2)[None, None]
            ground = ground_mask(semantic)
            yy = (torch.arange(512, device=self.device).float()[:, None] + 0.5).expand(512, 512)
            # The ground plane is world Y=0.3; transform its intersection to
            # camera depth even for rotated/translated viewpoints.
            xx = (torch.arange(512, device=self.device).float()[None] + 0.5).expand(512, 512)
            rays = torch.stack(((xx - 256) / focal_length, (yy - 256) / focal_length, torch.ones_like(xx)), dim=-1)
            world_rays = rays @ camera[:3, :3]
            center = -camera[:3, :3].T @ camera[:3, 3]
            denominator = world_rays[..., 1]
            ground_depth = (
                (0.3 - center[1])
                / torch.where(denominator.abs() > 1e-8, denominator, torch.full_like(denominator, 1e-8))
                / 1000
            )
            ground_valid = ground & (ground_depth > 0.001) & (ground_depth < 0.0048)
            target = torch.where(existing_mask, existing_depth, ground_depth[None, None])
            align = existing_mask | ground_valid[None, None]
            depth = self._depth(pixels, steps=depth_steps, seed=current_seed, target=target, align=align)
            depth = refine_disparity(depth, regions, ground)
            support = occluder_mask(semantic, depth) if gen_layer else torch.zeros_like(sky)
            foreground = functional.max_pool2d((~support).float(), 9, 1, 4) == 0
            original_pixels, original_depth = pixels, depth
            if foreground.any():
                from torchvision.transforms.functional import gaussian_blur

                generated = self._inpaint(
                    pixels,
                    foreground,
                    prompt=prompt or "road, building",
                    negative_prompt="tree, plant",
                    steps=inpaint_steps,
                    seed=current_seed + 1,
                )
                stitch = (functional.max_pool2d((~support).float(), 5, 1, 2) == 0).float()
                stitch = gaussian_blur(stitch, [3, 3], sigma=[1.0, 1.0]).clamp(0, 1)
                pixels = generated * stitch + pixels * (1 - stitch)
                background_semantic, background_regions = self.segmenter(image_pil(pixels))
                background_depth = self._depth(
                    pixels,
                    steps=depth_steps,
                    seed=current_seed + 1,
                    target=depth,
                    align=~foreground & (depth < 0.0048),
                    farther=foreground & (depth < (0.006 if index == 0 else 0.0048)),
                )
                background_depth = refine_disparity(
                    background_depth, background_regions, ground, existing_mask=~foreground, existing_depth=depth
                )
                background_depth = torch.where(background_depth < depth, depth + 0.0001, background_depth)
                depth = torch.where(foreground, background_depth, depth)
                sky = (background_semantic == 2)[None, None]
            start = len(scene.means) if scene is not None else 0
            if sky.any():
                sky_depth = torch.full_like(depth, 20.0)
                sky_normals = functional.normalize(-rays.permute(2, 0, 1)[None], dim=1)
                new_sky = GaussianSurfels.from_rgbd(
                    pixels,
                    sky_depth,
                    sky_normals,
                    intrinsic=intrinsic,
                    world_to_camera=camera,
                    mask=new_region & sky,
                    layer=0,
                )
                scene = new_sky if scene is None else new_sky.append(scene)
            normals = self._normals(pixels, steps=normal_steps, seed=current_seed + 2)
            base = GaussianSurfels.from_rgbd(
                (pixels * 1.01).clamp(0, 1),
                depth * 1000,
                normals,
                intrinsic=intrinsic,
                world_to_camera=camera,
                mask=new_region & ~sky,
                layer=1,
                footprint=0.7,
                opacity=0.15,
                thickness=0.7,
            ).remove_floaters(mean_squared_distance=0.0002)
            # New particles precede the frozen scene, including depth-sort ties.
            scene = base if scene is None else base.append(scene)
            target_image = image_tensor(image_pil(pixels), self.device)
            scene = scene.refine(target_image, camera, intrinsic, steps=refinement_steps, stop=len(base.means))
            if foreground.any():
                foreground_normals = self._normals(original_pixels, steps=normal_steps, seed=current_seed + 3)
                objects = GaussianSurfels.from_rgbd(
                    (original_pixels * 1.01).clamp(0, 1),
                    original_depth * 1000,
                    foreground_normals,
                    intrinsic=intrinsic,
                    world_to_camera=camera,
                    mask=new_region & foreground,
                    layer=2,
                    footprint=0.7,
                    opacity=0.15,
                    thickness=0.7,
                ).remove_floaters(mean_squared_distance=0.0002)
                scene = objects.append(scene)
                target_image = image_tensor(image_pil(original_pixels), self.device)
                scene = scene.refine(target_image, camera, intrinsic, steps=refinement_steps, stop=len(objects.means))
            preview, _, _ = scene.render(camera, intrinsic, height=512, width=512)
            image_pil(preview).save(directory / f"view_{index:03d}.png")
            image_pil(original_pixels).save(directory / f"keyframe_{index:03d}.png")
            np.save(directory / f"depth_{index:03d}.npy", (original_depth[0, 0] * 1000).cpu().numpy())
            views.append(
                {
                    "index": index,
                    "new_pixels": int(new_region.sum()),
                    "foreground_pixels": int(foreground.sum()),
                    "new_surfels": len(scene.means) - start,
                    "world_to_camera": camera.cpu().tolist(),
                }
            )
        if scene is None or not len(scene.means):
            raise RuntimeError("WonderWorld produced no Gaussian surfels")
        scene.save_splat(destination)
        metadata = {
            "model_id": "wonderworld",
            "runtime": "worldfoundry.wonderworld.native",
            "source_revision": "5cf11467ad21a9d7bca269f961855a235c7d352e",
            "seed": seed,
            "image_size": [512, 512],
            "focal_length": focal_length,
            "camera_convention": "opencv-world-to-camera",
            "scene_depth_scale": 1000,
            "surfel_count": len(scene.means),
            "layer_counts": {str(layer): int((scene.layers == layer).sum()) for layer in (0, 1, 2)},
            "depth_steps": depth_steps,
            "inpaint_steps": inpaint_steps,
            "normal_steps": normal_steps,
            "refinement_steps": refinement_steps,
            "views": views,
        }
        metadata_path = destination.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "model_id": "wonderworld",
            "artifact_kind": "generated_world",
            "artifact_path": str(destination),
            "preview_path": str(directory / "view_000.png"),
            "metadata_path": str(metadata_path),
            "run_dir": str(directory),
            "runtime": "worldfoundry.wonderworld.native",
            "metadata": metadata,
        }
