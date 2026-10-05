from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange

from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.depth_selector import (
    build_condition_db_in_memory,
    select_best_condition_index,
)
from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.moge_pano import (
    ensure_moge,
    select_tier,
)
from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.pano_postprocess import (
    postprocess_panorama,
)
from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.panogen import (
    Image2PanoramaDemo,
    ensure_hy3dworld,
)
from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.point_renderer import (
    TorchPointCloudRenderer,
)
from worldfoundry.core.io.paths import hfd_root_path

from . import moge_pano

_RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
# WorldFM follows the official MoGe-2 pipeline.
DEFAULT_WORLDFM_MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"
# Backward-compatible alias for existing imports.
DEFAULT_MOGE_REPO = DEFAULT_WORLDFM_MOGE2_REPO


def _local_moge_model_path_candidates(source_value: str) -> list[Path]:
    cache_root = hfd_root_path()
    names = []
    if source_value:
        names.append(source_value.replace("/", "--"))
        names.append(source_value.split("/")[-1])
    deduped = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return [(cache_root / name / "model.pt") for name in deduped]


def _resolve_moge_pretrained(
    pretrained_model_name_or_path: Optional[str | os.PathLike],
) -> str:
    if pretrained_model_name_or_path is None:
        for candidate in _local_moge_model_path_candidates(DEFAULT_WORLDFM_MOGE2_REPO):
            if candidate.is_file():
                return str(candidate.resolve())
        return DEFAULT_WORLDFM_MOGE2_REPO

    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        if candidate.is_file():
            return str(candidate.resolve())
        model_file = candidate / "model.pt"
        if model_file.is_file():
            return str(model_file.resolve())
        raise FileNotFoundError(f"Expected a MoGe checkpoint file at {model_file}")

    source_value = str(pretrained_model_name_or_path)
    for local_candidate in _local_moge_model_path_candidates(source_value):
        if local_candidate.is_file():
            return str(local_candidate.resolve())

    return source_value


def _ensure_utils3d_alias() -> None:
    if "utils3d" in sys.modules:
        return

    if importlib.util.find_spec("utils3d") is not None:
        importlib.import_module("utils3d")
        return

    from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist import (
        utils3d as vendored_utils3d,
    )

    sys.modules["utils3d"] = vendored_utils3d


def _to_rgb_pil_image(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")

    if isinstance(data, (str, os.PathLike)):
        return Image.open(data).convert("RGB")

    if isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        array = np.asarray(data)

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported WorldFM image input shape: {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if array.min() >= -1.0 and array.max() <= 1.0:
                if array.min() < 0.0:
                    array = (array + 1.0) * 127.5
                else:
                    array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


class WorldFMRepresentation:
    """
    WorldFM scene representation stack: panorama, MoGe, point cloud and condition rendering.

    WorldFM uses the canonical WorldFoundry MoGe base model and defaults to
    ``Ruicheng/moge-2-vitl-normal``.
    """

    def __init__(
        self,
        hw_path: Optional[str] = None,
        moge_path: Optional[str] = None,
        realesrgan_path: Optional[str] = None,
        zim_path: Optional[str] = None,
        moge_pretrained: Optional[str | os.PathLike] = DEFAULT_WORLDFM_MOGE2_REPO,
        render_size: int = 512,
        resolution_level: int = 30,
        fov_deg: float = 45.0,
        num_views: int = 42,
        merge_max_width: int = 4096,
        merge_max_height: int = 2048,
        batch_size: int = 4,
        panogen_seed: int = 42,
        panogen_fp8_attention: bool = False,
        panogen_fp8_gemm: bool = False,
        panogen_cache: bool = False,
        sample_grid: int = 10,
        center_grid: int = 15,
        center_frac: float = 0.5,
        eps_rel: float = 0.02,
        eps_abs: float = 0.0,
        px_radius: int = 0,
        max_view_angle_deg: float = 180.0,
        use_distance_weight: bool = True,
        dist_min_m: float = 1.0,
        dist_max_m: float = 20.0,
        weight_near: float = 1.0,
        weight_far: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        for name, value in {
            "hw_path": hw_path,
            "moge_path": moge_path,
            "realesrgan_path": realesrgan_path,
            "zim_path": zim_path,
        }.items():
            if value:
                raise RuntimeError(
                    f"WorldFM no longer accepts `{name}` external source checkout paths at runtime. "
                    "Use WorldFoundry base/runtime code, installed Python packages, or provide "
                    "panorama_image/panorama_path to skip panorama generation."
                )
        self.hw_path = hw_path
        self.moge_path = None
        self.realesrgan_path = None
        self.zim_path = None
        self.moge_pretrained = _resolve_moge_pretrained(moge_pretrained)
        self.render_size = int(render_size)
        self.resolution_level = int(resolution_level)
        self.fov_deg = float(fov_deg)
        self.num_views = int(num_views)
        self.merge_max_width = int(merge_max_width)
        self.merge_max_height = int(merge_max_height)
        self.batch_size = int(batch_size)
        self.panogen_seed = int(panogen_seed)
        self.panogen_fp8_attention = bool(panogen_fp8_attention)
        self.panogen_fp8_gemm = bool(panogen_fp8_gemm)
        self.panogen_cache = bool(panogen_cache)
        self.sample_grid = int(sample_grid)
        self.center_grid = int(center_grid)
        self.center_frac = float(center_frac)
        self.eps_rel = float(eps_rel)
        self.eps_abs = float(eps_abs)
        self.px_radius = int(px_radius)
        self.max_view_angle_deg = float(max_view_angle_deg)
        self.use_distance_weight = bool(use_distance_weight)
        self.dist_min_m = float(dist_min_m)
        self.dist_max_m = float(dist_max_m)
        self.weight_near = float(weight_near)
        self.weight_far = float(weight_far)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, device=None, **kwargs):
        if pretrained_model_path is not None:
            kwargs.setdefault("moge_pretrained", pretrained_model_path)
        return cls(device=device, **kwargs)

    def _torch_device(self) -> torch.device:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def _setup_external_repos(self, *, require_panorama_generator: bool) -> None:
        _ensure_utils3d_alias()
        ensure_moge()

        if not require_panorama_generator:
            return

        ensure_hy3dworld(self.hw_path)

    def _resolve_scene_dir(
        self,
        output_dir: Optional[str | os.PathLike],
        scene_name: str,
    ) -> Optional[Path]:
        if output_dir is None:
            return None
        base_dir = Path(output_dir).expanduser().resolve()
        scene_dir = base_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir

    def _materialize_input_image(
        self,
        images,
        image_path: Optional[str],
        scene_dir: Optional[Path],
    ) -> tuple[str, Optional[Path]]:
        if image_path:
            resolved = Path(image_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"WorldFM input image not found: {resolved}")
            return str(resolved), None

        if isinstance(images, (str, os.PathLike)):
            resolved = Path(images).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"WorldFM input image not found: {resolved}")
            return str(resolved), None

        image = _to_rgb_pil_image(images)
        if scene_dir is not None:
            materialized = scene_dir / "_worldfm_input.png"
            image.save(materialized)
            return str(materialized), None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            materialized = Path(handle.name)
        image.save(materialized)
        return str(materialized), materialized

    def _load_panorama(self, panorama_image=None, panorama_path: Optional[str] = None) -> Image.Image:
        if panorama_path is not None:
            path = Path(panorama_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"WorldFM panorama image not found: {path}")
            return Image.open(path).convert("RGB")
        if panorama_image is None:
            raise ValueError("Expected `panorama_image` or `panorama_path`.")
        return _to_rgb_pil_image(panorama_image)

    def _run_panorama_generation(self, image_path: str, scene_dir: Optional[Path]) -> Image.Image:
        if scene_dir is not None:
            cached_panorama = scene_dir / "panorama.png"
            if cached_panorama.exists():
                return Image.open(cached_panorama).convert("RGB")

        class _Args:
            fp8_attention = self.panogen_fp8_attention
            fp8_gemm = self.panogen_fp8_gemm
            cache = self.panogen_cache

        demo = Image2PanoramaDemo(_Args())
        return demo.run(
            prompt="",
            negative_prompt="",
            image_path=image_path,
            seed=self.panogen_seed,
            output_path=str(scene_dir) if scene_dir is not None else "output_panorama",
            save_to_disk=scene_dir is not None,
        )

    def _run_moge_pipeline(self, panorama_img: Image.Image):
        import utils3d

        image_rgb = np.asarray(panorama_img.convert("RGB"))
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

        orig_h, orig_w = image_rgb.shape[:2]
        tier = select_tier(orig_w)
        tgt_w, tgt_h = tier["width"], tier["height"]
        split_resolution = tier["split_res"]
        if orig_w != tgt_w or orig_h != tgt_h:
            interpolation = cv2.INTER_AREA if tgt_w < orig_w else cv2.INTER_LINEAR
            image_rgb = cv2.resize(image_rgb, (tgt_w, tgt_h), interpolation=interpolation)

        device = self._torch_device()
        model = moge_pano.MoGeModel.from_pretrained(self.moge_pretrained).to(device).eval()
        extrinsics, intrinsics = moge_pano._get_panorama_cameras(self.num_views, self.fov_deg)
        splitted_images = moge_pano.split_panorama_image(image_rgb, extrinsics, intrinsics, split_resolution)

        splitted_dist = []
        splitted_masks = []
        for index in trange(0, len(splitted_images), self.batch_size, desc="MoGe Infer", leave=False):
            batch = np.stack(splitted_images[index:index + self.batch_size])
            tensor = torch.tensor(batch / 255.0, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            fov_x, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[index:index + self.batch_size])))
            fov_x_tensor = torch.tensor(fov_x, dtype=torch.float32, device=device)
            outputs = model.infer(
                tensor,
                resolution_level=self.resolution_level,
                fov_x=fov_x_tensor,
                apply_mask=False,
            )
            splitted_dist.extend(list(outputs["points"].norm(dim=-1).cpu().numpy()))
            splitted_masks.extend(list(outputs["mask"].cpu().numpy()))

        merging_width = min(self.merge_max_width, image_rgb.shape[1])
        merging_height = min(self.merge_max_height, image_rgb.shape[0])
        panorama_depth, panorama_mask = moge_pano.merge_panorama_depth(
            merging_width,
            merging_height,
            splitted_dist,
            splitted_masks,
            extrinsics,
            intrinsics,
        )
        panorama_depth = panorama_depth.astype(np.float32)
        panorama_depth = cv2.resize(panorama_depth, (image_rgb.shape[1], image_rgb.shape[0]), cv2.INTER_LINEAR)
        panorama_mask = cv2.resize(
            panorama_mask.astype(np.uint8),
            (image_rgb.shape[1], image_rgb.shape[0]),
            cv2.INTER_NEAREST,
        ) > 0

        depth_raw = panorama_depth.copy()
        if panorama_mask.any():
            depth_raw[~panorama_mask] = panorama_depth[panorama_mask].max()
        depth_raw = depth_raw / 100.0

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pano_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return postprocess_panorama(pano_bgr, depth_raw, save_dir=None)

    def build_scene_context(
        self,
        *,
        images=None,
        image_path: Optional[str] = None,
        panorama_image=None,
        panorama_path: Optional[str] = None,
        K,
        scene_name: str,
        output_dir: Optional[str | os.PathLike] = None,
    ) -> Dict[str, Any]:
        scene_dir = self._resolve_scene_dir(output_dir, scene_name)
        use_panorama_input = panorama_image is not None or panorama_path is not None
        self._setup_external_repos(require_panorama_generator=not use_panorama_input)

        if use_panorama_input:
            panorama_img = self._load_panorama(panorama_image=panorama_image, panorama_path=panorama_path)
        else:
            materialized_path, cleanup_path = self._materialize_input_image(images, image_path, scene_dir)
            try:
                panorama_img = self._run_panorama_generation(materialized_path, scene_dir)
            finally:
                if cleanup_path is not None and cleanup_path.exists():
                    cleanup_path.unlink(missing_ok=True)

        pp_result = self._run_moge_pipeline(panorama_img)
        device = self._torch_device()
        renderer = TorchPointCloudRenderer(
            points_xyz=pp_result.ply_xyz,
            points_rgb=pp_result.ply_rgb / 255.0 if pp_result.ply_rgb.dtype == np.uint8 else pp_result.ply_rgb,
            width=self.render_size,
            height=self.render_size,
            device=str(device),
            mode="fast",
        )
        cond_db = build_condition_db_in_memory(
            condition_images=pp_result.condition_images,
            transforms_dict=pp_result.transforms,
            torch_renderer=renderer,
            device=device,
        )

        return {
            "scene_name": scene_name,
            "scene_dir": scene_dir,
            "K": np.asarray(K, dtype=np.float64),
            "renderer": renderer,
            "cond_db": cond_db,
            "condition_images": pp_result.condition_images,
        }

    def render_views(
        self,
        scene_context: Dict[str, Any],
        c2w_list: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        K = np.asarray(scene_context["K"], dtype=np.float64)
        renderer = scene_context["renderer"]
        cond_db = scene_context["cond_db"]
        condition_images = scene_context["condition_images"]

        rendered_views = []
        for c2w in c2w_list:
            c2w_array = np.asarray(c2w, dtype=np.float64)
            render_output = renderer.render_torch(
                K_3x3=K,
                c2w_4x4=c2w_array,
                c2w_is_camera_to_world=True,
            )
            best_index, best_hits, total_samples = select_best_condition_index(
                depth_cur=render_output.depth_f32,
                K_cur=K,
                c2w_cur=c2w_array,
                cond_db=cond_db,
                sample_grid=self.sample_grid,
                center_grid=self.center_grid,
                center_frac=self.center_frac,
                eps_rel=self.eps_rel,
                eps_abs=self.eps_abs,
                px_radius=self.px_radius,
                max_view_angle_deg=self.max_view_angle_deg,
                use_distance_weight=self.use_distance_weight,
                dist_min_m=self.dist_min_m,
                dist_max_m=self.dist_max_m,
                weight_near=self.weight_near,
                weight_far=self.weight_far,
            )
            cond_nearest_rgb = np.asarray(
                Image.fromarray(condition_images[int(best_index)], mode="RGB").resize(
                    (self.render_size, self.render_size),
                    resample=_RESAMPLE_BILINEAR,
                )
            )
            rendered_views.append(
                {
                    "render_rgb_u8": render_output.rgb_u8,
                    "cond_nearest_rgb": cond_nearest_rgb,
                    "c2w": c2w_array,
                    "condition_index": int(best_index),
                    "hits": int(best_hits),
                    "samples": int(total_samples),
                }
            )

        return rendered_views

    def get_representation(self, data):
        scene_context = data.get("scene_context")
        if scene_context is None:
            scene_context = self.build_scene_context(
                images=data.get("images"),
                image_path=data.get("image_path"),
                panorama_image=data.get("panorama_image"),
                panorama_path=data.get("panorama_path"),
                K=data["K"],
                scene_name=data.get("scene_name", "worldfm_scene"),
                output_dir=data.get("output_dir"),
            )

        c2w_list = data.get("c2w_list") or data.get("interactions")
        if c2w_list is None:
            raise ValueError("WorldFM representation requires `c2w_list`.")

        return {
            "scene_context": scene_context,
            "scene_name": scene_context["scene_name"],
            "K": scene_context["K"],
            "c2w_list": [np.asarray(pose, dtype=np.float64) for pose in c2w_list],
            "rendered_conditions": self.render_views(scene_context, c2w_list),
        }
