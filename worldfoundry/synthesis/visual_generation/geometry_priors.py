from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.base_synthesis import BaseSynthesis


_BASE_MODEL_TARGETS: Mapping[str, str] = {
    "dap": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1",
    "depth-anything-v2-prior": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v2",
    "depth-anything-v3-prior": "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3",
    "dust3r": "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
    "dust3r-base-model": "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
    "geocalib-prior": "worldfoundry.base_models.three_dimensions.general_3d.geocalib",
    "metric3d-prior": "worldfoundry.base_models.three_dimensions.depth.metric3d",
    "prior-depth-anything": "worldfoundry.base_models.three_dimensions.depth.priorda",
    "track-anything-prior": "worldfoundry.base_models.perception_core.tracking.track_anything",
    "unidepth-v2-prior": "worldfoundry.base_models.three_dimensions.depth.unidepth",
    "unik3d-prior": "worldfoundry.base_models.three_dimensions.depth.unik3d",
    "video-depth-anything-prior": "worldfoundry.base_models.three_dimensions.depth.videodepthanything",
}

_PROFILE_ID_OVERRIDES: Mapping[str, str] = {
    "dust3r": "dust3r-base-model",
}


def _checkpoint_root() -> Path:
    return Path(
        os.environ.get(
            "WORLDFOUNDRY_CKPT_DIR",
            str(Path(__file__).resolve().parents[5] / "ckpt"),
        )
    ).expanduser()


def _known_checkpoint_candidates(model_id: str) -> tuple[str, ...]:
    root = _checkpoint_root()
    table: Mapping[str, Sequence[Path]] = {
        "dap": (
            root / "DAP-weights" / "model.pth",
            root / "hfd" / "Insta360-Research--DAP-weights" / "model.pth",
        ),
        "depth-anything-v2-prior": (
            root / "Depth-Anything-V2-Large" / "depth_anything_v2_vitl.pth",
            root / "Prior-Depth-Anything" / "depth_anything_v2_vitl.pth",
        ),
        "depth-anything-v3-prior": (
            root / "DA3METRIC-LARGE" / "model.safetensors",
            root / "depth-anything--DA3METRIC-LARGE" / "model.safetensors",
        ),
        "dust3r": (
            root / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
            root / "DUSt3R_ViTLarge_BaseDecoder_512_dpt" / "model.safetensors",
        ),
        "dust3r-base-model": (
            root / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
            root / "DUSt3R_ViTLarge_BaseDecoder_512_dpt" / "model.safetensors",
        ),
        "geocalib-prior": (
            root / "GeoCalib" / "geocalib-pinhole.tar",
            Path(torch.hub.get_dir()) / "geocalib" / "pinhole.tar",
        ),
        "metric3d-prior": (
            root / "Metric3D" / "metric_depth_vit_giant2_800k.pth",
            root / "Metric3D" / "metric_depth_vit_large_800k.pth",
        ),
        "prior-depth-anything": (
            root / "Prior-Depth-Anything",
        ),
        "track-anything-prior": (
            root / "SAM" / "models" / "sams" / "sam_vit_b_01ec64.pth",
            root / "Track-Anything" / "R50_DeAOTL_PRE_YTB_DAV.pth",
            root / "GroundingDINO" / "groundingdino_swint_ogc.pth",
        ),
        "unidepth-v2-prior": (
            root / "unidepth-v2-vitl14",
            root / "lpiccinelli--unidepth-v2-vitl14",
        ),
        "unik3d-prior": (
            root / "unik3d-vitl",
        ),
        "video-depth-anything-prior": (
            root / "Video-Depth-Anything-Large" / "video_depth_anything_vitl.pth",
        ),
    }
    return tuple(str(path) for path in table.get(model_id, ()))


@dataclass(frozen=True)
class GeometryPriorRuntimeState:
    model_id: str
    profile_id: str
    base_model_target: str
    import_error: str | None
    checkpoint_paths: tuple[str, ...]

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.import_error:
            missing.append(f"base model import failed: {self.import_error}")
        for path in self.checkpoint_paths:
            if "://" in path:
                continue
            if not Path(path).expanduser().exists():
                missing.append(f"missing checkpoint: {path}")
        return tuple(missing)


class GeometryPriorSynthesis(BaseSynthesis):
    """In-tree runtime adapter for standalone geometry-prior integrations."""

    MODEL_ID: str | None = None

    def __init__(self, *, model_id: str, device: str = "cuda", options: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.model_id = model_id
        self.profile_id = _PROFILE_ID_OVERRIDES.get(model_id, model_id)
        self.device = device
        self.options = dict(options or {})
        self.profile = load_runtime_profile(self.profile_id, check_conda_env_exists=False)
        self.state = self._runtime_state()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "GeometryPriorSynthesis":
        options: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            options.update(model_path)
        elif model_path is not None:
            options["model_path"] = str(model_path)
        options.update(dict(required_components or {}))
        options.update(kwargs)
        model_id = str(options.get("profile_id") or options.get("runtime_profile") or options.get("model_id") or cls.MODEL_ID or "")
        if not model_id:
            raise ValueError("GeometryPriorSynthesis requires model_id/profile_id.")
        return cls(model_id=model_id, device=device, options=options)

    def _runtime_state(self) -> GeometryPriorRuntimeState:
        base_model_target = str(self.options.get("base_model_target") or _BASE_MODEL_TARGETS.get(self.model_id, ""))
        import_error = None
        if base_model_target:
            try:
                __import__(base_model_target)
            except Exception as exc:  # noqa: BLE001 - preserve import failure for Studio.
                import_error = f"{type(exc).__name__}: {exc}"
        profile_paths = self._checkpoint_paths()
        known_paths = _known_checkpoint_candidates(self.model_id)
        explicit = self._explicit_checkpoint()
        if self.model_id == "track-anything-prior":
            checkpoint_paths = tuple(
                self._expand_path(str(self.options.get(key) or default))
                for key, default in zip(
                    ("sam_ckpt_path", "aot_ckpt_path", "grounding_dino_ckpt_path"),
                    known_paths,
                )
            )
        elif explicit:
            checkpoint_paths = (self._expand_path(str(explicit)),)
        else:
            existing_known = next((path for path in known_paths if Path(path).expanduser().exists()), "")
            fallback_known = existing_known or (known_paths[0] if known_paths else "")
            checkpoint_paths = tuple(dict.fromkeys((*profile_paths, *(path for path in (fallback_known,) if path))))
        return GeometryPriorRuntimeState(
            model_id=self.model_id,
            profile_id=self.profile_id,
            base_model_target=base_model_target,
            import_error=import_error,
            checkpoint_paths=checkpoint_paths,
        )

    def _checkpoint_paths(self) -> list[str]:
        paths: list[str] = []
        for item in self.profile.checkpoints:
            if not isinstance(item, Mapping):
                continue
            local_dir = self._expand_path(item.get("local_dir"))
            if local_dir:
                paths.append(local_dir)
                continue
            local_path = self._expand_path(item.get("local_path") or item.get("path"))
            if local_path and not str(local_path).startswith(("http://", "https://")):
                paths.append(local_path)
                continue
            repo_id = item.get("repo_id")
            if isinstance(repo_id, str) and repo_id.strip():
                filename = item.get("filename")
                paths.append(f"hf://{repo_id.strip()}/{filename}" if filename else f"hf://{repo_id.strip()}")
                continue
            uri = item.get("uri") or item.get("path")
            if isinstance(uri, str) and uri.strip():
                paths.append(uri.strip())
        return paths

    @staticmethod
    def _expand_path(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        return os.path.expandvars(os.path.expanduser(value))

    def _explicit_checkpoint(self) -> Any:
        return (
            self.options.get("weights_path")
            or self.options.get("checkpoint_path")
            or self.options.get("ckpt_path")
            or self.options.get("weights_dir")
            or self.options.get("model_path")
        )

    def _existing_checkpoint(self, *, directory_ok: bool = False) -> str:
        explicit = self._explicit_checkpoint()
        candidates = [str(explicit)] if explicit else []
        candidates.extend(self.state.checkpoint_paths)
        for raw_path in candidates:
            if not raw_path or "://" in raw_path:
                continue
            path = Path(raw_path).expanduser()
            if path.is_file() or (directory_ok and path.is_dir()):
                return str(path)
        return ""

    def _output_dir(self, output_path: str | Path | None, explicit_output_dir: Any = None) -> Path:
        if explicit_output_dir:
            output_dir = Path(str(explicit_output_dir)).expanduser()
        elif output_path:
            path = Path(output_path).expanduser()
            output_dir = path if path.suffix == "" else path.parent
        else:
            output_dir = Path.cwd() / "geometry_prior_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
        path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def predict(
        self,
        *,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        missing = self.state.missing_requirements
        plan = {
            "model_id": self.model_id,
            "profile_id": self.state.profile_id,
            "base_model_target": self.state.base_model_target,
            "device": self.device,
            "has_images": images is not None,
            "has_video": video is not None,
            "interactions": list(interactions or ()),
            "requested_output_path": str(output_path) if output_path is not None else "",
            "checkpoint_paths": list(self.state.checkpoint_paths),
            "missing_requirements": list(missing),
            "runtime_options": {key: str(value) for key, value in kwargs.items()},
        }
        if missing:
            return {
                "status": "blocked",
                "runtime": "geometry_prior_preflight",
                "backend_quality": "in_tree_prior_checkpoint_or_dependency_required",
                "blocked_reasons": missing,
                "metadata": plan,
            }
        if not execute:
            return {
                "status": "blocked",
                "runtime": "geometry_prior_preflight",
                "backend_quality": "execute_required_for_prior_inference",
                "blocked_reason": "Geometry prior inference requires execute=True; preflight artifacts are not emitted.",
                "metadata": plan,
            }

        explicit_output_dir = kwargs.pop("output_dir", None)
        output_dir = self._output_dir(output_path, explicit_output_dir)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(0)

        if self.model_id == "geocalib-prior":
            return self._run_geocalib(images=images, output_dir=output_dir, **kwargs)
        if self.model_id == "video-depth-anything-prior":
            return self._run_video_depth(video=video, images=images, output_dir=output_dir, **kwargs)
        if self.model_id == "track-anything-prior":
            if prompt is not None:
                kwargs.setdefault("prompt", prompt)
            return self._run_track_anything(video=video, images=images, output_dir=output_dir, **kwargs)
        if self.model_id in {"dust3r", "dust3r-base-model"}:
            return self._run_dust3r(images=images, output_dir=output_dir, **kwargs)
        return self._run_single_image_depth(images=images, output_dir=output_dir, **kwargs)

    def _run_single_image_depth(self, *, images: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        rgb = _load_rgb_array(_first_available(images, kwargs.get("image_path"), kwargs.get("input_path")))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rgb_tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
        focal = float(kwargs.get("focal_length") or max(rgb.shape[:2]))

        from worldfoundry.base_models.three_dimensions.depth.base import DepthEstimationInput
        from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType

        ckpt = self._existing_checkpoint(directory_ok=self.model_id in {"prior-depth-anything", "unidepth-v2-prior", "unik3d-prior"})
        if self.model_id == "dap":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dap_adapter import DAPModel

            model = DAPModel(weights_path=ckpt, input_size=int(kwargs.get("input_size") or 518))
            src = DepthEstimationInput(rgb=rgb_tensor, camera_type=CameraType.PANORAMA)
        elif self.model_id == "depth-anything-v2-prior":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v2 import DepthAnythingDepthModel

            model = DepthAnythingDepthModel(model=str(kwargs.get("model") or "vitl"), weights_path=ckpt)
            src = DepthEstimationInput(rgb=rgb_tensor)
        elif self.model_id == "depth-anything-v3-prior":
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3 import DepthAnything3Model

            model = DepthAnything3Model(weights_path=ckpt)
            src = DepthEstimationInput(rgb=rgb_tensor, intrinsics=torch.tensor([focal], device=device), camera_type=CameraType.PINHOLE)
        elif self.model_id == "metric3d-prior":
            from worldfoundry.base_models.three_dimensions.depth.metric3d import Metric3DDepthModel

            model = Metric3DDepthModel(version=int(kwargs.get("version") or 2), model=str(kwargs.get("model") or "giant2"), weights_path=ckpt)
            src = DepthEstimationInput(rgb=rgb_tensor, intrinsics=torch.tensor([focal], device=device), camera_type=CameraType.PINHOLE)
        elif self.model_id == "prior-depth-anything":
            from worldfoundry.base_models.three_dimensions.depth.priorda import PriorDAModel

            prompt_depth = _load_prompt_depth(kwargs.get("prompt_depth_path") or kwargs.get("prior_depth_path"), rgb.shape[:2], device=device)
            model = PriorDAModel(weights_dir=ckpt, device=device)
            src = DepthEstimationInput(rgb=rgb_tensor, prompt_metric_depth=prompt_depth)
            with torch.inference_mode():
                result = model.estimate(src, pattern=str(kwargs.get("pattern") or "2000"))
            return _write_depth_artifacts(
                model_id=self.model_id,
                result=result,
                rgb=rgb,
                output_dir=output_dir,
                depth_kind=str(getattr(model.depth_type, "value", model.depth_type)),
                max_points=int(kwargs.get("max_points") or 5000),
                focal_length=focal,
                extra_metadata={"checkpoint_path": ckpt, "pattern": str(kwargs.get("pattern") or "2000")},
            )
        elif self.model_id == "unidepth-v2-prior":
            from worldfoundry.base_models.three_dimensions.depth.unidepth import UniDepth2Model

            model = UniDepth2Model(type=str(kwargs.get("type") or "l"), model_path=ckpt, device=device)
            src = DepthEstimationInput(rgb=rgb_tensor, intrinsics=torch.tensor([focal], device=device), camera_type=CameraType.PINHOLE)
        elif self.model_id == "unik3d-prior":
            from worldfoundry.base_models.three_dimensions.depth.unik3d import Unik3DModel

            model = Unik3DModel(type=str(kwargs.get("type") or "l"), model_path=ckpt)
            camera_model = str(kwargs.get("camera_model") or "auto").strip().lower()
            if camera_model == "panorama":
                src = DepthEstimationInput(rgb=rgb_tensor, camera_type=CameraType.PANORAMA)
            elif camera_model == "pinhole":
                src = DepthEstimationInput(
                    rgb=rgb_tensor,
                    intrinsics=torch.tensor([focal], device=device),
                    camera_type=CameraType.PINHOLE,
                )
            elif camera_model == "auto":
                src = DepthEstimationInput(rgb=rgb_tensor, camera_type=CameraType.PINHOLE)
            else:
                raise ValueError("UniK3D camera_model must be one of: auto, pinhole, panorama")
        else:
            raise RuntimeError(f"Unsupported geometry prior model: {self.model_id}")

        with torch.inference_mode():
            result = model.estimate(src)
        return _write_depth_artifacts(
            model_id=self.model_id,
            result=result,
            rgb=rgb,
            output_dir=output_dir,
            depth_kind=str(getattr(model.depth_type, "value", model.depth_type)),
            max_points=int(kwargs.get("max_points") or 5000),
            focal_length=focal,
            extra_metadata={
                "checkpoint_path": ckpt,
                **({"camera_model": camera_model} if self.model_id == "unik3d-prior" else {}),
            },
        )

    def _run_video_depth(self, *, video: Any, images: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        from worldfoundry.base_models.three_dimensions.depth.base import DepthEstimationInput
        from worldfoundry.base_models.three_dimensions.depth.videodepthanything import VideoDepthAnythingDepthModel

        frames = _load_video_frames(
            _first_available(video, kwargs.get("video_path"), kwargs.get("input_path"), images),
            max_frames=int(kwargs.get("max_frames") or 4),
        )
        ckpt = self._existing_checkpoint()
        model = VideoDepthAnythingDepthModel(
            model=str(kwargs.get("model") or "vitl"),
            input_size=int(kwargs.get("input_size") or 518),
            weights_path=ckpt,
        )
        with torch.inference_mode():
            result = model.estimate(
                DepthEstimationInput(
                    video_frame_list=[frame.astype(np.float32) / 255.0 for frame in frames]
                )
            )
        return _write_depth_artifacts(
            model_id=self.model_id,
            result=result,
            rgb=frames[0].astype(np.float32) / 255.0,
            output_dir=output_dir,
            depth_kind="affine_disp_video",
            max_points=int(kwargs.get("max_points") or 5000),
            extra_metadata={"checkpoint_path": ckpt, "num_frames": len(frames)},
        )

    def _run_geocalib(self, *, images: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        from worldfoundry.base_models.three_dimensions.general_3d.geocalib import GeoCalib

        rgb = _load_rgb_array(_first_available(images, kwargs.get("image_path"), kwargs.get("input_path")))
        ckpt = self._existing_checkpoint()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GeoCalib(weights=ckpt).to(device).eval()
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            calibration = model.calibrate(image_tensor, camera_model=str(kwargs.get("camera_model") or "pinhole"))

        preview_path = output_dir / "input_preview.png"
        Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)).save(preview_path)
        metadata_path = output_dir / "geocalib_result.json"
        self._write_json(
            metadata_path,
            {
                "model_id": self.model_id,
                "checkpoint_path": ckpt,
                "calibration": _safe_json(calibration),
            },
        )
        return {
            "status": "succeeded",
            "runtime": "geometry_prior_in_tree",
            "model_id": self.model_id,
            "artifact_path": str(metadata_path),
            "preview_image": str(preview_path),
            "metadata_path": str(metadata_path),
        }

    def _run_dust3r(self, *, images: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        from worldfoundry.base_models.three_dimensions.general_3d.dust3r import ensure_import_paths

        ensure_import_paths()
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images

        image_paths = _collect_image_paths(
            _first_available(images, kwargs.get("image_path"), kwargs.get("input_path")),
            output_dir=output_dir,
            max_images=int(kwargs.get("max_images") or 2),
        )
        if len(image_paths) == 1:
            image_paths = [image_paths[0], image_paths[0]]
        if len(image_paths) < 2:
            raise ValueError("DUSt3R inference requires at least one image path; a single image is duplicated for validation tests.")

        ckpt = self._existing_checkpoint(directory_ok=True)
        if not ckpt:
            raise FileNotFoundError("DUSt3R requires a local checkpoint file or a Hugging Face model directory.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(device).eval()
        dust3r_images = load_images(
            image_paths[: int(kwargs.get("max_images") or 2)],
            size=int(kwargs.get("image_size") or 512),
            verbose=bool(kwargs.get("verbose", False)),
        )
        pairs = make_pairs(
            dust3r_images,
            scene_graph=str(kwargs.get("scene_graph") or "complete"),
            prefilter=None,
            symmetrize=True,
        )
        with torch.no_grad():
            output = inference(
                pairs,
                model,
                device,
                batch_size=int(kwargs.get("batch_size") or 1),
                verbose=bool(kwargs.get("verbose", False)),
            )
        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PairViewer,
            min_conf_thr=float(kwargs.get("min_conf_thr") or 3.0),
        )

        imgs = [np.asarray(img) for img in scene.imgs]
        pts3d = [item.detach().cpu().numpy() for item in scene.get_pts3d()]
        masks = [item.detach().cpu().numpy().astype(bool) for item in scene.get_masks()]
        points = []
        colors = []
        for pts, img, mask in zip(pts3d, imgs, masks, strict=False):
            if mask.shape == pts.shape[:2]:
                points.append(pts[mask])
                colors.append(img[mask])
            else:
                points.append(pts.reshape(-1, 3))
                colors.append(img.reshape(-1, 3))

        ply_path = output_dir / "dust3r_point_cloud.ply"
        _write_colored_points_ply(
            ply_path,
            points=points,
            colors=colors,
            max_points=int(kwargs.get("max_points") or 20000),
        )
        intrinsics_path = output_dir / "dust3r_intrinsics.npy"
        poses_path = output_dir / "dust3r_camera_poses.npy"
        np.save(intrinsics_path, scene.get_intrinsics().detach().cpu().numpy())
        np.save(poses_path, scene.get_im_poses().detach().cpu().numpy())
        preview_path = output_dir / "dust3r_input_preview.png"
        Image.fromarray(np.clip(imgs[0] * 255.0, 0, 255).astype(np.uint8)).save(preview_path)
        metadata_path = output_dir / "dust3r_metadata.json"
        self._write_json(
            metadata_path,
            {
                "model_id": self.model_id,
                "checkpoint_path": ckpt,
                "input_images": image_paths,
                "num_pairs": len(pairs),
                "point_cloud_path": str(ply_path),
                "intrinsics_path": str(intrinsics_path),
                "camera_poses_path": str(poses_path),
                "runtime_options": _safe_json(kwargs),
            },
        )
        return {
            "status": "succeeded",
            "runtime": "geometry_prior_in_tree",
            "model_id": self.model_id,
            "artifact_path": str(ply_path),
            "preview_image": str(preview_path),
            "point_cloud_path": str(ply_path),
            "metadata_path": str(metadata_path),
        }

    def _run_track_anything(self, *, video: Any, images: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        from worldfoundry.base_models.perception_core.tracking.track_anything import TrackAnythingPipeline

        defaults = _known_checkpoint_candidates(self.model_id)

        def checkpoint_path(key: str, index: int) -> Path:
            return Path(str(kwargs.get(key) or self.options.get(key) or defaults[index])).expanduser()

        sam_ckpt = checkpoint_path("sam_ckpt_path", 0)
        aot_ckpt = checkpoint_path("aot_ckpt_path", 1)
        grounding_ckpt = checkpoint_path("grounding_dino_ckpt_path", 2)
        frames = _load_video_frames(
            _first_available(video, kwargs.get("video_path"), kwargs.get("input_path"), images),
            max_frames=int(kwargs.get("max_frames") or 3),
        )
        prompt = str(kwargs.get("mask_phrase") or kwargs.get("object_prompt") or kwargs.get("prompt") or "robot")
        gpu_id = int(kwargs.get("gpu_id") or 0)
        tracker = TrackAnythingPipeline(
            [prompt],
            sam_points_per_side=int(kwargs.get("sam_points_per_side") or 16),
            sam_run_gap=int(kwargs.get("sam_run_gap") or 10),
            sam_ckpt_path=str(sam_ckpt),
            aot_ckpt_path=str(aot_ckpt),
            grounding_dino_ckpt_path=str(grounding_ckpt),
            gpu_id=gpu_id,
        )

        masks = []
        phrases: dict[int, str] = {}
        for index, frame in enumerate(frames):
            rgb = torch.from_numpy(frame.astype(np.float32) / 255.0)
            mask, frame_phrases = tracker.track(SimpleNamespace(raw_frame_idx=index, rgb=rgb))
            masks.append(mask.detach().cpu().numpy().astype(np.uint8))
            phrases.update({int(key): str(value) for key, value in frame_phrases.items()})

        masks_np = np.stack(masks, axis=0)
        masks_path = output_dir / "track_masks.npz"
        np.savez_compressed(masks_path, masks=masks_np)
        preview_path = output_dir / "track_preview.png"
        _save_mask_overlay(frames[0], masks_np[0], preview_path)
        metadata_path = output_dir / "track_anything_metadata.json"
        self._write_json(
            metadata_path,
            {
                "model_id": self.model_id,
                "prompt": prompt,
                "num_frames": len(frames),
                "mask_shape": list(masks_np.shape),
                "unique_mask_ids": [int(item) for item in np.unique(masks_np)],
                "phrases": phrases,
                "sam_checkpoint": str(sam_ckpt),
                "aot_checkpoint": str(aot_ckpt),
                "grounding_dino_checkpoint": str(grounding_ckpt),
                "masks_path": str(masks_path),
                "preview_image": str(preview_path),
                "runtime_options": _safe_json(kwargs),
            },
        )
        return {
            "status": "succeeded",
            "runtime": "geometry_prior_in_tree",
            "model_id": self.model_id,
            "artifact_path": str(masks_path),
            "preview_image": str(preview_path),
            "masks_path": str(masks_path),
            "metadata_path": str(metadata_path),
        }


def _load_rgb_array(source: Any) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source.convert("RGB")
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            image_files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
            if not image_files:
                raise FileNotFoundError(f"No image files found in {path}")
            path = image_files[0]
        image = Image.open(path).convert("RGB")
    elif isinstance(source, torch.Tensor):
        array = source.detach().cpu().float().numpy()
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.moveaxis(array, 0, -1)
        if array.max(initial=0.0) > 2.0:
            array = array / 255.0
        return _ensure_rgb_float(array)
    elif isinstance(source, np.ndarray):
        array = source
        if array.ndim == 4:
            array = array[0]
        if array.max(initial=0.0) > 2.0:
            array = array.astype(np.float32) / 255.0
        return _ensure_rgb_float(array)
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        if not source:
            raise ValueError("Image sequence is empty.")
        return _load_rgb_array(source[0])
    else:
        raise ValueError("Geometry prior inference requires an image path, image upload, or image array.")
    return np.asarray(image).astype(np.float32) / 255.0


def _first_available(*items: Any) -> Any:
    for item in items:
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        return item
    return None


def _ensure_rgb_float(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    return np.clip(array, 0.0, 1.0)


def _load_video_frames(source: Any, *, max_frames: int) -> list[np.ndarray]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray, np.ndarray)):
        frames = [_as_uint8_frame(item) for item in source[:max_frames]]
    elif isinstance(source, (str, Path)) and Path(source).expanduser().suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        frames = _read_video_file(Path(source).expanduser(), max_frames=max_frames)
    else:
        frame = _as_uint8_frame(source)
        frames = [frame, frame.copy()]
    if not frames:
        raise ValueError("Video depth inference requires at least one frame.")
    return frames


def _read_video_file(path: Path, *, max_frames: int) -> list[np.ndarray]:
    try:
        import imageio.v3 as iio

        frames = []
        for index, frame in enumerate(iio.imiter(path)):
            if index >= max_frames:
                break
            frames.append(_as_uint8_frame(frame))
        return frames
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames


def _as_uint8_frame(item: Any) -> np.ndarray:
    rgb = _load_rgb_array(item)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _load_prompt_depth(path: Any, shape: tuple[int, int], *, device: str) -> torch.Tensor:
    if not path:
        raise ValueError(
            "Prior Depth Anything requires prompt_depth_path or prior_depth_path; "
            "a synthetic depth ramp is not a valid inference condition"
        )
    array = np.load(path) if str(path).endswith(".npy") else np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    if array.shape != shape:
        array = np.asarray(Image.fromarray(array.astype(np.float32)).resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)
    return torch.from_numpy(array.astype(np.float32)).to(device=device)


def _write_depth_artifacts(
    *,
    model_id: str,
    result: Any,
    rgb: np.ndarray,
    output_dir: Path,
    depth_kind: str,
    max_points: int,
    focal_length: float | None = None,
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    depth = getattr(result, "metric_depth", None)
    depth_source = "metric_depth"
    if depth is None:
        depth = getattr(result, "relative_inv_depth", None)
        depth_source = "relative_inv_depth"
    if depth is None:
        raise RuntimeError(f"{model_id} did not return metric_depth or relative_inv_depth.")
    depth_np = depth.detach().cpu().float().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth, dtype=np.float32)
    first_depth = depth_np[0] if depth_np.ndim == 3 else depth_np
    if first_depth.ndim == 3:
        first_depth = first_depth[0]

    depth_path = output_dir / "depth.npy"
    np.save(depth_path, depth_np)
    preview_path = output_dir / "depth_preview.png"
    _save_depth_preview(first_depth, preview_path)
    ply_path = output_dir / "point_cloud.ply"
    points = _result_points(result)
    if points is not None:
        _write_camera_points_ply(ply_path, rgb=rgb, points=points, max_points=max_points)
        point_source = "model_points"
    else:
        _write_point_cloud_ply(
            ply_path,
            rgb=rgb,
            depth=first_depth,
            inverse_depth=depth_source == "relative_inv_depth",
            max_points=max_points,
            focal_length=focal_length,
        )
        point_source = "depth_backprojection"
    metadata_path = output_dir / "depth_metadata.json"
    metadata = {
        "model_id": model_id,
        "depth_source": depth_source,
        "depth_kind": depth_kind,
        "depth_shape": list(depth_np.shape),
        "point_source": point_source,
        "coordinate_frame": "camera",
        "coordinate_convention": "x-right/y-up/z-forward",
        "preview_image": str(preview_path),
        "point_cloud_path": str(ply_path),
        **dict(extra_metadata),
    }
    metadata_path.write_text(json.dumps(_safe_json(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "succeeded",
        "runtime": "geometry_prior_in_tree",
        "model_id": model_id,
        "artifact_path": str(ply_path),
        "depth_path": str(depth_path),
        "preview_image": str(preview_path),
        "point_cloud_path": str(ply_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def _save_depth_preview(depth: np.ndarray, path: Path) -> None:
    finite = np.isfinite(depth)
    if finite.any():
        lo = float(np.percentile(depth[finite], 2))
        hi = float(np.percentile(depth[finite], 98))
        denom = max(hi - lo, 1e-6)
        norm = np.clip((depth - lo) / denom, 0.0, 1.0)
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    image = (norm * 255.0).astype(np.uint8)
    try:
        import cv2

        colored = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        Image.fromarray(colored).save(path)
    except Exception:
        Image.fromarray(image).save(path)


def _collect_image_paths(source: Any, *, output_dir: Path, max_images: int) -> list[str]:
    image_paths: list[str] = []

    def add_path(path: Path) -> None:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    image_paths.append(str(child))
                    if len(image_paths) >= max_images:
                        return
        elif path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            image_paths.append(str(path))

    if isinstance(source, (str, Path)):
        add_path(Path(source).expanduser())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray, np.ndarray)):
        for item in source:
            if len(image_paths) >= max_images:
                break
            if isinstance(item, (str, Path)):
                add_path(Path(item).expanduser())
            else:
                frame_path = output_dir / f"dust3r_input_{len(image_paths):03d}.png"
                Image.fromarray(_as_uint8_frame(item)).save(frame_path)
                image_paths.append(str(frame_path))
    elif source is not None:
        frame_path = output_dir / "dust3r_input_000.png"
        Image.fromarray(_as_uint8_frame(source)).save(frame_path)
        image_paths.append(str(frame_path))

    return image_paths[:max(max_images, 1)]


def _write_colored_points_ply(
    path: Path,
    *,
    points: Sequence[np.ndarray],
    colors: Sequence[np.ndarray],
    max_points: int,
) -> None:
    point_chunks = []
    color_chunks = []
    for pts, rgb in zip(points, colors, strict=False):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        rgb = np.asarray(rgb).reshape(-1, 3)
        if rgb.dtype != np.uint8:
            if rgb.size and float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        valid = np.isfinite(pts).all(axis=1)
        point_chunks.append(pts[valid])
        color_chunks.append(rgb[valid])

    all_points = np.concatenate(point_chunks, axis=0) if point_chunks else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.zeros((0, 3), dtype=np.uint8)
    if all_points.shape[0] > max_points:
        indices = np.linspace(0, all_points.shape[0] - 1, max_points).astype(np.int64)
        all_points = all_points[indices]
        all_colors = all_colors[indices]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {all_points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(all_points, all_colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _save_mask_overlay(frame: np.ndarray, mask: np.ndarray, path: Path) -> None:
    rgb = np.asarray(frame, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = np.asarray(Image.fromarray(mask).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST), dtype=np.uint8)
    overlay = rgb.astype(np.float32).copy()
    foreground = mask > 0
    overlay[foreground] = overlay[foreground] * 0.45 + np.array([255.0, 64.0, 32.0], dtype=np.float32) * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def _result_points(result: Any) -> np.ndarray | None:
    points = getattr(result, "points", None)
    if points is None:
        return None
    array = points.detach().cpu().float().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
    if array.ndim == 4:
        array = np.moveaxis(array, 1, -1)[0] if array.shape[1] == 3 else array[0]
    if array.ndim != 3:
        return None
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    return np.asarray(array, dtype=np.float32) if array.shape[-1] == 3 else None


def _write_camera_points_ply(path: Path, *, rgb: np.ndarray, points: np.ndarray, max_points: int) -> None:
    points = np.asarray(points, dtype=np.float32).copy()
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if colors.shape[:2] != points.shape[:2]:
        colors = np.asarray(Image.fromarray(colors).resize((points.shape[1], points.shape[0]), Image.BILINEAR))
    # Model outputs use OpenCV camera coordinates. Persist a user-facing
    # right/up/forward frame so exported PLY assets need no second transform.
    points[..., 1] *= -1.0
    _write_colored_points_ply(path, points=(points,), colors=(colors,), max_points=max_points)


def _write_point_cloud_ply(
    path: Path,
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    inverse_depth: bool,
    max_points: int,
    focal_length: float | None = None,
) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    if inverse_depth:
        values = depth[np.isfinite(depth)]
        scale = float(np.percentile(values, 95)) if values.size else 1.0
        z = 1.0 / np.maximum(depth / max(scale, 1e-6), 1e-3)
    else:
        z = depth
    h, w = z.shape[:2]
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if rgb_uint8.shape[:2] != (h, w):
        rgb_uint8 = np.asarray(Image.fromarray(rgb_uint8).resize((w, h), Image.BILINEAR))
    stride = max(int(np.sqrt(max(h * w, 1) / max(max_points, 1))), 1)
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = z[ys, xs]
    valid = np.isfinite(zs) & (zs > 0)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)
    if xs.size > max_points:
        indices = np.linspace(0, xs.size - 1, max_points).astype(np.int64)
        xs, ys, zs = xs[indices], ys[indices], zs[indices]
    colors = rgb_uint8[(ys.astype(np.int64)).clip(0, h - 1), (xs.astype(np.int64)).clip(0, w - 1)]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    focal = float(focal_length or max(h, w))
    points = np.stack(((xs - cx) / focal * zs, -(ys - cy) / focal * zs, zs), axis=1)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "to_dict"):
        try:
            return _safe_json(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Image.Image):
        return {"mode": value.mode, "size": list(value.size)}
    if hasattr(value, "__dict__"):
        return _safe_json(vars(value))
    return str(value)


__all__ = ["GeometryPriorSynthesis"]
