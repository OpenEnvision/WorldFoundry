from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

DEFAULT_VGGT_OMEGA_CHECKPOINT_NAME = "vggt_omega_1b_512.pt"


class VGGTOmegaRepresentation:
    """VGGT-Omega representation model for camera, depth, and point maps."""

    def __init__(self, model: Optional[Any] = None, device: Optional[str] = None) -> None:
        """
        Create a VGGT-Omega runtime wrapper.

        Args:
            model: Loaded official VGGT-Omega model instance.
            device: Torch device used for inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        enable_alignment: bool = False,
        repo_root: Optional[str] = None,
        **kwargs: Any,
    ) -> "VGGTOmegaRepresentation":
        """
        Load the official VGGT-Omega checkpoint.

        Args:
            pretrained_model_path: Checkpoint file, checkpoint directory, or HF-style model id.
            device: Torch device used for inference.
            enable_alignment: Whether to instantiate the text-alignment head.
            repo_root: Deprecated source-tree path argument. External checkouts are not used.
            **kwargs: Optional preprocessing configuration.
        """
        if repo_root is not None:
            raise RuntimeError(
                "VGGT-Omega no longer accepts external repo_root source trees. "
                "Install vggt_omega as a normal Python package or vendor it under WorldFoundry."
            )

        from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.vggt_omega.models import (
            VGGTOmega,
        )

        checkpoint_path = _resolve_checkpoint_file(pretrained_model_path, enable_alignment=enable_alignment)
        model = VGGTOmega(enable_alignment=enable_alignment)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)

        instance = cls(model=model, device=device)
        instance.preprocess_mode = kwargs.get("preprocess_mode", "balanced")
        instance.resolution = kwargs.get("resolution", 256 if enable_alignment else 512)
        instance.patch_size = kwargs.get("patch_size", 16)
        return instance

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run VGGT-Omega inference and return WorldFoundry-compatible arrays.

        Args:
            data: Input images plus preprocessing and prediction flags.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")

        image_list = _normalize_image_input(data["images"])
        preprocess_mode = data.get("preprocess_mode", self.preprocess_mode)
        resolution = data.get("resolution", self.resolution)
        patch_size = data.get("patch_size", self.patch_size)

        from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.vggt_omega.utils.load_fn import (
            load_and_preprocess_images,
        )
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.vggt_omega.utils.pose_enc import (
            encoding_to_camera,
        )

        images = load_and_preprocess_images(
            image_list,
            mode=_normalize_preprocess_mode(preprocess_mode),
            image_resolution=resolution,
            patch_size=patch_size,
        ).to(self.device)

        with torch.inference_mode():
            predictions = self.model(images)
            extrinsic, intrinsic = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

        depth_map = predictions["depth"].squeeze(0).detach().cpu().numpy()
        depth_conf = predictions["depth_conf"].squeeze(0).detach().cpu().numpy()
        extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
        intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
        point_map = _unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)

        results: Dict[str, Any] = {
            "extrinsic": extrinsic_np,
            "intrinsic": intrinsic_np,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "point_map": point_map,
            "point_conf": depth_conf,
            "point_map_from_depth": point_map,
        }

        if "camera_and_register_tokens" in predictions:
            results["camera_and_register_tokens"] = (
                predictions["camera_and_register_tokens"].detach().cpu().numpy()
            )
        if "text_alignment_embedding" in predictions:
            results["text_alignment_embedding"] = (
                predictions["text_alignment_embedding"].detach().cpu().numpy()
            )
        return results

def _resolve_checkpoint_file(pretrained_model_path: str, *, enable_alignment: bool = False) -> Path:
    """
    Resolve a VGGT-Omega checkpoint file.

    Args:
        pretrained_model_path: File path or directory containing released checkpoints.
        enable_alignment: Select the released 256-resolution text-alignment checkpoint.
    """
    path = Path(pretrained_model_path)
    if path.is_file():
        return path
    if path.is_dir():
        checkpoint_name = (
            "vggt_omega_1b_256_text.pt" if enable_alignment else DEFAULT_VGGT_OMEGA_CHECKPOINT_NAME
        )
        checkpoint = path / checkpoint_name
        if checkpoint.is_file():
            return checkpoint
        candidates = sorted(path.glob("*.pt"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"VGGT-Omega checkpoint file not found: {pretrained_model_path}")


def _normalize_image_input(images_input: Any) -> list[str]:
    """
    Normalize image inputs to official VGGT-Omega image paths.

    Args:
        images_input: Image path, directory, text list, or list of image paths.
    """
    if isinstance(images_input, (str, Path)):
        path = Path(images_input)
        if path.is_dir():
            return sorted(
                str(item)
                for item in path.iterdir()
                if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        if path.suffix.lower() == ".txt":
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [str(path)]
    if isinstance(images_input, list) and all(isinstance(item, (str, Path)) for item in images_input):
        return [str(item) for item in images_input]
    raise ValueError("VGGT-Omega official runtime requires image paths for this validation path.")


def _normalize_preprocess_mode(preprocess_mode: str) -> str:
    """
    Map WorldFoundry preprocessing names to official VGGT-Omega modes.

    Args:
        preprocess_mode: WorldFoundry or official preprocessing mode.
    """
    if preprocess_mode in {"balanced", "max_size"}:
        return preprocess_mode
    if preprocess_mode in {"crop", "square"}:
        return "balanced"
    raise ValueError(f"Unsupported VGGT-Omega preprocess_mode: {preprocess_mode}")


def _unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Convert VGGT-Omega depth maps to world-coordinate point maps.

    Args:
        depth_map: Depth maps with shape [S, H, W, 1] or [S, H, W].
        extrinsics: Camera-from-world matrices with shape [S, 3, 4].
        intrinsics: Camera intrinsics with shape [S, 3, 3].
    """
    depth = np.asarray(depth_map)
    if depth.ndim == 4:
        depth = depth[..., 0]

    point_maps = []
    for idx in range(depth.shape[0]):
        cur_depth = depth[idx]
        intrinsic = intrinsics[idx]
        extrinsic = extrinsics[idx]
        height, width = cur_depth.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x_cam = (xx - intrinsic[0, 2]) / intrinsic[0, 0] * cur_depth
        y_cam = (yy - intrinsic[1, 2]) / intrinsic[1, 1] * cur_depth
        cam_points = np.stack([x_cam, y_cam, cur_depth], axis=-1)
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        world_points = np.einsum(
            "ij,hwj->hwi",
            rotation.T,
            cam_points - translation.reshape(1, 1, 3),
        )
        point_maps.append(world_points)
    return np.stack(point_maps, axis=0)
