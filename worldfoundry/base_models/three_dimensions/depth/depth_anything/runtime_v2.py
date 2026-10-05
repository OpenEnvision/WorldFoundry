from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v2 import (
    DepthAnythingV2,
)
from worldfoundry.core.model_loading import load_torch_checkpoint

DEFAULT_DEPTH_ANYTHING2_SMALL_REPO = "depth-anything/Depth-Anything-V2-Small"
DEFAULT_DEPTH_ANYTHING2_BASE_REPO = "depth-anything/Depth-Anything-V2-Base"
DEFAULT_DEPTH_ANYTHING2_LARGE_REPO = "depth-anything/Depth-Anything-V2-Large"
DEFAULT_DEPTH_ANYTHING2_REPO = DEFAULT_DEPTH_ANYTHING2_LARGE_REPO

DEPTH_ANYTHING2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_DEPTH_ANYTHING2_REPOS = {
    "vits": DEFAULT_DEPTH_ANYTHING2_SMALL_REPO,
    "vitb": DEFAULT_DEPTH_ANYTHING2_BASE_REPO,
    "vitl": DEFAULT_DEPTH_ANYTHING2_LARGE_REPO,
}


def _checkpoint_filename(encoder: str) -> str:
    return f"depth_anything_v2_{encoder}.pth"


def _infer_encoder_from_name(value: str) -> Optional[str]:
    lowered = value.lower()
    encoder_hints = {
        "vits": ("depth_anything_v2_vits", "vits", "small"),
        "vitb": ("depth_anything_v2_vitb", "vitb", "base"),
        "vitl": ("depth_anything_v2_vitl", "vitl", "large"),
        "vitg": ("depth_anything_v2_vitg", "vitg", "giant"),
    }
    for encoder, hints in encoder_hints.items():
        if any(hint in lowered for hint in hints):
            return encoder
    return None


def _unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def _infer_encoder_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    embed_weight = state_dict.get("pretrained.cls_token")
    if embed_weight is None:
        embed_weight = state_dict.get("pretrained.pos_embed")
    if embed_weight is None:
        return None
    embed_dim = int(embed_weight.shape[-1])
    dim_to_encoder = {
        384: "vits",
        768: "vitb",
        1024: "vitl",
        1536: "vitg",
    }
    return dim_to_encoder.get(embed_dim)


class DepthAnything2Representation:
    """Representation wrapper around the vendored official Depth Anything V2 runtime."""

    def __init__(
        self,
        model: Optional[DepthAnythingV2] = None,
        device: Optional[str] = None,
        encoder: str = "vitl",
        default_input_size: int = 518,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder
        self.default_input_size = int(default_input_size)
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device).eval()

    @staticmethod
    def _resolve_checkpoint_path(
        pretrained_model_path: Optional[str],
        encoder: str,
    ) -> Tuple[str, str]:
        requested_encoder = str(encoder or "vitl")
        if requested_encoder not in DEPTH_ANYTHING2_MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported Depth Anything V2 encoder '{requested_encoder}'. "
                f"Expected one of {sorted(DEPTH_ANYTHING2_MODEL_CONFIGS)}."
            )

        if pretrained_model_path is None:
            if requested_encoder not in DEFAULT_DEPTH_ANYTHING2_REPOS:
                raise ValueError(
                    "Depth Anything V2 does not define a default Hugging Face repo for `vitg`. "
                    "Provide a local checkpoint path or explicit repo id."
                )
            repo_id = DEFAULT_DEPTH_ANYTHING2_REPOS[requested_encoder]
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=_checkpoint_filename(requested_encoder))
            return checkpoint_path, requested_encoder

        candidate = Path(pretrained_model_path).expanduser()
        hinted_encoder = _infer_encoder_from_name(str(pretrained_model_path)) or requested_encoder

        if candidate.exists():
            if candidate.is_file():
                return str(candidate.resolve()), hinted_encoder

            search_roots = [candidate, candidate / "checkpoints"]
            exact_candidates = []
            for root in search_roots:
                exact_candidates.append(root / _checkpoint_filename(requested_encoder))
                if hinted_encoder != requested_encoder:
                    exact_candidates.append(root / _checkpoint_filename(hinted_encoder))

            for checkpoint_path in exact_candidates:
                if checkpoint_path.is_file():
                    resolved_encoder = _infer_encoder_from_name(checkpoint_path.name) or hinted_encoder
                    return str(checkpoint_path.resolve()), resolved_encoder

            discovered = []
            for root in search_roots:
                if root.is_dir():
                    discovered.extend(sorted(root.glob("depth_anything_v2_*.pth")))
            if len(discovered) == 1:
                resolved_encoder = _infer_encoder_from_name(discovered[0].name) or hinted_encoder
                return str(discovered[0].resolve()), resolved_encoder

            if (candidate / "depth_anything_v2").is_dir():
                raise ValueError(
                    "Received a Depth-Anything-V2 code checkout without checkpoints. "
                    "Pass a checkpoint file, a directory containing `depth_anything_v2_<encoder>.pth`, "
                    "or a Hugging Face repo id such as `depth-anything/Depth-Anything-V2-Large`."
                )

            raise FileNotFoundError(
                f"Could not find a Depth-Anything-V2 checkpoint under {candidate}. "
                f"Expected `{_checkpoint_filename(requested_encoder)}`."
            )

        repo_id = pretrained_model_path
        repo_encoder = _infer_encoder_from_name(repo_id) or requested_encoder
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=_checkpoint_filename(repo_encoder))
        return checkpoint_path, repo_encoder

    @classmethod
    def load_model(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        **kwargs,
    ) -> Tuple[DepthAnythingV2, str]:
        del kwargs
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path, requested_encoder = cls._resolve_checkpoint_path(pretrained_model_path, encoder)

        # The checkpoint is consumed purely as a state_dict; load weights-only by
        # default and only fall back to unrestricted pickle on legacy wrapped
        # checkpoints that fail the safe path.
        checkpoint = load_torch_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            allow_unsafe_pickle_fallback=True,
        )

        state_dict = _unwrap_state_dict(checkpoint)
        resolved_encoder = _infer_encoder_from_state_dict(state_dict) or requested_encoder
        model_config = dict(DEPTH_ANYTHING2_MODEL_CONFIGS[resolved_encoder])

        if all(f"depth_head.projects.{idx}.weight" in state_dict for idx in range(4)):
            model_config["out_channels"] = [
                int(state_dict[f"depth_head.projects.{idx}.weight"].shape[0])
                for idx in range(4)
            ]

        model = DepthAnythingV2(**model_config)
        model.load_state_dict(state_dict, strict=True)
        return model.to(target_device).eval(), resolved_encoder

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        encoder: str = "vitl",
        device: Optional[str] = None,
        default_input_size: int = 518,
        **kwargs,
    ) -> "DepthAnything2Representation":
        model, resolved_encoder = cls.load_model(
            pretrained_model_path=pretrained_model_path,
            encoder=encoder,
            device=device,
            **kwargs,
        )
        return cls(
            model=model,
            device=device,
            encoder=resolved_encoder,
            default_input_size=default_input_size,
        )

    @staticmethod
    def _as_uint8_hwc(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = np.asarray(image)

        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Unsupported image shape for DepthAnything2: {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        if np.issubdtype(array.dtype, np.floating):
            if array.min() >= -1.0 and array.max() <= 1.0:
                if array.min() < 0.0:
                    array = (array + 1.0) * 127.5
                else:
                    array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        elif array.dtype == np.uint8:
            array = np.ascontiguousarray(array)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(array)

    @classmethod
    def _coerce_raw_bgr(cls, image: Any, color_order: str = "rgb") -> np.ndarray:
        array = cls._as_uint8_hwc(image)
        normalized_order = str(color_order or "rgb").lower()
        if normalized_order == "rgb":
            return array[:, :, ::-1].copy()
        if normalized_order == "bgr":
            return array
        if normalized_order == "auto":
            if array.shape[-1] == 3 and array[..., 0].mean() > array[..., 2].mean():
                return array[:, :, ::-1].copy()
            return array
        raise ValueError(f"Unsupported color_order '{color_order}'. Expected 'rgb', 'bgr', or 'auto'.")

    def _predict_depth(self, raw_bgr: np.ndarray, input_size: int) -> torch.Tensor:
        height, width = raw_bgr.shape[:2]
        tensor, _ = self.model.image2tensor(raw_bgr, input_size)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            depth = self.model.forward(tensor)
        return F.interpolate(
            depth[:, None],
            (height, width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("DepthAnything2 model not loaded. Use from_pretrained() first.")

        raw_bgr = data.get("raw_bgr")
        if raw_bgr is None:
            if "image" not in data:
                raise ValueError("DepthAnything2Representation requires `raw_bgr` or `image`.")
            raw_bgr = self._coerce_raw_bgr(
                data["image"],
                color_order=data.get("color_order", "rgb"),
            )
        else:
            raw_bgr = self._coerce_raw_bgr(
                raw_bgr,
                color_order=data.get("raw_bgr_color_order", "bgr"),
            )

        return_visualization = bool(data.get("return_visualization", False))
        grayscale = bool(data.get("grayscale", False))
        input_size = int(data.get("input_size", self.default_input_size))

        depth = self._predict_depth(raw_bgr, input_size=input_size)

        result = {"depth": depth}
        if return_visualization:
            from worldfoundry.core.media.artifacts import (
                depth_to_colormap_rgb,
                depth_to_uint8,
            )

            depth_uint8 = depth_to_uint8(depth.detach().cpu().numpy())
            if depth_uint8 is None:
                raise ValueError(f"Unexpected depth shape for visualization: {tuple(depth.shape)}")
            if grayscale:
                depth_vis = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = depth_to_colormap_rgb(depth_uint8)
            result["depth_visualization"] = depth_vis

        return result


__all__ = [
    "DEFAULT_DEPTH_ANYTHING2_BASE_REPO",
    "DEFAULT_DEPTH_ANYTHING2_LARGE_REPO",
    "DEFAULT_DEPTH_ANYTHING2_REPO",
    "DEFAULT_DEPTH_ANYTHING2_SMALL_REPO",
    "DepthAnything2Representation",
]
