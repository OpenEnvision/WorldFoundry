from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from worldfoundry.core.model_loading import load_torch_checkpoint

DEFAULT_REPO_ID = "robbyant/lingbot-map"
DEFAULT_CHECKPOINT_CANDIDATES = (
    "lingbot-map-long.pt",
    "lingbot-map.pt",
    "lingbot-map-stage1.pt",
    "model.pt",
    "pytorch_model.bin",
)


def _runtime_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_runtime_importable() -> None:
    root = str(_runtime_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _collect_image_paths(path: str | os.PathLike[str]) -> list[str]:
    path_obj = Path(path).expanduser()
    if path_obj.is_file():
        if path_obj.suffix.lower() == ".txt":
            return [
                line.strip()
                for line in path_obj.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return [str(path_obj)]
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    image_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sorted(
        str(candidate)
        for candidate in path_obj.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in image_exts
    )


def _resolve_checkpoint_path(pretrained_model_path: str | os.PathLike[str] | None, checkpoint_name: str | None) -> str:
    if pretrained_model_path is None:
        raise ValueError("LingBotMapRepresentation.from_pretrained requires a checkpoint path or repo directory.")

    candidate = Path(pretrained_model_path).expanduser()
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        names = (checkpoint_name,) if checkpoint_name else DEFAULT_CHECKPOINT_CANDIDATES
        for name in names:
            if name is None:
                continue
            checkpoint = candidate / name
            if checkpoint.is_file():
                return str(checkpoint)
        pt_files = sorted(candidate.glob("*.pt"))
        if pt_files:
            return str(pt_files[0])
        raise FileNotFoundError(
            f"No LingBot-Map checkpoint found in {candidate}. "
            f"Tried: {', '.join(DEFAULT_CHECKPOINT_CANDIDATES)}"
        )

    path_text = str(pretrained_model_path)
    if "/" in path_text and not candidate.is_absolute() and not path_text.startswith("."):
        from huggingface_hub import snapshot_download

        repo_root = Path(snapshot_download(path_text))
        return _resolve_checkpoint_path(repo_root, checkpoint_name)

    raise FileNotFoundError(f"LingBot-Map checkpoint path does not exist: {pretrained_model_path}")


def _pil_from_input(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = array.astype("float32")
            if array.size and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        if array.ndim == 2:
            pil = Image.fromarray(array, mode="L")
        else:
            pil = Image.fromarray(array[..., :3])
    else:
        pil = Image.open(image)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode == "RGBA":
        background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(background, pil)
    return pil.convert("RGB")


def _preprocess_image_objects(images: Iterable[Any], image_size: int, patch_size: int, mode: str) -> torch.Tensor:
    tensors = []
    for image in images:
        pil = _pil_from_input(image)
        width, height = pil.size
        if mode == "pad":
            if width >= height:
                new_width = image_size
                new_height = round(height * (new_width / width) / patch_size) * patch_size
            else:
                new_height = image_size
                new_width = round(width * (new_height / height) / patch_size) * patch_size
        elif mode == "crop":
            new_width = image_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            raise ValueError("preprocess_mode must be 'crop' or 'pad'.")

        pil = pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
        array = np.asarray(pil).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if mode == "crop" and new_height > image_size:
            start_y = (new_height - image_size) // 2
            tensor = tensor[:, start_y : start_y + image_size, :]
        if mode == "pad":
            h_padding = image_size - tensor.shape[1]
            w_padding = image_size - tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
        tensors.append(tensor)

    if not tensors:
        raise ValueError("LingBot-Map requires at least one image.")

    shapes = {tuple(tensor.shape[-2:]) for tensor in tensors}
    if len(shapes) > 1:
        max_h = max(shape[0] for shape in shapes)
        max_w = max(shape[1] for shape in shapes)
        padded = []
        for tensor in tensors:
            h_padding = max_h - tensor.shape[1]
            w_padding = max_w - tensor.shape[2]
            padded.append(
                F.pad(
                    tensor,
                    (w_padding // 2, w_padding - w_padding // 2, h_padding // 2, h_padding - h_padding // 2),
                    mode="constant",
                    value=1.0,
                )
            )
        tensors = padded
    return torch.stack(tensors)


class LingBotMapRepresentation:
    """In-tree LingBot-Map streaming 3D reconstruction wrapper."""

    def __init__(
        self,
        model: Any = None,
        device: Optional[str] = None,
        *,
        mode: str = "streaming",
        image_size: int = 518,
        patch_size: int = 14,
        num_scale_frames: int = 8,
        keyframe_interval: int | str | None = "auto",
        use_amp: bool = True,
        preprocess_mode: str = "crop",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.mode = mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_scale_frames = num_scale_frames
        self.keyframe_interval = keyframe_interval
        self.use_amp = use_amp
        self.preprocess_mode = preprocess_mode
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if capability >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | os.PathLike[str] | None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "LingBotMapRepresentation":
        _ensure_runtime_importable()
        mode = str(kwargs.get("mode", "streaming"))
        if mode not in {"streaming", "windowed"}:
            raise ValueError("LingBot-Map mode must be 'streaming' or 'windowed'.")

        checkpoint_path = _resolve_checkpoint_path(
            pretrained_model_path or kwargs.get("checkpoint_path") or kwargs.get("repo_root") or DEFAULT_REPO_ID,
            kwargs.get("checkpoint_name"),
        )
        if mode == "windowed":
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        model = GCTStream(
            img_size=int(kwargs.get("image_size", 518)),
            patch_size=int(kwargs.get("patch_size", 14)),
            enable_3d_rope=bool(kwargs.get("enable_3d_rope", True)),
            max_frame_num=int(kwargs.get("max_frame_num", 1024)),
            kv_cache_sliding_window=int(kwargs.get("kv_cache_sliding_window", 64)),
            kv_cache_scale_frames=int(kwargs.get("kv_cache_scale_frames", kwargs.get("num_scale_frames", 8))),
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=bool(kwargs.get("use_sdpa", True)),
            camera_num_iterations=int(kwargs.get("camera_num_iterations", 4)),
        )
        # The checkpoint is consumed purely as a state_dict; load weights-only by
        # default and only fall back to unrestricted pickle on legacy wrapped
        # checkpoints that fail the safe path.
        checkpoint = load_torch_checkpoint(
            checkpoint_path,
            map_location=device or "cpu",
            weights_only=True,
            allow_unsafe_pickle_fallback=True,
        )
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return cls(
            model=model,
            device=device,
            mode=mode,
            image_size=int(kwargs.get("image_size", 518)),
            patch_size=int(kwargs.get("patch_size", 14)),
            num_scale_frames=int(kwargs.get("num_scale_frames", 8)),
            keyframe_interval=kwargs.get("keyframe_interval", "auto"),
            use_amp=bool(kwargs.get("use_amp", True)),
            preprocess_mode=str(kwargs.get("preprocess_mode", "crop")),
        )


    def _prepare_images(self, images: Any, **kwargs: Any) -> torch.Tensor:
        _ensure_runtime_importable()
        image_size = int(kwargs.get("image_size", self.image_size))
        patch_size = int(kwargs.get("patch_size", self.patch_size))
        mode = str(kwargs.get("preprocess_mode", self.preprocess_mode))

        if isinstance(images, torch.Tensor):
            tensor = images.float()
        elif isinstance(images, (str, os.PathLike)):
            from lingbot_map.utils.load_fn import load_and_preprocess_images

            paths = _collect_image_paths(images)
            tensor = load_and_preprocess_images(paths, mode=mode, image_size=image_size, patch_size=patch_size)
        elif isinstance(images, (list, tuple)) and all(isinstance(item, (str, os.PathLike)) for item in images):
            from lingbot_map.utils.load_fn import load_and_preprocess_images

            tensor = load_and_preprocess_images(
                [str(item) for item in images],
                mode=mode,
                image_size=image_size,
                patch_size=patch_size,
            )
        elif isinstance(images, (list, tuple)):
            tensor = _preprocess_image_objects(images, image_size=image_size, patch_size=patch_size, mode=mode)
        else:
            tensor = _preprocess_image_objects([images], image_size=image_size, patch_size=patch_size, mode=mode)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _resolve_keyframe_interval(self, num_frames: int, raw_value: Any = None) -> int:
        value = self.keyframe_interval if raw_value is None else raw_value
        if value is None or value == 0 or (isinstance(value, str) and value.lower() == "auto"):
            return 1 if num_frames <= 320 else (num_frames + 319) // 320
        return max(1, int(value))

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("LingBot-Map model is not loaded. Use from_pretrained() first.")
        options = dict(data)
        raw_images = options.pop("images")
        images = self._prepare_images(raw_images, **options)
        num_frames = int(images.shape[0])
        output_device = torch.device("cpu") if data.get("offload_to_cpu", True) else None
        autocast_enabled = self.device == "cuda" and self.use_amp
        keyframe_interval = self._resolve_keyframe_interval(num_frames, data.get("keyframe_interval"))
        run_mode = str(data.get("mode") or self.mode)
        if run_mode not in {"streaming", "windowed"}:
            raise ValueError("LingBot-Map mode must be 'streaming' or 'windowed'.")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
            if run_mode == "windowed":
                if not hasattr(self.model, "inference_windowed"):
                    raise ValueError("Windowed inference requires loading LingBot-Map with mode='windowed'.")
                predictions = self.model.inference_windowed(
                    images,
                    window_size=int(data.get("window_size", 64)),
                    overlap_size=data.get("overlap_size", 16),
                    overlap_keyframes=data.get("overlap_keyframes"),
                    num_scale_frames=int(data.get("num_scale_frames", self.num_scale_frames)),
                    keyframe_interval=keyframe_interval,
                    flow_threshold=float(data.get("flow_threshold", 0.0)),
                    max_non_keyframe_gap=int(data.get("max_non_keyframe_gap", 30)),
                    output_device=output_device,
                )
            else:
                predictions = self.model.inference_streaming(
                    images,
                    num_scale_frames=int(data.get("num_scale_frames", self.num_scale_frames)),
                    keyframe_interval=keyframe_interval,
                    output_device=output_device,
                )

        _ensure_runtime_importable()
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        result: Dict[str, Any] = {
            "extrinsic": extrinsic.float().cpu().numpy().squeeze(0),
            "intrinsic": intrinsic.float().cpu().numpy().squeeze(0),
            "keyframe_interval": keyframe_interval,
            "mode": run_mode,
        }
        for key in ("depth", "depth_conf", "world_points", "world_points_conf", "images"):
            value = predictions.get(key)
            if isinstance(value, torch.Tensor):
                result[key] = value.float().cpu().numpy().squeeze(0)
            elif value is not None:
                result[key] = value
        result["input_images"] = images.detach().cpu().numpy()
        return result
