from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "checkpoints/helios-distilled"
DEFAULT_WAH_LORA = "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"


def _field(row: dict[str, str], *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_csv_path(value: str, csv_path: Path, *, required: bool = False) -> Path | None:
    value = str(value or "").strip()
    if not value:
        if required:
            raise ValueError(f"Missing required path in {csv_path}")
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [csv_path.parent / raw, RUNTIME_ROOT / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"Could not resolve {value!r} from {csv_path}")
    return candidates[0].resolve()


def load_demo_row(csv_path: Path) -> dict[str, Any]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing demo CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{csv_path} must contain exactly one data row, got {len(rows)}")
    row = rows[0]
    image_path = _resolve_csv_path(
        _field(row, "first_frame_path", "image_path", "image", "first_frame"),
        csv_path,
        required=True,
    )
    prompt = _field(row, "prompt", "prompts", "caption")
    prompt_path = _resolve_csv_path(_field(row, "prompt_path"), csv_path, required=False)
    if not prompt and prompt_path is not None:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{csv_path} must provide prompt or prompt_path")
    return {
        "csv_path": csv_path,
        "image_path": image_path,
        "prompt": prompt,
        "camera_poses_path": _resolve_csv_path(_field(row, "camera_poses_path", "camera_path"), csv_path),
        "warp_video_path": _resolve_csv_path(_field(row, "warp_video_path", "warp_path"), csv_path),
        "warp_visibility_mask_path": _resolve_csv_path(
            _field(
                row,
                "warp_visibility_mask_path",
                "warp_visibliry_mask_path",
                "visibility_mask_path",
                "warp_mask_path",
            ),
            csv_path,
        ),
    }


def load_camera_poses(path: Path, key: str) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera pose file: {path}")
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}. Available keys: {list(data.files)}")
        poses = np.asarray(data[key], dtype=np.float32)
        fps = int(round(float(data["fps"]))) if "fps" in data else 16  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera poses with shape [T, 4, 4], got {poses.shape}")
    return poses, fps


def load_video_frames(path: Path) -> tuple[list[np.ndarray], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing video file: {path}")
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        frames = [frame_to_uint8(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path} contains no frames")
    fps = int(round(float(meta.get("fps") or 16)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    return frames, fps


def torch_dtype_from_arg(dtype: str, device: str):
    import torch

    if dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def unwrap_video_frames(value: Any) -> list[Any]:
    if hasattr(value, "frames"):
        value = value.frames
    if isinstance(value, np.ndarray):
        if value.ndim == 5:
            value = value[0]
        if value.ndim == 4:
            return [value[i] for i in range(value.shape[0])]
        if value.ndim == 3:
            return [value]
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
            nested = value[0]
            if not (isinstance(nested, np.ndarray) and nested.ndim == 3):
                return unwrap_video_frames(nested)
        return list(value)
    raise TypeError(f"Unsupported pipeline output type: {type(value)!r}")


def frame_to_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        arr = np.asarray(frame)
        if arr.ndim != 3:
            raise ValueError(f"Expected frame with shape [H, W, C], got {arr.shape}")
        if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.round().astype(np.uint8)
    return arr


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame_to_uint8(frame))


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.is_absolute() and path.is_dir():
        return str(path)
    if not path.is_absolute():
        path = RUNTIME_ROOT / path
    path = Path(str(path.absolute()))
    checkpoints_root = Path(str((RUNTIME_ROOT / "checkpoints").absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"model_path must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {path}")
    return str(path)


def resolve_lora_path(lora_path: str | Path | None) -> str | None:
    if lora_path is None:
        return None
    value = str(lora_path).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute() and path.is_file():
        return str(path)
    if not path.is_absolute():
        path = RUNTIME_ROOT / path
    path = Path(str(path.absolute()))
    checkpoints_root = Path(str((RUNTIME_ROOT / "checkpoints").absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"lora_path must be under {checkpoints_root}, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing LoRA checkpoint: {path}")
    return str(path)


def disable_diffusers_optional_attention() -> None:
    try:
        import diffusers.utils.import_utils as diffusers_import_utils
    except Exception:
        return
    for attr in (
        "_xformers_available",
        "_flash_attn_available",
        "_flash_attn_3_available",
        "_aiter_available",
        "_sageattention_available",
    ):
        if hasattr(diffusers_import_utils, attr):
            setattr(diffusers_import_utils, attr, False)
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils
    except Exception:
        return
    for module in (transformers_utils, transformers_import_utils):
        for name in (
            "is_flash_attn_2_available",
            "is_flash_attn_greater_or_equal",
            "is_flash_attn_greater_or_equal_2_10",
        ):
            if hasattr(module, name):
                setattr(module, name, lambda *args, **kwargs: False)


def run_infer_from_csv(
    *,
    csv_path: Path,
    output: Path,
    model_path: str = DEFAULT_MODEL,
    lora_path: str | Path | None = DEFAULT_WAH_LORA,
    camera_key: str = "camera_poses",
    height: int = 384,
    width: int = 640,
    num_frames: int = 0,
    fps: int = 0,
    seed: int = 42,
    device: str = "cuda",
    dtype: str = "auto",
    no_lora: bool = False,
    warp_debug_dir: Path | None = None,
    enable_optional_attention: bool = False,
) -> dict[str, Any]:
    if not enable_optional_attention:
        disable_diffusers_optional_attention()

    sample = load_demo_row(csv_path)
    csv_path = sample["csv_path"]
    image_path = sample["image_path"]
    prompt = sample["prompt"]
    output = output.expanduser().resolve()

    warp_video = None
    warp_visibility_mask = None
    camera_poses = None
    conditioning_type = ""
    conditioning_frames = 0
    conditioning_fps = 16
    if sample["warp_video_path"] is not None:
        warp_video, conditioning_fps = load_video_frames(sample["warp_video_path"])
        conditioning_type = "warp_video"
        conditioning_frames = len(warp_video)
        if sample["warp_visibility_mask_path"] is not None:
            warp_visibility_mask, mask_fps = load_video_frames(sample["warp_visibility_mask_path"])
            if len(warp_visibility_mask) != conditioning_frames:
                raise ValueError(
                    f"warp visibility mask has {len(warp_visibility_mask)} frames, "
                    f"but warp video has {conditioning_frames} frames"
                )
            if int(fps) <= 0:
                conditioning_fps = mask_fps or conditioning_fps
    elif sample["camera_poses_path"] is not None:
        camera_poses, conditioning_fps = load_camera_poses(sample["camera_poses_path"], camera_key)
        conditioning_type = "camera_poses"
        conditioning_frames = int(camera_poses.shape[0])
    else:
        raise ValueError(f"{csv_path} must provide either warp_video_path or camera_poses_path")

    resolved_fps = int(fps) if int(fps) > 0 else int(conditioning_fps)
    resolved_num_frames = int(num_frames) if int(num_frames) > 0 else int(conditioning_frames)
    if conditioning_type == "camera_poses" and resolved_num_frames > conditioning_frames:
        raise ValueError(f"num_frames={resolved_num_frames} exceeds camera pose length {conditioning_frames}")

    import torch

    from . import WarpAsHistoryPipeline

    torch_dtype = torch_dtype_from_arg(dtype, device)
    generator = torch.Generator(device=device).manual_seed(int(seed)) if device.startswith("cuda") else None

    resolved_model_path = resolve_model_path(model_path)
    resolved_lora_path = None if no_lora else resolve_lora_path(lora_path)
    pipe = WarpAsHistoryPipeline.from_pretrained(resolved_model_path, torch_dtype=torch_dtype).to(device)
    pipe_kwargs = {
        "prompt": prompt,
        "image": Image.open(image_path).convert("RGB"),
        "lora_path": resolved_lora_path,
        "height": int(height),
        "width": int(width),
        "num_frames": resolved_num_frames,
        "generator": generator,
        "output_type": "np",
        "warp_debug_dir": warp_debug_dir,
        "warp_debug_fps": resolved_fps,
    }
    if conditioning_type == "warp_video":
        pipe_kwargs["warp_video"] = warp_video
        pipe_kwargs["warp_visibility_mask"] = warp_visibility_mask
    else:
        pipe_kwargs["camera_poses"] = camera_poses

    result = pipe(**pipe_kwargs)
    frames = unwrap_video_frames(result)
    write_video(output, frames, fps=resolved_fps)
    return {
        "event": "infer_done",
        "csv": str(csv_path),
        "conditioning_type": conditioning_type,
        "image": str(image_path),
        "output": str(output),
        "lora_path": resolved_lora_path,
        "frames": len(frames),
        "conditioning_frames": conditioning_frames,
        "num_frames": resolved_num_frames,
        "fps": resolved_fps,
    }


def infer_plan_kwargs(
    *,
    csv_path: Path,
    output: Path,
    model_path: str,
    lora_path: str | None,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    seed: int,
    device: str,
    dtype: str,
    warp_debug_dir: str | Path | None,
    enable_optional_attention: bool,
) -> dict[str, Any]:
    return {
        "csv_path": str(csv_path),
        "output": str(output),
        "model_path": model_path,
        "lora_path": lora_path,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "seed": int(seed),
        "device": device,
        "dtype": dtype,
        "no_lora": lora_path is None,
        "warp_debug_dir": None if warp_debug_dir is None else str(Path(warp_debug_dir).expanduser().resolve()),
        "enable_optional_attention": enable_optional_attention,
    }
