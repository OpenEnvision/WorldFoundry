"""Checkpoint-backed inference for the released Spatial-Forcing policies."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from worldfoundry.core.media.types import MediaKind, infer_media_kind
from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config


@dataclass(frozen=True)
class OfficialCheckpoint:
    """Immutable metadata for an official Spatial-Forcing Hub repository."""

    repo_id: str
    revision: str
    task_suite_name: str
    unnorm_key: str
    license: str
    action_head_file: str
    proprio_projector_file: str


_DATA_CONFIG = load_vla_va_wam_runtime_config("spatial-forcing")
_CHECKPOINT_CONFIGS = _DATA_CONFIG.get("checkpoints")
if not isinstance(_CHECKPOINT_CONFIGS, Mapping):
    raise TypeError("spatial-forcing data config requires a checkpoints mapping")
OFFICIAL_CHECKPOINTS: dict[str, OfficialCheckpoint] = {
    str(name): OfficialCheckpoint(
        repo_id=str(payload["repo_id"]),
        revision=str(payload["revision"]),
        task_suite_name=str(payload["task_suite_name"]),
        unnorm_key=str(payload["unnorm_key"]),
        license=str(payload["license"]),
        action_head_file=str(payload["action_head_file"]),
        proprio_projector_file=str(payload["proprio_projector_file"]),
    )
    for name, payload in _CHECKPOINT_CONFIGS.items()
    if isinstance(payload, Mapping)
}
OFFICIAL_CHECKPOINTS_BY_REPO = {item.repo_id: item for item in OFFICIAL_CHECKPOINTS.values()}
DEFAULT_CHECKPOINT = OFFICIAL_CHECKPOINTS[str(_DATA_CONFIG["default_checkpoint"])]

_REQUIRED_SNAPSHOT_FILES = tuple(str(item) for item in _DATA_CONFIG["required_snapshot_files"])
_INFERENCE_ALLOW_PATTERNS = tuple(str(item) for item in _DATA_CONFIG["inference_allow_patterns"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        try:
            return _jsonable(value.detach().cpu())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _checkpoint_spec(value: str | Path) -> OfficialCheckpoint | None:
    text = str(value)
    if text in OFFICIAL_CHECKPOINTS_BY_REPO:
        return OFFICIAL_CHECKPOINTS_BY_REPO[text]
    normalized = text.replace("--", "/")
    for spec in OFFICIAL_CHECKPOINTS.values():
        if spec.repo_id in normalized or spec.repo_id.rsplit("/", 1)[-1] in text:
            return spec
    return None


def _resolve_checkpoint(config: SpatialForcingRuntimeConfig) -> tuple[Path, OfficialCheckpoint | None, str]:
    location = config.checkpoint_location
    spec = _checkpoint_spec(location)
    revision = str(config.revision or (spec.revision if spec is not None else ""))
    try:
        local_dir = resolve_local_hf_model_path(location, required_files=_REQUIRED_SNAPSHOT_FILES)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Spatial-Forcing requires a complete checkpoint staged in local WorldFoundry/HF storage; "
            f"no runtime download is permitted for {location!r}"
        ) from error
    return local_dir, spec, revision


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_component_state_dict(path: Path) -> dict[str, Any]:
    import torch

    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Spatial-Forcing component checkpoint is not a state dict: {path}")
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError(
            f"Spatial-Forcing component checkpoint must be a non-empty string-to-tensor mapping: {path}"
        )
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _component_file(checkpoint_dir: Path, spec: OfficialCheckpoint | None, kind: str) -> Path:
    if kind not in {"action_head", "proprio_projector"}:
        raise ValueError(f"Unknown Spatial-Forcing component kind: {kind}")
    filename = getattr(spec, f"{kind}_file") if spec is not None else f"{kind}--latest_checkpoint.pt"
    candidate = checkpoint_dir / filename
    if candidate.is_file():
        return candidate
    matches = sorted(checkpoint_dir.glob(f"*{kind}*checkpoint*.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Spatial-Forcing {kind} checkpoint was not found in {checkpoint_dir}")
    raise FileExistsError(f"Spatial-Forcing found multiple {kind} checkpoints in {checkpoint_dir}: {matches}")


def _to_uint8_array(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Spatial-Forcing image path does not exist: {image_path}")
        if infer_media_kind(image_path) is not MediaKind.IMAGE:
            raise ValueError(f"Spatial-Forcing expected an image file, got: {image_path}")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    else:
        try:
            import torch

            if isinstance(image, torch.Tensor):
                tensor = image.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
            else:
                array = np.asarray(image)
        except ImportError:
            array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Spatial-Forcing image must be HxWxC or CxHxW, got shape {array.shape}")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Spatial-Forcing image must have 3 channels, got shape {array.shape}")
    if array.dtype.kind == "f":
        max_value = 1.0 if float(np.nanmax(array)) <= 1.0 else 255.0
        array = np.clip(array, 0.0, max_value) / max_value * 255.0
    return np.asarray(array, dtype=np.uint8)


def _prepare_image(image: Any, *, center_crop: bool) -> Any:
    from PIL import Image

    image_size = 224
    pil = Image.fromarray(_to_uint8_array(image)).convert("RGB")
    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if center_crop:
        crop_side = int(round(image_size * math.sqrt(0.9)))
        left = (image_size - crop_side) // 2
        top = (image_size - crop_side) // 2
        pil = pil.crop((left, top, left + crop_side, top + crop_side))
        pil = pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return pil


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value)


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if _present(value):
            return value
    return None


def _select_images(image: Any, observation: Mapping[str, Any], *, count: int) -> list[Any]:
    image_mapping = image if isinstance(image, Mapping) else {}
    primary = _first_present(observation, "full_image", "image", "rgb", "ref_image_path")
    if primary is None:
        primary = _first_present(image_mapping, "full_image", "image", "rgb")
    if primary is None and not isinstance(image, Mapping) and _present(image):
        primary = image
    if primary is None:
        raise ValueError("Spatial-Forcing requires a full_image/agent-view image")

    images = [primary]
    if count == 1:
        return images
    for mapping in (observation, image_mapping):
        keys = ["wrist_image", "full_image_wrist", "image_wrist", "wrist"]
        keys.extend(sorted(str(key) for key in mapping if "wrist" in str(key) and str(key) not in keys))
        for key in keys:
            value = mapping.get(key)
            if _present(value) and all(value is not previous for previous in images):
                images.append(value)
            if len(images) >= count:
                return images[:count]
    raise ValueError(f"Spatial-Forcing checkpoint expects {count} images, but only {len(images)} were provided")


def _resolve_unnorm_key(
    norm_stats: Mapping[str, Any],
    requested: str | None,
    task_suite_name: str | None,
) -> str:
    candidates = [requested, task_suite_name, f"{task_suite_name}_no_noops" if task_suite_name else None]
    for candidate in candidates:
        if candidate and candidate in norm_stats:
            return str(candidate)
    for key in norm_stats:
        if task_suite_name and str(key).startswith(task_suite_name):
            return str(key)
    if len(norm_stats) == 1:
        return str(next(iter(norm_stats)))
    raise KeyError(f"Spatial-Forcing unnorm_key was not found; requested={requested!r}, available={list(norm_stats)}")


def _normalize_proprio(proprio: Any, norm_stats: Mapping[str, Any]) -> Any:
    import numpy as np

    from .modeling import ACTION_PROPRIO_NORMALIZATION_TYPE, PROPRIO_DIM, NormalizationType

    array = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if array.shape[0] != PROPRIO_DIM:
        raise ValueError(f"Spatial-Forcing LIBERO state must have {PROPRIO_DIM} values, got {array.shape}")
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low = np.asarray(norm_stats["q01"], dtype=np.float32)
        high = np.asarray(norm_stats["q99"], dtype=np.float32)
        mask = np.asarray(norm_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        low = np.asarray(norm_stats["min"], dtype=np.float32)
        high = np.asarray(norm_stats["max"], dtype=np.float32)
        mask = np.asarray(norm_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    else:
        raise ValueError(f"Unsupported Spatial-Forcing proprio normalization: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    normalized = np.where(mask, 2 * (array - low) / (high - low + 1e-8) - 1, array)
    return np.clip(normalized, -1.0, 1.0)


@dataclass(frozen=True)
class SpatialForcingRuntimeConfig:
    """Options for one released Spatial-Forcing OpenVLA policy."""

    checkpoint_location: str
    revision: str | None = None
    device: str = "cuda"
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = True
    attn_implementation: str = "auto"
    unnorm_key: str | None = None
    task_suite_name: str | None = None
    num_images_in_input: int = 2
    center_crop: bool = True

    def __post_init__(self) -> None:
        if not self.local_files_only:
            raise ValueError("Spatial-Forcing runtime is local-only")
        if self.num_images_in_input < 1:
            raise ValueError("Spatial-Forcing num_images_in_input must be at least one")


class SpatialForcingRuntime:
    """Inference-only runtime for official Spatial-Forcing LIBERO checkpoints."""

    def __init__(self, config: SpatialForcingRuntimeConfig) -> None:
        self.config = config
        self.checkpoint_dir: Path | None = None
        self.checkpoint_spec: OfficialCheckpoint | None = None
        self.revision = ""
        self.device = ""
        self.dtype: Any | None = None
        self.dtype_name = ""
        self.attn_implementation = ""
        self.processor: Any | None = None
        self.model: Any | None = None
        self.action_head: Any | None = None
        self.proprio_projector: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        from transformers import LlamaTokenizerFast

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling import (
            ACTION_DIM,
            L1RegressionActionHead,
            OpenVLAConfig,
            OpenVLAForActionPrediction,
            PrismaticImageProcessor,
            PrismaticProcessor,
            ProprioProjector,
        )

        checkpoint_dir, spec, revision = _resolve_checkpoint(self.config)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_spec = spec
        self.revision = revision
        self.device = resolve_inference_device(self.config.device)
        self.dtype = resolve_inference_dtype(self.device, self.config.torch_dtype)
        self.dtype_name = str(self.dtype).removeprefix("torch.")

        image_processor = PrismaticImageProcessor.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        self.processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

        model_config = OpenVLAConfig.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        self.attn_implementation = resolve_transformers_attention_implementation(
            self.config.attn_implementation,
            self.device,
        )
        model_kwargs["attn_implementation"] = self.attn_implementation

        def load_main_model() -> Any:
            loaded, loading_info = OpenVLAForActionPrediction.from_pretrained(
                str(checkpoint_dir),
                output_loading_info=True,
                **model_kwargs,
            )
            problems = {
                key: loading_info.get(key, [])
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key, [])
            }
            if problems:
                raise RuntimeError(f"Spatial-Forcing checkpoint is not architecture-compatible: {problems}")
            return loaded

        try:
            model = load_main_model()
        except (ImportError, TypeError, ValueError):
            if self.attn_implementation == "eager":
                raise
            fallback = "sdpa" if self.attn_implementation == "flash_attention_2" else "eager"
            model_kwargs["attn_implementation"] = fallback
            try:
                model = load_main_model()
            except TypeError:
                model_kwargs.pop("attn_implementation", None)
                model = load_main_model()
                fallback = "eager"
            self.attn_implementation = fallback

        stats_path = checkpoint_dir / "dataset_statistics.json"
        model.norm_stats = _load_json(stats_path)
        model.vision_backbone.set_num_images_in_input(self.config.num_images_in_input)
        model = model.to(device=self.device, dtype=self.dtype).eval()

        action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=ACTION_DIM)
        action_head.load_state_dict(_load_component_state_dict(_component_file(checkpoint_dir, spec, "action_head")))
        action_head = action_head.to(device=self.device, dtype=self.dtype).eval()

        proprio_projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
        proprio_projector.load_state_dict(
            _load_component_state_dict(_component_file(checkpoint_dir, spec, "proprio_projector"))
        )
        proprio_projector = proprio_projector.to(device=self.device, dtype=self.dtype).eval()

        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not instruction:
            raise ValueError("Spatial-Forcing requires a non-empty language instruction")
        self.load()
        assert self.model is not None
        assert self.processor is not None

        import torch

        started = time.monotonic()
        spec = self.checkpoint_spec
        task_suite_name = self.config.task_suite_name or (spec.task_suite_name if spec is not None else None)
        requested_unnorm_key = self.config.unnorm_key or (spec.unnorm_key if spec is not None else None)
        unnorm_key = _resolve_unnorm_key(self.model.norm_stats, requested_unnorm_key, task_suite_name)
        images = [
            _prepare_image(item, center_crop=self.config.center_crop)
            for item in _select_images(image, observation, count=self.config.num_images_in_input)
        ]

        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, images[0]).to(self.device, dtype=self.dtype)
        if len(images) > 1:
            wrist_inputs = [self.processor(prompt, item).to(self.device, dtype=self.dtype) for item in images[1:]]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], *[item["pixel_values"] for item in wrist_inputs]],
                dim=1,
            )

        state = _first_present(observation, "state", "proprio", "robot_state")
        if state is None:
            raise ValueError("Spatial-Forcing requires observation.state with 8 LIBERO proprio values")
        proprio = _normalize_proprio(state, self.model.norm_stats[unnorm_key]["proprio"])

        with torch.inference_mode():
            actions, _ = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                noisy_action_projector=None,
                use_film=False,
            )

        return {
            "status": "completed",
            "actions": _jsonable(actions),
            "model_id": "spatial-forcing",
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_ref": spec.repo_id if spec is not None else "",
            "checkpoint_revision": self.revision,
            "runtime": "worldfoundry.spatial_forcing.in_tree_runtime",
            "backend_quality": "checkpoint_backed",
            "official_prompt": prompt,
            "task_suite_name": task_suite_name,
            "unnorm_key": unnorm_key,
            "num_images_in_input": self.config.num_images_in_input,
            "torch_dtype": self.dtype_name,
            "attention_implementation": self.attn_implementation,
            "device": self.device,
            "duration_seconds": round(time.monotonic() - started, 3),
        }


__all__ = [
    "DEFAULT_CHECKPOINT",
    "OFFICIAL_CHECKPOINTS",
    "OfficialCheckpoint",
    "SpatialForcingRuntime",
    "SpatialForcingRuntimeConfig",
]
