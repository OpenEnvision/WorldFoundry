# Inference-only OpenVLA source retained in-tree.
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        try:
            return _jsonable(value.numpy())
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_runtime_path(value: str | Path) -> Path:
    path = resolve_worldfoundry_path(value)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def select_openvla_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    unnorm_key: str | None,
) -> Path:
    if checkpoint_dir:
        explicit_path = _expand_runtime_path(checkpoint_dir)
        if not explicit_path.is_dir():
            raise FileNotFoundError(f"OpenVLA checkpoint directory does not exist: {explicit_path}")
        return explicit_path

    candidates: list[Mapping[str, Any]] = []
    candidates.extend(dict(item) for item in checkpoints)

    if unnorm_key and unnorm_key.lower().startswith("libero"):
        for item in candidates:
            role = str(item.get("role") or "").lower()
            local_dir = str(item.get("local_dir") or "")
            if "libero" in role or "libero" in local_dir.lower():
                path = _expand_runtime_path(local_dir)
                if path.exists():
                    return path

    for item in candidates:
        path_text = str(item.get("local_dir") or "")
        if not path_text:
            continue
        path = _expand_runtime_path(path_text)
        if path.exists():
            return path
    raise FileNotFoundError("No local OpenVLA checkpoint was found.")


def openvla_prompt(instruction: str, checkpoint_dir: Path) -> str:
    if "v01" in str(checkpoint_dir):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def _resolve_device(torch: Any, requested: str) -> str:
    device = requested or "cuda"
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("OpenVLA runtime requested CUDA, but torch.cuda.is_available() is false.")
    return device


def _resolve_dtype(torch: Any, *, device: str, requested: str | None) -> tuple[Any, str, Any, str]:
    dtype_name = str(requested or "auto").lower()
    if dtype_name == "auto":
        if device.startswith("cuda") and int(torch.__version__.split(".", 1)[0]) >= 2:
            dtype = torch.bfloat16
            dtype_name = "bfloat16"
        elif device.startswith("cuda"):
            dtype = torch.float16
            dtype_name = "float16"
        else:
            dtype = torch.float32
            dtype_name = "float32"
    elif dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
        dtype_name = "bfloat16"
    elif dtype_name in {"fp16", "float16", "half"}:
        dtype = torch.float16
        dtype_name = "float16"
    elif dtype_name in {"fp32", "float32"}:
        dtype = torch.float32
        dtype_name = "float32"
    else:
        raise ValueError(f"Unsupported OpenVLA torch_dtype value: {requested}")

    load_dtype = torch.bfloat16 if device.startswith("cuda") and dtype == torch.float16 else dtype
    load_dtype_name = "bfloat16" if load_dtype == torch.bfloat16 else dtype_name
    return dtype, dtype_name, load_dtype, load_dtype_name


def _load_image(image: Any):
    from PIL import Image

    if image is None:
        raise ValueError("OpenVLA runtime requires an RGB observation image.")
    if isinstance(image, Image.Image):
        return image.convert("RGB"), "in-memory:PIL.Image"
    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenVLA image path does not exist: {image_path}")
        return Image.open(image_path).convert("RGB"), str(image_path)
    if isinstance(image, Sequence) and not isinstance(image, (bytes, bytearray, str)):
        if not image:
            raise ValueError("OpenVLA received an empty image sequence.")
        return _load_image(image[0])

    try:
        import numpy as np
        import torch

        if isinstance(image, torch.Tensor):
            array = image.detach().cpu()
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                array = array.permute(1, 2, 0)
            array = array.numpy()
        else:
            array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"OpenVLA image array must be HxWxC or CxHxW, got shape {array.shape}.")
        if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype.kind == "f":
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.asarray(array, dtype=np.uint8)
        return Image.fromarray(array).convert("RGB"), "in-memory:array"
    except ImportError as exc:
        raise TypeError("OpenVLA image input must be a path, PIL image, tensor, or RGB array.") from exc


@dataclass(frozen=True)
class OpenVLARuntimeConfig:
    checkpoint_dir: Path | None
    unnorm_key: str = "bridge_orig"
    device: str = "cuda"
    torch_dtype: str = "auto"
    attn_implementation: str = "eager"
    use_cache: str | bool | None = None


class OpenVLARuntime:
    """In-tree OpenVLA inference runtime using vendored HF architecture code."""

    def __init__(self, config: OpenVLARuntimeConfig) -> None:
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device: str = ""
        self.dtype: Any | None = None
        self.dtype_name = ""
        self.load_dtype_name = ""

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoTokenizer

        from .configuration import OpenVLAConfig
        from .modeling import OpenVLAForActionPrediction
        from .preprocessing import PrismaticImageProcessor, PrismaticProcessor

        if self.config.checkpoint_dir is None:
            raise FileNotFoundError("No local OpenVLA checkpoint was found.")
        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"OpenVLA checkpoint directory does not exist: {checkpoint}")

        self.device = _resolve_device(torch, self.config.device)
        self.dtype, self.dtype_name, load_dtype, self.load_dtype_name = _resolve_dtype(
            torch,
            device=self.device,
            requested=self.config.torch_dtype,
        )

        image_processor = PrismaticImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, use_fast=True)
        self.processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

        model_config = OpenVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": load_dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        try:
            model = OpenVLAForActionPrediction.from_pretrained(str(checkpoint), **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = OpenVLAForActionPrediction.from_pretrained(str(checkpoint), **model_kwargs)

        stats_path = checkpoint / "dataset_statistics.json"
        if stats_path.is_file():
            model.norm_stats = _load_json(stats_path)

        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not instruction:
            raise ValueError("OpenVLA runtime requires a non-empty instruction prompt.")
        self.load()
        assert self.model is not None
        assert self.processor is not None

        import torch

        started = time.monotonic()
        rgb_image, image_source = _load_image(image)
        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        official_prompt = openvla_prompt(instruction, checkpoint)
        inputs = self.processor(official_prompt, rgb_image).to(self.device, dtype=self.dtype)
        if (
            "input_ids" in inputs
            and "attention_mask" in inputs
            and not torch.all(inputs["input_ids"][:, -1] == 29871)
        ):
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(inputs["attention_mask"][:, :1])],
                dim=1,
            )

        generate_kwargs: dict[str, Any] = {"do_sample": False}
        if self.config.use_cache in {"true", "false"}:
            generate_kwargs["use_cache"] = self.config.use_cache == "true"
        elif isinstance(self.config.use_cache, bool):
            generate_kwargs["use_cache"] = self.config.use_cache

        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.config.unnorm_key,
                **generate_kwargs,
            )

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-worldfoundry-openvla-runtime",
            "status": "success",
            "model_id": "openvla",
            "backend": "worldfoundry.openvla.in_tree_runtime.predict_action",
            "backend_quality": "in_tree_runtime",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(checkpoint),
            "image_source": image_source,
            "instruction": instruction,
            "official_prompt": official_prompt,
            "unnorm_key": self.config.unnorm_key,
            "device": self.device,
            "torch_dtype": self.dtype_name,
            "load_torch_dtype": self.load_dtype_name,
            "generate_kwargs": generate_kwargs,
            "action": _jsonable(action),
            "actions": [_jsonable(action)],
            "metadata": _jsonable(dict(extra_metadata or {})),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_hash = file_sha256(target)
        return {
            "status": "success",
            "model_id": "openvla",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_hash,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
