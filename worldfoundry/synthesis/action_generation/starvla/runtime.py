"""Flat, checkpoint-backed StarVLA inference runtime."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.action_normalization import (
    select_modality_statistics,
    unnormalize_action_values,
)
from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import (
    project_root,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.core.io.formats.serialization import jsonable

RUNTIME_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StarVLARuntimeConfig:
    checkpoint_dir: Path
    base_vlm: str
    action_model_type: str
    action_dim: int
    action_horizon: int
    device: str
    torch_dtype: str
    track: str
    attn_implementation: str = "auto"
    enable_official_runtime: bool = True
    base_world_model: str | None = None
    unnorm_key: str | None = None
    action_normalization: str = "min_max"

    def __post_init__(self) -> None:
        if self.action_dim < 1 or self.action_horizon < 1:
            raise ValueError("StarVLA action_dim and action_horizon must be positive.")


def _expand_path(value: str | Path) -> Path:
    path = resolve_worldfoundry_path(value)
    return path if path.is_absolute() else project_root() / path


def _path_or_model_id(value: str | Path, *, required_files: tuple[str, ...] = ()) -> str:
    text = str(value)
    try:
        return str(resolve_local_hf_model_path(text, required_files=required_files))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"StarVLA asset {text!r} is not staged locally; use hfd under "
            "WORLDFOUNDRY_HFD_ROOT or pass a complete local directory."
        ) from exc


def _select_checkpoint_by_needles(
    checkpoints: tuple[Mapping[str, Any], ...], needles: tuple[str, ...]
) -> Mapping[str, Any] | None:
    for item in checkpoints:
        blob = " ".join(str(item.get(key) or "") for key in ("repo_id", "local_dir", "role", "variant_id", "id")).lower()
        if any(needle in blob for needle in needles):
            return item
    return None


def _resolve_checkpoint_record(record: Mapping[str, Any]) -> Path:
    errors: list[str] = []
    for field in ("local_dir", "repo_id"):
        value = record.get(field)
        if not value:
            continue
        try:
            return resolve_local_hf_model_path(
                str(value), required_files=("config.yaml", "dataset_statistics.json")
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))
    detail = "\n".join(errors)
    raise FileNotFoundError(f"StarVLA checkpoint assets are not staged locally.\n{detail}")


def select_starvla_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
    variant_id: str | None = None,
    track: str | None = None,
) -> Path:
    candidate: str | Path | None = checkpoint_dir
    selector = f"{variant_id or ''} {track or ''}".lower()
    selected: Mapping[str, Any] | None = None
    if candidate is None and checkpoints:
        if any(key in selector for key in ("wm4a", "wan2", "world_action", "wam")):
            selected = _select_checkpoint_by_needles(checkpoints, ("wm4a", "wan2", "world_action"))
        elif any(key in selector for key in ("qwen", "vla")):
            selected = _select_checkpoint_by_needles(checkpoints, ("qwen3", "policy"))
        selected = selected or checkpoints[0]
        return _resolve_checkpoint_record(selected)
    if candidate is None:
        raise ValueError("StarVLA requires a checkpoint directory.")
    return resolve_local_hf_model_path(
        candidate, required_files=("config.yaml", "dataset_statistics.json")
    )


def select_starvla_base_vlm(
    *, base_vlm: str | Path | None, checkpoints: tuple[Mapping[str, Any], ...]
) -> str:
    if base_vlm:
        return _path_or_model_id(base_vlm, required_files=("config.json",))
    record = _select_checkpoint_by_needles(checkpoints, ("base_vlm", "base_vl", "qwen--qwen3"))
    if record is None:
        raise ValueError("StarVLA requires a base_vlm path or profile checkpoint record.")
    for field in ("local_dir", "repo_id"):
        if record.get(field):
            resolved = _path_or_model_id(record[field], required_files=("config.json",))
            if Path(resolved).is_dir():
                return resolved
    raise FileNotFoundError("StarVLA base VLM is not staged locally.")


def select_starvla_base_world_model(
    *, base_world_model: str | Path | None, checkpoints: tuple[Mapping[str, Any], ...]
) -> str | None:
    """Resolve the optional Wan base strictly from local hfd assets."""

    if base_world_model:
        return _path_or_model_id(base_world_model, required_files=("model_index.json",))
    record = _select_checkpoint_by_needles(
        checkpoints,
        ("base_world_model", "wan2.2-ti2v-5b-diffusers", "wan2d2_base"),
    )
    if record is None:
        return None
    for field in ("local_dir", "repo_id"):
        if record.get(field):
            try:
                return _path_or_model_id(
                    record[field], required_files=("model_index.json",)
                )
            except FileNotFoundError:
                continue
    return None


def _checkpoint_weight_file(checkpoint_dir: Path) -> Path:
    candidates = sorted((checkpoint_dir / "checkpoints").glob("*.pt"))
    candidates.extend(sorted((checkpoint_dir / "checkpoints").glob("*.safetensors")))
    if not candidates:
        raise FileNotFoundError(f"StarVLA policy weights were not found in {checkpoint_dir / 'checkpoints'}.")
    def checkpoint_rank(path: Path) -> tuple[int, str]:
        matches = re.findall(r"(?:step|steps)[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
        return (int(matches[-1]) if matches else -1, path.name)

    return max(candidates, key=checkpoint_rank)


def _load_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "config.yaml"
    if not path.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_starvla_plan_payload(
    *,
    config: StarVLARuntimeConfig,
    context: Mapping[str, Any],
    profile: Mapping[str, Any],
    runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_config = _load_checkpoint_config(config.checkpoint_dir)
    framework = checkpoint_config.get("framework") if isinstance(checkpoint_config.get("framework"), Mapping) else {}
    action_model = framework.get("action_model") if isinstance(framework.get("action_model"), Mapping) else {}
    framework_name = str(framework.get("name") or "QwenOFT")
    declared_action_model_type = str(action_model.get("action_model_type") or config.action_model_type)
    effective_action_model_type = (
        "MLP" if framework_name in {"QwenOFT", "WanOFT"} else declared_action_model_type
    )
    return {
        "schema_version": "worldfoundry-runtime-profile-starvla-plan",
        "profile": profile,
        "context": dict(context),
        "runtime": {
            "backend": "worldfoundry.starvla.in_tree_checkpoint_runtime",
            "backend_quality": "checkpoint_backed",
            "runtime_package": "worldfoundry.synthesis.action_generation.starvla.runtime",
            "runtime_root": str(RUNTIME_ROOT),
            "checkpoint_dir": str(config.checkpoint_dir),
            "checkpoint_file": str(_checkpoint_weight_file(config.checkpoint_dir)),
            "dataset_statistics": str(config.checkpoint_dir / "dataset_statistics.json"),
            "config_yaml": str(config.checkpoint_dir / "config.yaml"),
            "base_vlm": config.base_vlm,
            "base_world_model": config.base_world_model or "",
            "attention_implementation": config.attn_implementation,
            "checkpoint_runtime_enabled": config.enable_official_runtime,
            "unnorm_key": config.unnorm_key or "auto_single_key",
            "action_normalization": config.action_normalization,
            "framework_name": framework_name,
            "action_model_type": effective_action_model_type,
            "checkpoint_declared_action_model_type": declared_action_model_type,
            "action_dim": int(action_model.get("action_dim") or config.action_dim),
            "action_horizon": int(action_model.get("action_horizon") or config.action_horizon),
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "track": config.track,
        },
        "inference": {
            "instruction": str(runtime_options.get("instruction") or context.get("prompt") or ""),
            "action_space": jsonable(runtime_options.get("action_space")),
            "policy_controls": jsonable(runtime_options.get("policy_controls")),
        },
        "limitations": [
            "LIBERO simulator scoring is a separate benchmark run.",
            "WanOFT requires a Diffusers-format Wan2.2 TI2V base checkpoint.",
        ],
    }


class StarVLAPlanRuntime:
    def __init__(self, config: StarVLARuntimeConfig) -> None:
        self.config = config
        self._loaded: tuple[Any, Any] | None = None

    def write_plan(
        self,
        *,
        context: Mapping[str, Any],
        profile: Mapping[str, Any],
        runtime_options: Mapping[str, Any],
        plan_path: str | Path,
    ) -> dict[str, Any]:
        payload = build_starvla_plan_payload(
            config=self.config,
            context=context,
            profile=profile,
            runtime_options=runtime_options,
        )
        target = Path(plan_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def predict_action(
        self,
        *,
        prompt: str,
        images: Any,
        output_path: str | Path,
        state: Any = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.config.enable_official_runtime:
            raise RuntimeError("StarVLA checkpoint inference is disabled; set enable_official_runtime=True.")
        model, torch = self._load_model()
        import numpy as np

        started = time.monotonic()
        example: dict[str, Any] = {"image": _coerce_images(images), "lang": prompt}
        if state is not None:
            example["state"] = state
        with torch.inference_mode():
            prediction = model.predict_action([example])
        normalized_actions = np.asarray(prediction["normalized_actions"], dtype=np.float32)
        selected_unnorm_key, action_statistics = select_modality_statistics(
            model.norm_stats,
            modality="action",
            key=self.config.unnorm_key,
        )
        actions = unnormalize_action_values(
            normalized_actions,
            action_statistics,
            mode=self.config.action_normalization,
        ).astype(np.float32, copy=False)
        if not np.isfinite(actions).all():
            raise FloatingPointError("StarVLA produced non-finite actions.")

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        attention = getattr(getattr(model, "qwen_vl_interface", None), "attention_implementation", "")
        payload = {
            "schema_version": "worldfoundry-starvla-action-trace",
            "status": "success",
            "model_id": "starvla",
            "backend": "worldfoundry.starvla.in_tree_predict_action",
            "backend_quality": "checkpoint_backed",
            "artifact_kind": "action_trace",
            "instruction": prompt,
            "action_shape": list(actions.shape),
            "actions": actions.tolist(),
            "normalized_actions": normalized_actions.tolist(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": jsonable({
                "checkpoint_dir": self.config.checkpoint_dir,
                "checkpoint_file": _checkpoint_weight_file(self.config.checkpoint_dir),
                "base_vlm": self.config.base_vlm,
                "base_world_model": self.config.base_world_model,
                "attention_implementation": attention,
                "unnorm_key": selected_unnorm_key,
                "action_normalization": self.config.action_normalization,
                "track": self.config.track,
                **dict(extra_metadata or {}),
            }),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "model_id": "starvla",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "action_shape": payload["action_shape"],
            "duration_seconds": payload["duration_seconds"],
        }

    def _load_model(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        import torch

        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .modeling.base import load_starvla_model

        device = resolve_inference_device(self.config.device, allow_cpu_fallback=True)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        model = load_starvla_model(
            _checkpoint_weight_file(self.config.checkpoint_dir),
            base_vlm=self.config.base_vlm,
            base_world_model=self.config.base_world_model,
            attn_implementation=self.config.attn_implementation,
            device=device,
            torch_dtype=str(dtype).removeprefix("torch."),
        )
        model.requires_grad_(False).to(device=device, dtype=dtype).eval()
        self._loaded = (model, torch)
        return self._loaded


def _coerce_images(images: Any) -> list[Any]:
    if images is None:
        raise ValueError("StarVLA predict_action requires at least one RGB image.")
    if isinstance(images, Mapping):
        preferred = ("full_image", "image", "rgb", "agentview_image")
        keys = [key for key in preferred if key in images]
        keys.extend(sorted(key for key in images if key not in keys and "wrist" in str(key).lower()))
        keys.extend(sorted(key for key in images if key not in keys))
        candidates = [images[key] for key in keys]
    elif isinstance(images, (list, tuple)):
        candidates = list(images)
    else:
        candidates = [images]
    return [_coerce_image(item) for item in candidates]


def _coerce_image(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        text = str(image)
        if text.startswith("memory://"):
            raise ValueError(f"StarVLA cannot load placeholder image: {text}")
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"StarVLA image does not exist: {path}")
        return Image.open(path).convert("RGB")
    from .modeling.images import to_pil_preserve

    return to_pil_preserve(image)


__all__ = [
    "StarVLAPlanRuntime",
    "StarVLARuntimeConfig",
    "build_starvla_plan_payload",
    "select_starvla_base_vlm",
    "select_starvla_base_world_model",
    "select_starvla_checkpoint",
]
