from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path

from .compat import install_jax_compatibility


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _worldfoundry_repository_root() -> Path:
    return project_root()


def _expand_checkpoint_path(value: str | Path) -> Path:
    repo_root = _worldfoundry_repository_root()
    path = resolve_worldfoundry_path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def select_octo_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    variant: str | None,
    require_exists: bool = True,
) -> Path:
    """Select a local Octo checkpoint directory from explicit input or profile metadata."""
    candidates: list[Mapping[str, Any]] = []
    if checkpoint_dir:
        candidates.append({"local_dir": str(checkpoint_dir), "role": "explicit_checkpoint"})
    candidates.extend(dict(item) for item in checkpoints)

    requested = str(variant or "").lower()
    if requested:
        for item in candidates:
            role = str(item.get("role") or "").lower()
            local_dir = str(item.get("local_dir") or "").lower()
            if requested in role or requested in local_dir:
                path = _expand_checkpoint_path(str(item.get("local_dir") or ""))
                if not require_exists or path.exists():
                    return path

    for item in candidates:
        path_text = str(item.get("local_dir") or "")
        if not path_text:
            continue
        path = _expand_checkpoint_path(path_text)
        if not require_exists or path.exists():
            return path
    raise FileNotFoundError("No local Octo checkpoint was found.")


def _load_image_array(image: Any, *, size: tuple[int, int]) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image

    if image is None:
        raise ValueError("Octo runtime requires an RGB observation image.")
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB").resize(size)), "in-memory:PIL.Image"
    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Octo image path does not exist: {image_path}")
        return np.array(Image.open(image_path).convert("RGB").resize(size)), str(image_path)
    if isinstance(image, Sequence) and not isinstance(image, (bytes, bytearray, str)):
        if not image:
            raise ValueError("Octo received an empty image sequence.")
        return _load_image_array(image[0], size=size)

    array = np.asarray(image)
    if array.ndim == 5:
        array = array[0, -1]
    elif array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Octo image array must be HxWxC, got shape {array.shape}.")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    rgb = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB").resize(size)
    return np.array(rgb), "in-memory:array"


@dataclass(frozen=True)
class OctoRuntimeConfig:
    checkpoint_dir: Path
    variant: str = "small"
    dataset_key: str = "bridge_dataset"
    image_key: str = "image_primary"
    image_size: tuple[int, int] = (256, 256)
    jax_platform: str = "cpu"
    seed: int = 0


class OctoRuntime:
    """Direct in-tree Octo inference runtime."""

    def __init__(self, config: OctoRuntimeConfig) -> None:
        self.config = config
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        platform = self.config.jax_platform.lower()
        if platform == "gpu":
            platform = "cuda"
        if platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError(f"Unsupported Octo JAX platform: {self.config.jax_platform!r}")
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)
        install_jax_compatibility()
        from .modeling.model import OctoModel

        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Octo checkpoint directory does not exist: {checkpoint}")
        self.model = OctoModel.load_pretrained(str(checkpoint))

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not instruction:
            raise ValueError("Octo runtime requires a non-empty instruction prompt.")
        self.load()
        assert self.model is not None

        import numpy as np

        jax = importlib.import_module("jax")
        started = time.monotonic()
        image_array, image_source = _load_image_array(image, size=self.config.image_size)
        observation = {
            self.config.image_key: image_array[np.newaxis, np.newaxis, ...],
            "timestep_pad_mask": np.array([[True]]),
        }
        task = self.model.create_tasks(texts=[instruction])
        unnormalization_statistics = self.model.dataset_statistics[self.config.dataset_key]["action"]
        action = self.model.sample_actions(
            observation,
            task,
            unnormalization_statistics=unnormalization_statistics,
            rng=jax.random.PRNGKey(self.config.seed),
        )

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-octo-runtime",
            "status": "success",
            "model_id": "octo",
            "backend": "worldfoundry.octo.in_tree_runtime.sample_actions",
            "backend_quality": "in_tree_runtime",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(self.config.checkpoint_dir.expanduser().resolve()),
            "dataset_key": self.config.dataset_key,
            "image_key": self.config.image_key,
            "image_source": image_source,
            "instruction": instruction,
            "jax_platform": self.config.jax_platform,
            "seed": self.config.seed,
            "action": _jsonable(action),
            "actions": _jsonable(action),
            "metadata": _jsonable(dict(extra_metadata or {})),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_hash = file_sha256(target)
        return {
            "status": "success",
            "model_id": "octo",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_hash,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
