from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path

DEFAULT_INFINITE_WORLD_REPO = "MeiGen-AI/Infinite-World"
IN_TREE_BACKEND = "worldfoundry.infinite_world.in_tree_runtime"
REQUIRED_RUNTIME_FILES = (
    "checkpoints/infinite_world_model.ckpt",
    "checkpoints/models/Wan2.1_VAE.pth",
    "checkpoints/models/models_t5_umt5-xxl-enc-bf16.pth",
    "checkpoints/models/google/umt5-xxl/tokenizer.json",
    "checkpoints/models/google/umt5-xxl/spiece.model",
)
REQUIRED_PYTHON_MODULES = (
    "cv2",
    "einops",
    "ftfy",
    "omegaconf",
    "regex",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[6]


def find_default_model_root() -> Path | None:
    """
    Locate an already staged Infinite-World checkpoint tree.

    Args:
        None.
    """
    hub_root = Path(os.environ.get("HF_HUB_CACHE", "")).expanduser() if os.environ.get("HF_HUB_CACHE") else None
    candidates: list[Path] = [
        checkpoint_root_path("Infinite-World"),
        hfd_root_path("MeiGen-AI--Infinite-World"),
        hfd_root_path("custom--Infinite-World"),
        _project_root() / "cache" / "hfd" / "Infinite-World",
        _project_root() / "cache" / "hfd" / "MeiGen-AI--Infinite-World",
    ]
    if hub_root is not None:
        candidates.append(hub_root / "models--MeiGen-AI--Infinite-World")
    candidates.append(checkpoint_root_path("huggingface", "hub", "models--MeiGen-AI--Infinite-World"))

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            expanded.extend(path for path in snapshots.iterdir() if path.is_dir())

    for candidate in expanded:
        if candidate.is_dir() and not missing_runtime_files(candidate):
            return candidate.resolve()
    return None


def missing_runtime_files(model_root: str | Path) -> list[str]:
    """
    Return required checkpoint/tokenizer files missing from model_root.

    Args:
        model_root: Local Infinite-World model root containing the checkpoints directory.
    """
    root = Path(model_root).expanduser()
    return [item for item in REQUIRED_RUNTIME_FILES if not (root / item).is_file()]


def missing_python_modules() -> list[str]:
    """
    Return Python modules required by the Infinite-World inference path that are not importable.

    Args:
        None.
    """
    return [name for name in REQUIRED_PYTHON_MODULES if importlib.util.find_spec(name) is None]


def resolve_in_tree_model_root(pretrained_model_path: Any = None) -> Path:
    """
    Resolve a local Infinite-World model root without network downloads.

    Args:
        pretrained_model_path: Local checkpoint root, mapping with path keys, or None for default cache lookup.
    """
    candidate_value = pretrained_model_path
    if isinstance(pretrained_model_path, Mapping):
        candidate_value = (
            pretrained_model_path.get("pretrained_model_path")
            or pretrained_model_path.get("model_path")
            or pretrained_model_path.get("model_root")
            or pretrained_model_path.get("checkpoint_dir")
        )

    if candidate_value is None or str(candidate_value) in {DEFAULT_INFINITE_WORLD_REPO, "Infinite-World"}:
        default_root = find_default_model_root()
        if default_root is None:
            raise FileNotFoundError(_blocked_message(None))
        candidate = default_root
    else:
        candidate = Path(str(candidate_value)).expanduser()

    if not candidate.is_dir():
        raise FileNotFoundError(_blocked_message(candidate))

    candidate_options = [candidate]
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        candidate_options.extend(path for path in snapshots.iterdir() if path.is_dir())
    fallback_root = find_default_model_root()
    if fallback_root is not None and fallback_root not in candidate_options:
        candidate_options.append(fallback_root)

    for option in candidate_options:
        if option.is_dir() and not missing_runtime_files(option):
            candidate = option
            break

    missing = missing_runtime_files(candidate)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"{_blocked_message(candidate)} Missing required files: {missing_text}.")
    missing_modules = missing_python_modules()
    if missing_modules:
        missing_text = ", ".join(missing_modules)
        raise RuntimeError(f"Infinite-World runtime dependencies are missing: {missing_text}.")
    return candidate.resolve()


def _blocked_message(candidate: Path | None) -> str:
    location = "default cache roots" if candidate is None else str(candidate)
    return (
        "Infinite-World is restricted to in-tree/local assets; network snapshot downloads are disabled. "
        f"Stage the official MeiGen-AI/Infinite-World checkpoint tree under {location} or pass a local "
        "pretrained_model_path containing checkpoints/."
    )


@dataclass(frozen=True)
class InfiniteWorldRuntimePlan:
    model_root: str | None
    config_path: str
    device: str
    backend: str = IN_TREE_BACKEND
    backend_quality: str = "blocked_missing_weights"
    status: str = "blocked"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the Infinite-World runtime plan.

        Args:
            None.
        """
        missing = missing_runtime_files(self.model_root) if self.model_root else list(REQUIRED_RUNTIME_FILES)
        missing_modules = missing_python_modules()
        return {
            "status": self.status,
            "artifact_kind": "generated_video",
            "backend": self.backend,
            "backend_quality": self.backend_quality,
            "model_root": self.model_root,
            "config_path": self.config_path,
            "device": self.device,
            "required_files": list(REQUIRED_RUNTIME_FILES),
            "missing_files": missing,
            "required_python_modules": list(REQUIRED_PYTHON_MODULES),
            "missing_python_modules": missing_modules,
            "note": (
                "Infinite-World runtime code is vendored in-tree. Full generation requires the official "
                "DiT, VAE, T5 encoder, and tokenizer assets to be staged locally; this integration does "
                "not checkout an external repo or download weights at runtime."
            ),
        }
