"""Offline asset resolution for the native FantasyWorld integration."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path


DEFAULT_FANTASY_WORLD_WAN21_REPO = "acvlab/FantasyWorld-Wan2.1-I2V-14B-480P"
DEFAULT_FANTASY_WORLD_WAN22_REPO = "acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera"
DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B"
DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO = "alibaba-pai/Wan2.2-Fun-Reward-LoRAs"
DEFAULT_FANTASY_WORLD_MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"

DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards"
)

WAN22_LORA_HIGH_NAME = "Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors"
WAN22_LORA_LOW_NAME = "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"


def project_root() -> Path:
    current = Path(__file__).resolve()
    return next(
        (parent for parent in current.parents if (parent / "pyproject.toml").is_file()),
        current.parents[4],
    )


def checkpoint_root() -> Path:
    configured = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    return Path(configured).expanduser() if configured else project_root().parent / "ckpt"


def cache_root() -> Path:
    return project_root() / "cache" / "hfd"


def _candidate_paths(source: str | os.PathLike | None) -> Iterable[Path]:
    if source is None:
        return
    direct = Path(source).expanduser()
    if direct.exists():
        yield direct.resolve()
        return
    value = str(source)
    aliases = tuple(dict.fromkeys((value, value.replace("/", "--"), value.rsplit("/", 1)[-1])))
    for base in (checkpoint_root(), checkpoint_root() / "hfd", cache_root()):
        for alias in aliases:
            candidate = base / alias
            if candidate.exists():
                yield candidate.resolve()


def _first_file(
    sources: Sequence[str | os.PathLike | None],
    filenames: Sequence[str],
) -> Path | None:
    for source in sources:
        for candidate in _candidate_paths(source):
            if candidate.is_file() and candidate.name in filenames:
                return candidate
            for filename in filenames:
                path = candidate / filename
                if path.is_file():
                    return path.resolve()
    return None


def _first_directory(
    sources: Sequence[str | os.PathLike | None],
    required: Sequence[str],
) -> Path | None:
    for source in sources:
        for candidate in _candidate_paths(source):
            root = candidate if candidate.is_dir() else candidate.parent
            if all((root / item).exists() for item in required):
                return root.resolve()
    return None


def _offline_snapshot(repo_id: str, required: Sequence[str]) -> Path:
    try:
        return resolve_local_hf_model_path(repo_id, required_files=tuple(required))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"FantasyWorld asset {repo_id!r} is not staged locally; prepare it before inference"
        ) from error


def resolve_fantasy_world_wan21_checkpoint(source_value) -> Path:
    sources = (source_value, DEFAULT_FANTASY_WORLD_WAN21_REPO)
    path = _first_file(sources, ("model.pth",))
    if path is not None:
        return path
    return _offline_snapshot(DEFAULT_FANTASY_WORLD_WAN21_REPO, ("model.pth",)) / "model.pth"


def resolve_fantasy_world_wan22_checkpoint_dir(source_value) -> Path:
    required = ("high_noise_model.pth", "low_noise_model.pth")
    sources = (source_value, DEFAULT_FANTASY_WORLD_WAN22_REPO)
    root = _first_directory(sources, required)
    return root if root is not None else _offline_snapshot(DEFAULT_FANTASY_WORLD_WAN22_REPO, required)


def resolve_fantasy_world_wan21_base_dir(source_value) -> Path:
    required = (
        "diffusion_pytorch_model-00001-of-00007.safetensors",
        "diffusion_pytorch_model-00007-of-00007.safetensors",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl",
    )
    sources = (source_value, DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO)
    root = _first_directory(sources, required)
    return root if root is not None else _offline_snapshot(DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO, required)


def resolve_fantasy_world_wan22_base_dir(source_value) -> Path:
    required = (
        "high_noise_model",
        "low_noise_model",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl",
    )
    sources = (
        source_value,
        DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO,
        "Wan2.2-I2V-A14B",
        "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera",
        "PAI/Wan2.2-Fun-A14B-Control-Camera",
    )
    root = _first_directory(sources, required)
    return root if root is not None else _offline_snapshot(DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO, required)


def resolve_fantasy_world_wan22_lora_dir(source_value) -> Path:
    required = (WAN22_LORA_HIGH_NAME, WAN22_LORA_LOW_NAME)
    sources = (
        source_value,
        DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO,
        "PAI/Wan2.2-Fun-Reward-LoRAs",
        "Wan2.2-Fun-Reward-LoRAs",
    )
    root = _first_directory(sources, required)
    return root if root is not None else _offline_snapshot(DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO, required)


def resolve_moge_pretrained(source_value) -> str:
    sources = (source_value, DEFAULT_FANTASY_WORLD_MOGE2_REPO)
    path = _first_file(sources, ("model.pt",))
    if path is not None:
        return str(path)
    if source_value is not None and Path(str(source_value)).expanduser().exists():
        raise FileNotFoundError(f"MoGe checkpoint model.pt not found under {source_value}")
    return str(source_value or DEFAULT_FANTASY_WORLD_MOGE2_REPO)


def ensure_moge2_runtime(moge_path=None) -> None:
    """Bind MoGe to the in-tree utils3d implementation; no external source runtime."""

    if moge_path:
        raise RuntimeError(
            "FantasyWorld no longer accepts an external MoGe checkout; pass moge_pretrained weights"
        )
    from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist import (
        utils3d as vendored_utils3d,
    )

    for name in tuple(sys.modules):
        if name == "utils3d" or name.startswith("utils3d."):
            del sys.modules[name]
    sys.modules["utils3d"] = vendored_utils3d


__all__ = [
    "DEFAULT_FANTASY_WORLD_MOGE2_REPO",
    "DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO",
    "DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT",
    "DEFAULT_FANTASY_WORLD_WAN21_REPO",
    "DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO",
    "DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO",
    "DEFAULT_FANTASY_WORLD_WAN22_REPO",
    "WAN22_LORA_HIGH_NAME",
    "WAN22_LORA_LOW_NAME",
    "ensure_moge2_runtime",
    "resolve_fantasy_world_wan21_base_dir",
    "resolve_fantasy_world_wan21_checkpoint",
    "resolve_fantasy_world_wan22_base_dir",
    "resolve_fantasy_world_wan22_checkpoint_dir",
    "resolve_fantasy_world_wan22_lora_dir",
    "resolve_moge_pretrained",
]
