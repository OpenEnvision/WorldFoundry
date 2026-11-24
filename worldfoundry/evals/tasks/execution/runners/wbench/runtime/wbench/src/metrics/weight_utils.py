"""WBench metric asset paths backed by worldfoundry.base_models."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_WEIGHTS_DIR = os.environ.get("WBENCH_WEIGHTS_DIR")
_FALLBACK_WEIGHTS_DIR = Path(_LEGACY_WEIGHTS_DIR or _PROJECT_ROOT / "weights")

_DIR_ASSET_BY_SUBDIR = {
    "clip": "wbench_clip_vit_b32_checkpoint",
    "pyiqa": "wbench_musiq_koniq_checkpoint",
}

_LEGACY_ASSET_RELATIVE_PATHS = {
    "wbench_clip_vit_b32_checkpoint": ("clip/ViT-B-32.pt", "clip_model/ViT-B-32.pt"),
    "wbench_clip_vit_l14_checkpoint": ("clip/ViT-L-14.pt", "clip_model/ViT-L-14.pt"),
    "wbench_aesthetic_linear_checkpoint": (
        "aesthetic/sa_0_4_vit_l_14_linear.pth",
        "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
    ),
    "wbench_musiq_koniq_checkpoint": (
        "pyiqa/musiq_koniq_ckpt-e95806b9.pth",
        "pyiqa_model/musiq_koniq_ckpt-e95806b9.pth",
    ),
    "wbench_clip_vit_b16_model_dir": (
        "clip-vit-base-patch16",
        "hfd/openai--clip-vit-base-patch16",
        "openai--clip-vit-base-patch16",
    ),
}


def _legacy_asset_path(asset_id: str) -> Path | None:
    if not _LEGACY_WEIGHTS_DIR:
        return None
    root = Path(_LEGACY_WEIGHTS_DIR)
    for relative_path in _LEGACY_ASSET_RELATIVE_PATHS.get(asset_id, ()):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def wbench_asset_path(asset_id: str) -> Path:
    legacy_path = _legacy_asset_path(asset_id)
    if legacy_path is not None:
        return legacy_path

    for asset in BASE_MODEL_CAPABILITIES["wbench_quality_metric_assets"].assets:
        if asset.id != asset_id:
            continue
        status = asset.check()
        matched = status.get("matched_path")
        if matched:
            return Path(matched)
        candidates = "\n".join(f"  - {path}" for path in status.get("candidate_paths", ()))
        exports = "\n".join(f"  {command}" for command in status.get("export_commands", ()))
        raise FileNotFoundError(
            f"Required WBench asset {asset_id!r} is not staged.\n"
            f"Candidate paths:\n{candidates or '  <none>'}\n"
            f"Environment override:\n{exports or '  <none>'}"
        )
    raise KeyError(f"unknown WBench quality asset id: {asset_id}")


def clip_vit_b16_model_dir() -> Path:
    return wbench_asset_path("wbench_clip_vit_b16_model_dir")


def get_weights_dir(subdir: str = "") -> str:
    if _LEGACY_WEIGHTS_DIR and subdir in {"clip", "pyiqa"}:
        candidates = {
            "clip": ("clip", "clip_model"),
            "pyiqa": ("pyiqa", "pyiqa_model"),
        }[subdir]
        for candidate in candidates:
            path = _FALLBACK_WEIGHTS_DIR / candidate
            if path.exists():
                return str(path)
        path = _FALLBACK_WEIGHTS_DIR / candidates[0]
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    asset_id = _DIR_ASSET_BY_SUBDIR.get(subdir)
    if asset_id:
        return str(wbench_asset_path(asset_id).parent)
    path = _FALLBACK_WEIGHTS_DIR / subdir if subdir else _FALLBACK_WEIGHTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def setup_torch_hub_dir():
    import torch

    hub_dir = get_weights_dir("torch_hub")
    torch.hub.set_dir(hub_dir)
    return hub_dir
