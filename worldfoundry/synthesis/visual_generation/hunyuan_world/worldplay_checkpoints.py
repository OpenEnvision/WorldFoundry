"""Checkpoint resolution helpers for HY-WorldPlay inference."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from worldfoundry.core.io.paths import resolve_local_hf_model_path

LOGGER = logging.getLogger(__name__)

ACTION_CKPT_FILENAMES = (
    "ar_distilled_action_model/model.safetensors",
    "ar_model/diffusion_pytorch_model.safetensors",
    "bidirectional_model/diffusion_pytorch_model.safetensors",
    "ar_rl_model/diffusion_pytorch_model.safetensors",
    "model.safetensors",
)

WORLDPLAY_ENCODER_FALLBACK_DIRS = (
    "Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "hfd/Qwen--Qwen2.5-VL-7B-Instruct",
    "FLUX.1-Redux-dev",
    "black-forest-labs--FLUX.1-Redux-dev",
    "hfd/black-forest-labs--FLUX.1-Redux-dev",
    "siglip-base-patch16-224",
    "google--siglip-base-patch16-224",
)


def _looks_like_hf_repo_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "://" in text:
        return False
    if text.startswith(("~", ".", "/")):
        return False
    return text.count("/") == 1


def first_existing_worldplay_action_ckpt(root: str | os.PathLike[str]) -> str | None:
    root_path = Path(root).expanduser()
    if root_path.is_file():
        return str(root_path)
    if not root_path.is_dir():
        return None
    for relative_path in ACTION_CKPT_FILENAMES:
        candidate = root_path / relative_path
        if candidate.is_file():
            return str(candidate)
    candidates = sorted(path for path in root_path.rglob("*.safetensors") if path.is_file())
    return str(candidates[0]) if candidates else None


def _resolve_local_worldplay_action_ckpt(repo_id: str) -> str:
    root = resolve_local_hf_model_path(repo_id)
    checkpoint = first_existing_worldplay_action_ckpt(root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Local HY-WorldPlay snapshot {root} has no supported action checkpoint. "
            "Pre-download the complete pinned repository before inference."
        )
    return checkpoint


def _stage_worldplay_action_ckpt_if_requested(path: str) -> str:
    cache_root = os.environ.get("WORLDFOUNDRY_HY_WORLDPLAY_LOCAL_CKPT_CACHE_DIR") or os.environ.get(
        "WORLDFOUNDRY_LOCAL_CKPT_CACHE_DIR"
    )
    if not cache_root:
        return path

    source = Path(path).expanduser()
    if not source.is_file():
        return path

    target = Path(cache_root).expanduser() / "HY-WorldPlay" / source.parent.name / source.name
    try:
        if source.resolve() == target.resolve():
            return str(target)
    except OSError:
        pass

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    try:
        import fcntl

        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            source_size = source.stat().st_size
            if not target.is_file() or target.stat().st_size != source_size:
                tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
                try:
                    shutil.copy2(source, tmp_path)
                    os.replace(tmp_path, target)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return str(target)
    except Exception as exc:
        LOGGER.warning(
            "HY-WorldPlay local checkpoint staging failed for %s -> %s: %s. Falling back to source path.",
            source,
            target,
            exc,
        )
        return path


def _copy_file_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    if target.is_file() and target.stat().st_size == source_size:
        return
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _copy_tree_if_needed(source_root: Path, target_root: Path) -> None:
    if not source_root.exists():
        return
    if source_root.is_file():
        _copy_file_if_needed(source_root, target_root)
        return
    for source in source_root.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(source_root)
        _copy_file_if_needed(source, target_root / relative)


def _stage_worldplay_video_model_if_requested(path: str, transformer_version: str) -> str:
    cache_root = os.environ.get("WORLDFOUNDRY_HY_WORLDPLAY_LOCAL_CKPT_CACHE_DIR") or os.environ.get(
        "WORLDFOUNDRY_LOCAL_CKPT_CACHE_DIR"
    )
    if not cache_root:
        return path

    source_root = Path(path).expanduser()
    if not source_root.is_dir():
        return path

    target_root = Path(cache_root).expanduser() / "HunyuanVideo-1.5"
    try:
        if source_root.resolve() == target_root.resolve():
            return str(target_root)
    except OSError:
        pass

    lock_path = target_root / ".worldfoundry_stage.lock"
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            for filename in ("config.json",):
                _copy_tree_if_needed(source_root / filename, target_root / filename)
            for dirname in ("scheduler", "vae", "text_encoder", "vision_encoder"):
                _copy_tree_if_needed(source_root / dirname, target_root / dirname)
            _copy_tree_if_needed(
                source_root / "transformer" / transformer_version,
                target_root / "transformer" / transformer_version,
            )
            source_ckpt_root = source_root.parent
            for dirname in WORLDPLAY_ENCODER_FALLBACK_DIRS:
                source = source_ckpt_root / dirname
                if source.exists():
                    _copy_tree_if_needed(source, Path(cache_root).expanduser() / dirname)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return str(target_root)
    except Exception as exc:
        LOGGER.warning(
            "HY-WorldPlay video-model staging failed for %s -> %s: %s. Falling back to source path.",
            source_root,
            target_root,
            exc,
        )
        return path


def resolve_worldplay_action_ckpt(action_ckpt: str | os.PathLike[str] | None) -> str | None:
    override = os.environ.get("WORLDFOUNDRY_HY_WORLDPLAY_ACTION_CKPT")
    value = str(override or action_ckpt or "").strip()
    if not value:
        return None

    local = first_existing_worldplay_action_ckpt(value)
    if local is not None:
        return _stage_worldplay_action_ckpt_if_requested(local)

    if _looks_like_hf_repo_id(value):
        return _stage_worldplay_action_ckpt_if_requested(_resolve_local_worldplay_action_ckpt(value))

    raise FileNotFoundError(f"HY-WorldPlay action checkpoint is not available locally: {value}")


def resolve_worldplay_video_model_path(
    video_model_path: str | os.PathLike[str],
    transformer_version: str,
) -> str:
    override = os.environ.get("WORLDFOUNDRY_HY_WORLDPLAY_VIDEO_MODEL_PATH")
    value = str(override or video_model_path or "").strip()
    if not value:
        return value
    if Path(value).expanduser().is_dir():
        return _stage_worldplay_video_model_if_requested(value, transformer_version)
    if _looks_like_hf_repo_id(value):
        local_root = resolve_local_hf_model_path(value)
        return _stage_worldplay_video_model_if_requested(str(local_root), transformer_version)
    raise FileNotFoundError(f"HY-WorldPlay video model is not available locally: {value}")
