"""WorldFoundry synthesis runtime component."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from worldfoundry.core.io.paths import project_root
from worldfoundry.pipelines.gen3c.constants import DEFAULT_GEN3C_NEGATIVE_PROMPT

DEFAULT_GEN3C_ALIAS = "gen3c"
DEFAULT_GEN3C_MOGE1_REPO = "Ruicheng/moge-vitl"
_GEN3C_CHECKPOINT_COMPONENT_REPOS = {
    "Gen3C-Cosmos-7B": "nvidia/GEN3C-Cosmos-7B",
    "Cosmos-Tokenize1-CV8x8x8-720p": "nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
    os.path.join("google-t5", "t5-11b"): "google-t5/t5-11b",
}
def _env_truthy(key: str) -> bool:
    """Helper function to env truthy.

    Args:
        key: The key.

    Returns:
        The return value.
    """
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_local_files_only() -> bool:
    """Return whether WorldFoundry should restrict HF resolution to local cache."""
    return _env_truthy("WORLDFOUNDRY_HF_LOCAL_FILES_ONLY")


def _snapshot_download_repo(repo_id: str) -> Path:
    """Resolve a Hugging Face model repo through the standard hub cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency checked by env setup.
        raise ImportError("GEN3C requires huggingface_hub to resolve model snapshots.") from exc

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_files_only=_hf_local_files_only(),
    )
    return Path(snapshot).expanduser()


def _checkpoint_layout_complete(root: Path) -> bool:
    """Helper function to checkpoint layout complete.

    Args:
        root: The root.

    Returns:
        The return value.
    """
    required_files = [
        root / "Gen3C-Cosmos-7B" / "model.pt",
        root / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt",
        root / "google-t5" / "t5-11b" / "config.json",
    ]
    return all(path.is_file() for path in required_files)


def _hf_snapshot_files(root: Path, repo_id: str, filename: str) -> list[Path]:
    """Return files from a Hugging Face hub cache directory without downloading."""
    sanitized = repo_id.replace("/", "--")
    model_dir = root / f"models--{sanitized}"
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(snapshots.glob(f"*/{filename}"))


def _hf_snapshot_dirs(root: Path, repo_id: str) -> list[Path]:
    """Return snapshot directories from a Hugging Face hub cache directory."""
    sanitized = repo_id.replace("/", "--")
    snapshots = root / f"models--{sanitized}" / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _component_required_file(layout_name: str) -> Path:
    if layout_name == "Gen3C-Cosmos-7B":
        return Path("model.pt")
    if layout_name == "Cosmos-Tokenize1-CV8x8x8-720p":
        return Path("mean_std.pt")
    if layout_name == os.path.join("google-t5", "t5-11b"):
        return Path("config.json")
    return Path("config.json")


def _component_dir_candidates(repo_id: str, root: Path) -> list[Path]:
    """Helper function to component dir candidates.

    Args:
        repo_id: The repo id.
        root: The root.

    Returns:
        The return value.
    """
    raw_name = repo_id.split("/")[-1]
    sanitized_name = repo_id.replace("/", "--")
    google_nested = root / repo_id
    candidates = [
        google_nested,
        root / sanitized_name,
        root / raw_name,
    ]
    candidates.extend(_hf_snapshot_dirs(root, repo_id))
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _find_component_dir(repo_id: str, root: Path) -> Optional[Path]:
    """Helper function to find component dir.

    Args:
        repo_id: The repo id.
        root: The root.

    Returns:
        The return value.
    """
    for candidate in _component_dir_candidates(repo_id, root):
        if candidate.exists():
            return candidate.resolve()
    return None


def _find_component_dir_with_file(layout_name: str, repo_id: str, root: Path) -> Optional[Path]:
    required = _component_required_file(layout_name)
    for candidate in _component_dir_candidates(repo_id, root):
        if (candidate / required).is_file():
            return candidate.resolve()
    return None


def _direct_layout_component_sources(root: Path) -> Optional[dict[str, Path]]:
    """Find GEN3C components in a direct local checkpoint root."""
    component_sources: dict[str, Path] = {}
    alias_names = {
        "Gen3C-Cosmos-7B": (
            "Gen3C-Cosmos-7B",
            "GEN3C-Cosmos-7B",
            "nvidia--GEN3C-Cosmos-7B",
        ),
        "Cosmos-Tokenize1-CV8x8x8-720p": (
            "Cosmos-Tokenize1-CV8x8x8-720p",
            "nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
        ),
        os.path.join("google-t5", "t5-11b"): (
            os.path.join("google-t5", "t5-11b"),
            "google-t5--t5-11b",
            "t5-11b",
        ),
    }
    for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
        required = _component_required_file(layout_name)
        found = None
        for name in alias_names.get(layout_name, (layout_name, repo_id.replace("/", "--"), repo_id.split("/")[-1])):
            candidate = root / name
            if (candidate / required).is_file():
                found = candidate.resolve()
                break
        if found is None:
            found = _find_component_dir_with_file(layout_name, repo_id, root)
        if found is None:
            return None
        component_sources[layout_name] = found
    return component_sources


def _reset_dir(path: Path) -> None:
    """Helper function to reset dir.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Helper function to link or copy.

    Args:
        src: The src.
        dst: The dst.

    Returns:
        The return value.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _reset_dir(dst)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _iter_hfd_roots(path_value: Optional[Path]) -> list[Path]:
    """Helper function to iter hfd roots.

    Args:
        path_value: The path value.

    Returns:
        The return value.
    """
    roots = []
    if path_value is not None:
        for current in (path_value, *path_value.parents):
            if current.name == "hfd":
                roots.append(current.resolve())
                break
    for env_key in ("WORLDFOUNDRY_HFD_ROOT", "WORLDFOUNDRY_CKPT_DIR"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value).expanduser().resolve())
            if env_key == "WORLDFOUNDRY_CKPT_DIR":
                roots.append((Path(value).expanduser() / "hfd").resolve())
    adjacent_ckpt = project_root().parent / "ckpt"
    if adjacent_ckpt.is_dir():
        roots.append(adjacent_ckpt.resolve())
        roots.append((adjacent_ckpt / "hfd").resolve())
        roots.append((adjacent_ckpt / "huggingface" / "hub").resolve())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append((Path(hf_home).expanduser() / "hub").resolve())
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value).expanduser().resolve())
    default_hfd = (project_root() / "cache" / "hfd").resolve()
    roots.append(default_hfd)
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _iter_direct_checkpoint_roots() -> list[Path]:
    """Helper function to iter direct checkpoint roots.

    Returns:
        The return value.
    """
    roots = []
    env_value = os.environ.get("WORLDFOUNDRY_GEN3C_CHECKPOINT_DIR")
    if env_value:
        roots.append(Path(env_value).expanduser().resolve())
    adjacent_ckpt = project_root().parent / "ckpt"
    if adjacent_ckpt.is_dir():
        roots.append(adjacent_ckpt.resolve())
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _stage_hfd_checkpoints(hfd_root: Path) -> Optional[Path]:
    """Helper function to stage hfd checkpoints.

    Args:
        hfd_root: The hfd root.

    Returns:
        The return value.
    """
    component_sources = {}
    for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
        component_dir = _find_component_dir_with_file(layout_name, repo_id, hfd_root)
        if component_dir is None:
            return None
        component_sources[layout_name] = component_dir

    return _stage_component_sources(component_sources)


def _stage_component_sources(component_sources: dict[str, Path]) -> Path:
    """Stage component directories into GEN3C's official checkpoint layout."""
    stage_root = project_root() / "cache" / "runtime" / "gen3c_checkpoints"
    if stage_root.exists():
        _reset_dir(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    for layout_name, source_dir in component_sources.items():
        _link_or_copy(source_dir, stage_root / layout_name)
    return stage_root.resolve()


def _stage_hf_snapshot_checkpoints() -> Optional[Path]:
    """Build GEN3C's official checkpoint_dir layout from HF snapshots."""
    component_sources: dict[str, Path] = {}
    try:
        for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
            component_sources[layout_name] = _snapshot_download_repo(repo_id)
    except Exception:
        if _hf_local_files_only():
            return None
        raise

    return _stage_component_sources(component_sources)


def prepare_gen3c_checkpoint_root(checkpoint_dir: Optional[str]) -> str:
    """Prepare gen3c checkpoint root.

    Args:
        checkpoint_dir: The checkpoint dir.

    Returns:
        The return value.
    """
    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())

    for candidate in candidates:
        if candidate.exists() and _checkpoint_layout_complete(candidate):
            return str(candidate.resolve())
        if candidate.exists():
            sources = _direct_layout_component_sources(candidate)
            if sources is not None:
                staged = _stage_component_sources(sources)
                if _checkpoint_layout_complete(staged):
                    return str(staged)

    for candidate in _iter_direct_checkpoint_roots():
        if candidate.exists() and _checkpoint_layout_complete(candidate):
            return str(candidate.resolve())
        if candidate.exists():
            sources = _direct_layout_component_sources(candidate)
            if sources is not None:
                staged = _stage_component_sources(sources)
                if _checkpoint_layout_complete(staged):
                    return str(staged)

    for candidate in candidates:
        for hfd_root in _iter_hfd_roots(candidate if candidate.exists() else candidate.parent):
            staged = _stage_hfd_checkpoints(hfd_root)
            if staged is not None:
                return str(staged)

    for hfd_root in _iter_hfd_roots(None):
        staged = _stage_hfd_checkpoints(hfd_root)
        if staged is not None:
            return str(staged)

    staged = _stage_hf_snapshot_checkpoints()
    if staged is not None and _checkpoint_layout_complete(staged):
        return str(staged)

    raise FileNotFoundError(
        "Unable to locate a complete GEN3C checkpoint root. Expected either a directory containing "
        "'Gen3C-Cosmos-7B/', 'Cosmos-Tokenize1-CV8x8x8-720p/', and 'google-t5/t5-11b/', or cached "
        "HFD repos for those components under ${WORLDFOUNDRY_HFD_ROOT}."
    )
