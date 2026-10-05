from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
import requests
import shutil
import yaml

# ---------------------------------------------------------------------------
# Occlusion-variant helpers
# ---------------------------------------------------------------------------

_OCCLUSION_SUFFIXES = ("_cardboard", "_curtain", "_lightoff")


def occlusion_base(task_id: str) -> str:
    """Strip the occlusion suffix from a task ID, returning the base name.

    Examples:
        "ice_on_burner_cardboard" -> "ice_on_burner"
        "ice_on_burner"           -> "ice_on_burner"  (unchanged)
    """
    for s in _OCCLUSION_SUFFIXES:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id

# -----------------------------
# Output links / paths
# -----------------------------

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _resolve_video_source(link: str, outputs_json_path: Path) -> Tuple[str, Path]:
    """
    Returns ("url", dummy_path) or ("file", resolved_path).
    If file path is relative, resolve relative to outputs_json parent directory.
    """
    link = link.strip()
    if _is_url(link):
        return "url", Path(link)  # keep as "Path-like" for uniformity

    p = Path(link)
    if not p.is_absolute():
        p = (outputs_json_path.parent / p).resolve()
    else:
        p = p.resolve()
    return "file", p


# def _copy_or_symlink(src: Path, dst: Path, *, mode: str = "symlink") -> None:
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists():
#         dst.unlink()

#     if mode == "copy":
#         shutil.copy2(src, dst)
#     else:
#         # symlink by default (fast)
#         dst.symlink_to(src)

def _copy_or_symlink(src: Path, dst: Path, *, mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    # mode == "symlink"
    try:
        # On Windows this may fail without admin/dev mode
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


# -----------------------------
# Portable path helpers
# -----------------------------

def _path_to_rel(path: Path, base: Optional[Path] = None) -> str:
    """
    Serialize a path for storage in JSON.

    Returns a POSIX-style relative path from `base` (default: cwd / repo root).
    Falls back to an absolute POSIX path if `path` is not under `base`.
    Using POSIX separators makes stored paths portable across Windows and Linux.
    """
    if base is None:
        base = Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _path_from_rel(s: str, base: Optional[Path] = None) -> Path:
    """
    Deserialize a path stored by `_path_to_rel`.

    If `s` is already absolute, returns it as-is.
    Otherwise resolves relative to `base` (default: cwd / repo root).
    """
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    if base is None:
        base = Path.cwd()
    return (base / p).resolve()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)