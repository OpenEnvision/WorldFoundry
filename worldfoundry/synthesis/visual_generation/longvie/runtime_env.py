"""
This module provides utility functions for resolving file system paths related to the LongVie project
and its dependencies, including base models like Wan2.1 I2V and UMT5 tokenizer, and control weights.
It handles various ways of specifying paths, including local directories, Hugging Face Hub (HFD)
repositories, and environment variable expansions, ensuring that the correct model assets are found.
It also manages Python's import resolution path (`sys.path`) for specific LongVie runtime components.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.runtime.env import resolve_hfd_root
from worldfoundry.core.io.paths import checkpoint_root_path


# Default Hugging Face repository IDs for various LongVie components and dependencies.
DEFAULT_LONGVIE2_REPO = "Vchitect/LongVie2"
DEFAULT_WAN21_I2V_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
DEFAULT_WAN21_T2V_TOKENIZER_REPO = "Wan-AI/Wan2.1-T2V-1.3B"

# List of essential files expected to be present in the Wan2.1 I2V base model directory.
WAN21_I2V_REQUIRED_FILES = (
    *(f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)),
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
)


def runtime_root() -> Path:
    """
    Returns the root directory for the LongVie specific runtime components.
    This typically points to `longvie_runtime` adjacent to this file.
    """
    return Path(__file__).resolve().parent / "longvie_runtime"


def video_depth_anything_runtime_root() -> Path:
    """
    Returns the root directory for the `video_depth_anything_longvie` runtime.
    This path is relative to the current file, navigating up three directories
    then down into a specific `three_dimensions` structure.
    """
    return (
        Path(__file__).resolve().parents[3]
        / "three_dimensions"
        / "depth"
        / "video_depth_anything_longvie"
    )


def project_root() -> Path:
    """
    Attempts to find the root directory of the current project by searching for 'pyproject.toml'.
    It traverses parent directories upwards from the current file's location.
    If 'pyproject.toml' is not found, it falls back to a predefined depth (6 parents up).

    Returns:
        Path: The determined project root directory.
    """
    # Iterate through parent directories to find the one containing 'pyproject.toml'.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback if 'pyproject.toml' is not found within a reasonable ancestor.
    return Path(__file__).resolve().parents[6]


def ensure_longvie_runtime() -> Path:
    """
    Ensures that the LongVie runtime and its base-model dependencies are prioritized
    in Python's import resolution path (`sys.path`).

    It adds the `longvie_runtime` and `video_depth_anything_longvie` directories to the
    front of `sys.path`, removing them first if they already exist, to ensure precedence.

    Returns:
        Path: The resolved path to the LongVie runtime root.
    """
    root = runtime_root().resolve()
    video_depth_root = video_depth_anything_runtime_root().resolve()
    for import_root in (root, video_depth_root):
        root_str = str(import_root)
        # Remove the path if it already exists to prevent duplicates and ensure it's at the front.
        if root_str in sys.path:
            sys.path.remove(root_str)
        # Insert the path at the beginning of sys.path for highest priority.
        sys.path.insert(0, root_str)
    return root


def _local_bench_hfd_root() -> Path | None:
    """
    Identifies a local "hfd" (Hugging Face Dataset/Download) root directory within
    or adjacent to the project's 'ckpt' folder, often used for local development or benchmarks.

    Returns:
        Path | None: The path to the local HFD root if found, otherwise None.
    """
    repo_root = project_root()
    # Check common locations for a local HFD root.
    for candidate in (repo_root / "ckpt" / "hfd", repo_root.parent / "ckpt" / "hfd"):
        if candidate.exists():
            return candidate
    return None


def _hfd_roots() -> tuple[Path, ...]:
    """
    Gathers all potential Hugging Face Download (HFD) root directories.
    This includes the globally configured HFD root and any local benchmark roots.
    It ensures that all returned paths are unique and resolved.

    Returns:
        tuple[Path, ...]: A tuple of unique, resolved HFD root paths.
    """
    # Start with the globally configured HFD root from worldfoundry.runtime.
    roots = [resolve_hfd_root(), checkpoint_root_path()]
    # Add any local benchmark HFD root if it exists.
    local_root = _local_bench_hfd_root()
    if local_root is not None:
        roots.append(local_root)

    seen: set[Path] = set()
    unique: list[Path] = []
    # Deduplicate and resolve all candidate HFD roots.
    for root in roots:
        # Resolve path only if it exists; otherwise, keep as is for potential creation.
        resolved = root.resolve() if root.exists() else root
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def hfd_repo_dir(repo_id: str) -> Path:
    """
    Constructs the expected local directory path for a given Hugging Face repository ID
    under the primary HFD root. The '/' in repo_id is replaced with '--'.

    Args:
        repo_id (str): The ID of the Hugging Face repository (e.g., "Vchitect/LongVie2").

    Returns:
        Path: The constructed path to the repository's local directory.
    """
    # Use the first (primary) HFD root and convert '/' in repo_id to '--'.
    return _hfd_roots()[0] / repo_id.replace("/", "--")


def _hfd_repo_dirs(repo_id: str) -> Iterable[Path]:
    """
    Generates all possible local directory paths for a given Hugging Face repository ID,
    checking under all identified HFD root directories.

    Args:
        repo_id (str): The ID of the Hugging Face repository.

    Yields:
        Path: A candidate path for the repository's local directory.
    """
    # Format the repo_id for file system path (e.g., "Vchitect--LongVie2").
    repo_dir = repo_id.replace("/", "--")
    leaf = repo_id.rsplit("/", 1)[-1]
    # Yield a candidate path for each HFD root.
    for root in _hfd_roots():
        yield root / repo_dir
        yield root / leaf


def _hfd_roots_text() -> str:
    """
    Returns a comma-separated string of all identified HFD root directories,
    useful for error messages.

    Returns:
        str: A string representing all HFD root paths.
    """
    return ", ".join(str(root) for root in _hfd_roots())


def _expanded_path(value: str | os.PathLike) -> Path:
    """
    Expands a given path-like value, handling environment variables (especially `WORLDFOUNDRY_`)
    and user home directory (`~`).

    Args:
        value (str | os.PathLike): The path or string to expand.

    Returns:
        Path: The expanded and user-expanded Path object.
    """
    text = str(value)
    # Check for specific WORLDFOUNDRY_ environment variable patterns.
    if "$WORLDFOUNDRY_" in text or "${WORLDFOUNDRY_" in text:
        return expand_worldfoundry_path(text)
    # Otherwise, perform standard user home directory expansion.
    return Path(text).expanduser()


def _candidate_dirs(value: str | os.PathLike | None, default_repo_id: str) -> Iterable[Path]:
    """
    Generates a sequence of candidate directories where required files might be located.
    It prioritizes an explicitly provided `value`, then checks HFD repository directories,
    and finally the default HFD repository directory. Duplicate paths are removed.

    Args:
        value (str | os.PathLike | None): An explicit path provided by the user, or None.
        default_repo_id (str): The default Hugging Face repository ID to check if `value` is not sufficient.

    Yields:
        Path: A unique, resolved candidate directory path.
    """
    candidates: list[Path] = []
    if value:
        path = _expanded_path(value)
        candidates.append(path)
        text = str(value)
        # If the provided value looks like a Hugging Face repo ID (contains '/')
        # and the direct path doesn't exist, also search within HFD roots for it.
        if "/" in text and not path.exists():
            candidates.extend(_hfd_repo_dirs(text))
    # Always include the default repo ID's HFD directories as candidates.
    candidates.extend(_hfd_repo_dirs(default_repo_id))

    seen: set[Path] = set()
    # Deduplicate and yield resolved paths.
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _require_dir_with_files(
    value: str | os.PathLike | None,
    *,
    default_repo_id: str,
    required_files: Sequence[str],
    label: str,
) -> Path:
    """
    Attempts to locate a directory that contains a specified set of required files.
    It searches through a list of candidate directories generated from `value` and `default_repo_id`.

    Args:
        value (str | os.PathLike | None): An explicit path provided by the user, or None.
        default_repo_id (str): The default Hugging Face repository ID to check.
        required_files (Sequence[str]): A list of file names (relative to the directory)
                                        that must exist within the target directory.
        label (str): A human-readable label for the asset being searched (e.g., "Wan2.1 I2V model").

    Returns:
        Path: The resolved path to the directory containing all required files.

    Raises:
        FileNotFoundError: If no suitable directory containing all required files is found
                           among the candidates.
    """
    for candidate in _candidate_dirs(value, default_repo_id):
        # If the candidate path itself is a file, use its parent directory. Otherwise, use the candidate itself.
        directory = candidate.parent if candidate.is_file() else candidate
        # Check if all required files exist within the current candidate directory.
        if all((directory / name).is_file() for name in required_files):
            return directory.resolve()
    # If no directory is found after checking all candidates, raise an error.
    expected = ", ".join(required_files)
    raise FileNotFoundError(
        f"Unable to locate {label}. Expected files under a local directory: {expected}. "
        f"Set an explicit path or stage {default_repo_id} under one of: {_hfd_roots_text()}."
    )


def resolve_wan21_i2v_dir(value: str | os.PathLike | None = None) -> Path:
    """
    Resolves the directory containing the Wan2.1 I2V 14B base model files.
    It uses the `_require_dir_with_files` utility with predefined defaults and required files.

    Args:
        value (str | os.PathLike | None): An explicit path to the Wan2.1 I2V directory, or None.

    Returns:
        Path: The resolved path to the Wan2.1 I2V model directory.
    """
    return _require_dir_with_files(
        value,
        default_repo_id=DEFAULT_WAN21_I2V_REPO,
        required_files=WAN21_I2V_REQUIRED_FILES,
        label="Wan2.1 I2V 14B base model",
    )


def resolve_wan21_tokenizer_dir(value: str | os.PathLike | None = None) -> Path:
    """
    Resolves the directory containing the Wan2.1 UMT5 tokenizer files.
    It uses the `_require_dir_with_files` utility and then appends the specific
    subdirectories "google/umt5-xxl" to the found path.

    Args:
        value (str | os.PathLike | None): An explicit path to the Wan2.1 tokenizer directory, or None.

    Returns:
        Path: The resolved path to the Wan2.1 UMT5 tokenizer directory.
    """
    # Find the root containing the tokenizer files, then append the specific nested directory.
    return (
        _require_dir_with_files(
            value,
            default_repo_id=DEFAULT_WAN21_T2V_TOKENIZER_REPO,
            required_files=(
                "google/umt5-xxl/tokenizer_config.json",
                "google/umt5-xxl/spiece.model",
            ),
            label="Wan2.1 UMT5 tokenizer",
        )
        / "google"
        / "umt5-xxl"
    )


def resolve_longvie_weight_dir(value: str | os.PathLike | None = None) -> Path:
    """
    Resolves the directory containing the LongVie control checkpoint files.
    It uses the `_require_dir_with_files` utility with predefined defaults.

    Args:
        value (str | os.PathLike | None): An explicit path to the LongVie weight directory, or None.

    Returns:
        Path: The resolved path to the LongVie weights directory.
    """
    return _require_dir_with_files(
        value,
        default_repo_id=DEFAULT_LONGVIE2_REPO,
        required_files=("control.safetensors",),
        label="LongVie control checkpoint",
    )


def resolve_control_weight_path(
    value: str | os.PathLike | None = None,
    *,
    weight_dir: str | os.PathLike | None = None,
) -> Path:
    """
    Resolves the specific file path for the LongVie control checkpoint.
    It first checks if an explicit `value` path points directly to the file.
    If not, it resolves the weight directory using `weight_dir` (or defaults)
    and then constructs the path to "control.safetensors" within that directory.

    Args:
        value (str | os.PathLike | None): An explicit path to the control checkpoint file, or None.
        weight_dir (str | os.PathLike | None): An explicit path to the directory containing
                                                LongVie weights, or None.

    Returns:
        Path: The resolved path to the control.safetensors file.

    Raises:
        FileNotFoundError: If the control checkpoint file cannot be found.
    """
    # If an explicit file path is provided, try to resolve and return it.
    if value:
        path = _expanded_path(value)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"LongVie control checkpoint not found: {path}")
    # Otherwise, resolve the containing directory and construct the path to the default file.
    directory = resolve_longvie_weight_dir(weight_dir)
    path = directory / "control.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LongVie control checkpoint not found: {path}")
    return path.resolve()


def resolve_dit_weight_path(
    value: str | os.PathLike | None = None,
    *,
    weight_dir: str | os.PathLike | None = None,
    required: bool = False,
) -> Path | None:
    """
    Resolves the specific file path for the LongVie DiT (Diffusion Transformer) checkpoint.
    It first checks if an explicit `value` path points directly to the file.
    If not, it resolves the weight directory using `weight_dir` (or defaults)
    and then constructs the path to "dit.safetensors" within that directory.

    Args:
        value (str | os.PathLike | None): An explicit path to the DiT checkpoint file, or None.
        weight_dir (str | os.PathLike | None): An explicit path to the directory containing
                                                LongVie weights, or None.
        required (bool): If True, a FileNotFoundError will be raised if the DiT checkpoint
                         cannot be found. If False, returns None if not found.

    Returns:
        Path | None: The resolved path to the dit.safetensors file, or None if not found
                     and `required` is False.

    Raises:
        FileNotFoundError: If the DiT checkpoint file cannot be found and `required` is True.
    """
    # If an explicit file path is provided, try to resolve and return it.
    if value:
        path = _expanded_path(value)
        if path.is_file():
            return path.resolve()
        # If an explicit path was given but the file doesn't exist, raise an error if required.
        if required:
            raise FileNotFoundError(f"LongVie DiT checkpoint not found: {path}")
    # Otherwise, resolve the containing directory and construct the path to the default file.
    directory = resolve_longvie_weight_dir(weight_dir)
    path = directory / "dit.safetensors"
    if path.is_file():
        return path.resolve()
    # If the default file is not found, raise an error if required.
    if required:
        raise FileNotFoundError(f"LongVie DiT checkpoint not found: {path}")
    return None
