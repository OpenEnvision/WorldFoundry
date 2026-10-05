"""
This module provides utilities for resolving file paths and configuring the environment
for the MultiWorld project, particularly for "ittakestwo" components.

It helps locate the project root, the MultiWorld runtime root, and specific
configuration and checkpoint files, handling default values and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from worldfoundry.core.io.paths import package_module_root as package_root


DEFAULT_ITTAKESTWO_CONFIG_RELATIVE_PATH = "ittakestwo/configs/inference_480P_toy.yaml"
DEFAULT_ITTAKESTWO_CHECKPOINT_RELATIVE_PATH = "checkpoints/multiworld_480p_toydata.safetensors"
DEFAULT_MULTIWORLD_HF_REPO_ID = "Haoyuwu/MultiWorldCheckpoint"
DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME = "multiworld_480p_toydata.safetensors"
WAN_TI2V_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_TI2V_MODEL_DIR = "Wan2.2-TI2V-5B"
WAN_TI2V_HFD_DIR = "Wan-AI--Wan2.2-TI2V-5B"
WAN_TI2V_HF_CACHE_DIR = "models--Wan-AI--Wan2.2-TI2V-5B"
WAN_TI2V_DIT_FILENAMES = (
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
)
WAN_TI2V_REQUIRED_FILENAMES = (*WAN_TI2V_DIT_FILENAMES, "Wan2.2_VAE.pth")
IN_TREE_RUNTIME_ROOT = Path(__file__).resolve().parent
MULTIWORLD_CONFIG_DIR = package_root("worldfoundry") / "data" / "models" / "runtime" / "configs" / "multiworld"


def project_root() -> Path:
    """
    Determines the root directory of the current Python project.

    It searches upwards from the current file's location for a 'pyproject.toml' file.
    If 'pyproject.toml' is not found within a reasonable depth, it falls back
    to a predefined parent level, assuming a specific project structure.

    Returns:
        Path: The resolved path to the project root directory.
    """
    current = Path(__file__).resolve()
    # Iterate through parent directories to find 'pyproject.toml' indicating the project root.
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback if 'pyproject.toml' is not found after searching, assuming a specific project structure.
    return current.parents[6]


def _candidate_runtime_root(path_value: Path) -> Optional[Path]:
    """
    Checks if a given path (or a nested 'MultiWorld' directory within it) qualifies
    as a valid MultiWorld runtime root by checking for the presence of sentinel files.

    Args:
        path_value (Path): The potential root path to check.

    Returns:
        Optional[Path]: The validated runtime root path if found, otherwise None.
    """
    sentinels = [("runtime_env.py",)]
    # Check if the sentinel files/directories exist directly under path_value
    if all(path_value.joinpath(*parts).is_file() for parts in sentinels):
        return path_value

    nested = path_value / "worldfoundry" / "synthesis" / "visual_generation" / "multiworld"
    # Check if the sentinel files/directories exist under a nested 'MultiWorld' directory
    if all(nested.joinpath(*parts).is_file() for parts in sentinels):
        return nested

    return None


def local_runtime_candidates() -> list[Path]:
    """
    Generates a list of potential MultiWorld runtime root paths.

    This includes paths specified in environment variables (WORLDFOUNDRY_MULTIWORLD_RUNTIME_ROOT,
    MULTIWORLD_RUNTIME_ROOT) and a default official runtime repository path.

    Returns:
        list[Path]: A list of candidate paths for the MultiWorld runtime root.
    """
    candidates: list[Path] = []
    # Collect paths from specified environment variables
    for env_name in (
        "WORLDFOUNDRY_MULTIWORLD_RUNTIME_ROOT",
        "MULTIWORLD_RUNTIME_ROOT",
    ):
        value = os.environ.get(env_name)
        if value and str(value).strip():
            candidates.append(Path(value).expanduser())
    candidates.append(IN_TREE_RUNTIME_ROOT)
    return candidates


def default_runtime_root() -> Optional[Path]:
    """
    Attempts to find and validate the default MultiWorld runtime root.

    It iterates through a list of candidate paths (from environment variables
    and official repositories) and returns the first one that contains the
    expected MultiWorld runtime structure.

    Returns:
        Optional[Path]: The resolved default MultiWorld runtime root path, or None if not found.
    """
    for candidate in local_runtime_candidates():
        resolved = _candidate_runtime_root(candidate)
        # Return the resolved path if it's a valid directory
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def resolve_runtime_root(runtime_root: Optional[str | Path]) -> str:
    """
    Resolves the MultiWorld runtime root path.

    If `runtime_root` is not provided, it attempts to find a default.
    It then validates the chosen path to ensure it contains the expected
    MultiWorld runtime structure.

    Args:
        runtime_root (Optional[str | Path]): The explicit path to the MultiWorld
                                             runtime root, or None to auto-detect.

    Returns:
        str: The absolute and validated path to the MultiWorld runtime root.

    Raises:
        FileNotFoundError: If no valid MultiWorld runtime root can be found or
                           the provided path is invalid.
    """
    if runtime_root is None or str(runtime_root).strip() == "":
        resolved = default_runtime_root()
        if resolved is None:
            raise FileNotFoundError(
                "Unable to locate the in-tree MultiWorld runtime. "
                "Pass required_components['runtime_root'] only when using an explicit local override."
            )
        return str(resolved)

    candidate = Path(str(runtime_root)).expanduser()
    resolved = _candidate_runtime_root(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"MultiWorld runtime not found at '{runtime_root}'. "
            "Expected the native WorldFoundry multiworld package."
        )
    return str(resolved.resolve())


def _resolve_existing_file(path_value: str | Path, runtime_root: str | Path) -> Optional[Path]:
    """
    Resolves an existing file path, checking both the direct path and a path
    relative to the MultiWorld runtime root.

    Args:
        path_value (str | Path): The file path to resolve.
        runtime_root (str | Path): The MultiWorld runtime root directory.

    Returns:
        Optional[Path]: The absolute path to the existing file if found, otherwise None.
    """
    candidate = Path(str(path_value)).expanduser()
    # Check if the path exists directly
    if candidate.is_file():
        return candidate.resolve()

    # Check if the path exists relative to the runtime root
    runtime_candidate = Path(runtime_root).expanduser() / candidate
    if runtime_candidate.is_file():
        return runtime_candidate.resolve()

    return None


def _resolve_data_config_file(path_value: str | Path) -> Optional[Path]:
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        data_candidate = MULTIWORLD_CONFIG_DIR / candidate
        if data_candidate.is_file():
            return data_candidate.resolve()
    return None


def resolve_config_path(config_path: Optional[str | Path], runtime_root: str | Path) -> str:
    """
    Resolves the full path to the ItTakesTwo configuration file.

    It uses a default configuration path if `config_path` is not provided.
    It checks for the file at the given path and also relative to the MultiWorld runtime root.

    Args:
        config_path (Optional[str | Path]): The explicit path to the config file,
                                            or None to use the default.
        runtime_root (str | Path): The MultiWorld runtime root directory.

    Returns:
        str: The absolute and validated path to the ItTakesTwo configuration file.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    raw_value = config_path or DEFAULT_ITTAKESTWO_CONFIG_RELATIVE_PATH
    resolved = _resolve_data_config_file(raw_value)
    if resolved is None:
        resolved = _resolve_existing_file(raw_value, runtime_root)
    if resolved is None and config_path:
        resolved = _resolve_data_config_file(DEFAULT_ITTAKESTWO_CONFIG_RELATIVE_PATH)
        if resolved is None:
            resolved = _resolve_existing_file(DEFAULT_ITTAKESTWO_CONFIG_RELATIVE_PATH, runtime_root)
    if resolved is None:
        raise FileNotFoundError(
            f"MultiWorld ItTakesTwo config not found: {raw_value}. "
            f"Checked the given path, '{MULTIWORLD_CONFIG_DIR / str(raw_value)}', "
            f"and '{Path(runtime_root) / raw_value}'."
        )
    return str(resolved)


def _checkpoint_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("WORLDFOUNDRY_CKPT_DIR", "WORLD_EVALS_CKPT_DIR"):
        value = os.environ.get(env_name)
        if value and str(value).strip():
            roots.append(Path(value).expanduser())
    project_ckpt = project_root().parent / "ckpt"
    roots.append(project_ckpt)
    return roots


def _valid_wan_ti2v_root(path_value: Path) -> Optional[Path]:
    path_value = path_value.expanduser()
    if all((path_value / filename).is_file() for filename in WAN_TI2V_REQUIRED_FILENAMES):
        return path_value.resolve()
    return None


def _candidate_wan_ti2v_roots() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("WORLDFOUNDRY_WAN_TI2V_ROOT", "WAN_TI2V_ROOT"):
        value = os.environ.get(env_name)
        if value and str(value).strip():
            candidates.append(Path(value).expanduser())
    for root in _checkpoint_roots():
        candidates.extend(
            [
                root / WAN_TI2V_MODEL_DIR,
                root / "hfd" / WAN_TI2V_HFD_DIR,
                root / "huggingface" / "hub" / WAN_TI2V_HF_CACHE_DIR / "snapshots",
            ]
        )
    expanded: list[Path] = []
    for candidate in candidates:
        if candidate.name == "snapshots" and candidate.is_dir():
            expanded.extend(sorted(candidate.glob("*")))
        else:
            expanded.append(candidate)
    return expanded


def resolve_wan_ti2v_root(path_value: Optional[str | Path] = None) -> str:
    """Resolve the official Wan2.2-TI2V-5B root used by MultiWorld."""
    if path_value is not None and str(path_value).strip():
        resolved = _valid_wan_ti2v_root(Path(path_value))
        if resolved is None:
            raise FileNotFoundError(
                f"Wan2.2-TI2V-5B root is incomplete: {path_value}. "
                f"Expected files: {', '.join(WAN_TI2V_REQUIRED_FILENAMES)}."
            )
        return str(resolved)

    for candidate in _candidate_wan_ti2v_roots():
        resolved = _valid_wan_ti2v_root(candidate)
        if resolved is not None:
            return str(resolved)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Wan2.2-TI2V-5B files were not found locally and `huggingface_hub` is not installed. "
            f"Expected files: {', '.join(WAN_TI2V_REQUIRED_FILENAMES)}."
        ) from exc

    downloaded = Path(
        snapshot_download(
            repo_id=WAN_TI2V_REPO_ID,
            allow_patterns=list(WAN_TI2V_REQUIRED_FILENAMES),
            local_files_only=False,
        )
    ).expanduser()
    resolved = _valid_wan_ti2v_root(downloaded)
    if resolved is None:
        raise FileNotFoundError(
            f"Hugging Face download for {WAN_TI2V_REPO_ID} did not contain all required files: "
            f"{', '.join(WAN_TI2V_REQUIRED_FILENAMES)}."
        )
    return str(resolved)


def _candidate_checkpoint_paths(runtime_root: str | Path) -> list[Path]:
    paths: list[Path] = [
        Path(runtime_root).expanduser() / DEFAULT_ITTAKESTWO_CHECKPOINT_RELATIVE_PATH,
    ]
    for root in _checkpoint_roots():
        paths.extend(
            [
                root / "MultiWorld" / DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME,
                root / "multiworld" / DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME,
                root / "hfd" / "Haoyuwu--MultiWorldCheckpoint" / DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME,
                root
                / "huggingface"
                / "hub"
                / "models--Haoyuwu--MultiWorldCheckpoint"
                / "snapshots",
            ]
        )
    expanded: list[Path] = []
    for path in paths:
        if path.name == "snapshots" and path.is_dir():
            expanded.extend(sorted(path.glob(f"*/{DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME}")))
        else:
            expanded.append(path)
    return expanded


def _resolve_default_checkpoint(runtime_root: str | Path) -> Optional[Path]:
    for path in _candidate_checkpoint_paths(runtime_root):
        if path.is_file():
            return path.resolve()
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        return None
    downloaded = hf_hub_download(
        repo_id=DEFAULT_MULTIWORLD_HF_REPO_ID,
        filename=DEFAULT_MULTIWORLD_CHECKPOINT_FILENAME,
    )
    path = Path(downloaded).expanduser()
    return path.resolve() if path.is_file() else None


def resolve_checkpoint_path(checkpoint_path: Optional[str | Path], runtime_root: str | Path) -> str:
    """
    Resolves the full path to the ItTakesTwo checkpoint file.

    It uses a default checkpoint path if `checkpoint_path` is not provided.
    It checks for the file at the given path and also relative to the MultiWorld runtime root.

    Args:
        checkpoint_path (Optional[str | Path]): The explicit path to the checkpoint file,
                                                or None to use the default.
        runtime_root (str | Path): The MultiWorld runtime root directory.

    Returns:
        str: The absolute and validated path to the ItTakesTwo checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file cannot be found.
    """
    raw_value = checkpoint_path or DEFAULT_ITTAKESTWO_CHECKPOINT_RELATIVE_PATH
    resolved = _resolve_existing_file(raw_value, runtime_root)
    if resolved is None:
        resolved = _resolve_default_checkpoint(runtime_root)
    if resolved is None:
        raise FileNotFoundError(
            f"MultiWorld checkpoint not found: {raw_value}. "
            f"Checked the given path, '{Path(runtime_root) / raw_value}', local checkpoint roots, "
            f"and Hugging Face repo '{DEFAULT_MULTIWORLD_HF_REPO_ID}'."
        )
    return str(resolved)


def build_subprocess_env(runtime_root: str | Path) -> dict[str, str]:
    """
    Builds a modified environment dictionary suitable for running MultiWorld
    subprocesses.

    It copies the current environment, modifies the `PYTHONPATH` to include
    necessary project roots and the MultiWorld runtime root, and sets default
    Hugging Face environment variables.

    Args:
        runtime_root (str | Path): The MultiWorld runtime root directory.

    Returns:
        dict[str, str]: A dictionary representing the environment variables
                        for a subprocess.
    """
    env = os.environ.copy()
    pythonpath_entries = [
        str(project_root().resolve()),
        str(Path(runtime_root).expanduser().resolve()),
    ]
    # Append existing PYTHONPATH entries to maintain current environment's Python paths.
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    # Set default Hugging Face cache directory if not already set.
    env.setdefault("HF_HOME", str((project_root().parent / "ckpt" / "huggingface").resolve()))
    # Disable XET for Hugging Face Hub to prevent potential issues with certain file systems.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


__all__ = [
    "DEFAULT_ITTAKESTWO_CHECKPOINT_RELATIVE_PATH",
    "DEFAULT_ITTAKESTWO_CONFIG_RELATIVE_PATH",
    "MULTIWORLD_CONFIG_DIR",
    "build_subprocess_env",
    "default_runtime_root",
    "local_runtime_candidates",
    "project_root",
    "resolve_checkpoint_path",
    "resolve_config_path",
    "resolve_runtime_root",
    "resolve_wan_ti2v_root",
    "WAN_TI2V_DIT_FILENAMES",
]
