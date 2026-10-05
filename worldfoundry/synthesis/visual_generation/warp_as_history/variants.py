"""
This module defines data structures and utility functions for managing Warp-as-History model variants
and their associated paths and configurations. It provides a centralized way to access model
checkpoints, demo data, and other runtime artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class WarpAsHistoryVariant:
    """
    Represents a specific variant of the Warp-as-History model, encapsulating all
    its configuration parameters, paths, and default settings.

    Attributes:
        model_id (str): A unique identifier for the model variant.
        display_name (str): A human-readable name for the model variant.
        task (str): The primary task this model variant is designed for (e.g., 'image-to-video-camera-control').
        model_path (str): The file system path to the main model checkpoint.
        lora_path (str): The file system path to the LoRA weights associated with the model.
        demo_csv_path (str): The path to a CSV file containing demo data for the model.
        default_height (int): The default output video height.
        default_width (int): The default output video width.
        default_num_frames (int): The default number of frames for generated videos.
        default_fps (int): The default frames per second for generated videos.
        default_dtype (str): The default data type for model computations (e.g., "auto", "float16").
        artifact_kind (str): The type of artifact produced by this model (e.g., "generated_video").
        default_extension (str): The default file extension for generated artifacts.
        notes (str): Additional notes or information about the model variant.
    """
    model_id: str
    display_name: str
    task: str
    model_path: str
    lora_path: str
    demo_csv_path: str
    default_height: int = 384
    default_width: int = 640
    default_num_frames: int = 33
    default_fps: int = 16
    default_dtype: str = "auto"
    artifact_kind: str = "generated_video"
    default_extension: str = ".mp4"
    notes: str = ""


def runtime_root() -> Path:
    """
    Returns the root directory for the Warp-as-History runtime artifacts.
    This is typically a subdirectory named 'warp_as_history_runtime' sibling to the current file.

    Returns:
        Path: The absolute path to the runtime root directory.
    """
    return Path(__file__).resolve().parent / "warp_as_history_runtime"


def project_root() -> Path:
    """
    Identifies and returns the root directory of the current project.
    It does this by searching upwards from the current file for a 'pyproject.toml' file.
    If 'pyproject.toml' is not found, it falls back to a predefined parent level.

    Returns:
        Path: The absolute path to the project root directory.
    """
    current = Path(__file__).resolve()
    # Iterate through parent directories to find the project root marked by 'pyproject.toml'.
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback to a fixed parent level if 'pyproject.toml' is not found, assuming a specific project structure.
    return current.parents[6]


def _prefer_existing_path(*candidates: str | Path) -> str:
    """
    Given a list of candidate file system paths, returns the first one that exists.
    Paths are expanded (e.g., user home directory `~`) and resolved to absolute paths.

    Args:
        *candidates (str | Path): One or more potential file system paths.

    Returns:
        str: The string representation of the first existing and resolved path.
             If no candidate exists, returns the string representation of the first candidate,
             or an empty string if no candidates were provided.
    """
    for candidate in candidates:
        path = Path(candidate).expanduser()
        # Check if the candidate path actually exists on the file system.
        if path.exists():
            return str(path.resolve())
    # If no candidate path exists, return the string representation of the first candidate if available,
    # otherwise an empty string.
    return str(candidates[0]) if candidates else ""


def checkpoint_root() -> Path:
    """
    Determines and returns the root directory where model checkpoints are stored.
    It first checks for an 'ckpt/checkpoints' directory adjacent to the project root.
    If not found, it defaults to 'cache/checkpoints' within the project root.

    Returns:
        Path: The absolute path to the checkpoint root directory.
    """
    root = project_root()
    candidates = []
    env_root = os.getenv("WORLDFOUNDRY_CKPT_DIR") or os.getenv("WORLDEVALS_CKPT_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        (
            root.parent / "ckpt",
            root.parent / "ckpt" / "checkpoints",
            root / "cache" / "checkpoints",
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0] if candidates else root / "cache" / "checkpoints"


def test_cases_root() -> Path:
    """
    Returns the root directory for Warp-as-History specific test cases.

    Returns:
        Path: The absolute path to the test cases directory.
    """
    root = project_root()
    candidates = (
        root / "worldfoundry" / "data" / "test_cases" / "warp_as_history",
        root / "src" / "worldfoundry" / "data" / "test_cases" / "warp_as_history",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# A dictionary mapping model_id to WarpAsHistoryVariant objects, defining known model configurations.
WARP_AS_HISTORY_VARIANTS: dict[str, WarpAsHistoryVariant] = {
    "warp-as-history": WarpAsHistoryVariant(
        model_id="warp-as-history",
        display_name="Warp-as-History",
        task="image-to-video-camera-control",
        model_path=_prefer_existing_path(
            checkpoint_root() / "BestWishYsh--Helios-Distilled",
            checkpoint_root() / "Helios-Distilled",
            checkpoint_root() / "helios-distilled",
            checkpoint_root() / "hfd" / "custom--Helios-Distilled",
        ),
        lora_path=_prefer_existing_path(
            checkpoint_root() / "yyfz233--warp-as-history" / "visible_lora_state_step1000.safetensors",
            checkpoint_root() / "warp-as-history" / "visible_lora_state_step1000.safetensors",
        ),
        demo_csv_path="data/demo/bmx-trees.csv",
        notes=(
            "Official Warp-as-History source is vendored in-tree. Runtime checkpoints remain external: "
            "BestWishYsh/Helios-Distilled, yyfz233/warp-as-history, and yyfz233/Pi3X for camera-pose warping."
        ),
    ),
}

# A dictionary mapping common aliases or alternative names to their canonical model_id.
WARP_AS_HISTORY_ALIASES: dict[str, str] = {
    "wah": "warp-as-history",
    "warp_as_history": "warp-as-history",
    "yyfz233/warp-as-history": "warp-as-history",
}


def get_warp_as_history_variant(model_id: str | None = None) -> WarpAsHistoryVariant:
    """
    Retrieves a WarpAsHistoryVariant object based on the provided model ID or alias.

    Args:
        model_id (str | None): The identifier or alias of the desired Warp-as-History model variant.
                                If None, defaults to "warp-as-history".

    Returns:
        WarpAsHistoryVariant: The configuration object for the specified model.

    Raises:
        KeyError: If the provided model_id or alias does not correspond to a known variant.
    """
    # Normalize the input model_id and resolve any aliases to the canonical key.
    key = (model_id or "warp-as-history").strip()
    key = WARP_AS_HISTORY_ALIASES.get(key, key)
    if key not in WARP_AS_HISTORY_VARIANTS:
        # Construct an informative error message listing all known variants and aliases.
        known = ", ".join(sorted((*WARP_AS_HISTORY_VARIANTS, *WARP_AS_HISTORY_ALIASES)))
        raise KeyError(f"Unknown Warp-as-History variant {model_id!r}. Known variants: {known}")
    return WARP_AS_HISTORY_VARIANTS[key]
