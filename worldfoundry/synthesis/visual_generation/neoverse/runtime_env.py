"""Utilities for resolving NeoVerse model and checkpoint paths.

This module provides functions to locate the project root, cache directories,
and NeoVerse runtime metadata. It also
includes robust mechanisms for finding NeoVerse model directories,
reconstructor checkpoints, and LoRA files, supporting both local paths
and downloading from Hugging Face Hub.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_NEOVERSE_REPO = "Yuppie1204/NeoVerse"
DEFAULT_NEOVERSE_LORA_NAME = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"

# List of essential files that must exist in a valid NeoVerse model directory.
_NEOVERSE_REQUIRED_FILES = (
    "reconstructor.ckpt",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)
# Patterns to allow when downloading a NeoVerse snapshot from Hugging Face Hub.
_NEOVERSE_ALLOW_PATTERNS = (
    "reconstructor.ckpt",
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google/*",
    "loras/*",
)


def project_root() -> Path:
    """Discovers and returns the root directory of the current project.

    The project root is determined by searching parent directories for a
    'pyproject.toml' file. If not found, it falls back to a fixed
    ancestor path relative to the current file.

    Returns:
        Path: The absolute path to the project root directory.
    """
    # Iterate through parent directories to find 'pyproject.toml', indicating the project root.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback if pyproject.toml is not found, assuming a fixed project structure.
    return Path(__file__).resolve().parents[6]


def cache_root() -> Path:
    """Returns the root directory for cached Hugging Face data.

    This directory is typically located within the project's 'cache' folder.

    Returns:
        Path: The absolute path to the cache directory.
    """
    return project_root() / "cache" / "hfd"


def runtime_root() -> Path:
    """Returns the root directory for NeoVerse runtime files.

    This directory is a sibling to the current module's directory and
    contains any necessary runtime components.

    Returns:
        Path: The absolute path to the NeoVerse runtime directory.
    """
    return Path(__file__).resolve().parent / "neoverse_runtime"


def _iter_candidate_paths(source_value: str | os.PathLike | None) -> Iterable[Path]:
    """Iterates through potential paths for a given source value.

    This function first checks if the source value directly refers to an
    existing path. If not, it attempts to find the source within the
    cache root, considering variations of the source name.

    Args:
        source_value (Optional[str | os.PathLike]): The input string or path-like object
                                                    representing the source to resolve.

    Yields:
        Path: Resolved absolute paths that are candidates.
    """
    if not source_value:
        return

    raw = Path(source_value).expanduser()
    # If the raw path exists, it's a direct candidate.
    if raw.exists():
        yield raw.resolve()
        return

    source_str = str(source_value)
    candidate_names = []
    if source_str:
        # Generate candidate names for cached items,
        # handling common URL-like or path-like formats.
        candidate_names.append(source_str.replace("/", "--"))
        candidate_names.append(source_str.split("/")[-1])
    # Deduplicate candidate names to avoid redundant checks.
    deduped = []
    for name in candidate_names:
        if name and name not in deduped:
            deduped.append(name)
    # Check for candidates within the cache root.
    for name in deduped:
        candidate = cache_root() / name
        if candidate.exists():
            yield candidate.resolve()


def _download_snapshot(repo_id: str, allow_patterns: Sequence[str]) -> Path:
    """Downloads a snapshot from the Hugging Face Hub.

    This function requires the `WORLDFOUNDRY_ALLOW_REMOTE_DOWNLOADS` environment
    variable to be set to '1' for security reasons in specific environments.
    It uses `huggingface_hub.snapshot_download` to fetch the specified
    repository content, filtering by allowed patterns.

    Args:
        repo_id (str): The ID of the repository on Hugging Face Hub (e.g., "org/repo_name").
        allow_patterns (Sequence[str]): A list of glob patterns to filter which files
                                         are downloaded from the repository.

    Returns:
        Path: The absolute path to the downloaded snapshot directory.

    Raises:
        FileNotFoundError: If remote downloads are disabled by environment variable.
        ImportError: If `huggingface_hub` is not installed.
    """
    # Prevent remote downloads unless explicitly enabled via environment variable.
    if os.environ.get("WORLDFOUNDRY_ALLOW_REMOTE_DOWNLOADS") != "1":
        raise FileNotFoundError(
            f"NeoVerse remote snapshot download is disabled for in-tree workspace inference: {repo_id}. "
            "Provide a local checkpoint path or set WORLDFOUNDRY_ALLOW_REMOTE_DOWNLOADS=1 explicitly."
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(allow_patterns),
        )
    ).resolve()


def _looks_like_neoverse_model_dir(candidate: Path) -> bool:
    """Checks if a given path appears to be a valid NeoVerse model directory.

    A valid NeoVerse model directory must:
    1. Be a directory.
    2. Contain all files specified in `_NEOVERSE_REQUIRED_FILES`.
    3. Contain a subdirectory named "google".
    4. Contain at least one `diffusion_pytorch_model*.safetensors` file.

    Args:
        candidate (Path): The path to check.

    Returns:
        bool: True if the path looks like a NeoVerse model directory, False otherwise.
    """
    if not candidate.is_dir():
        return False
    # Check for all required fixed-name files.
    if not all((candidate / file_name).exists() for file_name in _NEOVERSE_REQUIRED_FILES):
        return False
    # Check for the existence of the 'google' subdirectory.
    if not (candidate / "google").is_dir():
        return False
    # Check for at least one diffusion model safetensors file.
    return len(list(candidate.glob("diffusion_pytorch_model*.safetensors"))) > 0


def _resolve_file_from_source(
    source_value: str | os.PathLike,
    *,
    expected_names: Sequence[str],
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Resolves a specific file from a given source, which can be a local path or a Hugging Face repo ID.

    This function first attempts to find the file locally. If `source_value`
    is a directory, it searches within that directory for files matching
    `expected_names`. If `source_value` doesn't exist locally, it tries
    to download a snapshot from Hugging Face Hub and then searches within
    the downloaded snapshot.

    Args:
        source_value (str | os.PathLike): The local path or Hugging Face repo ID.
        expected_names (Sequence[str]): A list of file names or glob patterns to look for.
        allow_patterns (Optional[Sequence[str]]): Specific patterns to allow when downloading
                                                  from Hugging Face. Defaults to `expected_names`.

    Returns:
        Path: The absolute path to the resolved file.

    Raises:
        FileNotFoundError: If the file cannot be found locally or in the downloaded snapshot.
    """
    candidate = Path(source_value).expanduser()
    if candidate.exists():
        # If the candidate is a file, return it directly.
        if candidate.is_file():
            return candidate.resolve()
        # If the candidate is a directory, search for expected files within it.
        for file_name in expected_names:
            direct = candidate / file_name
            if direct.is_file():
                return direct.resolve()
        # If direct file lookup fails, try glob patterns.
        matches = []
        for pattern in expected_names:
            matches.extend(sorted(candidate.glob(pattern)))
        if matches:
            return matches[0].resolve()
        raise FileNotFoundError(f"Unable to find any of {list(expected_names)} under {candidate}.")

    # If local path doesn't exist, try downloading from Hugging Face Hub.
    snapshot_root = _download_snapshot(
        str(source_value),
        allow_patterns=allow_patterns or expected_names,
    )
    # Search for expected files within the downloaded snapshot.
    for file_name in expected_names:
        direct = snapshot_root / file_name
        if direct.is_file():
            return direct.resolve()
    matches = []
    for pattern in expected_names:
        matches.extend(sorted(snapshot_root.glob(pattern)))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Unable to find any of {list(expected_names)} in snapshot for {source_value}.")


def resolve_neoverse_model_dir(source_value: str | os.PathLike | None) -> Path:
    """Resolves the absolute path to a valid NeoVerse model directory.

    This function searches for a NeoVerse model directory in several locations:
    1. Paths derived from `source_value` (local or cached).
    2. Within a 'NeoVerse' subdirectory if the initial candidate is not the root itself.
    3. If not found locally, it attempts to download a snapshot from Hugging Face Hub
       (using `DEFAULT_NEOVERSE_REPO` if `source_value` is None) and then checks
       the downloaded directory and its 'NeoVerse' subdirectory.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, a Hugging Face repo ID,
                                                    or None to use the default repo.

    Returns:
        Path: The absolute path to the resolved NeoVerse model directory.

    Raises:
        FileNotFoundError: If a valid NeoVerse model directory cannot be found.
    """
    # Iterate through candidate paths (local or cached) to find a model directory.
    for candidate in _iter_candidate_paths(source_value):
        candidate_dir = candidate if candidate.is_dir() else candidate.parent
        if _looks_like_neoverse_model_dir(candidate_dir):
            return candidate_dir.resolve()
        # Also check for a common nesting pattern: 'candidate_dir/NeoVerse'.
        nested = candidate_dir / "NeoVerse"
        if _looks_like_neoverse_model_dir(nested):
            return nested.resolve()

    # If a source value was provided but nothing was found locally, raise an error.
    if source_value and Path(str(source_value)).expanduser().exists():
        raise FileNotFoundError(f"Unable to locate a valid NeoVerse model directory under {source_value}.")

    # If no local or cached directory is found, attempt to download from Hugging Face Hub.
    repo_id = str(source_value) if source_value else DEFAULT_NEOVERSE_REPO
    snapshot_root = _download_snapshot(repo_id, _NEOVERSE_ALLOW_PATTERNS)
    # Check the downloaded root and its 'NeoVerse' subdirectory.
    if _looks_like_neoverse_model_dir(snapshot_root):
        return snapshot_root.resolve()
    nested = snapshot_root / "NeoVerse"
    if _looks_like_neoverse_model_dir(nested):
        return nested.resolve()
    raise FileNotFoundError(f"Unable to locate a valid NeoVerse checkpoint directory in snapshot for {repo_id}.")


def resolve_neoverse_reconstructor_path(
    model_dir: str | os.PathLike,
    override: str | os.PathLike | None = None,
) -> Path:
    """Resolves the absolute path to the NeoVerse reconstructor checkpoint.

    Args:
        model_dir (str | os.PathLike): The root directory of the NeoVerse model.
        override (Optional[str | os.PathLike]): An optional path to explicitly
                                                specify the reconstructor checkpoint.
                                                If provided, this path will be used.

    Returns:
        Path: The absolute path to the reconstructor checkpoint file.

    Raises:
        FileNotFoundError: If the reconstructor checkpoint cannot be found.
    """
    if override:
        return _resolve_file_from_source(
            override,
            expected_names=("reconstructor.ckpt", "*.ckpt"),
        )

    model_root = Path(model_dir).expanduser().resolve()
    default_path = model_root / "reconstructor.ckpt"
    if default_path.is_file():
        return default_path
    raise FileNotFoundError(f"NeoVerse reconstructor checkpoint not found under {model_root}.")


def resolve_neoverse_lora_path(
    model_dir: str | os.PathLike,
    override: str | os.PathLike | None = None,
    *,
    use_lora: bool = True,
) -> Path | None:
    """Resolves the absolute path to a NeoVerse LoRA checkpoint.

    Args:
        model_dir (str | os.PathLike): The root directory of the NeoVerse model.
        override (Optional[str | os.PathLike]): An optional path to explicitly
                                                specify the LoRA checkpoint.
                                                If an empty string, LoRA is disabled.
        use_lora (bool): Whether to attempt to resolve a LoRA path. If False,
                         the function will always return None.

    Returns:
        Optional[Path]: The absolute path to the LoRA checkpoint file, or None
                        if `use_lora` is False, `override` is an empty string,
                        or no LoRA file can be found.
    """
    if not use_lora:
        return None

    # An empty string for override explicitly disables LoRA.
    if override == "":
        return None

    if override:
        return _resolve_file_from_source(
            override,
            expected_names=(DEFAULT_NEOVERSE_LORA_NAME, "*.safetensors"),
            allow_patterns=(DEFAULT_NEOVERSE_LORA_NAME, "loras/*.safetensors", "*.safetensors"),
        )

    model_root = Path(model_dir).expanduser().resolve()
    # Check common default locations for the LoRA file.
    direct_candidates = [
        model_root / "loras" / DEFAULT_NEOVERSE_LORA_NAME,
        model_root / DEFAULT_NEOVERSE_LORA_NAME,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    # If not found in direct locations, perform a recursive search.
    recursive_matches = sorted(model_root.glob(f"**/{DEFAULT_NEOVERSE_LORA_NAME}"))
    if recursive_matches:
        return recursive_matches[0].resolve()
    return None
