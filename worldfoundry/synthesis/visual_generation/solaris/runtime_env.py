"""Utilities for resolving file paths, environment variables, and configuration for the Solaris inference runtime.

This module provides functions to locate the Solaris runtime, resolve paths to various
directories (e.g., pretrained models, evaluation data, output, checkpoints, JAX cache),
and manage environment variables for inference execution. It also handles parsing
and normalizing evaluation types for dataset configuration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

from worldfoundry.core import cuda_visible_devices_from_device

# Defines specifications for different evaluation types, including their Hydra key
# and corresponding dataset directory name.
EVAL_TYPE_SPECS = {
    "translation": {
        "hydra_key": "eval_translation",
        "dataset_dir_name": "translationEval",
    },
    "rotation": {
        "hydra_key": "eval_rotation",
        "dataset_dir_name": "rotationEval",
    },
    "structure": {
        "hydra_key": "eval_structure",
        "dataset_dir_name": "structureEval",
    },
    "turn_to_look": {
        "hydra_key": "eval_turn_to_look",
        "dataset_dir_name": "turnToLookEval",
    },
    "turn_to_look_opposite": {
        "hydra_key": "eval_turn_to_look_opposite",
        "dataset_dir_name": "turnToLookOppositeEval",
    },
    "one_looks_away": {
        "hydra_key": "eval_one_looks_away",
        "dataset_dir_name": "oneLooksAwayEval",
    },
    "both_look_away": {
        "hydra_key": "eval_both_look_away",
        "dataset_dir_name": "bothLookAwayEval",
    },
}

# Default directory names for various components within the runtime.
DEFAULT_PRETRAINED_DIRNAME = "pretrained"
DEFAULT_DATA_DIRNAME = "datasets"
DEFAULT_OUTPUT_DIRNAME = "output"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoint"
DEFAULT_JAX_CACHE_DIRNAME = "jax_cache"


def project_root() -> Path:
    """Discovers the project root directory by searching for `pyproject.toml`.

    It traverses up the directory tree from the current file's location until it
    finds a `pyproject.toml` file, which is typically indicative of a project root.

    Returns:
        Path: The discovered project root directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback if pyproject.toml is not found within a reasonable depth,
    # assuming a specific project structure.
    return current.parents[5]


def _candidate_runtime(path_value: Path) -> Optional[Path]:
    """Checks if a given path is a valid Solaris runtime root.

    A path is considered a valid runtime root if it contains specific sentinel
    files, either directly or within a nested 'solaris' subdirectory.

    Args:
        path_value (Path): The path to check.

    Returns:
        Optional[Path]: The validated runtime root path if found, otherwise None.
    """
    # Sentinel files expected in a valid Solaris runtime directory.
    sentinels = [
        ("src", "inference.py"),
    ]
    # Check if the sentinels exist directly under path_value.
    if all(path_value.joinpath(*parts).is_file() for parts in sentinels):
        return path_value
    # If not found directly, check if they exist under a 'solaris' subdirectory.
    nested = path_value / "solaris"
    if all(nested.joinpath(*parts).is_file() for parts in sentinels):
        return nested
    return None


def local_runtime_candidates() -> list[Path]:
    """Generates a list of candidate paths for the Solaris runtime on the local system.

    WorldFoundry ships the inference-only Solaris runtime in-tree.

    Returns:
        list[Path]: A list of potential Solaris runtime root paths.
    """
    in_tree_runtime = Path(__file__).resolve().parent / "solaris_runtime"
    return [
        in_tree_runtime,
    ]


def _resolve_env_path(env_keys: Iterable[str]) -> Optional[Path]:
    """Resolves a file system path from a list of environment variable keys.

    It iterates through the provided environment keys in order, returning the
    first non-empty, expanded path found.

    Args:
        env_keys (Iterable[str]): An iterable of environment variable names to check.

    Returns:
        Optional[Path]: The resolved and expanded path if an environment variable
                        is set, otherwise None.
    """
    for env_key in env_keys:
        raw_value = os.getenv(env_key, "").strip()
        if raw_value:
            return Path(raw_value).expanduser()
    return None


def resolve_runtime_root(runtime_root: Optional[str] = None) -> str:
    """Resolves the root directory of the Solaris inference runtime.

    It checks for a provided `runtime_root` argument, then environment variables,
    and finally common local candidate paths. The first valid path found is returned.

    Args:
        runtime_root (Optional[str], optional): An explicitly provided path to the
                                                runtime root. Defaults to None.

    Returns:
        str: The absolute path to the resolved Solaris runtime root directory.

    Raises:
        FileNotFoundError: If no valid Solaris runtime root can be found.
    """
    candidates: list[Path] = []
    # Add an explicitly provided runtime_root to the candidates if available.
    if runtime_root:
        candidates.append(Path(runtime_root).expanduser())
    # Add candidate paths from environment variables.
    env_candidate = _resolve_env_path(
        [
            "WORLDFOUNDRY_SOLARIS_RUNTIME_ROOT",
            "SOLARIS_RUNTIME_ROOT",
        ]
    )
    if env_candidate is not None:
        candidates.append(env_candidate)
    # Add common local runtime candidates.
    candidates.extend(local_runtime_candidates())

    for candidate in candidates:
        # Check if the candidate path (or its 'solaris' subdirectory) is a valid runtime.
        resolved = _candidate_runtime(candidate)
        if resolved is not None and resolved.is_dir():
            return str(resolved.resolve())

    # If no runtime root was found after checking all candidates, raise an error.
    if runtime_root:
        raise FileNotFoundError(
            f"Solaris runtime not found at '{runtime_root}'. "
            "Expected a directory containing 'src/inference.py'; bundled configs are under worldfoundry/data/models/runtime/configs/solaris."
        )
    raise FileNotFoundError(
        "Unable to locate the Solaris inference runtime. Expected the in-tree "
        "worldfoundry/synthesis/visual_generation/solaris/solaris_runtime directory "
        "or an explicit required_components['runtime_root'] override."
    )


def _resolve_path_under_runtime(
    raw_value: Optional[str],
    runtime_root: str,
    *,
    default_relative: Optional[str],
    env_keys: Iterable[str],
    label: str,
    must_exist: bool,
) -> str:
    """Resolves a file system path, prioritizing explicit values, then environment variables,
    then a default relative path under the runtime root.

    Args:
        raw_value (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.
        default_relative (Optional[str]): A path relative to `runtime_root` to use
                                          if no other value is provided.
        env_keys (Iterable[str]): A list of environment variables to check for the path.
        label (str): A human-readable label for the path being resolved (used in error messages).
        must_exist (bool): If True, raises FileNotFoundError if the resolved path does not exist.

    Returns:
        str: The absolute and resolved path.

    Raises:
        ValueError: If the path cannot be resolved from any source.
        FileNotFoundError: If `must_exist` is True and the resolved path does not exist.
    """
    candidate = None
    if raw_value:
        # Prioritize explicit raw_value.
        candidate = Path(raw_value).expanduser()
    else:
        # Fallback to environment variables.
        env_candidate = _resolve_env_path(env_keys)
        if env_candidate is not None:
            candidate = env_candidate
        # Fallback to default relative path under runtime_root.
        elif default_relative is not None:
            candidate = Path(runtime_root) / default_relative

    # If no candidate was found after checking all sources, raise an error.
    if candidate is None:
        raise ValueError(f"Unable to resolve Solaris {label}.")

    # Resolve absolute path: if relative, combine with runtime_root.
    if not candidate.is_absolute():
        candidate = (Path(runtime_root) / candidate).resolve()
    else:
        candidate = candidate.resolve()

    # Check existence if required.
    if must_exist and not candidate.exists():
        raise FileNotFoundError(
            f"Solaris {label} not found: {candidate}. "
            f"Pass required_components['{label}'] explicitly if it lives elsewhere."
        )
    return str(candidate)


def resolve_pretrained_model_dir(
    pretrained_model_dir: Optional[str],
    runtime_root: str,
) -> str:
    """Resolves the directory containing pretrained models.

    Args:
        pretrained_model_dir (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.

    Returns:
        str: The absolute path to the pretrained models directory.

    Raises:
        ValueError: If the path cannot be resolved.
        FileNotFoundError: If the resolved path does not exist.
    """
    return _resolve_path_under_runtime(
        pretrained_model_dir,
        runtime_root,
        default_relative=DEFAULT_PRETRAINED_DIRNAME,
        env_keys=[
            "WORLDFOUNDRY_SOLARIS_PRETRAINED_MODEL_DIR",
            "SOLARIS_PRETRAINED_MODEL_DIR",
        ],
        label="pretrained_model_dir",
        must_exist=True,
    )


def resolve_eval_data_dir(
    eval_data_dir: Optional[str],
    runtime_root: str,
) -> str:
    """Resolves the directory containing evaluation datasets.

    Args:
        eval_data_dir (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.

    Returns:
        str: The absolute path to the evaluation data directory.

    Raises:
        ValueError: If the path cannot be resolved.
        FileNotFoundError: If the resolved path does not exist.
    """
    return _resolve_path_under_runtime(
        eval_data_dir,
        runtime_root,
        default_relative=DEFAULT_DATA_DIRNAME,
        env_keys=[
            "WORLDFOUNDRY_SOLARIS_EVAL_DATA_DIR",
            "SOLARIS_EVAL_DATA_DIR",
            "WORLDFOUNDRY_SOLARIS_DATA_DIR",
            "SOLARIS_DATA_DIR",
        ],
        label="eval_data_dir",
        must_exist=True,
    )


def resolve_output_dir(
    output_dir: Optional[str],
    runtime_root: str,
) -> str:
    """Resolves the directory for storing output files.

    Args:
        output_dir (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.

    Returns:
        str: The absolute path to the output directory.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    return _resolve_path_under_runtime(
        output_dir,
        runtime_root,
        default_relative=DEFAULT_OUTPUT_DIRNAME,
        env_keys=[
            "WORLDFOUNDRY_SOLARIS_OUTPUT_DIR",
            "SOLARIS_OUTPUT_DIR",
        ],
        label="output_dir",
        must_exist=False,
    )


def resolve_checkpoint_dir(
    checkpoint_dir: Optional[str],
    runtime_root: str,
) -> str:
    """Resolves the directory for storing model checkpoints.

    Args:
        checkpoint_dir (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.

    Returns:
        str: The absolute path to the checkpoint directory.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    return _resolve_path_under_runtime(
        checkpoint_dir,
        runtime_root,
        default_relative=DEFAULT_CHECKPOINT_DIRNAME,
        env_keys=[
            "WORLDFOUNDRY_SOLARIS_CHECKPOINT_DIR",
            "SOLARIS_CHECKPOINT_DIR",
        ],
        label="checkpoint_dir",
        must_exist=False,
    )


def resolve_jax_cache_dir(
    jax_cache_dir: Optional[str],
    runtime_root: str,
) -> str:
    """Resolves the directory for JAX cache files.

    Args:
        jax_cache_dir (Optional[str]): An explicitly provided path.
        runtime_root (str): The determined root directory of the Solaris runtime.

    Returns:
        str: The absolute path to the JAX cache directory.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    return _resolve_path_under_runtime(
        jax_cache_dir,
        runtime_root,
        default_relative=DEFAULT_JAX_CACHE_DIRNAME,
        env_keys=[
            "WORLDFOUNDRY_SOLARIS_JAX_CACHE_DIR",
            "SOLARIS_JAX_CACHE_DIR",
        ],
        label="jax_cache_dir",
        must_exist=False,
    )


def resolve_model_weights_path(
    model_weights_path: Optional[str],
    runtime_root: str,
    pretrained_model_dir: str,
) -> str:
    """Resolves the full path to the model weights file.

    It prioritizes an explicit `model_weights_path`, then environment variables,
    and finally defaults to a 'solaris.pt' file within the `pretrained_model_dir`.
    If an explicit path is relative, it first tries to resolve it relative to
    `pretrained_model_dir`, then relative to `runtime_root`.

    Args:
        model_weights_path (Optional[str]): An explicitly provided path to the weights file.
        runtime_root (str): The determined root directory of the Solaris runtime.
        pretrained_model_dir (str): The determined directory for pretrained models.

    Returns:
        str: The absolute path to the model weights file.

    Raises:
        FileNotFoundError: If the resolved model weights file does not exist.
    """
    candidate = None
    if model_weights_path:
        candidate = Path(model_weights_path).expanduser()
        # If the provided path is relative, try resolving it against two base directories.
        if not candidate.is_absolute():
            pretrained_relative = (Path(pretrained_model_dir) / candidate).resolve()
            runtime_relative = (Path(runtime_root) / candidate).resolve()
            # Use the first one that exists, or default to runtime_relative if neither exist.
            candidate = pretrained_relative if pretrained_relative.exists() else runtime_relative
    else:
        # Check environment variables for model weights path.
        env_candidate = _resolve_env_path(
            [
                "WORLDFOUNDRY_SOLARIS_MODEL_WEIGHTS_PATH",
                "SOLARIS_MODEL_WEIGHTS_PATH",
            ]
        )
        # Default to 'solaris.pt' in the pretrained model directory.
        candidate = env_candidate if env_candidate is not None else Path(pretrained_model_dir) / "solaris.pt"

    # Ensure the final candidate path is absolute and resolved.
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Solaris model_weights_path not found: {candidate}. "
            "Expected a checkpoint path such as pretrained/solaris.pt."
        )
    return str(candidate)


def _normalize_token(value: str) -> str:
    """Normalizes a string by converting it to lowercase, replacing non-alphanumeric
    characters with underscores, and stripping leading/trailing underscores.

    Args:
        value (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_eval_types(eval_types: Optional[object]) -> list[str]:
    """Normalizes and validates a specification of evaluation types.

    It accepts `None`, a single string, or an iterable of strings. It splits
    strings by whitespace or commas, normalizes each token, and maps it to
    a canonical evaluation type using defined aliases (e.g., 'translation',
    'translationEval', 'eval_translation'). 'all' is a special token to
    select all available evaluation types.

    Args:
        eval_types (Optional[object]): The raw input for evaluation types.
                                       Can be None, a string, or an iterable of strings.

    Returns:
        list[str]: A list of canonical, unique evaluation type strings.

    Raises:
        TypeError: If `eval_types` is not None, a string, or an iterable of strings,
                   or if elements in an iterable are not strings.
        ValueError: If an unsupported evaluation type is provided or if the
                    resolved list of types is empty.
    """
    if eval_types is None:
        return list(EVAL_TYPE_SPECS)

    raw_values: list[str] = []
    if isinstance(eval_types, str):
        # Split a single string by whitespace or commas.
        raw_values.extend(part for part in re.split(r"[\s,]+", eval_types) if part)
    else:
        try:
            # Attempt to iterate over the input if it's not a string.
            iterator = iter(eval_types)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("Solaris eval_types must be None, a string, or an iterable of strings.") from error
        for value in iterator:
            if not isinstance(value, str):
                raise TypeError("Solaris eval_types entries must be strings.")
            # For each string in the iterable, split by whitespace or commas.
            raw_values.extend(part for part in re.split(r"[\s,]+", value) if part)

    # Build a dictionary of aliases for various ways to refer to an eval type.
    aliases: dict[str, str] = {}
    for eval_type, spec in EVAL_TYPE_SPECS.items():
        dataset_name = spec["dataset_dir_name"]
        hydra_key = spec["hydra_key"]
        aliases[_normalize_token(eval_type)] = eval_type
        aliases[_normalize_token(dataset_name)] = eval_type
        aliases[_normalize_token(hydra_key)] = eval_type
        aliases[_normalize_token(f"{eval_type}_eval")] = eval_type

    normalized: list[str] = []
    seen = set()
    for raw_value in raw_values:
        token = _normalize_token(raw_value)
        # Handle the special 'all' token to return all known eval types.
        if token == "all":
            return list(EVAL_TYPE_SPECS)
        if token not in aliases:
            supported = ", ".join(EVAL_TYPE_SPECS)
            raise ValueError(f"Unsupported Solaris eval type '{raw_value}'. Supported values: {supported}")
        eval_type = aliases[token]
        # Ensure only unique eval types are added to the list.
        if eval_type not in seen:
            normalized.append(eval_type)
            seen.add(eval_type)
    if not normalized:
        raise ValueError("Solaris eval_types resolved to an empty selection.")
    return normalized


def build_eval_dataset_overrides(eval_types: list[str]) -> list[str]:
    """Generates Hydra configuration overrides to disable unselected evaluation datasets.

    For any evaluation type not present in the `eval_types` list, it creates a Hydra
    override string to exclude that dataset from the configuration.

    Args:
        eval_types (list[str]): A list of canonical evaluation types that *should* be included.

    Returns:
        list[str]: A list of Hydra override strings to disable unselected datasets.
    """
    selected = set(eval_types)
    overrides: list[str] = []
    for eval_type, spec in EVAL_TYPE_SPECS.items():
        if eval_type not in selected:
            overrides.append(f"~dataset@eval_datasets.{spec['hydra_key']}")
    return overrides


def dataset_dir_for_eval_type(eval_type: str) -> str:
    """Retrieves the dataset directory name for a given evaluation type.

    Args:
        eval_type (str): The canonical evaluation type string (e.g., 'translation').

    Returns:
        str: The directory name associated with the evaluation type.
    """
    return str(EVAL_TYPE_SPECS[eval_type]["dataset_dir_name"])


def hydra_key_for_eval_type(eval_type: str) -> str:
    """Retrieves the Hydra key for a given evaluation type.

    Args:
        eval_type (str): The canonical evaluation type string (e.g., 'translation').

    Returns:
        str: The Hydra key associated with the evaluation type.
    """
    return str(EVAL_TYPE_SPECS[eval_type]["hydra_key"])


def _cuda_library_dirs_from_python(python_executable: Optional[str]) -> list[str]:
    """Attempts to find CUDA library directories within a Python virtual environment.

    It inspects the `site-packages` directories under a virtual environment
    (identified by `pyvenv.cfg` or assumed based on `python_executable` location)
    for `nvidia/*/lib` paths. These paths might be needed for `LD_LIBRARY_PATH`.

    Args:
        python_executable (Optional[str]): The path to the Python executable.

    Returns:
        list[str]: A sorted list of unique CUDA library directories found.
    """
    if not python_executable:
        return []

    python_path = Path(python_executable).expanduser()
    venv_root = None
    # Search for pyvenv.cfg to identify the virtual environment root.
    for candidate in [python_path.parent.parent, *python_path.parents]:
        if (candidate / "pyvenv.cfg").is_file():
            venv_root = candidate
            break
    # Fallback if pyvenv.cfg is not found, assumes standard venv structure.
    if venv_root is None:
        venv_root = python_path.resolve().parent.parent

    candidates: list[str] = []
    for lib_parent_name in ("lib", "lib64"):
        lib_parent = venv_root / lib_parent_name
        if not lib_parent.is_dir():
            continue
        # Iterate through python*/site-packages directories.
        for site_packages_dir in sorted(lib_parent.glob("python*/site-packages")):
            nvidia_root = site_packages_dir / "nvidia"
            if not nvidia_root.is_dir():
                continue
            # Look for 'lib' directories within 'nvidia/*' structure.
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                if lib_dir.is_dir():
                    candidates.append(str(lib_dir))

    # Remove duplicates while preserving order.
    unique_candidates: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def build_inference_env(
    device: Optional[str],
    python_executable: Optional[str] = None,
) -> dict[str, str]:
    """Builds a modified environment dictionary suitable for Solaris inference.

    It copies the current environment, sets default values for several variables
    (e.g., Hydra, Weights & Biases, JAX preallocation), configures `CUDA_VISIBLE_DEVICES`
    based on the `device` argument, and optionally manages `LD_LIBRARY_PATH` by
    adding CUDA library directories found within the Python environment.

    Args:
        device (Optional[str]): The target device for inference (e.g., 'cuda', 'cuda:0', 'cpu').
        python_executable (Optional[str], optional): The path to the Python executable
                                                     for inspecting CUDA library paths.
                                                     Defaults to None.

    Returns:
        dict[str, str]: The prepared environment variable dictionary.
    """
    env = os.environ.copy()
    # Set default environment variables for Hydra, Weights & Biases, and JAX.
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Set CUDA_VISIBLE_DEVICES based on the 'device' argument.
    visible_device = cuda_visible_devices_from_device(device, map_inherited=False)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_device

    # Check if LD_LIBRARY_PATH management is explicitly enabled via environment variable.
    manage_ld_library_path = env.get("WORLDFOUNDRY_SOLARIS_MANAGE_LD_LIBRARY_PATH", "").strip().lower()
    if manage_ld_library_path in {"1", "true", "yes", "on"}:
        extra_lib_dirs = _cuda_library_dirs_from_python(python_executable)
    else:
        extra_lib_dirs = []

    if extra_lib_dirs:
        # Sort so that /nvjitlink/ paths come first.
        extra_lib_dirs.sort(key=lambda item: 0 if "/nvjitlink/" in item else 1)
        # Get existing LD_LIBRARY_PATH components.
        existing = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
        # Merge new directories, prioritizing extra_lib_dirs and removing duplicates from existing.
        merged = extra_lib_dirs + [part for part in existing if part not in extra_lib_dirs]
        env["LD_LIBRARY_PATH"] = ":".join(merged)
    return env


__all__ = [
    "EVAL_TYPE_SPECS",
    "build_eval_dataset_overrides",
    "build_inference_env",
    "dataset_dir_for_eval_type",
    "hydra_key_for_eval_type",
    "normalize_eval_types",
    "resolve_checkpoint_dir",
    "resolve_eval_data_dir",
    "resolve_jax_cache_dir",
    "resolve_model_weights_path",
    "resolve_output_dir",
    "resolve_pretrained_model_dir",
    "resolve_runtime_root",
]
