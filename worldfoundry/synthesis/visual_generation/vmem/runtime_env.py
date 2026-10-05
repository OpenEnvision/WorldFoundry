"""
This module configures the in-tree VMem runtime used by WorldFoundry inference.
It handles finding the project root, determining the VMem runtime directory,
managing Python's `sys.path`, and applying runtime compatibility patches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from worldfoundry.core.io.paths import package_data_path


def project_root() -> Path:
    """
    Identifies the root directory of the current project by searching for a 'pyproject.toml' file.

    It traverses up the directory tree from the current file's location. If 'pyproject.toml'
    is found, that directory is considered the project root. As a fallback, it returns
    a fixed number of levels up from the current file, which is useful in certain
    deployment environments where pyproject.toml might not be directly available.

    Returns:
        Path: The determined project root directory.
    """
    # Iterate through parent directories, starting from the current file's location,
    # to find the project root marked by a 'pyproject.toml' file.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: if no pyproject.toml is found, return a fixed number of levels up.
    # This assumes a specific project structure in environments where the toml might be missing.
    return Path(__file__).resolve().parents[6]


# Default path for the VMem runtime, relative to the current file's parent directory.
DEFAULT_VMEM_RUNTIME_ROOT = Path(__file__).resolve().parent / "vmem_runtime"

# A tuple of top-level module names that are considered part of the VMem runtime.
# Used for identifying and purging conflicting modules from sys.modules.
_RUNTIME_TOP_LEVEL_MODULES = ("modeling", "utils", "add_ckpt_path", "models", "dust3r")


def _runtime_root_candidates() -> list[Path]:
    """
    Generates the supported VMem runtime root candidates.

    The candidates are ordered by priority:
    1. `VMEM_RUNTIME_ROOT` environment variable.
    2. The in-tree runtime beside this file (`DEFAULT_VMEM_RUNTIME_ROOT`).

    Returns:
        list[Path]: A deduped list of absolute paths that are potential VMem runtime roots.
    """
    candidates: list[Path] = []

    # Prioritize environment variable 'VMEM_RUNTIME_ROOT' if set.
    env_root = os.environ.get("VMEM_RUNTIME_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(DEFAULT_VMEM_RUNTIME_ROOT)

    # Deduplicate and resolve all candidate paths to ensure uniqueness and absolute representation.
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


def runtime_root(override: Optional[str | Path] = None) -> Path:
    """
    Determines the active VMem runtime root directory.

    If an `override` path is provided, it is used directly. Otherwise, it iterates through
    the supported candidates and returns the first one that exists and is a directory. If
    no valid directory is found, it falls back to the first candidate path, which might not
    exist.

    Args:
        override (Optional[str | Path]): An optional path to explicitly set as the runtime root.

    Returns:
        Path: The determined VMem runtime root directory.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    # Iterate through candidate paths and return the first one that is an existing directory.
    for candidate in _runtime_root_candidates():
        if candidate.is_dir():
            return candidate
    # If no existing directory is found among candidates, return the first candidate path.
    # This might return a non-existent path, which will be handled by `ensure_vmem_runtime`.
    return _runtime_root_candidates()[0]


def default_config_path(runtime_override: Optional[str | Path] = None) -> Path:
    """
    Constructs the path to the default inference configuration file within the VMem runtime.

    Args:
        runtime_override (Optional[str | Path]): An optional path to explicitly set the runtime root,
                                                 which will then be used to find the config.

    Returns:
        Path: The absolute path to the default inference.yaml configuration file.
    """
    if runtime_override is not None or os.environ.get("VMEM_RUNTIME_ROOT"):
        custom = runtime_root(runtime_override) / "configs" / "inference" / "inference.yaml"
        if custom.is_file():
            return custom
    return package_data_path('models', 'runtime', 'configs', 'vmem', 'inference/inference.yaml')


def canonical_cut3r_parent() -> Path:
    """
    Returns the in-tree parent directory that contains the canonical CUT3R package.

    Returns:
        Path: The absolute path to the canonical CUT3R parent package directory.
    """
    return project_root() / "worldfoundry" / "base_models" / "three_dimensions" / "point_clouds"


def canonical_cut3r_root() -> Path:
    """Return the canonical in-tree CUT3R package directory."""
    return canonical_cut3r_parent() / "cut3r"


def canonical_dust3r_parent() -> Path:
    """
    Returns the in-tree DUSt3R package parent used by VMem's optional camera preprocessor.

    The packaged DUSt3R code still imports itself as top-level ``dust3r`` internally, so this
    parent directory is prepended to ``sys.path`` when VMem is initialized.
    """
    return (
        project_root()
        / "worldfoundry"
        / "base_models"
        / "three_dimensions"
        / "general_3d"
        / "dust3r"
    )


def _iter_runtime_module_names() -> Iterable[str]:
    """
    Iterates through currently loaded modules in `sys.modules` and yields the names
    of those that belong to the VMem runtime's top-level modules.

    Yields:
        str: The full name of a loaded module that is part of the VMem runtime.
    """
    for module_name in list(sys.modules):
        # Extract the top-level package name from the full module name.
        top_level = module_name.split(".", maxsplit=1)[0]
        if top_level in _RUNTIME_TOP_LEVEL_MODULES:
            yield module_name


def _purge_conflicting_runtime_modules(root: Path) -> None:
    """
    Removes loaded modules from `sys.modules` that are identified as VMem runtime modules
    but are loaded from a path *outside* the specified VMem runtime root.

    This prevents conflicts when switching or reloading the VMem runtime environment.

    Args:
        root (Path): The designated VMem runtime root directory.
    """
    for module_name in _iter_runtime_module_names():
        module = sys.modules.get(module_name)
        # Skip modules without a __file__ attribute (e.g., built-in modules).
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            # Resolve the actual path of the module file.
            module_path = Path(module_file).resolve()
        except OSError:
            # Handle cases where the module file might not exist or be accessible.
            continue
        # If the module's path is not within the specified runtime root, remove it from sys.modules.
        if root not in module_path.parents:
            sys.modules.pop(module_name, None)


def _purge_module_tree(prefix: str) -> None:
    """
    Removes a module and all its submodules (e.g., 'dust3r' and 'dust3r.foo', 'dust3r.bar')
    from `sys.modules`.

    This is useful for ensuring a clean slate when reloading or patching modules.

    Args:
        prefix (str): The top-level module name or package prefix to purge.
    """
    # Iterate over a copy of sys.modules keys to allow modification during iteration.
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            sys.modules.pop(module_name, None)


def _prepend_sys_path(path: Path) -> None:
    """
    Adds a given path to the beginning of `sys.path`.

    If the path is already in `sys.path`, it is first removed and then re-inserted
    at the front, ensuring it has the highest priority for module imports.

    Args:
        path (Path): The directory path to add to `sys.path`.
    """
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _prefer_vmem_import_paths(root: Path) -> None:
    """
    Configures `sys.path` to prioritize VMem's internal import paths.

    Args:
        root (Path): The VMem runtime root directory.
    """
    ordered_paths = [root]
    dust3r_parent = canonical_dust3r_parent()
    if dust3r_parent.is_dir():
        ordered_paths.append(dust3r_parent)

    # Prepend paths in reverse order so earlier entries keep highest import priority.
    for path in reversed(ordered_paths):
        _prepend_sys_path(path)


def ensure_vmem_runtime(runtime_override: Optional[str | Path] = None) -> Path:
    """
    Ensures that the VMem runtime environment is correctly set up and ready for use.

    This function performs several critical steps:
    1. Determines the VMem runtime root directory.
    2. Verifies that the runtime root and canonical in-tree CUT3R package exist.
    3. Purges any conflicting VMem-related modules that might have been loaded from other locations.
    4. Prioritizes VMem's specific import paths in `sys.path`.

    Args:
        runtime_override (Optional[str | Path]): An optional path to explicitly set the runtime root.

    Returns:
        Path: The absolute path to the verified VMem runtime root directory.

    Raises:
        FileNotFoundError: If the VMem runtime root or canonical CUT3R package is not found.
    """
    root = runtime_root(runtime_override)

    # Verify that the determined VMem runtime root directory exists.
    if not root.is_dir():
        raise FileNotFoundError(
            f"VMem runtime root does not exist: {root}. "
            "Pass required_components={'runtime_root': '/path/to/vmem'} to override it."
        )

    cut3r_root = canonical_cut3r_root()
    if not (cut3r_root / "__init__.py").is_file():
        raise FileNotFoundError(
            f"VMem requires the canonical in-tree CUT3R package at {cut3r_root}."
        )

    # Purge any modules that conflict with the intended VMem runtime root.
    _purge_conflicting_runtime_modules(root)

    _prefer_vmem_import_paths(root)
    return root


__all__ = [
    "DEFAULT_VMEM_RUNTIME_ROOT",
    "canonical_cut3r_root",
    "canonical_cut3r_parent",
    "canonical_dust3r_parent",
    "default_config_path",
    "ensure_vmem_runtime",
    "project_root",
    "runtime_root",
]
