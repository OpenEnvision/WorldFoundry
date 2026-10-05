"""Conda environment specification loading and unified-env routing for WorldFoundry.

Loads per-model conda environment specs from the official runtime profile
manifest, then applies the unified-env override policy that routes compatible
GPU profiles into shared ``worldfoundry-unified-{tier}`` environments when
possible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import InvalidVersion, Version

from worldfoundry.core.io.manifests import load_manifest, load_manifest_collection
from worldfoundry.core.io.paths import (
    conda_envs_root_path,
    package_data_root,
    project_root,
    resolve_worldfoundry_path,
    worldfoundry_path_tokens,
)

from .cuda_tiers import (
    SUPPORTED_CUDA_TIERS,
    cuda_version_tuple,
    detect_nvidia_driver_cuda,
    preferred_unified_tier,
    resolve_cuda_tier,
    unified_env_enabled,
    unified_env_exists,
    unified_env_name,
)
from .env import resolve_ckpt_dir, resolve_hfd_root

# ── Defaults ─────────────────────────────────────────────────────────────────

# Same values evaluation.utils derives; resolved from core so the runtime
# layer never imports worldfoundry.evaluation (SA-10).
REPO_ROOT = project_root()
DATA_ROOT = package_data_root()

DEFAULT_ENV_MANIFEST = DATA_ROOT / "models" / "runtime" / "environments"
DEFAULT_ENV_ROOT = conda_envs_root_path()
RUNTIME_ENV_INSTALLING_MARKER = ".worldfoundry-installing"
RUNTIME_ENV_READY_MARKER = ".worldfoundry-ready"
_MODEL_ID_ALIASES = {
    "lyra1": "lyra-1",
}


def runtime_env_is_usable(prefix: str | Path) -> bool:
    """Return whether an env has Python and is not marked as mid-installation.

    Existing environments remain compatible without a ready marker. New installers
    write an adjacent installing marker before creating the prefix and remove it
    only after validation succeeds, so a failed partial install cannot shadow the
    unified fallback.
    """

    path = Path(prefix).expanduser()
    installing_markers = (
        path / RUNTIME_ENV_INSTALLING_MARKER,
        path.with_name(f"{path.name}{RUNTIME_ENV_INSTALLING_MARKER}"),
    )
    return (path / "bin" / "python").is_file() and not any(
        marker.exists() for marker in installing_markers
    )


def _canonical_model_id(model_id: str) -> str:
    """Return the canonical runtime model id while accepting legacy aliases."""
    text = str(model_id or "").strip()
    return _MODEL_ID_ALIASES.get(text, text)


def _expand_runtime_path(value: str | Path | None) -> Path | None:
    """Expand WorldFoundry path tokens in *value* using runtime environment helpers."""
    if value is None:
        return None
    tokens = worldfoundry_path_tokens()
    tokens.update({
        "WORLDFOUNDRY_BENCH_ROOT": os.environ.get("WORLDFOUNDRY_BENCH_ROOT", str(REPO_ROOT)),
        "WORLDFOUNDRY_REPO_ROOT": os.environ.get("WORLDFOUNDRY_REPO_ROOT", str(REPO_ROOT)),
        "WORLDFOUNDRY_CKPT_DIR": str(resolve_ckpt_dir()),
        "WORLDFOUNDRY_HFD_ROOT": str(resolve_hfd_root()),
    })
    return resolve_worldfoundry_path(value, tokens)


def _tuple_of_str(values: Any) -> tuple[str, ...]:
    """Normalize arbitrary values to a tuple of non-empty strings."""
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Sequence):
        return tuple(str(item) for item in values)
    return (str(values),)


@dataclass(frozen=True)
class RuntimeCondaEnvSpec:
    """Describe a per-model conda environment specification.

    Attributes:
        model_id: Benchmark or integration model identifier.
        env_name: Conda environment name (e.g. ``worldfoundry-animatediff``).
        python: Python version string for the environment.
        cuda_profile: CUDA wheel profile tag (e.g. ``cu128``).
        driver_status: Driver compatibility classification string.
        conda_packages: Conda package specification strings.
        pip_packages: Pip package specification strings.
        pip_extra_index_url: Optional alternate pip index URL for CUDA wheels.
        pip_find_links: Optional pip find-links pages for wheels published outside the simple index API.
        channels: Conda channel names for package resolution.
        validation_imports: Module names that should be importable after setup.
        source_requirement_files: Requirement file paths bundled with the model source.
        editable_install_dirs: Directories installed in editable mode.
        pythonpath_dirs: Paths appended to ``PYTHONPATH`` at runtime.
        notes: Human-readable notes about environment quirks.
        env_root: Root directory under which conda environments are stored.
    """

    model_id: str
    env_name: str
    python: str = "3.10"
    cuda_profile: str = "cu128"
    driver_status: str = "compatible"
    conda_packages: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    pip_extra_index_url: str | None = None
    pip_find_links: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    validation_imports: tuple[str, ...] = ()
    source_requirement_files: tuple[str, ...] = ()
    editable_install_dirs: tuple[str, ...] = ()
    pythonpath_dirs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    env_root: Path = field(default=DEFAULT_ENV_ROOT)

    @property
    def env_prefix(self) -> Path:
        """Return the full conda environment prefix path."""
        return self.env_root / self.env_name

    @property
    def resolved_env_name(self) -> str:
        """Return the final environment name after prefix resolution."""
        return self.env_prefix.name

    @property
    def python_executable(self) -> Path:
        """Return the ``python`` binary path inside the conda environment."""
        return self.env_prefix / "bin" / "python"

    def executable(self, name: str) -> Path:
        """Return the path to an executable named *name* in the environment ``bin/``."""
        return self.env_prefix / "bin" / name

    @property
    def exists(self) -> bool:
        """Return whether the environment exists and is not mid-installation."""
        return runtime_env_is_usable(self.env_prefix)

    @property
    def driver_compatible(self) -> bool:
        """Return whether the NVIDIA driver can run the requested CUDA profile."""
        return self.driver_status.startswith("compatible") or self.driver_status.startswith("requires_local")

    def to_dict(self, *, check_exists: bool = True) -> dict[str, Any]:
        """Serialize the environment spec.

        Args:
            check_exists: Whether to stat the configured Python executable.
        """
        return {
            "model_id": self.model_id,
            "env_name": self.env_name,
            "resolved_env_name": self.resolved_env_name,
            "env_prefix": str(self.env_prefix),
            "python": self.python,
            "python_executable": str(self.python_executable),
            "cuda_profile": self.cuda_profile,
            "driver_status": self.driver_status,
            "driver_compatible": self.driver_compatible,
            "exists": self.exists if check_exists else False,
            "conda_packages": list(self.conda_packages),
            "pip_packages": list(self.pip_packages),
            "pip_extra_index_url": self.pip_extra_index_url,
            "pip_find_links": list(self.pip_find_links),
            "channels": list(self.channels),
            "validation_imports": list(self.validation_imports),
            "source_requirement_files": list(self.source_requirement_files),
            "editable_install_dirs": list(self.editable_install_dirs),
            "pythonpath_dirs": list(self.pythonpath_dirs),
            "notes": list(self.notes),
        }


# ── Unified env routing policy tables ────────────────────────────────────────

_UNIFIED_PYTHON_MINORS = {(3, 10), (3, 11)}
_ALWAYS_ISOLATED_PACKAGES = {"jax", "jaxlib", "tensorflow", "tensorflow-cpu"}
_EXACT_ABI_PACKAGES = {"torch", "torchvision", "torchaudio", "xformers", "flash-attn"}
_MODEL_SPECIFIC_ISOLATED_PACKAGES = {
    "controlnet-aux",
    "denku",
    "e3nn",
    "giga-datasets",
    "giga-train",
    "lerobot",
    "open3d",
    "uniception",
}
_UNIFIED_MINIMUMS = {
    "transformers": Version("4.57.0"),
    "diffusers": Version("0.37.0"),
}
_UNIFIED_MAJOR_CAPS = {
    "transformers": Version("5.0.0"),
}


def _version_or_none(value: str) -> Version | None:
    """Parse a version string; return ``None`` on failure."""
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _python_version_tuple(value: str) -> tuple[int, int] | None:
    """Extract ``(major, minor)`` from a Python version string."""
    match = re.match(r"^\s*([0-9]+)(?:\.([0-9]+))?", str(value or ""))
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2) or 0))


def _requirement_name_and_specs(requirement: str) -> tuple[str, tuple[tuple[str, Version], ...]] | None:
    """Parse a pip requirement string into ``(name, version_specs)`` or ``None``."""
    text = str(requirement or "").strip()
    if not text or text.startswith(("-", "#")):
        return None
    text = text.split(";", maxsplit=1)[0].strip()
    text = text.split(" @ ", maxsplit=1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$", text)
    if not match:
        return None
    name = match.group(1).replace("_", "-").lower()
    specs = []
    for operator, version_text in re.findall(r"(===|==|~=|>=|<=|<|>)\s*([^,\s]+)", match.group(2)):
        version = _version_or_none(version_text)
        if version is not None:
            specs.append((operator, version))
    return name, tuple(specs)


def unified_env_blocker(spec: RuntimeCondaEnvSpec) -> str | None:
    """Return why a profile should not be routed into the shared infer env."""

    driver_status = spec.driver_status.lower()
    if "extension" in driver_status:
        return "local_extension_build_requires_isolated_env"
    for item in spec.source_requirement_files:
        text = str(item)
        if "${WORLDFOUNDRY_MODEL_SOURCE_DIR}" in text:
            return "external_official_requirements"

    python_version = _python_version_tuple(spec.python)
    if python_version is not None and python_version not in _UNIFIED_PYTHON_MINORS:
        return f"python_{spec.python}_outside_unified_py310_py311"

    for requirement in spec.pip_packages:
        parsed = _requirement_name_and_specs(requirement)
        if parsed is None:
            continue
        name, version_specs = parsed
        if name in _ALWAYS_ISOLATED_PACKAGES:
            return f"{name}_runtime_requires_isolated_env"
        if name in _MODEL_SPECIFIC_ISOLATED_PACKAGES:
            return f"{name}_model_specific_package"
        if name in _EXACT_ABI_PACKAGES and any(operator in {"==", "==="} for operator, _ in version_specs):
            return f"{name}_exact_abi_pin"
        minimum = _UNIFIED_MINIMUMS.get(name)
        if minimum is not None:
            for operator, version in version_specs:
                if operator in {"==", "==="} and version < minimum:
                    return f"{name}_{version}_below_unified_{minimum}"
                if operator == "<" and version <= minimum:
                    return f"{name}_upper_bound_below_unified_{minimum}"
        major_cap = _UNIFIED_MAJOR_CAPS.get(name)
        if major_cap is not None:
            for operator, version in version_specs:
                if operator in {"==", "===", ">=", "~="} and version >= major_cap:
                    return f"{name}_{version}_requires_new_major"
                # A strict upper bound is an ABI/API compatibility contract,
                # not merely a minimum-feature hint.  Routing it into a shared
                # environment lets an unrelated upgrade (for example
                # Transformers 5.x) silently violate the model profile.
                if operator == "<" and version <= major_cap:
                    return f"{name}_upper_bound_{version}_requires_isolated_env"
                if operator == "<=" and version < major_cap:
                    return f"{name}_upper_bound_{version}_requires_isolated_env"
    return None


def _load_manifest_mapping(path: Path) -> dict[str, Any]:
    """Load a manifest YAML file or directory of manifests into a mapping."""
    if not path.exists():
        return {}
    payload = load_manifest_collection(path, item_key="envs") if path.is_dir() else load_manifest(path)
    return payload if isinstance(payload, dict) else {}


def _path_cache_key(value: str | Path | None, default: Path | None = None) -> str:
    if value is None:
        return str(default or "")
    return str(Path(value))


def _conda_specs_env_cache_key() -> tuple[tuple[str, str], ...]:
    keys = (
        "WORLDFOUNDRY_CONDA_ENV_ROOT",
        "WORLDFOUNDRY_CKPT_DIR",
        "WORLDFOUNDRY_HFD_ROOT",
        "WORLDFOUNDRY_HOME",
    )
    return tuple((key, os.environ.get(key, "")) for key in keys)


def _unified_override_env_cache_key() -> tuple[tuple[str, str], ...]:
    keys = (
        "WORLDFOUNDRY_USE_UNIFIED_ENV",
        "WORLDFOUNDRY_CONDA_ENVS_ROOT",
        "WORLDFOUNDRY_CONDA_ENV_ROOT",
        "WORLDFOUNDRY_UNIFIED_ENV_PREFIX",
        "WORLDFOUNDRY_CUDA_PROFILE",
        "WORLDFOUNDRY_CUDA_TIER",
        "WORLDFOUNDRY_DETECTED_DRIVER_CUDA",
    )
    values = [(key, os.environ.get(key, "")) for key in keys]
    values.extend((f"unified_env_exists:{tier}", str(unified_env_exists(tier))) for tier in SUPPORTED_CUDA_TIERS)
    return tuple(values)


def _load_runtime_conda_env_specs_uncached(
    manifest_path: str | Path | None = None,
    *,
    env_root: str | Path | None = None,
) -> dict[str, RuntimeCondaEnvSpec]:
    resolved_manifest = Path(manifest_path) if manifest_path is not None else DEFAULT_ENV_MANIFEST
    payload = _load_manifest_mapping(resolved_manifest)
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), Mapping) else {}
    resolved_root = _expand_runtime_path(
        env_root or os.environ.get("WORLDFOUNDRY_CONDA_ENV_ROOT") or defaults.get("env_root") or DEFAULT_ENV_ROOT
    ) or DEFAULT_ENV_ROOT
    default_channels = _tuple_of_str(defaults.get("channels"))
    default_extra_index = defaults.get("torch_cuda_extra_index_url")

    specs: dict[str, RuntimeCondaEnvSpec] = {}
    for item in payload.get("envs") or ():
        if not isinstance(item, Mapping) or not item.get("model_id"):
            continue
        pip_extra_index = item.get("pip_extra_index_url") if "pip_extra_index_url" in item else default_extra_index
        spec = RuntimeCondaEnvSpec(
            model_id=str(item["model_id"]),
            env_name=str(item.get("env_name") or f"worldfoundry-{item['model_id']}"),
            python=str(item.get("python") or "3.10"),
            cuda_profile=str(item.get("cuda_profile") or defaults.get("cuda_profile") or "cu113"),
            driver_status=str(item.get("driver_status") or "compatible"),
            conda_packages=_tuple_of_str(item.get("conda_packages")),
            pip_packages=_tuple_of_str(item.get("pip_packages")),
            pip_extra_index_url=None if pip_extra_index is None else str(pip_extra_index),
            pip_find_links=_tuple_of_str(item.get("pip_find_links") or item.get("pip_find_links_url")),
            channels=_tuple_of_str(item.get("channels")) or default_channels,
            validation_imports=_tuple_of_str(item.get("validation_imports")),
            source_requirement_files=_tuple_of_str(item.get("source_requirement_files")),
            editable_install_dirs=_tuple_of_str(item.get("editable_install_dirs")),
            pythonpath_dirs=_tuple_of_str(item.get("pythonpath_dirs")),
            notes=_tuple_of_str(item.get("notes")),
            env_root=_expand_runtime_path(item.get("env_root")) or resolved_root,
        )
        specs[spec.model_id] = spec
    return specs


@lru_cache(maxsize=32)
def _load_runtime_conda_env_specs_cached(
    manifest_path_key: str,
    env_root_key: str,
    env_cache_key: tuple[tuple[str, str], ...],
) -> dict[str, RuntimeCondaEnvSpec]:
    del env_cache_key
    return _load_runtime_conda_env_specs_uncached(
        manifest_path_key or None,
        env_root=env_root_key or None,
    )


def load_runtime_conda_env_specs(
    manifest_path: str | Path | None = None,
    *,
    env_root: str | Path | None = None,
) -> dict[str, RuntimeCondaEnvSpec]:
    """Load all model conda environment specs from the official manifest.

    Args:
        manifest_path: Path to the runtime profile manifest file or directory.
        env_root: Override for the conda environments root directory.

    Returns:
        Mapping of ``model_id`` to :class:`RuntimeCondaEnvSpec`.
    """
    manifest_path_key = _path_cache_key(manifest_path, DEFAULT_ENV_MANIFEST)
    env_root_key = "" if env_root is None else str(env_root)
    return dict(
        _load_runtime_conda_env_specs_cached(
            manifest_path_key,
            env_root_key,
            _conda_specs_env_cache_key(),
        )
    )


def _resolved_unified_env_root(env_root: str | Path | None = None) -> Path:
    """Resolve the root directory for unified-tier conda environments."""
    return _expand_runtime_path(
        env_root
        or os.environ.get("WORLDFOUNDRY_CONDA_ENVS_ROOT")
        or os.environ.get("WORLDFOUNDRY_CONDA_ENV_ROOT")
        or DEFAULT_ENV_ROOT
    ) or DEFAULT_ENV_ROOT


def apply_unified_env_override(
    spec: RuntimeCondaEnvSpec,
    *,
    env_root: str | Path | None = None,
) -> RuntimeCondaEnvSpec:
    """Route runnable GPU profiles to the shared unified tier env when enabled."""

    if not unified_env_enabled():
        return spec
    if spec.model_id.startswith("_"):
        return spec
    if spec.cuda_profile in {"", "cpu", "prepare_only"}:
        return spec
    if not spec.driver_compatible:
        return spec
    blocker = unified_env_blocker(spec)
    if blocker:
        return spec

    tier = resolve_cuda_tier(
        spec.cuda_profile,
        driver_cuda=os.environ.get("WORLDFOUNDRY_DETECTED_DRIVER_CUDA", ""),
        preferred_tier=preferred_unified_tier(),
    )
    if tier in {"", "cpu", "prepare_only"}:
        return spec

    return RuntimeCondaEnvSpec(
        model_id=spec.model_id,
        env_name=unified_env_name(tier),
        python=spec.python,
        cuda_profile=tier,
        driver_status=spec.driver_status,
        conda_packages=spec.conda_packages,
        pip_packages=spec.pip_packages,
        pip_extra_index_url=spec.pip_extra_index_url,
        pip_find_links=spec.pip_find_links,
        channels=spec.channels,
        validation_imports=spec.validation_imports,
        source_requirement_files=spec.source_requirement_files,
        editable_install_dirs=spec.editable_install_dirs,
        pythonpath_dirs=spec.pythonpath_dirs,
        notes=spec.notes + (f"routed_to_unified_env:{tier}",),
        env_root=_resolved_unified_env_root(env_root),
    )


def load_runtime_conda_env_spec(
    model_id: str,
    manifest_path: str | Path | None = None,
    *,
    env_root: str | Path | None = None,
) -> RuntimeCondaEnvSpec | None:
    """Load a single model's conda spec with unified-env override applied.

    Args:
        model_id: Model identifier to look up.
        manifest_path: Path to the runtime profile manifest file or directory.
        env_root: Override for the conda environments root directory.

    Returns:
        The resolved :class:`RuntimeCondaEnvSpec` or ``None`` if not found.
    """
    spec = load_runtime_conda_env_specs(manifest_path, env_root=env_root).get(_canonical_model_id(model_id))
    if spec is None:
        return None
    return apply_unified_env_override(spec, env_root=env_root)


def load_runtime_conda_env_specs_with_overrides(
    manifest_path: str | Path | None = None,
    *,
    env_root: str | Path | None = None,
) -> dict[str, RuntimeCondaEnvSpec]:
    """Load all model specs with unified-env overrides applied.

    Args:
        manifest_path: Path to the runtime profile manifest file or directory.
        env_root: Override for the conda environments root directory.

    Returns:
        Mapping of ``model_id`` to overridden :class:`RuntimeCondaEnvSpec`.
    """
    manifest_path_key = _path_cache_key(manifest_path, DEFAULT_ENV_MANIFEST)
    env_root_key = "" if env_root is None else str(env_root)
    return dict(
        _load_runtime_conda_env_specs_with_overrides_cached(
            manifest_path_key,
            env_root_key,
            _conda_specs_env_cache_key(),
            _unified_override_env_cache_key(),
        )
    )


@lru_cache(maxsize=32)
def _load_runtime_conda_env_specs_with_overrides_cached(
    manifest_path_key: str,
    env_root_key: str,
    specs_env_cache_key: tuple[tuple[str, str], ...],
    override_env_cache_key: tuple[tuple[str, str], ...],
) -> dict[str, RuntimeCondaEnvSpec]:
    del specs_env_cache_key, override_env_cache_key
    env_root = env_root_key or None
    return {
        model_id: apply_unified_env_override(spec, env_root=env_root)
        for model_id, spec in load_runtime_conda_env_specs(manifest_path_key or None, env_root=env_root).items()
    }


def clear_runtime_conda_env_cache() -> None:
    """Clear cached runtime conda environment manifests and overrides."""
    _load_runtime_conda_env_specs_cached.cache_clear()
    _load_runtime_conda_env_specs_with_overrides_cached.cache_clear()


def is_cuda_profile_supported(cuda_profile: str, driver_cuda: str | None = None) -> bool:
    """Check whether a CUDA profile can run on the local NVIDIA driver.

    Args:
        cuda_profile: CUDA wheel profile tag (e.g. ``cu128``).
        driver_cuda: Detected driver CUDA version string; auto-detected if ``None``.
    """
    if cuda_profile in {"", "cpu", "prepare_only"}:
        return True
    match = re.fullmatch(r"cu([0-9]{2})([0-9])", cuda_profile)
    if not match:
        return True
    required = (int(match.group(1)), int(match.group(2)))
    return cuda_version_tuple(driver_cuda or detect_nvidia_driver_cuda()) >= required


def resolve_conda_env_context(
    model_id: str,
    *,
    manifest_path: str | Path | None = None,
    env_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a model's conda env context with driver compatibility info.

    Args:
        model_id: Model identifier to look up.
        manifest_path: Path to the runtime profile manifest file or directory.
        env_root: Override for the conda environments root directory.

    Returns:
        Serialized environment context dict, or ``{}`` if the model is not found.
    """
    spec = load_runtime_conda_env_spec(model_id, manifest_path, env_root=env_root)
    if spec is None:
        return {}
    data = spec.to_dict()
    data["driver_cuda"] = detect_nvidia_driver_cuda()
    data["cuda_profile_supported"] = is_cuda_profile_supported(spec.cuda_profile, data["driver_cuda"])
    return data


def resolve_conda_executable(
    model_id: str,
    executable: str,
    *,
    manifest_path: str | Path | None = None,
    env_root: str | Path | None = None,
) -> str | None:
    """Resolve the path to an executable inside a model's conda environment.

    Args:
        model_id: Model identifier to look up.
        executable: Name of the binary (e.g. ``python``, ``pip``).
        manifest_path: Path to the runtime profile manifest file or directory.
        env_root: Override for the conda environments root directory.

    Returns:
        Absolute path string, or ``None`` if the environment or binary is missing.
    """
    spec = load_runtime_conda_env_spec(model_id, manifest_path, env_root=env_root)
    if spec is None or not spec.exists:
        return None
    candidate = spec.executable(executable)
    return str(candidate) if candidate.is_file() else None

__all__ = [
    "RuntimeCondaEnvSpec",
    "apply_unified_env_override",
    "clear_runtime_conda_env_cache",
    "cuda_version_tuple",
    "detect_nvidia_driver_cuda",
    "is_cuda_profile_supported",
    "load_runtime_conda_env_spec",
    "load_runtime_conda_env_specs",
    "load_runtime_conda_env_specs_with_overrides",
    "resolve_conda_env_context",
    "resolve_conda_executable",
    "unified_env_blocker",
]
