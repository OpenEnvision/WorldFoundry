"""Runtime environment manifests for model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import (
    MODEL_RUNTIME_ENVIRONMENTS_ROOT,
)
from worldfoundry.runtime.conda import RuntimeCondaEnvSpec, load_runtime_conda_env_specs_with_overrides

from ._shared import iter_manifest_mappings, schema_version_or_none, tuple_of_str, yaml_manifest_paths

# Canonical directory containing per-model conda environment specification files.
DEFAULT_RUNTIME_ENVIRONMENTS_ROOT = MODEL_RUNTIME_ENVIRONMENTS_ROOT

# Module-local aliases for the shared runtime loader helpers.
_tuple_of_str = tuple_of_str


# ── Core data model ──────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeEnvironmentProfile:
    """Resolved environment requirements for one model runtime.

    Attributes:
        environment_id: Unique identifier for this environment profile.
        model_id: Model identifier that this environment serves.
        env_name: Conda environment name (e.g. ``"worldfoundry-cogvideox"``).
        python: Required Python version (default ``"3.10"``).
        cuda_profile: CUDA toolkit profile label (e.g. ``"cu113"``).
        driver_status: NVIDIA driver compatibility status.
        conda_packages: Conda package specifications to install.
        pip_packages: Pip package specifications to install.
        pip_extra_index_url: Optional alternate pip index URL.
        pip_find_links: Optional pip find-links pages for wheels published outside the simple index API.
        channels: Conda channels to source packages from.
        validation_imports: Python module names to validate after setup.
        source_requirement_files: Requirement-file paths to install from.
        editable_install_dirs: Package directories for editable installs.
        pythonpath_dirs: Directories to prepend to ``PYTHONPATH``.
        env_required: Required environment variable names.
        env_optional: Optional environment variable names.
        setup_commands: Shell commands to run during environment setup.
        notes: Free-text notes and blocker descriptions.
        env_root: Base directory where conda environments are stored.
        metadata: Arbitrary metadata mapping for the profile.
        source: Provenance label — ``"target"`` or ``"legacy"``.
        schema_version: Schema version marker; only ``2`` is currently supported.
    """

    environment_id: str
    model_id: str
    env_name: str
    python: str = "3.10"
    cuda_profile: str = "cu113"
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
    env_required: tuple[str, ...] = ()
    env_optional: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    env_root: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "target"
    schema_version: int | None = None

    @property
    def env_prefix(self) -> Path | None:
        """Return the full path to the conda environment prefix directory."""
        return self.env_root / self.env_name if self.env_root is not None else None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "target") -> "RuntimeEnvironmentProfile":
        """Build a :class:`RuntimeEnvironmentProfile` from a raw YAML mapping.

        Supports nested ``conda``, ``system``, ``env``, and ``validation``
        sections alongside flat legacy key names.

        Args:
            data: Raw mapping loaded from a YAML environment manifest.
            source: Provenance label for the profile.

        Returns:
            A validated :class:`RuntimeEnvironmentProfile` instance.
        """
        conda = data.get("conda") if isinstance(data.get("conda"), Mapping) else {}
        system = data.get("system") if isinstance(data.get("system"), Mapping) else {}
        top_packages = data.get("packages") if isinstance(data.get("packages"), Mapping) else {}
        env = data.get("env") if isinstance(data.get("env"), Mapping) else {}
        commands = data.get("commands") if isinstance(data.get("commands"), Mapping) else {}
        validation = data.get("validation") if isinstance(data.get("validation"), Mapping) else {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        packages = conda.get("packages") if isinstance(conda.get("packages"), Mapping) else {}
        environment_id = str(data.get("environment_id") or data.get("id") or data.get("model_id") or "")
        model_id = str(data.get("model_id") or data.get("id") or environment_id)
        env_name = str(conda.get("env_name") or data.get("env_name") or f"worldfoundry-{model_id}")
        env_root = conda.get("env_root") or data.get("env_root")
        profile = cls(
            environment_id=environment_id,
            model_id=model_id,
            env_name=env_name,
            schema_version=_schema_version(data.get("schema_version")),
            python=str(conda.get("python") or data.get("python") or "3.10"),
            cuda_profile=str(
                conda.get("cuda_profile")
                or system.get("cuda")
                or data.get("cuda_profile")
                or "cu113"
            ),
            driver_status=str(conda.get("driver_status") or data.get("driver_status") or "compatible"),
            conda_packages=_tuple_of_str(
                packages.get("conda")
                or top_packages.get("conda")
                or conda.get("conda_packages")
                or data.get("conda_packages")
            ),
            pip_packages=_tuple_of_str(
                packages.get("pip")
                or top_packages.get("pip")
                or conda.get("pip_packages")
                or data.get("pip_packages")
            ),
            pip_extra_index_url=(
                str(conda.get("pip_extra_index_url") or data.get("pip_extra_index_url"))
                if conda.get("pip_extra_index_url") or data.get("pip_extra_index_url")
                else None
            ),
            pip_find_links=_tuple_of_str(conda.get("pip_find_links") or data.get("pip_find_links")),
            channels=_tuple_of_str(conda.get("channels") or data.get("channels")),
            validation_imports=_tuple_of_str(
                validation.get("imports") or conda.get("validation_imports") or data.get("validation_imports")
            ),
            source_requirement_files=_tuple_of_str(
                conda.get("source_requirement_files") or data.get("source_requirement_files")
            ),
            editable_install_dirs=_tuple_of_str(conda.get("editable_install_dirs") or data.get("editable_install_dirs")),
            pythonpath_dirs=_tuple_of_str(conda.get("pythonpath_dirs") or data.get("pythonpath_dirs")),
            env_required=_tuple_of_str(env.get("required") or data.get("env_required")),
            env_optional=_tuple_of_str(env.get("optional") or data.get("env_optional")),
            setup_commands=_tuple_of_str(commands.get("setup") or data.get("setup_commands")),
            notes=_tuple_of_str(data.get("notes")),
            env_root=Path(env_root).expanduser() if env_root else None,
            metadata=dict(metadata),
            source=source,
        )
        profile.validate()
        return profile

    @classmethod
    def from_conda_spec(
        cls,
        spec: RuntimeCondaEnvSpec,
        *,
        environment_id: str | None = None,
        source: str = "legacy",
    ) -> "RuntimeEnvironmentProfile":
        """Build a :class:`RuntimeEnvironmentProfile` from a legacy conda environment spec.

        Args:
            spec: Pre-resolved :class:`RuntimeCondaEnvSpec` instance.
            environment_id: Override for the environment identifier.
            source: Provenance label (defaults to ``"legacy"``).

        Returns:
            A validated :class:`RuntimeEnvironmentProfile` instance.
        """
        profile = cls(
            environment_id=environment_id or spec.model_id,
            model_id=spec.model_id,
            env_name=spec.env_name,
            python=spec.python,
            cuda_profile=spec.cuda_profile,
            driver_status=spec.driver_status,
            conda_packages=spec.conda_packages,
            pip_packages=spec.pip_packages,
            pip_extra_index_url=spec.pip_extra_index_url or None,
            pip_find_links=spec.pip_find_links,
            channels=spec.channels,
            validation_imports=spec.validation_imports,
            source_requirement_files=spec.source_requirement_files,
            editable_install_dirs=spec.editable_install_dirs,
            pythonpath_dirs=spec.pythonpath_dirs,
            env_required=(),
            env_optional=(),
            setup_commands=(),
            notes=spec.notes,
            env_root=spec.env_root,
            source=source,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Raise ``ValueError`` if the profile is missing required fields or uses an unsupported schema."""
        if self.schema_version is not None and self.schema_version != 2:
            raise ValueError(
                f"runtime environment {self.environment_id!r} uses unsupported schema_version {self.schema_version!r}."
            )
        if not self.environment_id:
            raise ValueError("runtime environment profile requires environment_id.")
        if not self.model_id:
            raise ValueError(f"runtime environment {self.environment_id!r} requires model_id.")
        if not self.env_name:
            raise ValueError(f"runtime environment {self.environment_id!r} requires env_name.")

    def to_dict(self, *, check_exists: bool = False) -> dict[str, Any]:
        """Convert the environment profile to a plain dictionary suitable for serialization.

        Args:
            check_exists: When ``True``, verify that the conda ``python``
                executable exists on disk and populate the ``"exists"`` key.
        """
        env_prefix = self.env_prefix
        python_executable = env_prefix / "bin" / "python" if env_prefix is not None else None
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "model_id": self.model_id,
            "env_name": self.env_name,
            "env_root": str(self.env_root) if self.env_root is not None else "",
            "env_prefix": str(env_prefix) if env_prefix is not None else "",
            "python": self.python,
            "python_executable": str(python_executable) if python_executable is not None else "",
            "exists": bool(python_executable and python_executable.is_file()) if check_exists else False,
            "cuda_profile": self.cuda_profile,
            "driver_status": self.driver_status,
            "conda_packages": list(self.conda_packages),
            "pip_packages": list(self.pip_packages),
            "pip_extra_index_url": self.pip_extra_index_url,
            "pip_find_links": list(self.pip_find_links),
            "channels": list(self.channels),
            "validation_imports": list(self.validation_imports),
            "source_requirement_files": list(self.source_requirement_files),
            "editable_install_dirs": list(self.editable_install_dirs),
            "pythonpath_dirs": list(self.pythonpath_dirs),
            "env_required": list(self.env_required),
            "env_optional": list(self.env_optional),
            "setup_commands": list(self.setup_commands),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "source": self.source,
        }


# ── Schema version / manifest scanning (shared helpers) ───────

_schema_version = schema_version_or_none


def _manifest_paths(root: str | Path | None) -> tuple[Path, ...]:
    """Retrieve all YAML manifest paths found under a root directory recursively."""
    return yaml_manifest_paths(root, default_root=DEFAULT_RUNTIME_ENVIRONMENTS_ROOT)


def _iter_environment_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load and iterate over environment mappings defined in a YAML manifest file."""
    return iter_manifest_mappings(
        path,
        collection_keys=("environments", "envs"),
        id_keys=("model_id", "environment_id", "id"),
        kind="runtime environment",
    )


# ── Public loaders ────────────────────────────────────────────


def load_runtime_environment_profile(path: str | Path) -> RuntimeEnvironmentProfile:
    """Load a single :class:`RuntimeEnvironmentProfile` from a YAML file path.

    Args:
        path: Path to a YAML file containing exactly one environment profile.

    Raises:
        ValueError: If the file contains zero or more than one profile.
    """
    entries = _iter_environment_mappings(Path(path))
    if len(entries) != 1:
        raise ValueError(f"expected one runtime environment in {path}, found {len(entries)}")
    return RuntimeEnvironmentProfile.from_mapping(entries[0])


def load_runtime_environment_profiles(
    root: str | Path | None = None,
    *,
    env_root: str | Path | None = None,
) -> dict[str, RuntimeEnvironmentProfile]:
    """Load and aggregate all :class:`RuntimeEnvironmentProfile` entries.

    Loads conda-backed environment specs from ``root`` (default:
    ``data/models/runtime/environments``), then overlays YAML profiles found
    under the same tree.

    Args:
        root: Directory root to scan for environment manifests.
        env_root: Base directory where conda environments are stored.

    Returns:
        A ``dict`` keyed by ``model_id`` mapping to resolved
        :class:`RuntimeEnvironmentProfile` instances.
    """
    resolved_root = root or DEFAULT_RUNTIME_ENVIRONMENTS_ROOT
    profiles = {
        model_id: RuntimeEnvironmentProfile.from_conda_spec(spec)
        for model_id, spec in load_runtime_conda_env_specs_with_overrides(
            resolved_root,
            env_root=env_root,
        ).items()
    }
    for path in _manifest_paths(resolved_root):
        for data in _iter_environment_mappings(path):
            profile = RuntimeEnvironmentProfile.from_mapping(data, source="target")
            profiles[profile.model_id] = profile
    return profiles


def load_runtime_environment_profile_by_id(model_id: str, **kwargs: Any) -> RuntimeEnvironmentProfile:
    """Load a single :class:`RuntimeEnvironmentProfile` by model/profile ID.

    Args:
        model_id: The unique identifier of the desired environment profile.
        **kwargs: Forwarded to :func:`load_runtime_environment_profiles`.

    Raises:
        KeyError: If ``model_id`` is not found among loaded profiles.
    """
    profiles = load_runtime_environment_profiles(**kwargs)
    if model_id not in profiles:
        raise KeyError(f"unknown runtime environment profile: {model_id}")
    return profiles[model_id]


resolve_runtime_environment_profile = load_runtime_environment_profile_by_id


__all__ = [
    "DEFAULT_RUNTIME_ENVIRONMENTS_ROOT",
    "RuntimeEnvironmentProfile",
    "load_runtime_environment_profile",
    "load_runtime_environment_profile_by_id",
    "load_runtime_environment_profiles",
    "resolve_runtime_environment_profile",
]
