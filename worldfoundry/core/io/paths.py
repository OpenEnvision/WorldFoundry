"""WorldFoundry path and directory resolution engine.

Runtimes and benchmarks must not hard-code host absolute paths. Logical
tokens (``${WORLDFOUNDRY_CACHE_DIR}``, checkpoint roots, hfd, conda) expand
the same way on PAI DLC, a laptop, or a container.

Why a single resolver:

- Portability: manifests and evaluators stay unmodified across hosts.
- Predictability: if an env var is missing, walk up to ``pyproject.toml``
  instead of silently using ``/tmp``.
- Hermetic caches: Hugging Face / scratch stay inside WorldFoundry roots
  unless the caller overrides them — otherwise a job contaminates the
  shared user cache.

Prefer :func:`resolve_worldfoundry_path` / :func:`resolve_hf_path` (in
``hf``) over concatenating strings in model code.
"""

from __future__ import annotations

import os
import sysconfig
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping, Sequence

# ──────────────────────────────────────────────────────────────────────────
# Package / repo roots — walk to pyproject.toml; never hard-code host paths
# ──────────────────────────────────────────────────────────────────────────


def package_root() -> Path:
    """Returns the resolved absolute path of the installed `worldfoundry` package root."""
    return Path(__file__).resolve().parents[2]


def package_data_root() -> Path:
    """Resolve bundled WorldFoundry model and benchmark metadata."""

    candidates = (
        package_root() / "data",
        Path(sysconfig.get_path("data")) / "worldfoundry" / "data",
    )
    for candidate in candidates:
        if (candidate / "models").is_dir() or (candidate / "benchmarks").is_dir():
            return candidate
    return candidates[0]


def package_data_path(*parts: str | Path) -> Path:
    """Resolve a path below the bundled WorldFoundry data root."""

    return package_data_root().joinpath(*(Path(part) for part in parts))


def package_module_root(package: str) -> Path:
    """Resolve the source directory for an importable package."""

    spec = find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Could not resolve package: {package}")
    return Path(spec.origin).resolve().parent


def project_root(start: str | Path | None = None) -> Path:
    """Walks upward from a starting path to locate the root repository containing `pyproject.toml`.

    This helper provides robust local development support, falling back to a package-relative
    root if executed from a system-wide python site-packages deployment.

    Args:
        start: File or directory from which to search upward. ``None`` starts
            from this module's installed source path.

    Returns:
        First ancestor containing ``pyproject.toml``, or the package-relative
        fallback when no repository marker is found.
    """
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return package_root().parents[1]


def worldfoundry_path_tokens(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Generates the dictionary of logical path-token replacements used across the system.

    Builds dynamic mappings for artifact, checkpoint, data, conda, and repo paths.
    Prioritizes explicit environment overrides (such as `WORLDFOUNDRY_HOME` or `WORLDFOUNDRY_CACHE_DIR`)
    and falls back to user-home cache directories when variables are unset.

    Args:
        env: Environment mapping to resolve instead of ``os.environ``. Passing
            a mapping makes resolution deterministic in tests.

    Returns:
        Token name to expanded path-string mapping.
    """
    environ = dict(os.environ if env is None else env)
    root = project_root()
    package = package_root()
    project_parent = root.parent
    adjacent_ckpt = project_parent / "ckpt"
    adjacent_conda = project_parent / "conda"
    adjacent_conda_envs = adjacent_conda / "envs"
    explicit_home = environ.get("WORLDFOUNDRY_HOME")
    home = Path(explicit_home or Path.home() / ".cache" / "worldfoundry").expanduser()
    cache_default = home / "cache" if explicit_home else home
    cache = Path(environ.get("WORLDFOUNDRY_CACHE_DIR") or cache_default).expanduser()
    data_dir = Path(
        environ.get("WORLDFOUNDRY_DATA_DIR")
        or environ.get("WORLDFOUNDRY_BENCHMARK_DATA_ROOT")
        or (home / "data" if explicit_home else cache / "data")
    ).expanduser()
    default_hfd_dataset_root = (
        data_dir if data_dir.name in {"datasets", "hfd_datasets"} else data_dir / "datasets"
    )
    hfd_dataset_root = Path(
        environ.get("WORLDFOUNDRY_HFD_DATASET_ROOT")
        or environ.get("WORLDFOUNDRY_LOCAL_DATA_ROOT")
        or environ.get("WORLDFOUNDRY_LOCAL_CACHE_DATA_ROOT")
        or default_hfd_dataset_root
    ).expanduser()
    artifact_dir = Path(
        environ.get("WORLDFOUNDRY_ARTIFACT_DIR")
        or environ.get("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
        or (home / "artifacts" if explicit_home else cache / "artifacts")
    ).expanduser()
    model_dir = Path(
        environ.get("WORLDFOUNDRY_MODEL_DIR") or (home / "models" if explicit_home else cache / "models")
    ).expanduser()
    default_model_source = cache / "official_runtime_repos"
    model_source = Path(environ.get("WORLDFOUNDRY_MODEL_SOURCE_DIR") or default_model_source).expanduser()
    default_ckpt_dir = (
        home / "checkpoints"
        if explicit_home
        else (adjacent_ckpt if adjacent_ckpt.is_dir() else cache / "checkpoints")
    )
    ckpt_dir = Path(environ.get("WORLDFOUNDRY_CKPT_DIR") or default_ckpt_dir).expanduser()
    hfd_root = Path(environ.get("WORLDFOUNDRY_HFD_ROOT") or ckpt_dir / "hfd").expanduser()
    ckpt_sibling = (
        ckpt_dir.with_name("ckpts" if ckpt_dir.name == "ckpt" else "ckpt")
        if ckpt_dir.name in {"ckpt", "ckpts"}
        else ckpt_dir
    )
    default_conda_root = (
        home / "conda"
        if explicit_home
        else (adjacent_conda if adjacent_conda.is_dir() else cache / "conda")
    )
    conda_root = Path(environ.get("WORLDFOUNDRY_CONDA_ROOT") or default_conda_root).expanduser()
    default_conda_envs_root = (
        home / "conda_envs"
        if explicit_home
        else (adjacent_conda_envs if adjacent_conda_envs.is_dir() else cache / "conda_envs")
    )
    conda_envs_root = Path(
        environ.get("WORLDFOUNDRY_CONDA_ENVS_ROOT")
        or environ.get("WORLDFOUNDRY_CONDA_ENV_ROOT")
        or default_conda_envs_root
    ).expanduser()
    return {
        "WORLDFOUNDRY_REPO_ROOT": str(root),
        "WORLDFOUNDRY_PACKAGE_ROOT": str(package),
        "WORLDFOUNDRY_DATA_ROOT": str(package_data_root()),
        "WORLDFOUNDRY_CACHE_DIR": str(cache),
        "WORLDFOUNDRY_HOME": str(home),
        "WORLDFOUNDRY_DATA_DIR": str(data_dir),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(hfd_dataset_root),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(artifact_dir),
        "WORLDFOUNDRY_MODEL_DIR": str(model_dir),
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": str(model_source),
        "WORLDFOUNDRY_CKPT_DIR": str(ckpt_dir),
        "WORLDFOUNDRY_CKPT_SIBLING_DIR": str(ckpt_sibling),
        "WORLDFOUNDRY_HFD_ROOT": str(hfd_root),
        "WORLDFOUNDRY_CONDA_ROOT": str(conda_root),
        "WORLDFOUNDRY_CONDA_ENVS_ROOT": str(conda_envs_root),
    }


def resolve_worldfoundry_path(value: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expands structural WorldFoundry path tokens (e.g. `${WORLDFOUNDRY_CKPT_DIR}`) and home markers (~).

    Performs precise regex-free variable mapping replacement while preserving subfolder hierarchies.

    Args:
        value: Path containing optional ``$NAME`` or ``${NAME}`` tokens.
        env: Environment mapping used to build/override WorldFoundry tokens.

    Returns:
        Expanded ``Path``. The target is not created or required to exist.
    """
    replacements = worldfoundry_path_tokens(env)
    if env is not None:
        replacements.update(
            {name: str(replacement) for name, replacement in env.items() if name.startswith("WORLDFOUNDRY_")}
        )
    expanded = str(value)
    for name, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        expanded = expanded.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
    return Path(os.path.expandvars(expanded)).expanduser()


def official_runtime_repo_path(
    repo_name: str,
    *,
    specific_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolves the checkout path of an official repository or model library dependency.

    Prioritizes specific system environmental overrides (e.g. custom paths for particular repos)
    before applying standard model source resolutions.
    """
    environ = dict(os.environ if env is None else env)
    if specific_env and environ.get(specific_env):
        return resolve_worldfoundry_path(environ[specific_env], environ)
    if environ.get("WORLDFOUNDRY_GITHUB_REPOS_ROOT"):
        return resolve_worldfoundry_path(Path(environ["WORLDFOUNDRY_GITHUB_REPOS_ROOT"]) / repo_name, environ)
    return resolve_worldfoundry_path(Path("${WORLDFOUNDRY_MODEL_SOURCE_DIR}") / repo_name, environ)


def model_source_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory containing official third-party codebases and model packages."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_MODEL_SOURCE_DIR}", env)


# ──────────────────────────────────────────────────────────────────────────
# Cache / checkpoint / hfd tokens — env first, then repo-relative fallback
# ──────────────────────────────────────────────────────────────────────────


def cache_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the standard WorldFoundry cached download directory."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CACHE_DIR}", env)


def scratch_directory(prefix: str, env: Mapping[str, str] | None = None) -> Path:
    """Create a unique directory under ``${WORLDFOUNDRY_CACHE_DIR}/scratch``.

    Prefer this over ``tempfile.mkdtemp()`` for generated videos and run
    artifacts so long-lived jobs do not fill ``/tmp`` (XC-17).
    """

    root = cache_root_path(env) / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))


def local_data_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root location containing local datasets, evaluation splits, and physical assets."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_DATA_DIR}", env)


def local_model_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory containing local model weights, configs, and adapters."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_MODEL_DIR}", env)


def artifact_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root directory where run scorecards and generated log artifacts are serialized."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_ARTIFACT_DIR}", env)


def checkpoint_root_path(
    *parts: str | Path,
    specific_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolves a target model checkpoint directory path with nested subfolders.

    Avoids host-specific hardcoding by querying general and model-specific variables.

    Args:
        *parts: Child path components appended to the resolved checkpoint root.
        specific_env: Optional model-specific environment variable that takes
            precedence over ``WORLDFOUNDRY_CKPT_DIR``.
        env: Environment mapping used instead of ``os.environ``.

    Returns:
        Resolved checkpoint path without creating it.
    """
    environ = dict(os.environ if env is None else env)
    if specific_env and environ.get(specific_env):
        root = resolve_worldfoundry_path(environ[specific_env], environ)
    else:
        root = resolve_worldfoundry_path("${WORLDFOUNDRY_CKPT_DIR}", environ)
    return root.joinpath(*(Path(part) for part in parts))


def checkpoint_root_candidates(
    *parts: str | Path,
    specific_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return the configured checkpoint root and its conventional singular/plural sibling.

    Existing WorldFoundry installations use both ``ckpt`` and ``ckpts`` next
    to the repository. Keep the configured root first, then allow strictly
    local discovery in the sibling without changing the meaning of
    :func:`checkpoint_root_path` for callers that need one write target.
    """

    primary = checkpoint_root_path(specific_env=specific_env, env=env)
    roots = [primary]
    if primary.name in {"ckpt", "ckpts"}:
        sibling_name = "ckpts" if primary.name == "ckpt" else "ckpt"
        roots.append(primary.with_name(sibling_name))
    suffix = tuple(Path(part) for part in parts)
    return tuple(root.joinpath(*suffix) for root in dict.fromkeys(roots))


def hfd_root_path(*parts: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Resolves the hfd-style local downloader checkpoint directory."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_HFD_ROOT}", env).joinpath(*(Path(part) for part in parts))


def hfd_dataset_root_path(*parts: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Resolve the canonical local root for Hugging Face benchmark datasets."""

    return resolve_worldfoundry_path("${WORLDFOUNDRY_HFD_DATASET_ROOT}", env).joinpath(
        *(Path(part) for part in parts)
    )


def resolve_local_hf_model_path(
    model_id_or_path: str | Path,
    *,
    required_files: Sequence[str] = (),
    revision: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a Hugging Face model strictly from WorldFoundry-local storage.

    The resolver understands direct export directories, hfd-style names such as
    ``owner--repo``, and the Hub cache ``snapshots/<revision>`` layout. It never
    contacts the network, which keeps model initialization deterministic and
    works in inference environments that do not install PyTorch.
    """

    def hfd_download_complete(directory: Path) -> bool:
        """Reject hfd snapshots that are still being transferred or truncated."""

        try:
            for candidate in directory.rglob("*"):
                # Direct hfd exports retain Hugging Face download metadata in
                # ``.cache``.  A stale partial there does not make the fully
                # materialized file in the export root incomplete.
                if ".cache" in candidate.relative_to(directory).parts:
                    continue
                if candidate.name.endswith((".aria2", ".incomplete", ".gstmp")):
                    return False
                if (
                    candidate.name == "._____temp"
                    and candidate.is_dir()
                    and next(candidate.iterdir(), None) is not None
                ):
                    return False
        except OSError:
            return False
        manifest = directory / ".hfd" / "manifest"
        if not manifest.is_file():
            return True
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                size_text, separator, relative_name = line.partition("\t")
                if not separator or not size_text.isdigit() or not relative_name:
                    return False
                target = directory / relative_name
                if not target.is_file() or target.stat().st_size != int(size_text):
                    return False
        except OSError:
            return False
        return True

    value = str(model_id_or_path)
    direct = resolve_worldfoundry_path(value, env)

    environ = dict(os.environ if env is None else env)
    ckpt_roots = checkpoint_root_candidates(env=environ)
    injected_home = Path(environ.get("HOME") or Path.home()).expanduser()
    xdg_cache_home = Path(
        environ.get("XDG_CACHE_HOME") or injected_home / ".cache"
    ).expanduser()
    hf_home = Path(
        environ.get("HF_HOME") or xdg_cache_home / "huggingface"
    ).expanduser()
    hf_hub_cache = Path(
        environ.get("HF_HUB_CACHE")
        or environ.get("HUGGINGFACE_HUB_CACHE")
        or hf_home / "hub"
    ).expanduser()
    roots = [hfd_root_path(env=environ)]
    for local_root in ckpt_roots:
        roots.extend(
            (
                local_root / "hfd_models",
                local_root / "hfd",
                local_root / "huggingface" / "hub",
                local_root / "hf_cache" / "hub",
                local_root,
            )
        )
    roots.insert(0, hf_home / "hub")
    roots.insert(0, hf_hub_cache)
    normalized = value.replace("/", "--")
    leaf = value.rsplit("/", 1)[-1]
    names = tuple(dict.fromkeys((normalized, f"models--{normalized}", value, leaf)))

    def usable(directory: Path) -> Path | None:
        """Accept a Hub-style cache dir only when the download marker and snapshot exist."""

        if not directory.is_dir():
            return None
        if not hfd_download_complete(directory):
            return None
        snapshots = directory / "snapshots"
        if snapshots.is_dir():
            revisions: list[Path] = []
            if revision is not None:
                requested = str(revision).strip()
                if not requested:
                    raise ValueError("Hugging Face revision cannot be empty")
                revisions.append(snapshots / requested)
                revision_ref = directory / "refs" / requested
                if revision_ref.is_file():
                    resolved_ref = revision_ref.read_text(encoding="utf-8").strip()
                    if resolved_ref:
                        revisions.append(snapshots / resolved_ref)
            else:
                main_ref = directory / "refs" / "main"
                if main_ref.is_file():
                    main_revision = main_ref.read_text(encoding="utf-8").strip()
                    if main_revision:
                        revisions.append(snapshots / main_revision)
                revisions.extend(sorted(snapshots.iterdir(), reverse=True))
            for snapshot in dict.fromkeys(revisions):
                if snapshot.is_dir() and all((snapshot / name).is_file() for name in required_files):
                    return snapshot.resolve()
            return None
        if all((directory / name).is_file() for name in required_files):
            return directory.resolve()
        return None

    resolved_direct = usable(direct)
    if resolved_direct is not None:
        return resolved_direct

    checked: list[Path] = []
    for root in dict.fromkeys(path.expanduser() for path in roots):
        for name in names:
            candidate = root / name
            checked.append(candidate)
            resolved = usable(candidate)
            if resolved is not None:
                return resolved
        if root.is_dir():
            for candidate in sorted(root.glob(f"*--{leaf}")):
                checked.append(candidate)
                resolved = usable(candidate)
                if resolved is not None:
                    return resolved

    locations = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"Local Hugging Face assets for {value!r} are missing. Checked:\n{locations}")


def resolve_local_checkpoint_file(
    model_id_or_path: str | Path,
    filename: str,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one checkpoint file from explicit or WorldFoundry-local storage.

    An explicit file path is accepted directly. Directory paths and repository
    identifiers are resolved through :func:`resolve_local_hf_model_path`, so
    incomplete hfd transfers are rejected and this function never contacts a
    model hub.
    """

    direct = resolve_worldfoundry_path(model_id_or_path, env)
    if direct.is_file():
        if direct.name.endswith((".aria2", ".incomplete", ".gstmp")):
            raise FileNotFoundError(f"Checkpoint transfer is incomplete: {direct}")
        partial_markers = tuple(
            direct.with_name(f"{direct.name}{suffix}") for suffix in (".aria2", ".incomplete", ".gstmp", "_.gstmp")
        )
        if direct.stat().st_size <= 0 or any(marker.exists() for marker in partial_markers):
            raise FileNotFoundError(f"Checkpoint transfer is incomplete: {direct}")
        return direct.resolve()
    root = resolve_local_hf_model_path(
        model_id_or_path,
        required_files=(filename,),
        env=env,
    )
    checkpoint = root / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local checkpoint file is missing: {checkpoint}")
    return checkpoint.resolve()


def conda_envs_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the root path containing model-specific python environments and dependencies."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CONDA_ENVS_ROOT}", env)


def conda_root_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolves the base installation folder of the system conda package manager."""
    return resolve_worldfoundry_path("${WORLDFOUNDRY_CONDA_ROOT}", env)


def resolve_package_path(*parts: str | Path) -> Path:
    """Resolves a subpath relative to the active `worldfoundry` package source directory."""
    return package_root().joinpath(*(Path(part) for part in parts))


def resolve_data_path(*parts: str | Path) -> Path:
    """Resolves a subpath under the internal `worldfoundry/data` asset directory."""
    return resolve_package_path("data", *parts)


def repo_relative_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Squeezes a path relative to the active repository parent to keep logs clean and short.

    If the path is outside the repository tree, falls back gracefully to a fully resolved POSIX string.
    """
    resolved = Path(path).expanduser().resolve()
    repo = Path(root).expanduser().resolve() if root is not None else project_root()
    try:
        return resolved.relative_to(repo).as_posix()
    except ValueError:
        return resolved.as_posix()


# Canonical roots consumed by runtime/evaluation. Kept here so lower layers
# do not import ``worldfoundry.evaluation.utils``.
REPO_ROOT = project_root()
DATA_ROOT = package_data_root()
BENCHMARKS_DATA_ROOT = DATA_ROOT / "benchmarks"


__all__ = [
    "checkpoint_root_candidates",
    "checkpoint_root_path",
    "artifact_root_path",
    "BENCHMARKS_DATA_ROOT",
    "DATA_ROOT",
    "REPO_ROOT",
    "cache_root_path",
    "conda_envs_root_path",
    "conda_root_path",
    "hfd_dataset_root_path",
    "hfd_root_path",
    "local_data_root_path",
    "local_model_root_path",
    "resolve_local_checkpoint_file",
    "resolve_local_hf_model_path",
    "model_source_root_path",
    "package_module_root",
    "official_runtime_repo_path",
    "package_root",
    "project_root",
    "repo_relative_path",
    "resolve_data_path",
    "resolve_package_path",
    "resolve_worldfoundry_path",
    "worldfoundry_path_tokens",
]
