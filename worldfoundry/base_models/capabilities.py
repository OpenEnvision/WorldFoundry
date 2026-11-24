"""Registry and readiness checks for reusable base-model capabilities.

A *capability* is an in-tree foundation model (depth, segmentation, judges, …)
or a benchmark data asset that runners share. Its declaration — Python imports,
pip packages, checkpoints and datasets with their expected layout — lives in
``worldfoundry/data/base_models``:

* ``capabilities/*.yaml`` — one mapping of capability id → spec per file.
* ``stacks.yaml`` — named bundles of capabilities; members may name other stacks.
* ``benchmark_assets.yaml`` — benchmark id → data-asset capabilities it stages.

This module loads that data on first use and answers readiness questions:
which imports and assets are present, and which pip/download/export commands
would repair the rest. Nothing here installs or downloads.

A capability entry takes the fields of :class:`BaseModelCapability` (minus
``id``, which is the mapping key) and a list of ``assets`` with the fields of
:class:`BaseModelAsset`. Omitted fields default as follows:

* ``canonical_path`` — this module, for data-only capabilities;
  ``canonical_owner`` — ``canonical_path`` as a dotted module name.
* ``asset_env`` — the union of the assets' ``env`` lists.
* ``install_packages`` — the import names themselves.

Three templates cover the repetitive benchmark entries (set ``template:``):

* ``hf_dataset`` — ``datasets: [{id, env?, repo_id, local_name, revision?,
  alternate_local_names?, min_file_count?, required?, role?, note?}]``;
  each becomes a Hugging Face dataset directory under the HFD dataset root.
* ``in_tree_source`` — ``path, kind?, env?, upstream, upstream_revision?,
  role?, note?``; an upstream benchmark repository that is vendored in-tree
  and must never be cloned during readiness. ``env`` is the runner's
  override for the runtime root.
* ``external_official_runtime`` — ``upstream, root_env, checkout, dataset?,
  requires?, note?``; an official simulator stack installed outside
  WorldFoundry whose rollout results WorldFoundry scores.

Unknown fields, duplicate ids, and stack or benchmark references to unknown
capabilities fail at import with :class:`CapabilityRegistryError`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import os
import shlex
import signal
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.core.io.paths import worldfoundry_path_tokens

REPO_ROOT = Path(
    os.environ.get("WORLDFOUNDRY_REPO_ROOT") or worldfoundry_path_tokens()["WORLDFOUNDRY_REPO_ROOT"]
).expanduser()
REGISTRY_DIR = Path(__file__).resolve().parents[1] / "data" / "base_models"
# Owner recorded for data-only capabilities that have no model code of their own.
_REGISTRY_OWNER_PATH = "worldfoundry/base_models/capabilities.py"


# ── Path tokens ─────────────────────────────────────────────────────────


def _path_tokens(env: Mapping[str, str] | None = None) -> dict[str, str]:
    environ = os.environ if env is None else env
    core_tokens = worldfoundry_path_tokens(environ)
    repo_root = str(Path(environ.get("WORLDFOUNDRY_REPO_ROOT") or core_tokens["WORLDFOUNDRY_REPO_ROOT"]).expanduser())
    cache_root = Path(core_tokens["WORLDFOUNDRY_CACHE_DIR"]).expanduser()
    hfd_root = Path(core_tokens["WORLDFOUNDRY_HFD_ROOT"]).expanduser()
    workspace_root = Path(
        environ.get("WORLDFOUNDRY_WORKSPACE_ROOT") or environ.get("WORLDFOUNDRY_BENCH_ROOT") or hfd_root.parent
    ).expanduser()
    worldscore_asset_dir = Path(
        environ.get("WORLDFOUNDRY_WORLDSCORE_ASSET_DIR") or cache_root / "assets" / "worldscore_official"
    ).expanduser()
    worldscore_dataset_dir = Path(
        environ.get("WORLDFOUNDRY_WORLDSCORE_DATASET_DIR") or worldscore_asset_dir / "data" / "WorldScore-Dataset"
    ).expanduser()
    worldscore_checkpoint_dir = Path(
        environ.get("WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR") or worldscore_asset_dir / "metrics" / "checkpoints"
    ).expanduser()
    return {
        "WORLDFOUNDRY_REPO_ROOT": repo_root,
        "WORLDFOUNDRY_WORKSPACE_ROOT": str(workspace_root),
        "WORLDFOUNDRY_BENCH_ROOT": str(Path(environ.get("WORLDFOUNDRY_BENCH_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_CACHE_DIR": str(cache_root),
        # Roots the VBench / VBench-2.0 runtimes search for metric checkpoints.
        "WORLDFOUNDRY_VBENCH_CACHE_DIR": str(
            Path(environ.get("WORLDFOUNDRY_VBENCH_CACHE_DIR") or cache_root / "models" / "vbench").expanduser()
        ),
        "WORLDFOUNDRY_VBENCH2_CACHE_DIR": str(
            Path(environ.get("WORLDFOUNDRY_VBENCH2_CACHE_DIR") or cache_root / "models" / "vbench2").expanduser()
        ),
        "WORLDFOUNDRY_DATA_DIR": str(Path(core_tokens["WORLDFOUNDRY_DATA_DIR"]).expanduser()),
        "WORLDFOUNDRY_HFD_ROOT": str(hfd_root),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(Path(core_tokens["WORLDFOUNDRY_HFD_DATASET_ROOT"]).expanduser()),
        "WORLDFOUNDRY_CKPT_DIR": core_tokens["WORLDFOUNDRY_CKPT_DIR"],
        "WORLDFOUNDRY_MODEL_DIR": core_tokens["WORLDFOUNDRY_MODEL_DIR"],
        "WORLDFOUNDRY_WORLDSCORE_ASSET_DIR": str(worldscore_asset_dir),
        "WORLDFOUNDRY_WORLDSCORE_DATASET_DIR": str(worldscore_dataset_dir),
        "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR": str(worldscore_checkpoint_dir),
    }


def _substitute_tokens(text: str, tokens: Mapping[str, str]) -> str:
    for name, replacement in tokens.items():
        text = text.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
    return text


def expand_capability_path(value: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expand known WorldFoundry path tokens and tilde, anchoring relative paths at the repository root.

    Use the supplied environment mapping, or os.environ when omitted, to derive
    known token defaults; unknown tokens remain literal. No paths are created
    or checked, and the result is not resolved through symlinks.
    """
    tokens = _path_tokens(env)
    path = Path(_substitute_tokens(str(value), tokens)).expanduser()
    return path if path.is_absolute() else Path(tokens["WORLDFOUNDRY_REPO_ROOT"]) / path


# ── Probes ──────────────────────────────────────────────────────────────


def _has_min_files(path: Path, min_file_count: int | None) -> bool:
    if min_file_count is None:
        return True
    count = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    count += 1
                    if count >= min_file_count:
                        return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    return list({str(path): path for path in paths}.values())


@lru_cache(maxsize=256)
def _python_import_available(python_executable: str, module_name: str, pythonpath: str) -> bool:
    executable = Path(python_executable).expanduser()
    if not executable.is_file():
        return False
    timeout = float(os.environ.get("WORLDFOUNDRY_EXTERNAL_IMPORT_PROBE_TIMEOUT", "3"))
    kill_timeout = float(os.environ.get("WORLDFOUNDRY_EXTERNAL_IMPORT_PROBE_KILL_TIMEOUT", "0.5"))
    command = [
        str(executable),
        "-c",
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
        module_name,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
        )
    except OSError:
        return False
    try:
        return process.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        # Some broken external envs can block in uninterruptible filesystem
        # or loader state. Readiness must report that import as unavailable
        # instead of blocking the whole benchmark inventory.
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=kill_timeout)
        return False


def _import_available(module_name: str, python_executable: str | Path | None = None) -> bool:
    if python_executable is None:
        try:
            return importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:  # a dotted name whose parent package is missing
            return False
    return _python_import_available(str(python_executable), module_name, os.environ.get("PYTHONPATH", ""))


def _hf_download_command(
    repo_id: str,
    *,
    filename: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    local_dir: str | Path,
) -> list[str]:
    command = ["hf", "download", repo_id]
    if filename:
        command.append(filename)
    if repo_type:
        command.extend(["--repo-type", repo_type])
    if revision:
        command.extend(["--revision", revision])
    command.extend(["--local-dir", str(local_dir)])
    return command


def _curl_download_command(url: str, output_path: str | Path) -> list[str]:
    return ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2", url, "-o", str(output_path)]


def _to_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    return value


# ── Declarations ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaseModelAsset:
    """Checkpoint, model directory, or benchmark data asset owned by a reusable capability.

    ``kind`` is ``"dir"`` or ``"file"``. A path is ready when it exists with that
    kind and meets the size, file-count and required-entry constraints.
    """

    id: str
    kind: str
    role: str
    env: tuple[str, ...]
    local_path: str
    alternate_paths: tuple[str, ...] = ()
    hf_repo_id: str | None = None
    hf_repo_type: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = None
    git_repo_url: str | None = None
    git_revision: str | None = None
    download_url: str | None = None
    download_command_template: str | None = None
    manual_action: str | None = None
    min_size_bytes: int | None = None
    min_file_count: int | None = None
    required_files: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    required_any_paths: tuple[str, ...] = ()
    required: bool = True
    note: str | None = None

    def is_ready(self, path: Path) -> bool:
        """Return whether ``path`` satisfies this asset's kind and layout constraints."""
        try:
            if self.kind != "dir":
                return path.is_file() and (self.min_size_bytes is None or path.stat().st_size >= self.min_size_bytes)
            return (
                path.is_dir()
                and all((path / name).is_file() for name in self.required_files)
                and all((path / name).exists() for name in self.required_paths)
                and (not self.required_any_paths or any((path / name).exists() for name in self.required_any_paths))
                and _has_min_files(path, self.min_file_count)
            )
        except OSError:
            return False

    def local_dir(self, env: Mapping[str, str] | None = None) -> Path:
        """Return the expanded local directory, or the parent of a file asset path.

        Use env (os.environ by default) for path tokens; asset-specific environment
        overrides are not consulted and the directory is not created.
        """
        local_path = expand_capability_path(self.local_path, env)
        return local_path if self.kind == "dir" else local_path.parent

    def download_command(self, env: Mapping[str, str] | None = None) -> list[str] | None:
        """Build download argv for the declared local path without executing it.

        Expand path tokens using env, or os.environ by default. A shell template
        takes precedence; Git sources and assets without a download source return
        None. URL downloads use mkdir and curl; Hugging Face downloads honor the
        repo type, revision and filename. Model directories with required file
        constraints download those files and the first required-any path only.
        """
        if self.download_command_template:
            return ["bash", "-lc", _substitute_tokens(self.download_command_template, _path_tokens(env))]
        if self.git_repo_url:
            # Source repositories must be integrated in-tree. Readiness/job setup
            # should never clone or update Git repositories.
            return None
        if not self.hf_repo_id:
            if not self.download_url:
                return None
            local_path = expand_capability_path(self.local_path, env)
            curl = shlex.join(_curl_download_command(self.download_url, local_path))
            return ["bash", "-lc", f"mkdir -p {shlex.quote(str(local_path.parent))} && {curl}"]
        local_dir = str(self.local_dir(env))
        hf_options = {"repo_type": self.hf_repo_type, "revision": self.hf_revision, "local_dir": local_dir}
        if (
            not self.hf_filename
            and self.kind == "dir"
            and self.role in {"model_dir", "text_encoder_model_dir"}
            and (self.required_files or self.required_any_paths)
        ):
            filenames = [*self.required_files, *self.required_any_paths[:1]]
            commands = [
                f"mkdir -p {shlex.quote(local_dir)}",
                *(shlex.join(_hf_download_command(self.hf_repo_id, filename=name, **hf_options)) for name in filenames),
            ]
            return ["bash", "-lc", " && ".join(commands)]
        return _hf_download_command(self.hf_repo_id, filename=self.hf_filename, **hf_options)

    def check(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Inspect environment overrides, then local and alternate paths, for a ready asset.

        Use os.environ when env is omitted. Check file size or directory contents
        against the asset constraints and select the first ready path. Return
        metadata, per-variable status, candidate paths, readiness and suggested
        download/export commands; export to the matched path or the local default.
        Read the filesystem without creating files, downloading or modifying env.
        """
        environ = os.environ if env is None else env
        local_path = expand_capability_path(self.local_path, environ)
        alternate_paths = _unique_paths(expand_capability_path(path, environ) for path in self.alternate_paths)
        env_status = []
        env_matches: list[Path] = []
        for name in self.env:
            raw_value = environ.get(name)
            path = expand_capability_path(raw_value, environ) if raw_value else None
            ready = path is not None and self.is_ready(path)
            if ready:
                env_matches.append(path)
            env_status.append(
                {"name": name, "present": bool(raw_value), "path": None if path is None else str(path), "ready": ready}
            )

        candidates = _unique_paths([local_path, *alternate_paths])
        matched = env_matches[0] if env_matches else next((path for path in candidates if self.is_ready(path)), None)
        download_command = self.download_command(environ)
        export_target = matched or local_path
        return {
            "id": self.id,
            "kind": self.kind,
            "role": self.role,
            "required": self.required,
            "env": list(self.env),
            "env_status": env_status,
            "local_path": str(local_path),
            "alternate_paths": [str(path) for path in alternate_paths],
            "candidate_paths": [str(path) for path in candidates],
            "matched_path": None if matched is None else str(matched),
            "env_ready": bool(env_matches),
            "ready": matched is not None,
            "hf_repo_id": self.hf_repo_id,
            "hf_repo_type": self.hf_repo_type,
            "hf_filename": self.hf_filename,
            "hf_revision": self.hf_revision,
            "git_repo_url": self.git_repo_url,
            "git_revision": self.git_revision,
            "download_command_template": self.download_command_template,
            "manual_action": self.manual_action,
            "min_file_count": self.min_file_count,
            "required_files": list(self.required_files),
            "required_paths": list(self.required_paths),
            "required_any_paths": list(self.required_any_paths),
            "download_command": download_command,
            "download_command_text": None if download_command is None else shlex.join(download_command),
            "export_commands": [f"export {name}={shlex.quote(str(export_target))}" for name in self.env],
            "note": self.note,
        }


@dataclass(frozen=True)
class BaseModelCapability:
    """Reusable in-tree foundation-model capability used by benchmark runtimes.

    ``install_packages`` pairs positionally with ``package_imports``; when it is
    empty each pip-installable import installs under its own name.
    """

    id: str
    family: str
    canonical_owner: str
    canonical_path: str
    package_imports: tuple[str, ...] = ()
    install_packages: tuple[str, ...] = ()
    non_pip_imports: tuple[str, ...] = ()
    manual_install_actions: tuple[str, ...] = ()
    asset_env: tuple[str, ...] = ()
    asset_env_policy: str = "any"
    assets: tuple[BaseModelAsset, ...] = ()
    gpu_required: bool = True

    def owner_path(self) -> Path:
        """Return canonical_path unchanged if absolute, otherwise relative to the module-level REPO_ROOT.

        Do not expand tokens or tilde, resolve symlinks, or check existence.
        """
        path = Path(self.canonical_path)
        return path if path.is_absolute() else REPO_ROOT / path

    def pip_package_by_import(self) -> dict[str, str]:
        """Map pip-installable imports to package names, excluding non-pip imports and empty packages.

        Equal-length import/package lists pair positionally before exclusions.
        Otherwise pair packages with the filtered imports, falling back to each
        import name when no corresponding package is supplied. No packages are installed.
        """
        non_pip = set(self.non_pip_imports)
        if len(self.install_packages) == len(self.package_imports):
            pairs: Iterable[tuple[str, str]] = zip(self.package_imports, self.install_packages, strict=True)
        else:
            pip_imports = [name for name in self.package_imports if name not in non_pip]
            packages = self.install_packages or tuple(pip_imports)
            pairs = ((name, packages[i] if i < len(packages) else name) for i, name in enumerate(pip_imports))
        return {name: package for name, package in pairs if name not in non_pip and package}

    def to_dict(self) -> dict[str, Any]:
        """Serialize capability metadata and asset specifications into a JSON-compatible dictionary.

        Convert tuples to lists and derive install_packages from the import mapping.
        Keep asset paths unexpanded; do not probe imports, inspect assets or install anything.
        """
        payload = _to_json(self)
        payload["install_packages"] = list(self.pip_package_by_import().values())
        return payload


@dataclass(frozen=True)
class BaseModelStack:
    """Named bundle of reusable base-model capabilities."""

    id: str
    family: str
    capability_ids: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stack ID, family, ordered capability IDs and note as JSON-compatible metadata."""
        return _to_json(self)


# ── Registry loading ────────────────────────────────────────────────────


class CapabilityRegistryError(ValueError):
    """Raised when ``worldfoundry/data/base_models`` contains an invalid declaration."""


def _build(cls: type, spec: Mapping[str, Any], where: str, **values: Any) -> Any:
    """Construct dataclass ``cls`` from a YAML mapping, rejecting unknown keys and converting lists to tuples."""

    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(spec) - set(fields))
    if unknown:
        raise CapabilityRegistryError(f"{where}: unknown field(s) {', '.join(unknown)}")
    for name, value in spec.items():
        if name in values:
            continue
        if isinstance(value, str) and str(fields[name].type).startswith("tuple"):
            value = (value,)
        values[name] = tuple(value) if isinstance(value, list) else value
    try:
        return cls(**values)
    except TypeError as exc:
        raise CapabilityRegistryError(f"{where}: {exc}") from exc


def _owner_from_path(canonical_path: str) -> str:
    return canonical_path.removesuffix(".py").replace("/", ".")


def _plain_capability(capability_id: str, spec: Mapping[str, Any], where: str) -> BaseModelCapability:
    if not isinstance(spec.get("assets", []), list):
        raise CapabilityRegistryError(f"{where}: assets must be a list")
    assets = tuple(
        _build(BaseModelAsset, asset, f"{where} asset {asset.get('id', index)!r}")
        for index, asset in enumerate(spec.get("assets", []))
    )
    canonical_path = spec.get("canonical_path", _REGISTRY_OWNER_PATH)
    return _build(
        BaseModelCapability,
        spec,
        where,
        id=capability_id,
        canonical_path=canonical_path,
        canonical_owner=spec.get("canonical_owner") or _owner_from_path(canonical_path),
        asset_env=tuple(spec["asset_env"])
        if "asset_env" in spec
        else tuple(dict.fromkeys(e for a in assets for e in a.env)),
        assets=assets,
    )


def _hf_dataset_capability(capability_id: str, spec: Mapping[str, Any], where: str) -> BaseModelCapability:
    """Expand ``template: hf_dataset`` — Hugging Face dataset repos staged under the HFD dataset root."""

    _require_keys(spec, {"datasets"}, where)
    assets = []
    for item in spec["datasets"]:
        _require_keys(
            item,
            {
                "id",
                "env",
                "repo_id",
                "local_name",
                "revision",
                "alternate_local_names",
                "min_file_count",
                "required",
                "role",
                "note",
            },
            f"{where} dataset {item.get('id')!r}",
        )
        alternates = tuple(
            path
            for name in item.get("alternate_local_names", ())
            for path in (
                f"${{WORLDFOUNDRY_HFD_DATASET_ROOT}}/{name}",
                f"${{WORLDFOUNDRY_WORKSPACE_ROOT}}/data/hfd_datasets/{name}",
            )
        )
        assets.append(
            BaseModelAsset(
                id=item["id"],
                kind="dir",
                role=item.get("role", "benchmark_dataset"),
                env=(item["env"],) if item.get("env") else (),
                local_path=f"${{WORLDFOUNDRY_HFD_DATASET_ROOT}}/{item['local_name']}",
                alternate_paths=alternates,
                hf_repo_id=item["repo_id"],
                hf_repo_type="dataset",
                hf_revision=item.get("revision"),
                min_file_count=item.get("min_file_count", 1),
                required=item.get("required", True),
                note=item.get("note"),
            )
        )
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner=_owner_from_path(_REGISTRY_OWNER_PATH),
        canonical_path=_REGISTRY_OWNER_PATH,
        asset_env=tuple(env for asset in assets for env in asset.env),
        asset_env_policy="all" if len(assets) > 1 else "any",
        assets=tuple(assets),
        gpu_required=False,
    )


def _in_tree_source_capability(capability_id: str, spec: Mapping[str, Any], where: str) -> BaseModelCapability:
    """Expand ``template: in_tree_source`` — an upstream benchmark repo vendored into this repository."""

    _require_keys(spec, {"path", "kind", "env", "upstream", "upstream_revision", "role", "note"}, where)
    kind = spec.get("kind", "dir")
    path = spec["path"]
    upstream = spec["upstream"]
    reference = f"{upstream} at {spec['upstream_revision']}" if spec.get("upstream_revision") else upstream
    note = (
        "In-tree WorldFoundry integration replacing an upstream source repository. "
        f"Upstream reference: {reference}. "
        "WorldFoundry base_models does not clone external source repositories during readiness or job setup."
    )
    if spec.get("note"):
        note = f"{note} Original upstream context: {spec['note']}"
    asset = BaseModelAsset(
        id=f"{capability_id}_in_tree",
        kind=kind,
        role=spec.get("role", "benchmark_in_tree_source"),
        env=(spec["env"],) if spec.get("env") else (),
        local_path=f"${{WORLDFOUNDRY_REPO_ROOT}}/{path}",
        min_file_count=1 if kind == "dir" else None,
        manual_action=(
            f"Restore the checked-in WorldFoundry integration at {path}; "
            f"do not stage or clone {upstream} for base_models readiness."
        ),
        note=note,
    )
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner=_owner_from_path(_REGISTRY_OWNER_PATH),
        canonical_path=_REGISTRY_OWNER_PATH,
        assets=(asset,),
        gpu_required=False,
    )


def _external_official_runtime_capability(
    capability_id: str, spec: Mapping[str, Any], where: str
) -> BaseModelCapability:
    """Expand ``template: external_official_runtime`` — an official simulator stack run outside WorldFoundry.

    WorldFoundry scores the official rollout results; it never clones or installs the runtime itself.
    """

    _require_keys(spec, {"upstream", "root_env", "checkout", "dataset", "requires", "note"}, where)
    upstream = spec["upstream"]
    requires = f" ({', '.join(spec['requires'])})" if spec.get("requires") else ""
    assets = [
        BaseModelAsset(
            id=f"{capability_id}_checkout",
            kind="dir",
            role="external_official_runtime",
            env=(spec["root_env"],),
            local_path=spec["checkout"],
            min_file_count=1,
            manual_action=(
                f"Install the official runtime from {upstream}{requires} and set {spec['root_env']} to its "
                "checkout; WorldFoundry scores its official rollout results and does not clone it."
            ),
            note=spec.get("note"),
        )
    ]
    if spec.get("dataset"):
        assets.append(
            BaseModelAsset(
                id=f"{capability_id}_dataset",
                kind="dir",
                role="external_benchmark_dataset",
                env=(),
                local_path=spec["dataset"],
                min_file_count=1,
                manual_action=f"Stage the benchmark assets the official {upstream} runtime reads under {spec['dataset']}.",
            )
        )
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner=_owner_from_path(_REGISTRY_OWNER_PATH),
        canonical_path=_REGISTRY_OWNER_PATH,
        assets=tuple(assets),
        gpu_required=False,
    )


_TEMPLATES = {
    "hf_dataset": _hf_dataset_capability,
    "in_tree_source": _in_tree_source_capability,
    "external_official_runtime": _external_official_runtime_capability,
}


def _require_keys(spec: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(spec) - allowed - {"template"})
    if unknown:
        raise CapabilityRegistryError(f"{where}: unknown field(s) {', '.join(unknown)}")


def _read_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader) or {}


def _load_capabilities(directory: Path) -> dict[str, BaseModelCapability]:
    capabilities: dict[str, BaseModelCapability] = {}
    for path in sorted(directory.glob("*.yaml")):
        for capability_id, spec in _read_yaml(path).items():
            where = f"{path.relative_to(REGISTRY_DIR)}: {capability_id}"
            if capability_id in capabilities:
                raise CapabilityRegistryError(f"{where}: duplicate capability id")
            template = spec.get("template")
            if template is None:
                capabilities[capability_id] = _plain_capability(capability_id, spec, where)
            elif template in _TEMPLATES:
                capabilities[capability_id] = _TEMPLATES[template](capability_id, spec, where)
            else:
                raise CapabilityRegistryError(f"{where}: unknown template {template!r}")
    return capabilities


def _load_stacks(path: Path, capabilities: Mapping[str, BaseModelCapability]) -> dict[str, BaseModelStack]:
    specs = _read_yaml(path)

    def expand(stack_id: str, trail: tuple[str, ...]) -> list[str]:
        if stack_id in trail:
            raise CapabilityRegistryError(f"stacks.yaml: cycle {' -> '.join((*trail, stack_id))}")
        members: list[str] = []
        for member in specs[stack_id]["capabilities"]:
            if member in specs:
                members.extend(expand(member, (*trail, stack_id)))
            elif member in capabilities:
                members.append(member)
            else:
                raise CapabilityRegistryError(f"stacks.yaml: {stack_id}: unknown capability or stack {member!r}")
        return list(dict.fromkeys(members))

    stacks = {}
    for stack_id, spec in specs.items():
        _require_keys(spec, {"family", "capabilities", "note"}, f"stacks.yaml: {stack_id}")
        stacks[stack_id] = BaseModelStack(
            id=stack_id, family=spec["family"], capability_ids=tuple(expand(stack_id, ())), note=spec["note"]
        )
    return stacks


def _load_benchmark_assets(path: Path, capabilities: Mapping[str, BaseModelCapability]) -> dict[str, tuple[str, ...]]:
    mapping = {str(benchmark_id): tuple(ids) for benchmark_id, ids in _read_yaml(path).items()}
    for benchmark_id, ids in mapping.items():
        unknown = [capability_id for capability_id in ids if capability_id not in capabilities]
        if unknown:
            raise CapabilityRegistryError(f"benchmark_assets.yaml: {benchmark_id}: unknown capability {unknown}")
    return mapping


@dataclass(frozen=True)
class _Registry:
    capabilities: dict[str, BaseModelCapability]
    stacks: dict[str, BaseModelStack]
    benchmark_assets: dict[str, tuple[str, ...]]


@lru_cache(maxsize=1)
def _registry() -> _Registry:
    capabilities = _load_capabilities(REGISTRY_DIR / "capabilities")
    return _Registry(
        capabilities=capabilities,
        stacks=_load_stacks(REGISTRY_DIR / "stacks.yaml", capabilities),
        benchmark_assets=_load_benchmark_assets(REGISTRY_DIR / "benchmark_assets.yaml", capabilities),
    )


_REGISTRY_ATTRIBUTES = {
    "BASE_MODEL_CAPABILITIES": "capabilities",
    "BASE_MODEL_STACKS": "stacks",
    "BENCHMARK_DATA_ASSET_CAPABILITIES": "benchmark_assets",
}


def __getattr__(name: str) -> Any:
    # The registry is read on first use, so importing this module (the CLI
    # does so for every command) stays cheap on slow network filesystems.
    if name in _REGISTRY_ATTRIBUTES:
        return getattr(_registry(), _REGISTRY_ATTRIBUTES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@lru_cache(maxsize=1)
def base_model_stack_ids() -> tuple[str, ...]:
    """Return the declared stack ids from ``stacks.yaml`` alone, without loading or validating capabilities."""

    return tuple(sorted(_read_yaml(REGISTRY_DIR / "stacks.yaml")))


# ── Lookup ──────────────────────────────────────────────────────────────


def benchmark_data_asset_capability_ids(benchmark_id: str) -> tuple[str, ...]:
    """Return benchmark-owned data asset capabilities staged with base-model assets."""

    return _registry().benchmark_assets.get(str(benchmark_id), ())


def benchmark_base_model_dependency_ids(
    benchmark_id: str,
    base_model_dependencies: tuple[str, ...] | list[str] = (),
    optional_base_model_dependencies: tuple[str, ...] | list[str] = (),
    *,
    include_optional: bool = False,
    include_data_assets: bool = True,
) -> tuple[str, ...]:
    """Return base-model stack/capability ids needed to prepare a benchmark."""

    ids: list[str] = [str(item) for item in base_model_dependencies if str(item).strip()]
    if include_optional:
        ids.extend(str(item) for item in optional_base_model_dependencies if str(item).strip())
    if include_data_assets:
        ids.extend(benchmark_data_asset_capability_ids(benchmark_id))
    return tuple(dict.fromkeys(ids))


def resolve_base_model_capability_ids(capability_or_stack_ids: list[str] | tuple[str, ...]) -> list[str]:
    """Expand capability and stack IDs into a deduplicated list in first-seen order.

    Convert each input to str and expand stacks by their declared member order.
    Raise KeyError with the available IDs for an unknown input; an empty input
    returns an empty list.
    """
    registry = _registry()
    resolved: list[str] = []
    for raw_id in capability_or_stack_ids:
        item_id = str(raw_id)
        if item_id in registry.capabilities:
            resolved.append(item_id)
        elif item_id in registry.stacks:
            resolved.extend(registry.stacks[item_id].capability_ids)
        else:
            available = ", ".join(sorted([*registry.capabilities, *registry.stacks]))
            raise KeyError(f"unknown base-model capability or stack {item_id!r}; available: {available}")
    return list(dict.fromkeys(resolved))


def get_base_model_capability(capability_id: str) -> BaseModelCapability:
    """Return the registered capability object for an exact ID, without checking readiness.

    Raise KeyError listing available capability IDs when the ID is unknown;
    stack IDs are not expanded.
    """
    try:
        return _registry().capabilities[capability_id]
    except KeyError as exc:
        available = ", ".join(sorted(_registry().capabilities))
        raise KeyError(f"unknown base-model capability {capability_id!r}; available: {available}") from exc


# ── VBench metric assets ────────────────────────────────────────────────


def _vbench_metric_asset_status(asset_id: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    capability = get_base_model_capability("vbench_metric_checkpoint_assets")
    asset = next((item for item in capability.assets if item.id == asset_id), None)
    if asset is None:
        raise KeyError(f"unknown VBench asset id: {asset_id}")
    status = asset.check(env)
    if status.get("ready") and status.get("matched_path"):
        return status
    candidates = "\n".join(f"  - {path}" for path in status.get("candidate_paths", ()))
    exports = "\n".join(f"  {command}" for command in status.get("export_commands", ()))
    raise FileNotFoundError(
        f"Required VBench asset {asset_id!r} is not staged.\n"
        f"Candidate paths:\n{candidates or '  <none>'}\n"
        f"Environment override:\n{exports or '  <none>'}"
    )


def vbench_asset_path(asset_id: str, *, env: Mapping[str, str] | None = None) -> Path:
    """Resolve a staged VBench/VBench++ metric asset to a local path."""

    return Path(_vbench_metric_asset_status(asset_id, env)["matched_path"])


def vbench_dino_hub_config(
    *,
    env: Mapping[str, str] | None = None,
    read_frame: bool | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Return a local ``torch.hub.load`` config for the DINO ViT-B/16 metric."""

    source = Path(_vbench_metric_asset_status("vbench_dino_source", env)["matched_path"])
    checkpoint = Path(_vbench_metric_asset_status("vbench_dino_vitb16_checkpoint", env)["matched_path"])
    config: dict[str, Any] = {
        "repo_or_dir": str(source),
        "path": str(checkpoint),
        "model": "dino_vitb16",
        "source": "local",
    }
    if read_frame is not None:
        config["read_frame"] = read_frame
    if resolution is not None:
        config["resolution"] = resolution
    return config


# ── Readiness ───────────────────────────────────────────────────────────


def check_base_model_capability(
    capability_id: str,
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
    check_imports: bool = True,
) -> dict[str, Any]:
    """Report owner-path, package-discovery and required-asset readiness with repair suggestions.

    Use env (os.environ by default) for assets, not for import probes. By default,
    find package specs in the current interpreter; python_executable selects
    cached subprocess probes instead. Dotted spec lookup may import parents.
    With check_imports=False, report imports as skipped and do not gate on them.
    Read asset paths and owner existence, but do not install or download.
    The returned metadata includes statuses, repair_hints and ok; gpu_required
    is metadata only, not a GPU probe. Unknown capability IDs raise KeyError.
    """
    environ = os.environ if env is None else env
    capability = get_base_model_capability(capability_id)
    imports = [
        {
            "name": name,
            "available": _import_available(name, python_executable) if check_imports else None,
            "python_executable": None if python_executable is None else str(python_executable),
            "skipped": not check_imports,
        }
        for name in capability.package_imports
    ]
    asset_env = [{"name": name, "present": bool(environ.get(name))} for name in capability.asset_env]
    assets = [asset.check(environ) for asset in capability.assets]
    if assets:
        required_assets_ready = all(asset["ready"] for asset in assets if asset["required"])
    elif not asset_env:
        required_assets_ready = True
    elif capability.asset_env_policy == "all":
        required_assets_ready = all(item["present"] for item in asset_env)
    else:
        required_assets_ready = any(item["present"] for item in asset_env)

    package_by_import = capability.pip_package_by_import()
    missing_imports = [str(item["name"]) for item in imports if check_imports and not item["available"]]
    missing_non_pip_imports = [name for name in missing_imports if name in capability.non_pip_imports]
    missing_packages = [
        package
        for name in missing_imports
        if name not in capability.non_pip_imports and (package := package_by_import.get(name, name))
    ]
    manual_install_actions = list(capability.manual_install_actions) if missing_non_pip_imports else []
    missing_assets = [asset for asset in assets if asset["required"] and not asset["ready"]]
    manual_asset_actions = [
        asset.get("manual_action") or f"stage {asset['id']} at one of: {', '.join(asset['candidate_paths'])}"
        for asset in missing_assets
        if not asset["download_command_text"]
    ]
    repair_hints = {
        "pip_install_packages": missing_packages,
        "pip_command": None if not missing_packages else "python -m pip install " + " ".join(missing_packages),
        "missing_non_pip_imports": missing_non_pip_imports,
        "manual_install_actions": manual_install_actions,
        "download_command_argvs": [asset["download_command"] for asset in missing_assets if asset["download_command"]],
        "download_commands": [
            asset["download_command_text"] for asset in missing_assets if asset["download_command_text"]
        ],
        "export_commands": [
            command
            for asset in assets
            if asset["required"] and not asset["env_ready"]
            for command in asset["export_commands"]
        ],
        "manual_actions": [*manual_install_actions, *manual_asset_actions],
    }
    owner_path = capability.owner_path()
    imports_ok = not check_imports or all(item["available"] for item in imports)
    return {
        **capability.to_dict(),
        "owner_path": str(owner_path),
        "owner_path_exists": owner_path.exists(),
        "imports": imports,
        "import_checks_skipped": not check_imports,
        "asset_env_status": asset_env,
        "asset_status": assets,
        "required_assets_ready": required_assets_ready,
        "repair_hints": repair_hints,
        "ok": owner_path.exists() and imports_ok and required_assets_ready,
    }


def check_base_model_dependencies(
    capability_ids: list[str] | tuple[str, ...],
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
    check_imports: bool = True,
) -> dict[str, Any]:
    """Expand stacks and aggregate capability readiness checks into one status dictionary.

    Preserve requested IDs and stack IDs, deduplicate resolved capabilities,
    and forward env, python_executable and check_imports to each check.
    Defaults use process environment and current-interpreter package discovery.
    Return per-capability checks and their combined ok (True for empty input).
    Unknown IDs raise KeyError; checks inspect paths and may probe subprocesses
    but do not install or download.
    """
    checks = [
        check_base_model_capability(
            capability_id, env=env, python_executable=python_executable, check_imports=check_imports
        )
        for capability_id in resolve_base_model_capability_ids(capability_ids)
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "python_executable": None if python_executable is None else str(python_executable),
        "import_checks_skipped": not check_imports,
        "requested_ids": [str(capability_id) for capability_id in capability_ids],
        "stack_ids": [str(item) for item in capability_ids if str(item) in _registry().stacks],
        "capability_ids": [check["id"] for check in checks],
        "checks": checks,
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _ids_by_family(items: Mapping[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item_id, item in items.items():
        groups.setdefault(item.family, []).append(item_id)
    return {family: sorted(ids) for family, ids in sorted(groups.items())}


def base_model_inventory() -> dict[str, Any]:
    """Return versioned catalog metadata, family groupings and asset counts without readiness probes.

    Include capabilities and stacks sorted by ID, role counts and benchmark
    data-asset mappings. No imports, downloads or asset checks are performed.
    """
    registry = _registry()
    all_assets = [asset for capability in registry.capabilities.values() for asset in capability.assets]
    return {
        "schema_version": "worldfoundry-base-model-inventory-v1",
        "capability_count": len(registry.capabilities),
        "stack_count": len(registry.stacks),
        "asset_count": len(all_assets),
        "required_asset_count": sum(1 for asset in all_assets if asset.required),
        "capabilities_by_family": _ids_by_family(registry.capabilities),
        "stacks_by_family": _ids_by_family(registry.stacks),
        "asset_counts_by_role": _count(asset.role for asset in all_assets),
        "benchmark_data_asset_count": len(registry.benchmark_assets),
        "benchmark_data_assets": {
            benchmark_id: list(capability_ids)
            for benchmark_id, capability_ids in sorted(registry.benchmark_assets.items())
        },
        "capabilities": [registry.capabilities[item].to_dict() for item in sorted(registry.capabilities)],
        "stacks": [registry.stacks[item].to_dict() for item in sorted(registry.stacks)],
    }


def _base_model_plan_summary(
    checks: list[dict[str, Any]],
    *,
    download_commands: list[str],
    export_commands: list[str],
    pip_install_packages: list[str],
    manual_actions: list[str],
) -> dict[str, Any]:
    family_counts: dict[str, dict[str, int]] = {}
    for check in checks:
        row = family_counts.setdefault(
            str(check.get("family") or "unknown"),
            {
                "capability_count": 0,
                "ok_count": 0,
                "missing_count": 0,
                "asset_count": 0,
                "missing_required_asset_count": 0,
            },
        )
        row["capability_count"] += 1
        row["ok_count" if check.get("ok") else "missing_count"] += 1
        assets = [asset for asset in check.get("asset_status") or () if isinstance(asset, Mapping)]
        row["asset_count"] += len(assets)
        row["missing_required_asset_count"] += sum(
            1 for asset in assets if not asset.get("ready") and asset.get("required")
        )

    assets = [asset for check in checks for asset in check.get("asset_status") or () if isinstance(asset, Mapping)]
    ready_asset_ids = [str(asset.get("id")) for asset in assets if asset.get("ready") and asset.get("id")]
    missing_required_asset_ids = [
        str(asset.get("id")) for asset in assets if not asset.get("ready") and asset.get("required") and asset.get("id")
    ]
    missing_capability_ids = [str(check.get("id")) for check in checks if not check.get("ok") and check.get("id")]
    return {
        "capability_count": len(checks),
        "ok_capability_count": sum(1 for check in checks if check.get("ok")),
        "missing_capability_count": sum(1 for check in checks if not check.get("ok")),
        "missing_capability_ids": missing_capability_ids,
        "asset_count": sum(len(check.get("asset_status") or ()) for check in checks),
        "ready_asset_count": len(ready_asset_ids),
        "missing_required_asset_count": len(missing_required_asset_ids),
        "missing_required_asset_ids": missing_required_asset_ids,
        "asset_counts_by_role": _count(str(asset.get("role") or "asset") for asset in assets),
        "family_counts": dict(sorted(family_counts.items())),
        "download_command_count": len(download_commands),
        "export_command_count": len(export_commands),
        "pip_install_package_count": len(pip_install_packages),
        "manual_action_count": len(manual_actions),
    }


def base_model_materialization_plan(
    capability_ids: list[str] | tuple[str, ...] | None = None,
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Build a versioned readiness and repair plan for selected capabilities or stacks.

    None selects every capability sorted by ID; an empty sequence selects none.
    Use env (os.environ by default) for asset paths and python_executable for
    package probes, with current-interpreter discovery by default. Aggregate
    checks, stack metadata, summary counts, download/export commands, sorted
    unique pip packages and manual actions. Probes read paths and may start
    subprocesses; no suggested command is executed. Unknown IDs raise KeyError.
    """
    requested_ids = sorted(_registry().capabilities) if capability_ids is None else list(capability_ids)
    selected_ids = resolve_base_model_capability_ids(requested_ids)
    stack_ids = [str(item) for item in requested_ids if str(item) in _registry().stacks]
    checks = [
        check_base_model_capability(capability_id, env=env, python_executable=python_executable)
        for capability_id in selected_ids
    ]

    def gather(key: str) -> list[Any]:
        return [item for check in checks for item in check["repair_hints"][key]]

    download_commands = gather("download_commands")
    export_commands = gather("export_commands")
    pip_install_packages = sorted(set(gather("pip_install_packages")))
    manual_actions = gather("manual_actions")
    return {
        "schema_version": "worldfoundry-base-model-assets-v1",
        "ok": all(check["ok"] for check in checks),
        "python_executable": None if python_executable is None else str(python_executable),
        "requested_ids": requested_ids,
        "stack_ids": stack_ids,
        "stacks": [_registry().stacks[stack_id].to_dict() for stack_id in stack_ids],
        "capability_ids": selected_ids,
        "checks": checks,
        "summary": _base_model_plan_summary(
            checks,
            download_commands=download_commands,
            export_commands=export_commands,
            pip_install_packages=pip_install_packages,
            manual_actions=manual_actions,
        ),
        "download_commands": download_commands,
        "download_command_argvs": gather("download_command_argvs"),
        "export_commands": export_commands,
        "pip_install_packages": pip_install_packages,
        "manual_actions": manual_actions,
    }
