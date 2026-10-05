"""Module for base_models -> capabilities.py functionality."""

from __future__ import annotations

import importlib.util
import os
import shlex
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import worldfoundry_path_tokens

REPO_ROOT = Path(os.environ.get("WORLDFOUNDRY_REPO_ROOT") or worldfoundry_path_tokens()["WORLDFOUNDRY_REPO_ROOT"]).expanduser()


def _path_tokens(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Helper function to path tokens.

    Args:
        env: The env.

    Returns:
        The return value.
    """
    environ = os.environ if env is None else env
    core_tokens = worldfoundry_path_tokens(environ)
    repo_root_path = Path(environ.get("WORLDFOUNDRY_REPO_ROOT") or core_tokens["WORLDFOUNDRY_REPO_ROOT"]).expanduser()
    repo_root = str(repo_root_path)
    cache_root = Path(core_tokens["WORLDFOUNDRY_CACHE_DIR"]).expanduser()
    data_root = Path(core_tokens["WORLDFOUNDRY_DATA_DIR"]).expanduser()
    hfd_root = Path(core_tokens["WORLDFOUNDRY_HFD_ROOT"]).expanduser()
    hfd_dataset_root = Path(core_tokens["WORLDFOUNDRY_HFD_DATASET_ROOT"]).expanduser()
    default_workspace_root = hfd_root.parent
    workspace_root = str(
        Path(
            environ.get("WORLDFOUNDRY_WORKSPACE_ROOT")
            or environ.get("WORLDFOUNDRY_BENCH_ROOT")
            or default_workspace_root
        ).expanduser()
    )
    worldscore_asset_dir = Path(
        environ.get("WORLDFOUNDRY_WORLDSCORE_ASSET_DIR")
        or cache_root / "assets" / "worldscore_official"
    ).expanduser()
    return {
        "WORLDFOUNDRY_REPO_ROOT": repo_root,
        "WORLDFOUNDRY_WORKSPACE_ROOT": workspace_root,
        "WORLDFOUNDRY_BENCH_ROOT": str(Path(environ.get("WORLDFOUNDRY_BENCH_ROOT") or repo_root).expanduser()),
        "WORLDFOUNDRY_CACHE_DIR": str(cache_root),
        "WORLDFOUNDRY_DATA_DIR": str(data_root),
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": core_tokens["WORLDFOUNDRY_MODEL_SOURCE_DIR"],
        "WORLDFOUNDRY_HFD_ROOT": str(hfd_root),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(hfd_dataset_root),
        "WORLDFOUNDRY_CKPT_DIR": core_tokens["WORLDFOUNDRY_CKPT_DIR"],
        "WORLDFOUNDRY_MODEL_DIR": core_tokens["WORLDFOUNDRY_MODEL_DIR"],
        "WORLDFOUNDRY_WORLDSCORE_ASSET_DIR": str(worldscore_asset_dir),
        "WORLDFOUNDRY_WORLDSCORE_DATASET_DIR": str(
            Path(environ.get("WORLDFOUNDRY_WORLDSCORE_DATASET_DIR") or worldscore_asset_dir / "data" / "WorldScore-Dataset")
            .expanduser()
        ),
        "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR": str(
            Path(
                environ.get("WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR")
                or worldscore_asset_dir / "metrics" / "checkpoints"
            ).expanduser()
        ),
    }


def expand_capability_path(value: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expand capability path.

    Args:
        value: The value.
        env: The env.

    Returns:
        The return value.
    """
    expanded = str(value)
    tokens = _path_tokens(env)
    for name, replacement in tokens.items():
        expanded = expanded.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else Path(tokens["WORLDFOUNDRY_REPO_ROOT"]) / path


def _dir_file_count_ready(path: Path, min_file_count: int | None) -> bool:
    """Helper function to dir file count ready.

    Args:
        path: The path.
        min_file_count: The min file count.

    Returns:
        The return value.
    """
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


def _required_files_ready(path: Path, required_files: tuple[str, ...]) -> bool:
    """Helper function to required files ready.

    Args:
        path: The path.
        required_files: The required files.

    Returns:
        The return value.
    """
    return all((path / filename).is_file() for filename in required_files)


def _required_paths_ready(path: Path, required_paths: tuple[str, ...]) -> bool:
    """Helper function to required paths ready.

    Args:
        path: The path.
        required_paths: The required paths.

    Returns:
        The return value.
    """
    return all((path / name).exists() for name in required_paths)


def _required_any_paths_ready(path: Path, required_any_paths: tuple[str, ...]) -> bool:
    """Helper function to required any paths ready.

    Args:
        path: The path.
        required_any_paths: The required any paths.

    Returns:
        The return value.
    """
    return not required_any_paths or any((path / name).exists() for name in required_any_paths)


def _path_ready(
    path: Path,
    *,
    kind: str,
    min_size_bytes: int | None = None,
    min_file_count: int | None = None,
    required_files: tuple[str, ...] = (),
    required_paths: tuple[str, ...] = (),
    required_any_paths: tuple[str, ...] = (),
) -> bool:
    """Helper function to path ready.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    try:
        if kind == "dir":
            return (
                path.is_dir()
                and _required_files_ready(path, required_files)
                and _required_paths_ready(path, required_paths)
                and _required_any_paths_ready(path, required_any_paths)
                and _dir_file_count_ready(path, min_file_count)
            )
        if not path.is_file():
            return False
        return min_size_bytes is None or path.stat().st_size >= min_size_bytes
    except OSError:
        return False


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Helper function to unique paths.

    Args:
        paths: The paths.

    Returns:
        The return value.
    """
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


@lru_cache(maxsize=256)
def _python_import_available(python_executable: str, module_name: str, pythonpath: str) -> bool:
    """Helper function to python import available.

    Args:
        python_executable: The python executable.
        module_name: The module name.
        pythonpath: The pythonpath.

    Returns:
        The return value.
    """
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
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            # Some broken external envs can block in uninterruptible filesystem
            # or loader state. Readiness must report that import as unavailable
            # instead of blocking the whole benchmark inventory.
            pass
        return False


def _import_available(module_name: str, python_executable: str | Path | None = None) -> bool:
    """Helper function to import available.

    Args:
        module_name: The module name.
        python_executable: The python executable.

    Returns:
        The return value.
    """
    if python_executable is None:
        return importlib.util.find_spec(module_name) is not None
    return _python_import_available(str(python_executable), module_name, os.environ.get("PYTHONPATH", ""))


def _hf_download_command(
    repo_id: str,
    *,
    filename: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    local_dir: str | Path,
) -> list[str]:
    """Helper function to hf download command.

    Args:
        repo_id: The repo id.

    Returns:
        The return value.
    """
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
    """Helper function to curl download command.

    Args:
        url: The url.
        output_path: The output path.

    Returns:
        The return value.
    """
    return [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        url,
        "-o",
        str(output_path),
    ]


@dataclass(frozen=True)
class BaseModelAsset:
    """Checkpoint, model directory, or benchmark data asset owned by a reusable capability."""

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

    def local_dir(self, env: Mapping[str, str] | None = None) -> Path:
        """Local dir.

        Args:
            env: The env.

        Returns:
            The return value.
        """
        local_path = expand_capability_path(self.local_path, env)
        return local_path if self.kind == "dir" else local_path.parent

    def download_command(self, env: Mapping[str, str] | None = None) -> list[str] | None:
        """Download command.

        Args:
            env: The env.

        Returns:
            The return value.
        """
        if self.download_command_template:
            command = str(self.download_command_template)
            for name, replacement in _path_tokens(env).items():
                command = command.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
            return ["bash", "-lc", command]
        if self.git_repo_url:
            # Source repositories must be integrated in-tree. Readiness/job setup
            # should never clone or update Git repositories.
            return None
        if not self.hf_repo_id:
            if not self.download_url:
                return None
            local_path = expand_capability_path(self.local_path, env)
            shell_command = (
                f"mkdir -p {shlex.quote(str(local_path.parent))} "
                f"&& {shlex.join(_curl_download_command(self.download_url, local_path))}"
            )
            return ["bash", "-lc", shell_command]
        local_dir = str(self.local_dir(env))
        if (
            not self.hf_filename
            and self.kind == "dir"
            and self.role in {"model_dir", "text_encoder_model_dir"}
            and (self.required_files or self.required_any_paths)
        ):
            commands: list[str] = [f"mkdir -p {shlex.quote(local_dir)}"]
            filenames = [*self.required_files]
            if self.required_any_paths:
                filenames.append(self.required_any_paths[0])
            for filename in filenames:
                commands.append(
                    shlex.join(
                        _hf_download_command(
                            self.hf_repo_id,
                            filename=filename,
                            repo_type=self.hf_repo_type,
                            revision=self.hf_revision,
                            local_dir=local_dir,
                        )
                    )
                )
            return ["bash", "-lc", " && ".join(commands)]

        return _hf_download_command(
            self.hf_repo_id,
            filename=self.hf_filename,
            repo_type=self.hf_repo_type,
            revision=self.hf_revision,
            local_dir=local_dir,
        )

    def check(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Check.

        Args:
            env: The env.

        Returns:
            The return value.
        """
        environ = os.environ if env is None else env
        local_path = expand_capability_path(self.local_path, environ)
        alternate_paths = _unique_paths([expand_capability_path(path, environ) for path in self.alternate_paths])
        env_status = []
        env_matches: list[Path] = []
        for name in self.env:
            raw_value = environ.get(name)
            path = expand_capability_path(raw_value, environ) if raw_value else None
            ready = bool(
                path
                and _path_ready(
                    path,
                    kind=self.kind,
                    min_size_bytes=self.min_size_bytes,
                    min_file_count=self.min_file_count,
                    required_files=self.required_files,
                    required_paths=self.required_paths,
                    required_any_paths=self.required_any_paths,
                )
            )
            if path and ready:
                env_matches.append(path)
            env_status.append(
                {
                    "name": name,
                    "present": bool(raw_value),
                    "path": None if path is None else str(path),
                    "ready": ready,
                }
            )

        candidates = _unique_paths([local_path, *alternate_paths])
        matched = next(
            (
                path
                for path in [*env_matches, *candidates]
                if _path_ready(
                    path,
                    kind=self.kind,
                    min_size_bytes=self.min_size_bytes,
                    min_file_count=self.min_file_count,
                    required_files=self.required_files,
                    required_paths=self.required_paths,
                    required_any_paths=self.required_any_paths,
                )
            ),
            None,
        )
        env_ready = bool(env_matches)
        download_command = self.download_command(environ)
        export_target = matched or local_path
        export_commands = [f"export {name}={shlex.quote(str(export_target))}" for name in self.env]
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
            "env_ready": env_ready,
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
            "export_commands": export_commands,
            "note": self.note,
        }


@dataclass(frozen=True)
class BaseModelCapability:
    """Reusable in-tree foundation-model capability used by benchmark runtimes."""

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
        """Owner path.

        Returns:
            The return value.
        """
        path = Path(self.canonical_path)
        return path if path.is_absolute() else REPO_ROOT / path

    def pip_package_by_import(self) -> dict[str, str]:
        """Pip package by import.

        Returns:
            The return value.
        """
        non_pip = set(self.non_pip_imports)
        if not self.package_imports:
            return {}
        if self.install_packages and len(self.install_packages) == len(self.package_imports):
            pairs = zip(self.package_imports, self.install_packages)
            return {
                import_name: package
                for import_name, package in pairs
                if import_name not in non_pip and package
            }
        pip_imports = [name for name in self.package_imports if name not in non_pip]
        install_packages = self.install_packages or tuple(pip_imports)
        return {
            import_name: install_packages[index] if index < len(install_packages) else import_name
            for index, import_name in enumerate(pip_imports)
            if (install_packages[index] if index < len(install_packages) else import_name)
        }

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            The return value.
        """
        return {
            "id": self.id,
            "family": self.family,
            "canonical_owner": self.canonical_owner,
            "canonical_path": self.canonical_path,
            "package_imports": list(self.package_imports),
            "install_packages": list(self.pip_package_by_import().values()),
            "non_pip_imports": list(self.non_pip_imports),
            "manual_install_actions": list(self.manual_install_actions),
            "asset_env": list(self.asset_env),
            "asset_env_policy": self.asset_env_policy,
            "assets": [
                {
                    "id": asset.id,
                    "kind": asset.kind,
                    "role": asset.role,
                    "env": list(asset.env),
                    "local_path": asset.local_path,
                    "alternate_paths": list(asset.alternate_paths),
                    "hf_repo_id": asset.hf_repo_id,
                    "hf_repo_type": asset.hf_repo_type,
                    "hf_filename": asset.hf_filename,
                    "hf_revision": asset.hf_revision,
                    "git_repo_url": asset.git_repo_url,
                    "git_revision": asset.git_revision,
                    "download_url": asset.download_url,
                    "download_command_template": asset.download_command_template,
                    "manual_action": asset.manual_action,
                    "min_size_bytes": asset.min_size_bytes,
                    "min_file_count": asset.min_file_count,
                    "required_files": list(asset.required_files),
                    "required_paths": list(asset.required_paths),
                    "required_any_paths": list(asset.required_any_paths),
                    "required": asset.required,
                    "note": asset.note,
                }
                for asset in self.assets
            ],
            "gpu_required": self.gpu_required,
        }


@dataclass(frozen=True)
class BaseModelStack:
    """Named bundle of reusable base-model capabilities."""

    id: str
    family: str
    capability_ids: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            The return value.
        """
        return {
            "id": self.id,
            "family": self.family,
            "capability_ids": list(self.capability_ids),
            "note": self.note,
        }


def _hf_dataset_asset(
    *,
    asset_id: str,
    role: str,
    env: str,
    repo_id: str,
    local_name: str,
    revision: str | None = None,
    alternate_local_names: tuple[str, ...] = (),
    min_file_count: int = 1,
    required: bool = True,
    note: str | None = None,
) -> BaseModelAsset:
    """Helper function to hf dataset asset.

    Returns:
        The return value.
    """
    alternates = tuple(
        path
        for name in alternate_local_names
        for path in (
            f"${{WORLDFOUNDRY_HFD_DATASET_ROOT}}/{name}",
            f"${{WORLDFOUNDRY_WORKSPACE_ROOT}}/data/hfd_datasets/{name}",
        )
    )
    return BaseModelAsset(
        id=asset_id,
        kind="dir",
        role=role,
        env=(env,),
        local_path=f"${{WORLDFOUNDRY_HFD_DATASET_ROOT}}/{local_name}",
        alternate_paths=alternates,
        hf_repo_id=repo_id,
        hf_repo_type="dataset",
        hf_revision=revision,
        min_file_count=min_file_count,
        required=required,
        note=note,
    )


def _benchmark_dataset_capability(
    capability_id: str,
    specs: tuple[dict[str, Any], ...],
) -> BaseModelCapability:
    """Helper function to benchmark dataset capability.

    Args:
        capability_id: The capability id.
        specs: The specs.

    Returns:
        The return value.
    """
    assets = tuple(
        _hf_dataset_asset(
            asset_id=str(spec["asset_id"]),
            role=str(spec.get("role") or "benchmark_dataset"),
            env=str(spec["env"]),
            repo_id=str(spec["repo_id"]),
            revision=spec.get("revision"),
            local_name=str(spec["local_name"]),
            alternate_local_names=tuple(str(item) for item in spec.get("alternate_local_names", ())),
            min_file_count=int(spec.get("min_file_count") or 1),
            required=bool(spec.get("required", True)),
            note=spec.get("note"),
        )
        for spec in specs
    )
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=tuple(env for asset in assets for env in asset.env),
        asset_env_policy="all" if len(assets) > 1 else "any",
        assets=assets,
        gpu_required=False,
    )


_IN_TREE_SOURCE_ASSETS: dict[str, tuple[str, str]] = {
    "bridgedata_v2_source_assets": ("file", "worldfoundry/data/benchmarks/tasks/external/bridgedata-v2.yaml"),
    "calvin_source_assets": ("dir", "worldfoundry/evaluation/tasks/embodied/simulators/calvin"),
    "libero_source_assets": ("dir", "worldfoundry/evaluation/tasks/embodied/simulators/libero"),
    "maniskill_source_assets": ("file", "worldfoundry/data/benchmarks/tasks/external/maniskill.yaml"),
    "metaworld_source_assets": ("file", "worldfoundry/data/benchmarks/tasks/external/metaworld.yaml"),
    "rlbench_source_assets": ("dir", "worldfoundry/evaluation/tasks/embodied/simulators/rlbench"),
    "robocasa_source_assets": ("dir", "worldfoundry/evaluation/tasks/embodied/simulators/robocasa"),
    "simpler_env_source_assets": ("dir", "worldfoundry/evaluation/tasks/embodied/simulators/simpler"),
    "iworld_bench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/iworldbench"),
    "t2vworldbench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/t2vworldbench"),
    "vmbench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/vmbench"),
    "mirabench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/mirabench"),
    "devil_dynamics_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/devil_dynamics"),
    "phygenbench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/phygenbench"),
    "physvidbench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/physvidbench"),
    "visual_chronometer_source_assets": (
        "dir",
        "worldfoundry/evaluation/tasks/execution/runners/phyfps_bench_gen/runtime/visual_chronometer",
    ),
    "videophy_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/videophy"),
    "physics_iq_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/physics_iq"),
    "t2v_safety_bench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/t2v_safety_bench"),
    "ipv_bench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/ipv_bench"),
    "videoscience_bench_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/videoscience_bench"),
    "phyeduvideo_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/phyeduvideo"),
    "worldarena_source_assets": ("dir", "worldfoundry/evaluation/tasks/execution/runners/worldarena"),
}


def _source_repo_capability(
    *,
    capability_id: str,
    env: str,
    repo_url: str,
    local_name: str,
    revision: str | None = None,
    alternate_paths: tuple[str, ...] = (),
    required_files: tuple[str, ...] = (),
    required_paths: tuple[str, ...] = (),
    role: str = "benchmark_source_repo",
    note: str | None = None,
) -> BaseModelCapability:
    """Helper function to source repo capability.

    Returns:
        The return value.
    """
    try:
        in_tree_kind, in_tree_path = _IN_TREE_SOURCE_ASSETS[capability_id]
    except KeyError as exc:
        raise ValueError(
            f"{capability_id} declares an external source repository but has no in-tree integration path"
        ) from exc
    source_note = (
        "In-tree WorldFoundry integration replacing an upstream source repository. "
        f"Upstream reference: {repo_url}. "
        "WorldFoundry base_models does not clone external source repositories during readiness or job setup."
    )
    if note:
        source_note = f"{source_note} Original upstream context: {note}"
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=(),
        assets=(
            BaseModelAsset(
                id=f"{capability_id}_in_tree",
                kind=in_tree_kind,
                role=role.replace("source_repo", "in_tree_source"),
                env=(),
                local_path=f"${{WORLDFOUNDRY_REPO_ROOT}}/{in_tree_path}",
                min_file_count=1 if in_tree_kind == "dir" else None,
                manual_action=(
                    f"Restore the checked-in WorldFoundry integration at {in_tree_path}; "
                    f"do not stage or clone {repo_url} for base_models readiness."
                ),
                note=source_note,
            ),
        ),
        gpu_required=False,
    )


def _external_dir_capability(
    *,
    capability_id: str,
    env: str,
    local_name: str,
    alternate_paths: tuple[str, ...] = (),
    role: str = "external_benchmark_asset",
    required: bool = True,
    required_any_paths: tuple[str, ...] = (),
    download_command_template: str | None = None,
    manual_action: str | None = None,
    note: str | None = None,
) -> BaseModelCapability:
    """Helper function to external dir capability.

    Returns:
        The return value.
    """
    return BaseModelCapability(
        id=capability_id,
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=(env,),
        assets=(
            BaseModelAsset(
                id=f"{capability_id}_dir",
                kind="dir",
                role=role,
                env=(env,),
                local_path=f"${{WORLDFOUNDRY_DATA_DIR}}/datasets/{local_name}",
                alternate_paths=alternate_paths,
                download_command_template=download_command_template,
                manual_action=manual_action,
                min_file_count=1,
                required_any_paths=required_any_paths,
                required=required,
                note=note,
            ),
        ),
        gpu_required=False,
    )


def _perception_data_capability(
    *,
    capability_id: str,
    env: str,
    local_path: str,
    alternate_paths: tuple[str, ...] = (),
    required_files: tuple[str, ...] = (),
    required_paths: tuple[str, ...] = (),
    required_any_paths: tuple[str, ...] = (),
    min_file_count: int = 1,
    download_command_template: str | None = None,
    manual_action: str | None = None,
    role: str = "perception_eval_dataset",
    note: str | None = None,
) -> BaseModelCapability:
    """Helper function to perception data capability.

    Returns:
        The return value.
    """
    return BaseModelCapability(
        id=capability_id,
        family="perception_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=(env,),
        assets=(
            BaseModelAsset(
                id=f"{capability_id}_dir",
                kind="dir",
                role=role,
                env=(env,),
                local_path=local_path,
                alternate_paths=alternate_paths,
                download_command_template=download_command_template,
                manual_action=manual_action,
                min_file_count=min_file_count,
                required_files=required_files,
                required_paths=required_paths,
                required_any_paths=required_any_paths,
                note=note,
            ),
        ),
        gpu_required=False,
    )


BASE_MODEL_CAPABILITIES: dict[str, BaseModelCapability] = {
    "vigeo": BaseModelCapability(
        id="vigeo",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.vigeo",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/vigeo",
        package_imports=("torch", "numpy", "einops", "huggingface_hub"),
        install_packages=("torch", "numpy", "einops", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_VIGEO_MODEL_DIR", "EVOKE_VIGEO_WEIGHTS"),
        assets=(
            BaseModelAsset(
                id="vigeo_model",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_VIGEO_MODEL_DIR", "EVOKE_VIGEO_WEIGHTS"),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/pkqbajng--ViGeo1.1",
                hf_repo_id="pkqbajng/ViGeo1.1",
                hf_revision="49103e6eeab888bae974251d3578b496bec711d7",
                required_files=("vigeo.pt",),
                note="Shared ViGeo video geometry model; checkpoint weights are CC-BY-NC-4.0.",
            ),
        ),
    ),
    "depth_anything_v3": BaseModelCapability(
        id="depth_anything_v3",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/depth_anything/depth_anything_v3",
        package_imports=("torch", "torchvision", "cv2", "numpy", "PIL", "omegaconf", "einops", "huggingface_hub", "safetensors"),
        install_packages=(
            "torch",
            "torchvision",
            "opencv-python",
            "numpy",
            "Pillow",
            "omegaconf",
            "einops",
            "huggingface-hub",
            "safetensors",
        ),
        asset_env=("WORLDFOUNDRY_DEPTH_ANYTHING_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="depth_anything_v3_model",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_DEPTH_ANYTHING_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/depth-anything--DA3-LARGE-1.1",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/depth-anything--DA3-LARGE-1.1",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/depth-anything--DA3-LARGE",
                ),
                hf_repo_id="depth-anything/DA3-LARGE-1.1",
                min_file_count=1,
                required_files=("config.json", "model.safetensors"),
                note="Default Depth Anything 3 model used by the in-tree DA3 pipeline.",
            ),
        ),
    ),
    "droid_slam": BaseModelCapability(
        id="droid_slam",
        family="slam",
        canonical_owner="worldfoundry.base_models.three_dimensions.slam.droid_slam",
        canonical_path="worldfoundry/base_models/three_dimensions/slam/droid_slam",
        package_imports=("torch", "cv2", "numpy", "lietorch", "droid_backends", "torch_scatter"),
        install_packages=("torch", "opencv-python", "numpy", "torch-scatter"),
        non_pip_imports=("lietorch", "droid_backends"),
        manual_install_actions=(
            "Build/stage importable DROID-SLAM CUDA extensions (`lietorch` and `droid_backends`) from source in the "
            "active benchmark env; do not install the unrelated PyPI `lietorch` package. Build against the same "
            "Torch/CUDA ABI before rerunning readiness.",
        ),
        asset_env=("WORLDFOUNDRY_DROID_SLAM_CKPT",),
        assets=(
            BaseModelAsset(
                id="droid_slam_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_DROID_SLAM_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/droid.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/droid.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/shape-of-motion/checkpoints/droid.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/metrics/checkpoints/droid.pth",
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/droid.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/droid.pth",
                ),
                min_size_bytes=15_000_000,
                note="DROID-SLAM checkpoint reused by WorldScore camera metrics.",
            ),
        ),
    ),
    "keye_vl": BaseModelCapability(
        id="keye_vl",
        family="multimodal_judge",
        canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.keye_vl",
        canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/keye_vl",
        package_imports=("torch", "transformers"),
        install_packages=("torch", "transformers"),
        manual_install_actions=(
            "Install `keye-vl-utils` and `qwen-vl-utils` in the active benchmark environment when running "
            "4DWorldBench Keye-VL judge dimensions.",
        ),
        asset_env=("WORLDFOUNDRY_KEYE_VL_MODEL", "WORLDFOUNDRY_4DWORLDBENCH_KEYE_MODEL", "KEYE_MODEL_PATH"),
        assets=(
            BaseModelAsset(
                id="keye_vl_1_5_8b_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_KEYE_VL_MODEL", "WORLDFOUNDRY_4DWORLDBENCH_KEYE_MODEL", "KEYE_MODEL_PATH"),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/Kwai-Keye--Keye-VL-1_5-8B",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/hfd/Kwai-Keye--Keye-VL-1_5-8B",
                    "${WORLDFOUNDRY_CKPT_DIR}/Kwai-Keye--Keye-VL-1_5-8B",
                    "${WORLDFOUNDRY_CKPT_DIR}/Keye-VL-1_5-8B",
                ),
                hf_repo_id="Kwai-Keye/Keye-VL-1_5-8B",
                min_file_count=3,
                required_files=("config.json",),
                required_any_paths=("model.safetensors", "pytorch_model.bin", "model-00001-of-00004.safetensors"),
                note="Keye-VL judge model used by 4DWorldBench alignment, motion-QA, and caption-helper dimensions.",
            ),
        ),
    ),
    "qwen3_vl_8b_instruct": BaseModelCapability(
        id="qwen3_vl_8b_instruct",
        family="multimodal_judge",
        canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.qwen",
        canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/qwen",
        package_imports=("torch", "transformers"),
        install_packages=("torch", "transformers"),
        asset_env=("WORLDFOUNDRY_QWEN3_VL_MODEL_DIR", "QWENVL_MODEL_PATH"),
        assets=(
            BaseModelAsset(
                id="qwen3_vl_8b_instruct_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_QWEN3_VL_MODEL_DIR", "QWENVL_MODEL_PATH"),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/Qwen--Qwen3-VL-8B-Instruct",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/hfd/Qwen--Qwen3-VL-8B-Instruct",
                    "${WORLDFOUNDRY_CKPT_DIR}/Qwen3-VL-8B-Instruct",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/Qwen--Qwen3-VL-8B-Instruct",
                ),
                hf_repo_id="Qwen/Qwen3-VL-8B-Instruct",
                min_file_count=3,
                required_files=("config.json",),
                required_any_paths=(
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model-00001-of-00005.safetensors",
                ),
                note="Qwen3-VL-8B-Instruct VLM judge used by the WorldOlympiad physical, interaction, and geometry tracks; reused in place of an ad-hoc --weights-dir/QwenVL checkout.",
            ),
        ),
    ),
    "grounding_dino": BaseModelCapability(
        id="grounding_dino",
        family="detection",
        canonical_owner="worldfoundry.base_models.perception_core.detection.grounding_dino",
        canonical_path="worldfoundry/base_models/perception_core/detection/grounding_dino",
        package_imports=("torch", "cv2", "numpy", "transformers", "timm"),
        install_packages=("torch", "opencv-python", "numpy", "transformers", "timm"),
        asset_env=("WORLDFOUNDRY_GROUNDING_DINO_CKPT",),
        assets=(
            BaseModelAsset(
                id="grounding_dino_swint_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_GROUNDING_DINO_CKPT",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/ShilongLiu--GroundingDINO/groundingdino_swint_ogc.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/ShilongLiu--GroundingDINO/groundingdino_swint_ogc.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/GroundingDINO/groundingdino_swint_ogc.pth",
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/groundingdino_swint_ogc.pth",
                ),
                hf_repo_id="ShilongLiu/GroundingDINO",
                hf_filename="groundingdino_swint_ogc.pth",
                min_size_bytes=600_000_000,
                note="GroundingDINO Swin-T checkpoint shared by detection and Grounded-SAM paths.",
            ),
            BaseModelAsset(
                id="grounding_dino_bert_base_uncased_model_dir",
                kind="dir",
                role="text_encoder_model_dir",
                env=("WORLDFOUNDRY_GROUNDING_DINO_TEXT_ENCODER_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/bert-base-uncased",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/bert-base-uncased",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/bert-base-uncased",
                ),
                hf_repo_id="bert-base-uncased",
                min_file_count=3,
                required_files=("config.json", "vocab.txt"),
                required_any_paths=("model.safetensors", "pytorch_model.bin"),
                note="BERT text encoder used by the in-tree GroundingDINO model.",
            ),
        ),
    ),
    "clip_vit_b32": BaseModelCapability(
        id="clip_vit_b32",
        family="vision_language_embedding",
        canonical_owner="worldfoundry.base_models.perception_core.general_perception.open_clip",
        canonical_path="worldfoundry/base_models/perception_core/general_perception/open_clip",
        package_imports=("torch", "numpy", "ftfy", "regex", "tqdm", "huggingface_hub", "transformers"),
        install_packages=("torch>=2.6", "numpy", "ftfy", "regex", "tqdm", "huggingface-hub", "transformers"),
        asset_env=("WORLDFOUNDRY_CLIP_VIT_B32_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="clip_vit_b32_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_CLIP_VIT_B32_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/openai--clip-vit-base-patch32",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/openai--clip-vit-base-patch32",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/openai--clip-vit-base-patch32",
                ),
                hf_repo_id="openai/clip-vit-base-patch32",
                min_file_count=1,
                required_files=("config.json", "pytorch_model.bin"),
                note="CLIP ViT-B/32 model directory shared by text-video alignment and temporal CLIP metrics.",
            ),
        ),
    ),
    "dino_vitb16": BaseModelCapability(
        id="dino_vitb16",
        family="visual_embedding",
        canonical_owner="worldfoundry.base_models.perception_core.general_perception.dino_embeddings",
        canonical_path="worldfoundry/base_models/perception_core/general_perception/dino_embeddings.py",
        package_imports=("torch", "torchvision", "numpy", "transformers"),
        install_packages=("torch", "torchvision", "numpy", "transformers"),
        asset_env=("WORLDFOUNDRY_DINO_VITB16_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="dino_vitb16_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_DINO_VITB16_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--dino-vitb16",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/dino-vitb16",
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/dino-vitb16",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/dino-vitb16",
                ),
                hf_repo_id="facebook/dino-vitb16",
                min_file_count=1,
                required_files=("config.json", "pytorch_model.bin"),
                note="DINO ViT-B/16 visual embedding model used by temporal DINO consistency metrics.",
            ),
        ),
    ),
    "dinov2_base": BaseModelCapability(
        id="dinov2_base",
        family="visual_embedding",
        canonical_owner="worldfoundry.base_models.perception_core.general_perception.dinov2",
        canonical_path="worldfoundry/base_models/perception_core/general_perception/dinov2",
        package_imports=("torch", "torchvision", "numpy", "transformers"),
        install_packages=("torch", "torchvision", "numpy", "transformers"),
        asset_env=("WORLDFOUNDRY_DINOV2_BASE_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="dinov2_base_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_DINOV2_BASE_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--dinov2-base",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/dinov2-base",
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/facebook--dinov2-base",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--dinov2-base",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/dinov2-base",
                ),
                hf_repo_id="facebook/dinov2-base",
                min_file_count=1,
                required_files=("config.json", "model.safetensors"),
                note="DINOv2 base visual embedding model used by DINO-style temporal consistency metrics.",
            ),
        ),
    ),
    "dinov3_base": BaseModelCapability(
        id="dinov3_base",
        family="visual_embedding",
        canonical_owner="worldfoundry.base_models.perception_core.general_perception.dinov3",
        canonical_path="worldfoundry/base_models/perception_core/general_perception/dinov3",
        package_imports=("torch", "torchvision", "numpy", "transformers"),
        install_packages=("torch", "torchvision", "numpy", "transformers"),
        asset_env=("WORLDFOUNDRY_DINOV3_BASE_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="dinov3_base_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_DINOV3_BASE_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--dinov3-vitb16-pretrain-lvd1689m",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/facebook--dinov3-vitb16-pretrain-lvd1689m",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--dinov3-vitb16-pretrain-lvd1689m",
                ),
                hf_repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
                min_file_count=1,
                required_files=("config.json", "model.safetensors"),
                note="DINOv3 ViT-B/16 base visual embedding model. Weights require gated access — request via https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ before downloading.",
            ),
        ),
    ),
    "siglip_so400m": BaseModelCapability(
        id="siglip_so400m",
        family="vision_language_embedding",
        canonical_owner="worldfoundry.base_models.perception_core.general_perception.open_clip",
        canonical_path="worldfoundry/base_models/perception_core/general_perception/open_clip",
        package_imports=("torch", "numpy", "timm", "ftfy", "regex", "tqdm", "huggingface_hub"),
        install_packages=("torch", "numpy", "timm", "ftfy", "regex", "tqdm", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_SIGLIP_SO400M_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="siglip_so400m_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_SIGLIP_SO400M_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/google--siglip-so400m-patch14-384",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/google--siglip-so400m-patch14-384",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/google--siglip-so400m-patch14-384",
                ),
                hf_repo_id="google/siglip-so400m-patch14-384",
                min_file_count=1,
                note="SigLIP SO400M image encoder reused by image-conditioned video and preference metrics.",
            ),
        ),
    ),
    "moge_vitl": BaseModelCapability(
        id="moge_vitl",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.moge",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/moge",
        package_imports=("torch", "cv2", "numpy", "scipy", "huggingface_hub"),
        install_packages=("torch", "opencv-python", "numpy", "scipy", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_MOGE_VITL_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="moge_vitl_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_MOGE_VITL_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/Ruicheng--moge-vitl",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/Ruicheng--moge-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/Ruicheng--moge-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/Ruicheng--moge-vitl",
                    "${WORLDFOUNDRY_HFD_ROOT}/moge-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/moge-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/moge-vitl",
                ),
                hf_repo_id="Ruicheng/moge-vitl",
                min_file_count=1,
                required_files=("model.pt",),
                note="MoGe v1 ViT-L metric-depth model used by Gen3C, ViPE, and world geometry pipelines.",
            ),
        ),
    ),
    "moge_v2_vitl_normal": BaseModelCapability(
        id="moge_v2_vitl_normal",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.moge",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/moge",
        package_imports=("torch", "cv2", "numpy", "scipy", "huggingface_hub"),
        install_packages=("torch", "opencv-python", "numpy", "scipy", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_MOGE_V2_VITL_NORMAL_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="moge_v2_vitl_normal_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_MOGE_V2_VITL_NORMAL_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/Ruicheng--moge-2-vitl-normal",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/Ruicheng--moge-2-vitl-normal",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/Ruicheng--moge-2-vitl-normal",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/Ruicheng--moge-2-vitl-normal",
                    "${WORLDFOUNDRY_HFD_ROOT}/moge-2-vitl-normal",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/moge-2-vitl-normal",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/moge-2-vitl-normal",
                    "${WORLDFOUNDRY_HFD_ROOT}/Ruicheng--moge-2-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/Ruicheng--moge-2-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/Ruicheng--moge-2-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/moge-2-vitl",
                ),
                hf_repo_id="Ruicheng/moge-2-vitl-normal",
                min_file_count=1,
                required_files=("model.pt",),
                note="MoGe v2 ViT-L normal model used as the preferred shared geometry prior.",
            ),
        ),
    ),
    "unidepth_v2_vitl14": BaseModelCapability(
        id="unidepth_v2_vitl14",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.unidepth",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/unidepth",
        package_imports=("torch", "torchvision", "numpy", "einops", "huggingface_hub", "safetensors", "xformers.components.attention"),
        install_packages=("torch", "torchvision", "numpy", "einops", "huggingface-hub", "safetensors", "xformers==0.0.28.post3"),
        asset_env=("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="unidepth_v2_vitl14_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/lpiccinelli--unidepth-v2-vitl14",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/lpiccinelli--unidepth-v2-vitl14",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/lpiccinelli--unidepth-v2-vitl14",
                ),
                hf_repo_id="lpiccinelli/unidepth-v2-vitl14",
                hf_revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
                min_file_count=1,
                required_files=("config.json", "model.safetensors"),
                note="UniDepth v2 ViT-L depth prior reused by ViPE-style geometric perception paths.",
            ),
        ),
    ),
    "unik3d_vitl": BaseModelCapability(
        id="unik3d_vitl",
        family="depth_geometry",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.unik3d",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/unik3d",
        package_imports=("torch", "torchvision", "numpy", "einops", "huggingface_hub", "safetensors"),
        install_packages=("torch", "torchvision", "numpy", "einops", "huggingface-hub", "safetensors"),
        asset_env=("WORLDFOUNDRY_UNIK3D_VITL_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="unik3d_vitl_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_UNIK3D_VITL_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/lpiccinelli--unik3d-vitl",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/unik3d-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/lpiccinelli--unik3d-vitl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/unik3d-vitl",
                ),
                hf_repo_id="lpiccinelli/unik3d-vitl",
                min_file_count=2,
                required_files=("config.json", "model.safetensors"),
                note="Official UniK3D ViT-L universal-camera metric-3D model directory.",
            ),
        ),
    ),
    "midas_v21_small_256": BaseModelCapability(
        id="midas_v21_small_256",
        family="depth",
        canonical_owner="worldfoundry.base_models.three_dimensions.depth.midas",
        canonical_path="worldfoundry/base_models/three_dimensions/depth/midas",
        package_imports=("torch", "torchvision", "cv2", "numpy", "timm"),
        install_packages=("torch", "torchvision", "opencv-python", "numpy", "timm"),
        asset_env=("WORLDFOUNDRY_MIDAS_V21_SMALL_CKPT",),
        assets=(
            BaseModelAsset(
                id="midas_v21_small_256_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_MIDAS_V21_SMALL_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/midas_v21_small_256.pt",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/midas_v21_small_256.pt",
                    "${WORLDFOUNDRY_CKPT_DIR}/MiDaS/midas_v21_small_256.pt",
                ),
                download_url="https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
                min_size_bytes=80_000_000,
                note="Official MiDaS v2.1 Small 256 checkpoint from isl-org/MiDaS.",
            ),
        ),
    ),
    "mobile_sam": BaseModelCapability(
        id="mobile_sam",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.mobile_sam",
        canonical_path="worldfoundry/base_models/perception_core/segment/mobile_sam",
        package_imports=("torch", "torchvision", "numpy"),
        install_packages=("torch", "torchvision", "numpy"),
        asset_env=("WORLDFOUNDRY_MOBILE_SAM_CKPT",),
        assets=(
            BaseModelAsset(
                id="mobile_sam_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_MOBILE_SAM_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/MobileSAM/mobile_sam.pt",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/mobile_sam.pt",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/MobileSAM/mobile_sam.pt",
                ),
                download_url="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
                min_size_bytes=40_000_000,
                note="Official MobileSAM TinyViT checkpoint from ChaoningZhang/MobileSAM.",
            ),
        ),
    ),
    "repvit_sam": BaseModelCapability(
        id="repvit_sam",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.repvit_sam",
        canonical_path="worldfoundry/base_models/perception_core/segment/repvit_sam",
        package_imports=("torch", "torchvision", "numpy", "timm"),
        install_packages=("torch", "torchvision", "numpy", "timm"),
        asset_env=("WORLDFOUNDRY_REPVIT_SAM_CKPT",),
        assets=(
            BaseModelAsset(
                id="repvit_sam_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_REPVIT_SAM_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/RepViT/repvit_sam.pt",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/repvit_sam.pt",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/repvit_sam.pt",
                ),
                download_url="https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt",
                min_size_bytes=100_000_000,
                note="Official RepViT-SAM checkpoint from the THU-MIG/RepViT v1.0 release.",
            ),
        ),
    ),
    "sam2": BaseModelCapability(
        id="sam2",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.sam2",
        canonical_path="worldfoundry/base_models/perception_core/segment/sam2",
        package_imports=("torch", "numpy", "hydra", "iopath"),
        install_packages=("torch", "numpy", "hydra-core", "iopath"),
        asset_env=("WORLDFOUNDRY_SAM2_CKPT",),
        assets=(
            BaseModelAsset(
                id="sam2_base_plus_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_SAM2_CKPT",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt",
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt",
                ),
                hf_repo_id="facebook/sam2.1-hiera-base-plus",
                hf_filename="sam2.1_hiera_base_plus.pt",
                min_size_bytes=300_000_000,
                note="SAM2.1 base-plus checkpoint used by video segmentation metrics.",
            ),
        ),
    ),
    "sam3": BaseModelCapability(
        id="sam3",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.sam3",
        canonical_path="worldfoundry/base_models/perception_core/segment/sam3",
        package_imports=("torch", "numpy", "huggingface_hub", "hydra", "omegaconf", "iopath", "pycocotools"),
        install_packages=("torch", "numpy", "huggingface-hub", "hydra-core", "omegaconf", "iopath", "pycocotools"),
        asset_env=("WORLDFOUNDRY_SAM3_CKPT",),
        assets=(
            BaseModelAsset(
                id="sam3_multiplex_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_SAM3_CKPT",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--sam3.1/sam3.1_multiplex.pt",
                alternate_paths=(
                    "${WORLDFOUNDRY_HFD_ROOT}/facebook--sam3/sam3.pt",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--sam3/sam3.pt",
                    "${WORLDFOUNDRY_CKPT_DIR}/sam3.1_multiplex.pt",
                    "${WORLDFOUNDRY_CKPT_DIR}/sam3.pt",
                ),
                hf_repo_id="facebook/sam3.1",
                hf_filename="sam3.1_multiplex.pt",
                min_size_bytes=1_000_000_000,
                note="SAM3 checkpoint for unified image/video segmentation and tracking metrics.",
            ),
        ),
    ),
    "sam_v1": BaseModelCapability(
        id="sam_v1",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.sam_v1",
        canonical_path="worldfoundry/base_models/perception_core/segment/sam_v1",
        package_imports=("torch", "torchvision", "numpy"),
        install_packages=("torch", "torchvision", "numpy"),
        asset_env=("WORLDFOUNDRY_SAM_VIT_H_CKPT",),
        assets=(
            BaseModelAsset(
                id="sam_vit_h_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_SAM_VIT_H_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/sam_vit_h_4b8939.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_HFD_ROOT}/facebook--sam-vit-huge/sam_vit_h_4b8939.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--sam-vit-huge/sam_vit_h_4b8939.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/code/sam_vit_h_4b8939.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/shape-of-motion/checkpoints/sam_vit_h_4b8939.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WonderJourney/sam_vit_h_4b8939.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/sam_vit_h_4b8939.pth",
                ),
                download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                min_size_bytes=2_000_000_000,
                note="SAM ViT-H checkpoint shared by Grounded-SAM and WorldScore segmentation paths.",
            ),
        ),
    ),
    "sam_vit_b": BaseModelCapability(
        id="sam_vit_b",
        family="segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.segment.sam_v1",
        canonical_path="worldfoundry/base_models/perception_core/segment/sam_v1",
        package_imports=("torch", "torchvision", "numpy"),
        install_packages=("torch", "torchvision", "numpy"),
        asset_env=("WORLDFOUNDRY_SAM_VIT_B_CKPT",),
        assets=(
            BaseModelAsset(
                id="sam_vit_b_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_SAM_VIT_B_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/sam_vit_b_01ec64.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/sam_vit_b_01ec64.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/sam_vit_b_01ec64.pth",
                ),
                download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                min_size_bytes=350_000_000,
                note="SAM ViT-B checkpoint used by several lightweight official segmentation metric paths.",
            ),
        ),
    ),
    "aot_deaot_l": BaseModelCapability(
        id="aot_deaot_l",
        family="video_object_segmentation",
        canonical_owner="worldfoundry.base_models.perception_core.tracking.aot",
        canonical_path="worldfoundry/base_models/perception_core/tracking/aot",
        package_imports=("torch", "cv2", "numpy"),
        install_packages=("torch", "opencv-python", "numpy"),
        asset_env=("WORLDFOUNDRY_DEAOT_R50_CKPT",),
        assets=(
            BaseModelAsset(
                id="deaot_r50_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_DEAOT_R50_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/R50_DeAOTL_PRE_YTB_DAV.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
                ),
                min_size_bytes=150_000_000,
                note="DeAOT-L R50 checkpoint used by Segment-and-Track-Anything object attribute metrics.",
            ),
        ),
    ),
    "sea_raft": BaseModelCapability(
        id="sea_raft",
        family="optical_flow",
        canonical_owner="worldfoundry.base_models.perception_core.optical_flow.sea_raft",
        canonical_path="worldfoundry/base_models/perception_core/optical_flow/sea_raft",
        package_imports=("torch", "cv2", "numpy", "huggingface_hub"),
        install_packages=("torch", "opencv-python", "numpy", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_SEA_RAFT_CKPT",),
        assets=(
            BaseModelAsset(
                id="sea_raft_spring_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_SEA_RAFT_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/Tartan-C-T-TSKH-spring540x960-M.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/Tartan-C-T-TSKH-spring540x960-M.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/Tartan-C-T-TSKH-spring540x960-M.pth",
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/sea_raft_gdrive/Tartan-C-T-TSKH-spring540x960-M.pth",
                ),
                min_size_bytes=70_000_000,
                note="SEA-RAFT optical-flow checkpoint reused by WorldScore motion metrics.",
            ),
        ),
    ),
    "flowformerplusplus": BaseModelCapability(
        id="flowformerplusplus",
        family="optical_flow",
        canonical_owner="worldfoundry.base_models.perception_core.optical_flow.flowformerplusplus",
        canonical_path="worldfoundry/base_models/perception_core/optical_flow/flowformerplusplus",
        package_imports=("torch", "cv2", "numpy", "PIL", "scipy", "timm", "yacs", "einops", "loguru"),
        install_packages=("torch", "opencv-python", "numpy", "Pillow", "scipy", "timm", "yacs", "einops", "loguru"),
        asset_env=("WORLDFOUNDRY_FLOWFORMERPLUSPLUS_CKPT",),
        assets=(
            BaseModelAsset(
                id="flowformerplusplus_things_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_FLOWFORMERPLUSPLUS_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/things_288960.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/things_288960.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/things_288960.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/metrics/checkpoints/things_288960.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/things_288960.pth",
                ),
                manual_action=(
                    "Stage FlowFormer++ `things_288960.pth` under "
                    "$WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR or set $WORLDFOUNDRY_FLOWFORMERPLUSPLUS_CKPT. "
                    "Use the official FlowFormer++/WorldScore checkpoint release if the file is not in Hugging Face."
                ),
                min_size_bytes=50_000_000,
                note="FlowFormer++ Things checkpoint used by WorldScore optical-flow and motion-accuracy metrics.",
            ),
        ),
    ),
    "vfimamba": BaseModelCapability(
        id="vfimamba",
        family="frame_interpolation",
        canonical_owner="worldfoundry.base_models.perception_core.frame_interpolation.vfimamba",
        canonical_path="worldfoundry/base_models/perception_core/frame_interpolation/vfimamba",
        package_imports=("torch", "cv2", "numpy", "timm", "einops", "mamba_ssm", "huggingface_hub", "transformers"),
        install_packages=("torch", "opencv-python", "numpy", "timm", "einops", "mamba-ssm", "huggingface-hub", "transformers"),
        asset_env=("WORLDFOUNDRY_VFIMAMBA_CKPT", "WORLDFOUNDRY_VFIMAMBA_S_CKPT"),
        assets=(
            BaseModelAsset(
                id="vfimamba_vfimamba_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_VFIMAMBA_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/VFIMamba.pkl",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/VFIMamba.pkl",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/VFIMamba.pkl",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/metrics/checkpoints/VFIMamba.pkl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/VFIMamba.pkl",
                ),
                manual_action=(
                    "Stage VFIMamba.pkl under $WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR or set "
                    "$WORLDFOUNDRY_VFIMAMBA_CKPT. The runner can also load Hugging Face model ids through "
                    "`Model.from_pretrained()` when an official Hub checkpoint is available."
                ),
                min_size_bytes=50_000_000,
                note="VFIMamba checkpoint used by WorldScore motion-smoothness interpolation metrics.",
            ),
            BaseModelAsset(
                id="vfimamba_vfimamba_s_checkpoint",
                kind="file",
                role="checkpoint",
                required=False,
                env=("WORLDFOUNDRY_VFIMAMBA_S_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/VFIMamba_S.pkl",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/VFIMamba_S.pkl",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/VFIMamba_S.pkl",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/metrics/checkpoints/VFIMamba_S.pkl",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/VFIMamba_S.pkl",
                ),
                manual_action=(
                    "Stage VFIMamba_S.pkl under $WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR or set "
                    "$WORLDFOUNDRY_VFIMAMBA_S_CKPT when using the small VFIMamba configuration."
                ),
                min_size_bytes=30_000_000,
                note="Optional VFIMamba-S checkpoint for the small interpolation configuration.",
            ),
        ),
    ),
    "fastvqa": BaseModelCapability(
        id="fastvqa",
        family="video_quality",
        canonical_owner="worldfoundry.base_models.perception_core.video_quality.fastvqa",
        canonical_path="worldfoundry/base_models/perception_core/video_quality/fastvqa",
        package_imports=("torch", "torchvision", "cv2", "numpy", "decord", "timm", "yaml"),
        install_packages=("torch", "torchvision", "opencv-python", "numpy", "decord", "timm", "pyyaml"),
        asset_env=("WORLDFOUNDRY_FASTVQA_CKPT", "WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT"),
        assets=(
            BaseModelAsset(
                id="fastvqa_fastervqa_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_FASTVQA_CKPT", "WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT"),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/FAST-VQA/FAST_VQA_3D_1_1_Scr.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/fastvqa/FAST_VQA_3D_1_1_Scr.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/4dworldbench/FAST-VQA/FAST_VQA_3D_1_1_Scr.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/metrics/checkpoints/FAST_VQA_3D_1_1_Scr.pth",
                ),
                manual_action=(
                    "Stage the official FAST_VQA_3D_1_1_Scr.pth checkpoint under $WORLDFOUNDRY_CKPT_DIR/FAST-VQA "
                    "or set $WORLDFOUNDRY_FASTVQA_CKPT/$WORLDFOUNDRY_4DWORLDBENCH_FASTVQA_CKPT."
                ),
                min_size_bytes=10_000_000,
                note="Default FasterVQA checkpoint used by 4DWorldBench perceptual_fastvqa.",
            ),
            BaseModelAsset(
                id="fastvqa_fast_b_checkpoint",
                kind="file",
                role="checkpoint",
                required=False,
                env=("WORLDFOUNDRY_FASTVQA_B_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/FAST-VQA/FAST_VQA_B_1_4.pth",
                alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/fastvqa/FAST_VQA_B_1_4.pth",),
                manual_action="Stage FAST_VQA_B_1_4.pth only when using the FAST-VQA model variant.",
                min_size_bytes=10_000_000,
                note="Optional FAST-VQA-B checkpoint.",
            ),
            BaseModelAsset(
                id="fastvqa_fast_m_checkpoint",
                kind="file",
                role="checkpoint",
                required=False,
                env=("WORLDFOUNDRY_FASTVQA_M_CKPT",),
                local_path="${WORLDFOUNDRY_CKPT_DIR}/FAST-VQA/FAST_VQA_M_1_4.pth",
                alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/fastvqa/FAST_VQA_M_1_4.pth",),
                manual_action="Stage FAST_VQA_M_1_4.pth only when using the FAST-VQA-M model variant.",
                min_size_bytes=10_000_000,
                note="Optional FAST-VQA-M checkpoint.",
            ),
        ),
    ),
    "raft": BaseModelCapability(
        id="raft",
        family="optical_flow",
        canonical_owner="worldfoundry.base_models.perception_core.optical_flow.raft",
        canonical_path="worldfoundry/base_models/perception_core/optical_flow/raft",
        package_imports=("torch", "cv2", "numpy"),
        install_packages=("torch", "opencv-python", "numpy"),
        asset_env=("WORLDFOUNDRY_RAFT_THINGS_CKPT",),
        assets=(
            BaseModelAsset(
                id="raft_things_checkpoint",
                kind="file",
                role="checkpoint",
                env=("WORLDFOUNDRY_RAFT_THINGS_CKPT",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/models/raft-things.pth",
                alternate_paths=(
                    "${WORLDFOUNDRY_CKPT_DIR}/models/raft-things.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/raft-things.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/WorldScore/models/raft-things.pth",
                    "${WORLDFOUNDRY_CKPT_DIR}/monst3r_raft_models/raft-things.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/models/raft-things.pth",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/raft-things.pth",
                ),
                download_command_template=(
                    "RAFT_CKPT=${WORLDFOUNDRY_RAFT_THINGS_CKPT:-${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/models/raft-things.pth}; "
                    "RAFT_ZIP=${WORLDFOUNDRY_RAFT_MODELS_ZIP:-${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}/models.zip}; "
                    "mkdir -p \"$(dirname \"$RAFT_CKPT\")\" \"$(dirname \"$RAFT_ZIP\")\"; "
                    "if [ ! -f \"$RAFT_CKPT\" ]; then "
                    "curl -L --fail --retry 5 --retry-delay 2 https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip -o \"$RAFT_ZIP\" "
                    "&& python -c 'import pathlib, sys, zipfile; "
                    "member=\"models/raft-things.pth\"; "
                    "archive=zipfile.ZipFile(sys.argv[1]); "
                    "info=archive.getinfo(member); "
                    "target=pathlib.Path(sys.argv[2]); "
                    "target.parent.mkdir(parents=True, exist_ok=True); "
                    "target.write_bytes(archive.read(info)); "
                    "print(target)' \"$RAFT_ZIP\" \"$RAFT_CKPT\"; "
                    "fi"
                ),
                min_size_bytes=20_000_000,
                note="RAFT Things checkpoint reused by fallback optical-flow metric paths; downloaded from the upstream RAFT models.zip bundle.",
            ),
        ),
    ),
    "vggt_1b": BaseModelCapability(
        id="vggt_1b",
        family="multi_view_geometry",
        canonical_owner="worldfoundry.base_models.three_dimensions.point_clouds.vggt",
        canonical_path="worldfoundry/base_models/three_dimensions/point_clouds/vggt",
        package_imports=("torch", "numpy", "einops", "huggingface_hub", "safetensors"),
        install_packages=("torch", "numpy", "einops", "huggingface-hub", "safetensors"),
        asset_env=("WORLDFOUNDRY_VGGT_1B_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="vggt_1b_model_dir",
                kind="dir",
                role="model_dir",
                env=("WORLDFOUNDRY_VGGT_1B_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/facebook--VGGT-1B",
                alternate_paths=(
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/facebook--VGGT-1B",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/facebook--VGGT-1B",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/facebook--VGGT-1B",
                    "${WORLDFOUNDRY_HFD_ROOT}/VGGT-1B",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/VGGT-1B",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/VGGT-1B",
                ),
                hf_repo_id="facebook/VGGT-1B",
                min_file_count=1,
                required_files=("config.json", "model.safetensors"),
                note="VGGT 1B multi-view geometry prior for camera, depth, point, and track estimates.",
            ),
        ),
    ),
    "coco2017_detection_segmentation_assets": _perception_data_capability(
        capability_id="coco2017_detection_segmentation_assets",
        env="WORLDFOUNDRY_COCO2017_ROOT",
        local_path="${WORLDFOUNDRY_DATA_DIR}/perception/coco2017",
        alternate_paths=(
            "${WORLDFOUNDRY_REPO_ROOT}/data/perception/coco2017",
            "${WORLDFOUNDRY_BENCH_ROOT}/data/coco2017",
            "${WORLDFOUNDRY_HFD_DATASET_ROOT}/coco2017",
        ),
        required_paths=("val2017", "annotations/instances_val2017.json"),
        min_file_count=10,
        download_command_template=(
            "COCO_ROOT=${WORLDFOUNDRY_COCO2017_ROOT:-${WORLDFOUNDRY_DATA_DIR}/perception/coco2017}; "
            "mkdir -p \"$COCO_ROOT\" && cd \"$COCO_ROOT\"; "
            "if [ ! -d val2017 ]; then "
            "curl -L --fail --retry 5 --retry-delay 2 http://images.cocodataset.org/zips/val2017.zip -o val2017.zip "
            "&& python -m zipfile -e val2017.zip .; "
            "fi; "
            "if [ ! -f annotations/instances_val2017.json ]; then "
            "curl -L --fail --retry 5 --retry-delay 2 http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip "
            "&& python -m zipfile -e annotations_trainval2017.zip .; "
            "fi"
        ),
        note="COCO 2017 val images and instance annotations shared by detection and segmentation evaluator validation runs.",
    ),
    "davis2017_video_segmentation_assets": _perception_data_capability(
        capability_id="davis2017_video_segmentation_assets",
        env="WORLDFOUNDRY_DAVIS2017_ROOT",
        local_path="${WORLDFOUNDRY_DATA_DIR}/perception/davis2017/DAVIS",
        alternate_paths=(
            "${WORLDFOUNDRY_REPO_ROOT}/data/perception/davis2017/DAVIS",
            "${WORLDFOUNDRY_REPO_ROOT}/data/perception/DAVIS",
            "${WORLDFOUNDRY_BENCH_ROOT}/data/DAVIS",
        ),
        required_paths=("JPEGImages/480p", "Annotations/480p", "ImageSets/2017/val.txt"),
        min_file_count=20,
        download_command_template=(
            "DAVIS_ROOT=${WORLDFOUNDRY_DAVIS2017_ROOT:-${WORLDFOUNDRY_DATA_DIR}/perception/davis2017/DAVIS}; "
            "DAVIS_PARENT=$(dirname \"$DAVIS_ROOT\"); "
            "mkdir -p \"$DAVIS_PARENT\"; "
            "if [ ! -d \"$DAVIS_ROOT/JPEGImages/480p\" ]; then "
            "curl -L --fail --retry 5 --retry-delay 2 https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip "
            "-o \"$DAVIS_PARENT/DAVIS-2017-trainval-480p.zip\" "
            "&& python -m zipfile -e \"$DAVIS_PARENT/DAVIS-2017-trainval-480p.zip\" \"$DAVIS_PARENT\"; "
            "fi"
        ),
        note="DAVIS 2017 trainval 480p bundle for video object segmentation and tracking metric validation.",
    ),
    "tum_rgbd_slam_sample_assets": _perception_data_capability(
        capability_id="tum_rgbd_slam_sample_assets",
        env="WORLDFOUNDRY_TUM_RGBD_SLAM_SAMPLE_ROOT",
        local_path="${WORLDFOUNDRY_DATA_DIR}/perception/tum_rgbd/rgbd_dataset_freiburg1_desk",
        alternate_paths=(
            "${WORLDFOUNDRY_REPO_ROOT}/data/perception/tum_rgbd/rgbd_dataset_freiburg1_desk",
            "${WORLDFOUNDRY_BENCH_ROOT}/data/tum_rgbd/rgbd_dataset_freiburg1_desk",
        ),
        required_files=("rgb.txt", "depth.txt", "groundtruth.txt"),
        required_paths=("rgb", "depth"),
        min_file_count=10,
        download_command_template=(
            "TUM_ROOT=${WORLDFOUNDRY_TUM_RGBD_SLAM_SAMPLE_ROOT:-${WORLDFOUNDRY_DATA_DIR}/perception/tum_rgbd/rgbd_dataset_freiburg1_desk}; "
            "TUM_PARENT=$(dirname \"$TUM_ROOT\"); "
            "mkdir -p \"$TUM_PARENT\"; "
            "if [ ! -f \"$TUM_ROOT/rgb.txt\" ]; then "
            "curl -L --fail --retry 5 --retry-delay 2 https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz "
            "-o \"$TUM_PARENT/rgbd_dataset_freiburg1_desk.tgz\" "
            "&& tar -xzf \"$TUM_PARENT/rgbd_dataset_freiburg1_desk.tgz\" -C \"$TUM_PARENT\"; "
            "fi"
        ),
        note="Small official TUM RGB-D Freiburg1 desk sequence for SLAM/depth validation and trajectory sanity checks.",
    ),
    "kitti_depth_slam_assets": _perception_data_capability(
        capability_id="kitti_depth_slam_assets",
        env="WORLDFOUNDRY_KITTI_ROOT",
        local_path="${WORLDFOUNDRY_DATA_DIR}/perception/kitti",
        alternate_paths=(
            "${WORLDFOUNDRY_REPO_ROOT}/data/perception/kitti",
            "${WORLDFOUNDRY_BENCH_ROOT}/data/kitti",
        ),
        required_any_paths=("sequences", "dataset/sequences", "data_odometry_gray", "data_depth_annotated"),
        min_file_count=10,
        manual_action=(
            "Stage KITTI odometry/depth assets under $WORLDFOUNDRY_KITTI_ROOT. "
            "Accepted markers include sequences/, dataset/sequences/, data_odometry_gray/, or data_depth_annotated/. "
            "Use the official KITTI download pages because the full assets require their terms/download flow."
        ),
        note="KITTI odometry/depth staging root for outdoor SLAM, depth, and driving perception evaluations.",
    ),
    "worldscore_official_assets": BaseModelCapability(
        id="worldscore_official_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_WORLDSCORE_DATASET_DIR", "WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR"),
        asset_env_policy="all",
        assets=(
            BaseModelAsset(
                id="worldscore_prompt_dataset",
                kind="dir",
                role="benchmark_dataset",
                env=("WORLDFOUNDRY_WORLDSCORE_DATASET_DIR",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_DATASET_DIR}",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_DIR}/data/WorldScore-Dataset",
                    "${WORLDFOUNDRY_BENCH_ROOT}/data/WorldScore-Dataset",
                ),
                min_file_count=4,
                note="WorldScore static/dynamic prompt metadata used by bounded and full official metric runs.",
            ),
            BaseModelAsset(
                id="worldscore_metric_checkpoint_dir",
                kind="dir",
                role="metric_checkpoint_bundle",
                env=("WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR",),
                local_path="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR}",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORLDSCORE_ASSET_DIR}/metrics/checkpoints",
                ),
                min_file_count=8,
                note="WorldScore metric checkpoint bundle containing DROID, SAM, SAM2, RAFT, SEA-RAFT, and quality checkpoints.",
            ),
        ),
        gpu_required=False,
    ),
    "videoscore_reward_model_v1_1": BaseModelCapability(
        id="videoscore_reward_model_v1_1",
        family="video_reward_model",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        package_imports=("torch", "transformers", "accelerate", "decord", "PIL", "numpy", "huggingface_hub"),
        install_packages=("torch", "transformers", "accelerate", "decord", "Pillow", "numpy", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_VIDEOSCORE_MODEL_DIR",),
        assets=(
            BaseModelAsset(
                id="videoscore_v1_1_model_dir",
                kind="dir",
                role="reward_model_dir",
                env=("WORLDFOUNDRY_VIDEOSCORE_MODEL_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/TIGER-Lab--VideoScore-v1.1",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/TIGER-Lab--VideoScore-v1.1",
                    "${WORLDFOUNDRY_BENCH_ROOT}/ckpt/hfd/TIGER-Lab--VideoScore-v1.1",
                    "${WORLDFOUNDRY_HFD_ROOT}/TIGER-Lab--VideoScore",
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/TIGER-Lab--VideoScore",
                ),
                hf_repo_id="TIGER-Lab/VideoScore-v1.1",
                min_file_count=8,
                required_files=("config.json", "processor_config.json", "model.safetensors.index.json"),
                note="VideoScore v1.1 reward model used by the bounded official scorer validation and VideoScore-Bench metrics.",
            ),
        ),
    ),
    "visual_chronometer_fps_predictor": BaseModelCapability(
        id="visual_chronometer_fps_predictor",
        family="video_metric_model",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        package_imports=("torch", "numpy", "cv2", "omegaconf", "huggingface_hub"),
        install_packages=("torch", "numpy", "opencv-python", "omegaconf", "huggingface-hub"),
        asset_env=("WORLDFOUNDRY_VISUAL_CHRONOMETER_CKPT_DIR",),
        assets=(
            BaseModelAsset(
                id="visual_chronometer_fps_ckpt",
                kind="file",
                role="metric_checkpoint",
                env=("WORLDFOUNDRY_VISUAL_CHRONOMETER_CKPT_DIR",),
                local_path="${WORLDFOUNDRY_HFD_ROOT}/xiangbog--Visual_Chronometer/vc_common_10_60fps.ckpt",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/xiangbog--Visual_Chronometer/vc_common_10_60fps.ckpt",
                    "${WORLDFOUNDRY_VISUAL_CHRONOMETER_ROOT}/inference/ckpts/vc_common_10_60fps.ckpt",
                ),
                hf_repo_id="xiangbog/Visual_Chronometer",
                hf_filename="vc_common_10_60fps.ckpt",
                min_file_count=1,
                note="Visual Chronometer PhyFPS predictor checkpoint used by in-tree official-run surfaces.",
            ),
        ),
    ),
    "videoscore_bench_dataset_assets": BaseModelCapability(
        id="videoscore_bench_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_VIDEOSCORE_BENCH_DATASET_ROOT", "WORLDFOUNDRY_VIDEOFEEDBACK_DATASET_ROOT"),
        assets=(
            BaseModelAsset(
                id="videoscore_bench_dataset_dir",
                kind="dir",
                role="benchmark_dataset",
                env=("WORLDFOUNDRY_VIDEOSCORE_BENCH_DATASET_ROOT",),
                local_path="${WORLDFOUNDRY_HFD_DATASET_ROOT}/TIGER-Lab__VideoScore-Bench",
                alternate_paths=(
                    "${WORLDFOUNDRY_WORKSPACE_ROOT}/data/hfd_datasets/TIGER-Lab__VideoScore-Bench",
                ),
                hf_repo_id="TIGER-Lab/VideoScore-Bench",
                hf_repo_type="dataset",
                min_file_count=4,
                required_files=("README.md", "video_feedback/test-00000-of-00001.parquet"),
                note="VideoScore-Bench metadata and frame archives for the bounded VideoScore official scorer path.",
            ),
            _hf_dataset_asset(
                asset_id="videofeedback_dataset_dir",
                role="benchmark_dataset",
                env="WORLDFOUNDRY_VIDEOFEEDBACK_DATASET_ROOT",
                repo_id="TIGER-Lab/VideoFeedback",
                local_name="TIGER-Lab__VideoFeedback",
                alternate_local_names=("TIGER-Lab/VideoFeedback",),
                note="VideoFeedback split referenced by VideoScore reward-model benchmark manifests.",
            ),
        ),
        gpu_required=False,
    ),
    "camerabench_dataset_assets": BaseModelCapability(
        id="camerabench_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_CAMERABENCH_DATASET_ROOT",),
        assets=(
            _hf_dataset_asset(
                asset_id="camerabench_dataset_dir",
                role="benchmark_dataset",
                env="WORLDFOUNDRY_CAMERABENCH_DATASET_ROOT",
                repo_id="syCen/CameraBench",
                revision="659f73c8899d81ffebd85876e3cdd137435b5e1b",
                local_name="syCen__CameraBench",
                alternate_local_names=("syCen/CameraBench",),
                note="CameraBench official prompt/video metadata used by camera-motion classification metrics.",
            ),
        ),
        gpu_required=False,
    ),
    "t2v_compbench_dataset_assets": BaseModelCapability(
        id="t2v_compbench_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_T2V_COMPBENCH_DATASET_ROOT",),
        assets=(
            _hf_dataset_asset(
                asset_id="t2v_compbench_videos_dir",
                role="benchmark_dataset",
                env="WORLDFOUNDRY_T2V_COMPBENCH_DATASET_ROOT",
                repo_id="Kaiyue/T2V-CompBench-Videos",
                revision="92f9ef4642f244567e8aa2789827ec301500a2ff",
                local_name="Kaiyue__T2V-CompBench-Videos",
                alternate_local_names=("Kaiyue/T2V-CompBench-Videos",),
                note="T2V-CompBench reference videos and annotations for compositional video evaluation.",
            ),
        ),
        gpu_required=False,
    ),
    "fetv_dataset_assets": BaseModelCapability(
        id="fetv_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_FETV_DATASET_ROOT",),
        assets=(
            _hf_dataset_asset(
                asset_id="fetv_dataset_dir",
                role="benchmark_dataset",
                env="WORLDFOUNDRY_FETV_DATASET_ROOT",
                repo_id="lyx97/FETV",
                revision="e9a6c057cb6ee9257f29e44d427117e8bd0d704f",
                local_name="lyx97__FETV",
                alternate_local_names=("lyx97/FETV",),
                note="FETV fine-grained text-to-video evaluation prompts, annotations, and references.",
            ),
        ),
        gpu_required=False,
    ),
    "fetv_blip_retrieval_source": BaseModelCapability(
        id="fetv_blip_retrieval_source",
        family="video_metric_model",
        canonical_owner="worldfoundry.base_models.perception_core.video_text.fetv_blip",
        canonical_path="worldfoundry/base_models/perception_core/video_text/fetv_blip",
        package_imports=("torch", "transformers", "PIL", "numpy", "timm", "fairscale"),
        install_packages=("torch", "transformers", "Pillow", "numpy", "timm", "fairscale"),
        gpu_required=True,
    ),
    "fetv_stylegan_v_fvd_source": BaseModelCapability(
        id="fetv_stylegan_v_fvd_source",
        family="video_metric_model",
        canonical_owner="worldfoundry.base_models.perception_core.video_quality.fetv_stylegan_v",
        canonical_path="worldfoundry/base_models/perception_core/video_quality/fetv_stylegan_v",
        package_imports=("torch", "torchvision", "PIL", "numpy", "scipy", "click", "omegaconf"),
        install_packages=("torch", "torchvision", "Pillow", "numpy", "scipy", "click", "omegaconf"),
        gpu_required=True,
    ),
    "vbench2_dataset_assets": BaseModelCapability(
        id="vbench2_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=(
            "WORLDFOUNDRY_VBENCH2_SAMPLED_VIDEOS_ROOT",
            "WORLDFOUNDRY_VBENCH2_HUMAN_ANNOTATION_ROOT",
            "WORLDFOUNDRY_VBENCH2_HUMAN_ANOMALY_ROOT",
        ),
        asset_env_policy="all",
        assets=(
            _hf_dataset_asset(
                asset_id="vbench2_sampled_videos_dir",
                role="reference_video_dataset",
                env="WORLDFOUNDRY_VBENCH2_SAMPLED_VIDEOS_ROOT",
                repo_id="Vchitect/VBench-2.0_sampled_videos",
                revision="f9ee15efcfb86d96bd5379e8d85f8d69b901aad1",
                local_name="Vchitect__VBench-2.0_sampled_videos",
                alternate_local_names=("Vchitect/VBench-2.0_sampled_videos",),
                note="VBench-2.0 sampled videos used by full-reference and human-alignment dimensions.",
            ),
            _hf_dataset_asset(
                asset_id="vbench2_human_annotation_dir",
                role="benchmark_annotation_dataset",
                env="WORLDFOUNDRY_VBENCH2_HUMAN_ANNOTATION_ROOT",
                repo_id="Vchitect/VBench-2.0_human_annotation",
                revision="3301a1a2d1cdb743b6dfdd641033c354335d1d48",
                local_name="Vchitect__VBench-2.0_human_annotation",
                alternate_local_names=("Vchitect/VBench-2.0_human_annotation",),
                note="VBench-2.0 human preference annotations.",
            ),
            _hf_dataset_asset(
                asset_id="vbench2_human_anomaly_dir",
                role="benchmark_annotation_dataset",
                env="WORLDFOUNDRY_VBENCH2_HUMAN_ANOMALY_ROOT",
                repo_id="Vchitect/VBench-2.0_human_anomaly",
                revision="c84fad03a01bb93446588080f55004fe46d40125",
                local_name="Vchitect__VBench-2.0_human_anomaly",
                alternate_local_names=("Vchitect/VBench-2.0_human_anomaly",),
                note="VBench-2.0 human anomaly annotations.",
            ),
        ),
        gpu_required=False,
    ),
    "video_bench_dataset_assets": BaseModelCapability(
        id="video_bench_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_VIDEOBENCH_ANNOTATION_ROOT", "WORLDFOUNDRY_VIDEOBENCH_OFFICIAL_VIDEOS_ROOT"),
        asset_env_policy="all",
        assets=(
            _hf_dataset_asset(
                asset_id="video_bench_human_annotation_dir",
                role="benchmark_annotation_dataset",
                env="WORLDFOUNDRY_VIDEOBENCH_ANNOTATION_ROOT",
                repo_id="Video-Bench/Video-Bench_human_annotation",
                revision="a8371b69c4a91ef1162d8b6af12c1759f1f19983",
                local_name="Video-Bench__Video-Bench_human_annotation",
                alternate_local_names=("Video-Bench/Video-Bench_human_annotation",),
                note="Video-Bench human annotation split used by official-result normalization.",
            ),
            _hf_dataset_asset(
                asset_id="video_bench_official_videos_dir",
                role="reference_video_dataset",
                env="WORLDFOUNDRY_VIDEOBENCH_OFFICIAL_VIDEOS_ROOT",
                repo_id="Video-Bench/Video-Bench_videos",
                revision="9000d1e14055106c8b8383c59e48da3d31648225",
                local_name="Video-Bench__Video-Bench_videos",
                alternate_local_names=("Video-Bench/Video-Bench_videos",),
                note="Gated Video-Bench official videos; hf download works only after access is granted.",
            ),
        ),
        gpu_required=False,
    ),
    "worldbench_dataset_assets": BaseModelCapability(
        id="worldbench_dataset_assets",
        family="benchmark_data_asset",
        canonical_owner="worldfoundry.base_models.capabilities",
        canonical_path="worldfoundry/base_models/capabilities.py",
        asset_env=("WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT",),
        assets=(
            _hf_dataset_asset(
                asset_id="worldbench_dataset_dir",
                role="benchmark_dataset",
                env="WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT",
                repo_id="worldbenchmark/WorldBench",
                revision="c72084cfc4500035032450b812bfd6b2956ed3e9",
                local_name="worldbenchmark__WorldBench",
                alternate_local_names=(
                    "worldbenchmark/WorldBench",
                    "worldbenchmark/IntuitivePhysics",
                    "worldbenchmark__IntuitivePhysics",
                ),
                note="WorldBench public HF dataset used by world-understanding normalizer surfaces.",
            ),
        ),
        gpu_required=False,
    ),
}


BASE_MODEL_CAPABILITIES.update(
    {
        "wbench_dataset_assets": BaseModelCapability(
            id="wbench_dataset_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.base_models.capabilities",
            canonical_path="worldfoundry/base_models/capabilities.py",
            asset_env=("WORLDFOUNDRY_WBENCH_DATASET_ROOT",),
            assets=(
                _hf_dataset_asset(
                    asset_id="wbench_dataset_dir",
                    role="benchmark_dataset",
                    env="WORLDFOUNDRY_WBENCH_DATASET_ROOT",
                    repo_id="meituan-longcat/WBench",
                    local_name="meituan-longcat__WBench",
                    alternate_local_names=("meituan-longcat/WBench",),
                    note="WBench public HF dataset used by the in-tree official runtime.",
                ),
            ),
            gpu_required=False,
        ),
        "wbench_quality_metric_assets": BaseModelCapability(
            id="wbench_quality_metric_assets",
            family="video_quality_metric_model",
            canonical_owner="worldfoundry.base_models.perception_core.general_perception.openai_clip",
            canonical_path="worldfoundry/base_models/perception_core/general_perception/openai_clip.py",
            package_imports=("pyiqa", "transformers", "torch", "torchvision", "ftfy", "regex", "tqdm"),
            install_packages=("pyiqa", "transformers", "torch", "torchvision", "ftfy", "regex", "tqdm"),
            asset_env=(
                "WORLDFOUNDRY_WBENCH_CLIP_VIT_B32_PT",
                "WORLDFOUNDRY_WBENCH_CLIP_VIT_L14_PT",
                "WORLDFOUNDRY_WBENCH_AESTHETIC_LINEAR_CKPT",
                "WORLDFOUNDRY_WBENCH_MUSIQ_SPAQ_CKPT",
                "WORLDFOUNDRY_WBENCH_CLIP_VIT_B16_DIR",
            ),
            asset_env_policy="all",
            assets=(
                BaseModelAsset(
                    id="wbench_clip_vit_b32_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_CLIP_VIT_B32_PT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/clip_model/ViT-B-32.pt",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/clip_model/ViT-B-32.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/clip/ViT-B-32.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/clip_model/ViT-B-32.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/VBench/clip_model/ViT-B-32.pt",
                    ),
                    download_url="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                    min_size_bytes=300_000_000,
                    note="OpenAI CLIP ViT-B/32 checkpoint used by WBench background consistency.",
                ),
                BaseModelAsset(
                    id="wbench_clip_vit_l14_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_CLIP_VIT_L14_PT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/clip_model/ViT-L-14.pt",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/clip_model/ViT-L-14.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/clip/ViT-L-14.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/clip_model/ViT-L-14.pt",
                        "${WORLDFOUNDRY_CKPT_DIR}/VBench/clip_model/ViT-L-14.pt",
                    ),
                    download_url="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
                    min_size_bytes=800_000_000,
                    note="OpenAI CLIP ViT-L/14 checkpoint used by WBench aesthetic quality.",
                ),
                BaseModelAsset(
                    id="wbench_aesthetic_linear_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_AESTHETIC_LINEAR_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/aesthetic/sa_0_4_vit_l_14_linear.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/VBench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
                    ),
                    download_url="https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
                    min_size_bytes=1_000,
                    note="LAION aesthetic predictor linear head used by WBench aesthetic quality.",
                ),
                BaseModelAsset(
                    id="wbench_musiq_spaq_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_MUSIQ_SPAQ_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/pyiqa/musiq_spaq_ckpt-358bb6af.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/VBench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
                    ),
                    download_url="https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
                    min_size_bytes=100_000_000,
                    note="MUSIQ-SPAQ checkpoint used by WBench imaging quality.",
                ),
                BaseModelAsset(
                    id="wbench_clip_vit_b16_model_dir",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_WBENCH_CLIP_VIT_B16_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/openai--clip-vit-base-patch16",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/openai--clip-vit-base-patch16",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/clip-vit-base-patch16",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/hfd/openai--clip-vit-base-patch16",
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/openai--clip-vit-base-patch16",
                    ),
                    hf_repo_id="openai/clip-vit-base-patch16",
                    min_file_count=1,
                    required_files=("config.json",),
                    required_any_paths=("model.safetensors", "pytorch_model.bin"),
                    note="HF CLIP ViT-B/16 model used by WBench subject consistency.",
                ),
            ),
        ),
        "wbench_transnetv2": BaseModelCapability(
            id="wbench_transnetv2",
            family="video_quality",
            canonical_owner="worldfoundry.base_models.perception_core.video_quality.transnetv2",
            canonical_path="worldfoundry/base_models/perception_core/video_quality/transnetv2.py",
            package_imports=("torch", "cv2", "numpy", "tqdm"),
            install_packages=("torch", "opencv-python", "numpy", "tqdm"),
            asset_env=("WORLDFOUNDRY_WBENCH_TRANSNETV2_CKPT",),
            assets=(
                BaseModelAsset(
                    id="wbench_transnetv2_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_TRANSNETV2_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/transnetv2/transnetv2-pytorch-weights.pth",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/transnetv2/transnetv2-pytorch-weights.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/transnetv2/transnetv2-pytorch-weights.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/wbench/transnetv2/transnetv2-pytorch-weights.pth",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    hf_filename="transnetv2/transnetv2-pytorch-weights.pth",
                    min_size_bytes=10_000_000,
                    note="TransNetV2 scene-cut checkpoint used by WBench segment_continuity.",
                ),
            ),
        ),
        "wbench_hpsv3": BaseModelCapability(
            id="wbench_hpsv3",
            family="video_reward_model",
            canonical_owner="worldfoundry.base_models.perception_core.video_quality.hpsv3",
            canonical_path="worldfoundry/base_models/perception_core/video_quality/hpsv3.py",
            package_imports=("torch", "transformers", "safetensors", "peft", "trl", "yaml"),
            install_packages=("torch", "transformers", "safetensors", "peft", "trl", "PyYAML"),
            asset_env=(
                "WORLDFOUNDRY_WBENCH_HPSV3_CKPT",
                "WORLDFOUNDRY_WBENCH_HPSV3_CONFIG",
                "WORLDFOUNDRY_WBENCH_HPSV3_QWEN2_VL_DIR",
            ),
            asset_env_policy="all",
            assets=(
                BaseModelAsset(
                    id="wbench_hpsv3_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_HPSV3_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/HPSv3/HPSv3.safetensors",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/HPSv3/HPSv3.safetensors",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/HPSv3/HPSv3.safetensors",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    hf_filename="HPSv3/HPSv3.safetensors",
                    min_size_bytes=100_000_000,
                    note="HPSv3 reward checkpoint used by WBench hpsv3_quality.",
                ),
                BaseModelAsset(
                    id="wbench_hpsv3_config",
                    kind="file",
                    role="config",
                    env=("WORLDFOUNDRY_WBENCH_HPSV3_CONFIG",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/HPSv3/HPSv3_7B_local.yaml",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/HPSv3/HPSv3_7B_local.yaml",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/HPSv3/HPSv3_7B_local.yaml",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    hf_filename="HPSv3/HPSv3_7B_local.yaml",
                    min_size_bytes=100,
                    note="HPSv3 local inference config used by WBench.",
                ),
                BaseModelAsset(
                    id="wbench_hpsv3_qwen2_vl_model",
                    kind="dir",
                    role="text_encoder_model_dir",
                    env=("WORLDFOUNDRY_WBENCH_HPSV3_QWEN2_VL_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/Qwen--Qwen2-VL-7B-Instruct",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/Qwen/Qwen2-VL-7B-Instruct",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/Qwen--Qwen2-VL-7B-Instruct",
                        "${WORLDFOUNDRY_CKPT_DIR}/Qwen2-VL-7B-Instruct",
                    ),
                    hf_repo_id="Qwen/Qwen2-VL-7B-Instruct",
                    min_file_count=1,
                    required_any_paths=("config.json",),
                    note="Qwen2-VL-7B base model referenced by HPSv3.",
                ),
            ),
        ),
        "wbench_pavrm_qwen3vl": BaseModelCapability(
            id="wbench_pavrm_qwen3vl",
            family="video_reward_model",
            canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.qwen.wbench_visual_plausibility",
            canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/qwen/wbench_visual_plausibility.py",
            package_imports=("torch", "transformers"),
            install_packages=("torch", "transformers"),
            asset_env=("WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR", "PAVRM_MODEL_PATH"),
            assets=(
                BaseModelAsset(
                    id="wbench_pavrm_qwen3vl_model",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR", "PAVRM_MODEL_PATH"),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/qwen3vl-a3b-visual-plausibility",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/qwen3vl-a3b-visual-plausibility",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/qwen3vl-a3b-visual-plausibility",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/qwen3vl-a3b-visual-plausibility",
                    ),
                    min_file_count=1,
                    required_any_paths=("config.json", "model.safetensors"),
                    manual_action=(
                        "Stage the WBench PAVRM Qwen3-VL visual plausibility model directory and set "
                        "WORLDFOUNDRY_WBENCH_PAVRM_MODEL_DIR or PAVRM_MODEL_PATH."
                    ),
                    note="PAVRM Qwen3-VL reward model used by WBench visual_plausibility.",
                ),
            ),
        ),
        "wbench_dreamsim": BaseModelCapability(
            id="wbench_dreamsim",
            family="visual_embedding",
            canonical_owner="worldfoundry.base_models.perception_core.video_quality.dreamsim",
            canonical_path="worldfoundry/base_models/perception_core/video_quality/dreamsim.py",
            package_imports=("torch", "torchvision", "peft", "timm", "open_clip"),
            install_packages=("torch", "torchvision", "peft", "timm", "open-clip-torch"),
            asset_env=("WORLDFOUNDRY_WBENCH_DREAMSIM_CACHE_DIR",),
            assets=(
                BaseModelAsset(
                    id="wbench_dreamsim_cache_dir",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_WBENCH_DREAMSIM_CACHE_DIR",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/dreamsim",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/dreamsim",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/dreamsim",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    min_file_count=1,
                    required_any_paths=("ensemble_lora/adapter_model.safetensors", "dino_vitb16_pretrain.pth"),
                    note="DreamSim local cache used by WBench spatial_consistency.",
                ),
            ),
        ),
        "wbench_megasam": BaseModelCapability(
            id="wbench_megasam",
            family="geometry",
            canonical_owner="worldfoundry.base_models.three_dimensions.slam.megasam",
            canonical_path="worldfoundry/base_models/three_dimensions/slam/megasam.py",
            package_imports=("torch", "cv2", "numpy", "lietorch", "droid_backends"),
            install_packages=("torch", "opencv-python", "numpy"),
            non_pip_imports=("lietorch", "droid_backends"),
            manual_install_actions=(
                "Build MegaSAM CUDA extensions from "
                "worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base with `pip install -e .`.",
            ),
            asset_env=(
                "WORLDFOUNDRY_WBENCH_MEGASAM_CKPT",
                "WORLDFOUNDRY_WBENCH_MEGASAM_DEPTH_ANYTHING_CKPT",
            ),
            asset_env_policy="all",
            assets=(
                BaseModelAsset(
                    id="wbench_megasam_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_MEGASAM_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/megasam/megasam_final.pth",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/megasam/megasam_final.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/megasam/megasam_final.pth",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    hf_filename="megasam/megasam_final.pth",
                    min_size_bytes=10_000_000,
                    note="MegaSAM camera-pose checkpoint used by WBench navigation/spatial metrics.",
                ),
                BaseModelAsset(
                    id="wbench_megasam_depth_anything_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_WBENCH_MEGASAM_DEPTH_ANYTHING_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/wbench/megasam/depth_anything_vitl14.pth",
                    alternate_paths=(
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/megasam/depth_anything_vitl14.pth",
                        "${WORLDFOUNDRY_CKPT_DIR}/WBench/weights/megasam/depth_anything_vitl14.pth",
                    ),
                    hf_repo_id="meituan-longcat/WBench-weights",
                    hf_filename="megasam/depth_anything_vitl14.pth",
                    min_size_bytes=100_000_000,
                    note="Depth-Anything ViT-L checkpoint used by MegaSAM mono-depth preprocessing.",
                ),
            ),
        ),
    }
)


BASE_MODEL_CAPABILITIES.update(
    {
        "aigcbench_dataset_assets": _benchmark_dataset_capability(
            "aigcbench_dataset_assets",
            (
                {
                    "asset_id": "aigcbench_dataset_dir",
                    "env": "WORLDFOUNDRY_AIGCBENCH_DATASET_ROOT",
                    "repo_id": "stevenfan/AIGCBench_v1.0",
                    "local_name": "stevenfan__AIGCBench_v1.0",
                    "alternate_local_names": ("stevenfan/AIGCBench_v1.0",),
                    "note": "AIGCBench public dataset for prompt/reference assets and official-result normalization.",
                },
            ),
        ),
        "chronomagic_dataset_assets": _benchmark_dataset_capability(
            "chronomagic_dataset_assets",
            (
                {
                    "asset_id": "chronomagic_bench_dataset_dir",
                    "env": "WORLDFOUNDRY_CHRONOMAGIC_BENCH_DATASET_ROOT",
                    "repo_id": "BestWishYsh/ChronoMagic-Bench",
                    "local_name": "BestWishYsh__ChronoMagic-Bench",
                    "alternate_local_names": ("BestWishYsh/ChronoMagic-Bench",),
                    "note": "ChronoMagic-Bench reference dataset for bounded and full official temporal reasoning runs.",
                },
                {
                    "asset_id": "chronomagic_pro_dataset_dir",
                    "env": "WORLDFOUNDRY_CHRONOMAGIC_PRO_DATASET_ROOT",
                    "repo_id": "BestWishYsh/ChronoMagic-Pro",
                    "local_name": "BestWishYsh__ChronoMagic-Pro",
                    "alternate_local_names": ("BestWishYsh/ChronoMagic-Pro",),
                    "note": "ChronoMagic-Pro reference dataset for extended temporal reasoning evaluation.",
                },
                {
                    "asset_id": "chronomagic_proh_dataset_dir",
                    "env": "WORLDFOUNDRY_CHRONOMAGIC_PROH_DATASET_ROOT",
                    "repo_id": "BestWishYsh/ChronoMagic-ProH",
                    "local_name": "BestWishYsh__ChronoMagic-ProH",
                    "alternate_local_names": ("BestWishYsh/ChronoMagic-ProH",),
                    "note": "ChronoMagic-ProH hard split assets for full official evaluation.",
                },
            ),
        ),
        "ewmbench_dataset_assets": _benchmark_dataset_capability(
            "ewmbench_dataset_assets",
            (
                {
                    "asset_id": "ewmbench_dataset_dir",
                    "env": "WORLDFOUNDRY_EWMBENCH_DATASET_ROOT",
                    "repo_id": "agibot-world/EWMBench",
                    "local_name": "agibot-world__EWMBench",
                    "alternate_local_names": ("agibot-world/EWMBench",),
                    "note": "EWMBench official HF dataset for embodied world-model evaluation inputs.",
                },
            ),
        ),
        "evalcrafter_dataset_assets": _benchmark_dataset_capability(
            "evalcrafter_dataset_assets",
            (
                {
                    "asset_id": "evalcrafter_dataset_dir",
                    "env": "WORLDFOUNDRY_EVALCRAFTER_DATASET_ROOT",
                    "repo_id": "RaphaelLiu/EvalCrafter_T2V_Dataset",
                    "local_name": "RaphaelLiu__EvalCrafter_T2V_Dataset",
                    "alternate_local_names": (
                        "RaphaelLiu/EvalCrafter_T2V_Dataset",
                        "EvalCrafter_T2V_Dataset",
                    ),
                    "note": "EvalCrafter public T2V dataset used by official-result normalization.",
                },
            ),
        ),
        "genai_bench_dataset_assets": _benchmark_dataset_capability(
            "genai_bench_dataset_assets",
            (
                {
                    "asset_id": "genai_bench_dataset_dir",
                    "env": "WORLDFOUNDRY_GENAI_BENCH_DATASET_ROOT",
                    "repo_id": "TIGER-Lab/GenAI-Bench",
                    "local_name": "TIGER-Lab__GenAI-Bench",
                    "alternate_local_names": ("TIGER-Lab/GenAI-Bench",),
                    "note": "GenAI-Bench public annotations and prompts for official result normalization.",
                },
            ),
        ),
        "t2v_metrics_source_assets": BaseModelCapability(
            id="t2v_metrics_source_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.base_models.perception_core.video_text.vqa_score",
            canonical_path="worldfoundry/base_models/perception_core/video_text/vqa_score",
            asset_env=(),
            assets=(
                BaseModelAsset(
                    id="t2v_metrics_in_tree_source",
                    kind="dir",
                    role="benchmark_source",
                    env=(),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/base_models/perception_core/video_text/vqa_score",
                    min_file_count=8,
                    required_files=("constants.py", "models/model.py", "models/vqascore_models/__init__.py"),
                    required_paths=("models",),
                    note=(
                        "In-tree VQAScore model backend shared by evaluation scorers."
                    ),
                ),
            ),
            gpu_required=False,
        ),
        "t2v_metrics_clip_flant5_xxl": BaseModelCapability(
            id="t2v_metrics_clip_flant5_xxl",
            family="video_metric_model",
            canonical_owner="worldfoundry.base_models.perception_core.video_text.vqa_score",
            canonical_path="worldfoundry/base_models/perception_core/video_text/vqa_score",
            package_imports=("torch", "transformers", "PIL", "numpy", "huggingface_hub"),
            install_packages=("torch", "transformers", "Pillow", "numpy", "huggingface-hub"),
            asset_env=("WORLDFOUNDRY_T2V_METRICS_CLIP_FLANT5_XXL_DIR",),
            assets=(
                BaseModelAsset(
                    id="t2v_metrics_clip_flant5_xxl_dir",
                    kind="dir",
                    role="metric_model_dir",
                    env=("WORLDFOUNDRY_T2V_METRICS_CLIP_FLANT5_XXL_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/zhiqiulin--clip-flant5-xxl",
                    alternate_paths=(
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/zhiqiulin--clip-flant5-xxl",
                        "${WORLDFOUNDRY_HFD_ROOT}/zhiqiulin/clip-flant5-xxl",
                    ),
                    hf_repo_id="zhiqiulin/clip-flant5-xxl",
                    min_file_count=2,
                    required_files=("config.json",),
                    note="CLIP-FlanT5-XXL VQAScore checkpoint used by GenAI-Bench and PhyGenBench alignment scoring.",
                ),
                BaseModelAsset(
                    id="t2v_metrics_flan_t5_xxl_tokenizer_dir",
                    kind="dir",
                    role="metric_tokenizer_dir",
                    env=("WORLDFOUNDRY_T2V_METRICS_FLAN_T5_XXL_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/google--flan-t5-xxl",
                    alternate_paths=(
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/google--flan-t5-xxl",
                    ),
                    hf_repo_id="google/flan-t5-xxl",
                    min_file_count=2,
                    required_files=("config.json", "tokenizer_config.json"),
                    note="Tokenizer dependency for clip-flant5-xxl VQAScore.",
                ),
            ),
        ),
        "t2v_metrics_blip2_itm": BaseModelCapability(
            id="t2v_metrics_blip2_itm",
            family="video_metric_model",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners._scorers.itm_score",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/_scorers/itm_score",
            package_imports=("torch", "transformers", "PIL", "numpy", "timm"),
            install_packages=("torch", "transformers", "Pillow", "numpy", "timm"),
            asset_env=("WORLDFOUNDRY_T2V_METRICS_BLIP2_ITM_DIR",),
            assets=(
                BaseModelAsset(
                    id="t2v_metrics_blip2_itm_dir",
                    kind="dir",
                    role="metric_model_dir",
                    env=("WORLDFOUNDRY_T2V_METRICS_BLIP2_ITM_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/Salesforce--blip2-opt-2.7b",
                    alternate_paths=(
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/Salesforce--blip2-opt-2.7b",
                    ),
                    hf_repo_id="Salesforce/blip2-opt-2.7b",
                    min_file_count=2,
                    required_files=("config.json",),
                    note="BLIP-2 ITMScore backbone used by CameraBench and t2v_metrics ITM paths.",
                ),
            ),
        ),
        "t2v_metrics_internvideo2_itm": BaseModelCapability(
            id="t2v_metrics_internvideo2_itm",
            family="video_metric_model",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners._scorers.itm_score",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/_scorers/itm_score",
            package_imports=("torch", "transformers", "decord", "PIL", "numpy"),
            install_packages=("torch", "transformers", "decord", "Pillow", "numpy"),
            asset_env=("WORLDFOUNDRY_T2V_METRICS_INTERNVIDEO2_ITM_DIR",),
            assets=(
                BaseModelAsset(
                    id="t2v_metrics_internvideo2_itm_dir",
                    kind="dir",
                    role="metric_model_dir",
                    env=("WORLDFOUNDRY_T2V_METRICS_INTERNVIDEO2_ITM_DIR",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/OpenGVLab--InternVideo2-Stage2_1B-224p-f4",
                    alternate_paths=(
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/OpenGVLab--InternVideo2-Stage2_1B-224p-f4",
                    ),
                    hf_repo_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
                    min_file_count=2,
                    required_files=("config.json",),
                    note="InternVideo2 ITMScore checkpoint for video-text matching metrics.",
                ),
            ),
        ),
        "jedi_source_assets": BaseModelCapability(
            id="jedi_source_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.evaluation.tasks.metrics.jedi",
            canonical_path="worldfoundry/evaluation/tasks/metrics/jedi",
            asset_env=("WORLDFOUNDRY_JEDI_ROOT",),
            assets=(
                BaseModelAsset(
                    id="jedi_in_tree_source",
                    kind="dir",
                    role="benchmark_source",
                    env=("WORLDFOUNDRY_JEDI_ROOT",),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/evaluation/tasks/metrics/jedi",
                    min_file_count=4,
                    required_files=("JEDi.py", "V_JEPA.py", "mmd_polynomial.py", "utils.py"),
                    note=(
                        "In-tree VideoJEDi metric package vendored from oooolga/JEDi. "
                        "WORLDFOUNDRY_JEDI_ROOT is only an optional upstream parity override."
                    ),
                ),
                BaseModelAsset(
                    id="jedi_bundled_config",
                    kind="file",
                    role="benchmark_config",
                    env=("WORLDFOUNDRY_JEDI_CONFIG_PATH",),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/benchmarks/assets/jedi/vith16_ssv2_16x2x3.yaml",
                    required_files=("vith16_ssv2_16x2x3.yaml",),
                    note="Bundled V-JEPA config for official JEDi feature extraction.",
                ),
            ),
            gpu_required=False,
        ),
        "jedi_vjepa_vith16": BaseModelCapability(
            id="jedi_vjepa_vith16",
            family="video_metric_model",
            canonical_owner="worldfoundry.evaluation.tasks.metrics.jedi",
            canonical_path="worldfoundry/evaluation/tasks/metrics/jedi",
            package_imports=("torch", "vjepa", "numpy", "yaml"),
            install_packages=("torch", "vjepa", "numpy", "pyyaml"),
            asset_env=("WORLDFOUNDRY_JEDI_MODEL_DIR", "WORLDFOUNDRY_JEDI_VJEPA_DIR"),
            assets=(
                BaseModelAsset(
                    id="jedi_vith16_encoder_checkpoint",
                    kind="file",
                    role="metric_model_file",
                    env=("WORLDFOUNDRY_JEDI_VITH16_CHECKPOINT",),
                    local_path="${WORLDFOUNDRY_JEDI_MODEL_DIR}/vith16.pth.tar",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/jedi-vjepa/vith16.pth.tar",
                    ),
                    download_url="https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
                    required_files=("vith16.pth.tar",),
                    note="V-JEPA ViT-H/16 encoder checkpoint used by VideoJEDi feature extraction.",
                ),
            ),
        ),
        "jedi_vjepa_ssv2_probe": BaseModelCapability(
            id="jedi_vjepa_ssv2_probe",
            family="video_metric_model",
            canonical_owner="worldfoundry.evaluation.tasks.metrics.jedi",
            canonical_path="worldfoundry/evaluation/tasks/metrics/jedi",
            package_imports=("torch", "vjepa", "numpy", "yaml"),
            install_packages=("torch", "vjepa", "numpy", "pyyaml"),
            asset_env=("WORLDFOUNDRY_JEDI_MODEL_DIR", "WORLDFOUNDRY_JEDI_VJEPA_DIR"),
            assets=(
                BaseModelAsset(
                    id="jedi_ssv2_probe_checkpoint",
                    kind="file",
                    role="metric_model_file",
                    env=("WORLDFOUNDRY_JEDI_SSV2_PROBE_CHECKPOINT",),
                    local_path="${WORLDFOUNDRY_JEDI_MODEL_DIR}/ssv2-probe.pth.tar",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/jedi-vjepa/ssv2-probe.pth.tar",
                    ),
                    download_url="https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar",
                    required_files=("ssv2-probe.pth.tar",),
                    note="Fine-tuned SSV2 attentive probe used by VideoJEDi feature aggregation.",
                ),
            ),
        ),
        "iworld_bench_dataset_assets": _benchmark_dataset_capability(
            "iworld_bench_dataset_assets",
            (
                {
                    "asset_id": "iworld_bench_dataset_dir",
                    "env": "WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT",
                    "repo_id": "EmbodiedCity/iWorld-Bench-Dataset",
                    "revision": "530c48cd3d8e9bb20ed685a96521c3a9b68cf6d2",
                    "local_name": "EmbodiedCity__iWorld-Bench-Dataset",
                    "alternate_local_names": ("EmbodiedCity/iWorld-Bench-Dataset",),
                    "min_file_count": 100,
                    "note": "iWorld-Bench released HF dataset with dataset/all_pack metadata and first-frame assets.",
                },
            ),
        ),
        "libero_dataset_assets": _benchmark_dataset_capability(
            "libero_dataset_assets",
            (
                {
                    "asset_id": "libero_lerobot_dataset_dir",
                    "env": "WORLDFOUNDRY_LIBERO_LEROBOT_DATASET_ROOT",
                    "repo_id": "lerobot/libero",
                    "local_name": "lerobot__libero",
                    "alternate_local_names": ("lerobot/libero",),
                    "note": "LeRobot LIBERO conversion used for dataset-level preflight and offline normalization.",
                },
                {
                    "asset_id": "libero_official_dataset_dir",
                    "env": "WORLDFOUNDRY_LIBERO_DATASETS_ROOT",
                    "repo_id": "yifengzhu-hf/LIBERO-datasets",
                    "local_name": "yifengzhu-hf__LIBERO-datasets",
                    "alternate_local_names": ("yifengzhu-hf/LIBERO-datasets",),
                    "note": "Official LIBERO dataset archive bundle for simulator/evaluation setup.",
                },
            ),
        ),
        "likephys_dataset_assets": _benchmark_dataset_capability(
            "likephys_dataset_assets",
            (
                {
                    "asset_id": "likephys_dataset_dir",
                    "env": "WORLDFOUNDRY_LIKEPHYS_DATASET_ROOT",
                    "repo_id": "JianhaoDYDY/LikePhys-Benchmark",
                    "local_name": "JianhaoDYDY__LikePhys-Benchmark",
                    "alternate_local_names": (
                        "JianhaoDYDY/LikePhys-Benchmark",
                        "datasets--JianhaoDYDY--LikePhys-Benchmark",
                        "LikePhys-Benchmark",
                    ),
                    "note": "LikePhys paired valid/impossible physics clips probed by the likelihood-preference evaluator.",
                },
            ),
        ),
        "phyground_dataset_assets": _benchmark_dataset_capability(
            "phyground_dataset_assets",
            (
                {
                    "asset_id": "phyground_dataset_dir",
                    "env": "WORLDFOUNDRY_PHYGROUND_DATASET_ROOT",
                    "repo_id": "NU-World-Model-Embodied-AI/phyground",
                    "local_name": "NU-World-Model-Embodied-AI__phyground",
                    "alternate_local_names": ("NU-World-Model-Embodied-AI/phyground",),
                    "note": "PhyGround official HF dataset for grounded physical-reasoning evaluation.",
                },
            ),
        ),
        "rbench_dataset_assets": _benchmark_dataset_capability(
            "rbench_dataset_assets",
            (
                {
                    "asset_id": "rbench_dataset_dir",
                    "env": "WORLDFOUNDRY_RBENCH_DATASET_ROOT",
                    "repo_id": "DAGroup-PKU/RBench",
                    "local_name": "DAGroup-PKU__RBench",
                    "alternate_local_names": (
                        "DAGroup-PKU/RBench",
                        "datasets--DAGroup-PKU--RBench",
                        "RBench",
                    ),
                    "note": "RBench prompts and first-frame conditioning images for robotics image-to-video evaluation.",
                },
            ),
        ),
        "physvidbench_dataset_assets": _benchmark_dataset_capability(
            "physvidbench_dataset_assets",
            (
                {
                    "asset_id": "physvidbench_dataset_dir",
                    "env": "WORLDFOUNDRY_PHYSVIDBENCH_DATASET_ROOT",
                    "repo_id": "Anonymousny/PhysVidBench",
                    "local_name": "Anonymousny__PhysVidBench",
                    "alternate_local_names": (
                        "Anonymousny/PhysVidBench",
                        "datasets--Anonymousny--PhysVidBench",
                        "PhysVidBench",
                    ),
                    "note": "PhysVidBench public prompt/question data for physical video consistency metrics.",
                },
            ),
        ),
        "robotwin_dataset_assets": _benchmark_dataset_capability(
            "robotwin_dataset_assets",
            (
                {
                    "asset_id": "robotwin2_dataset_dir",
                    "env": "WORLDFOUNDRY_ROBOTWIN2_DATASET_ROOT",
                    "repo_id": "TianxingChen/RoboTwin2.0",
                    "local_name": "TianxingChen__RoboTwin2.0",
                    "alternate_local_names": ("TianxingChen/RoboTwin2.0",),
                    "note": "RoboTwin 2.0 official dataset bundle for simulator asset preparation.",
                },
                {
                    "asset_id": "robotwin_lerobot_dataset_dir",
                    "env": "WORLDFOUNDRY_ROBOTWIN_UNIFIED_DATASET_ROOT",
                    "repo_id": "lerobot/robotwin_unified",
                    "local_name": "lerobot__robotwin_unified",
                    "alternate_local_names": ("lerobot/robotwin_unified",),
                    "note": "LeRobot unified RoboTwin dataset for offline checks and normalized result import.",
                },
            ),
        ),
        "videophy2_dataset_assets": _benchmark_dataset_capability(
            "videophy2_dataset_assets",
            (
                {
                    "asset_id": "videophy2_test_dataset_dir",
                    "env": "WORLDFOUNDRY_VIDEOPHY2_TEST_DATASET_ROOT",
                    "repo_id": "videophysics/videophy2_test",
                    "local_name": "videophysics__videophy2_test",
                    "alternate_local_names": ("videophysics/videophy2_test",),
                    "note": "VideoPhy2 public test split for physics-alignment evaluation.",
                },
                {
                    "asset_id": "videophy2_train_dataset_dir",
                    "env": "WORLDFOUNDRY_VIDEOPHY2_TRAIN_DATASET_ROOT",
                    "repo_id": "videophysics/videophy2_train",
                    "local_name": "videophysics__videophy2_train",
                    "alternate_local_names": ("videophysics/videophy2_train",),
                    "note": "VideoPhy2 public train/reference split used by official tooling.",
                },
            ),
        ),
        "videophy2_autoeval_model": BaseModelCapability(
            id="videophy2_autoeval_model",
            family="mllm",
            canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.videophy2_autoeval",
            canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/videophy2_autoeval",
            package_imports=("torch", "transformers", "peft", "pandas", "numpy", "decord", "PIL", "einops"),
            install_packages=("torch", "transformers", "peft", "pandas", "numpy", "decord", "Pillow", "einops"),
            asset_env=("WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_CKPT",),
            assets=(
                BaseModelAsset(
                    id="videophy2_autoeval_checkpoint_dir",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_CKPT",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/hbXNov--videophy_2_auto",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/videophysics--videophy_2_auto",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/hbXNov--videophy_2_auto",
                        "${WORLDFOUNDRY_CKPT_DIR}/VideoPhy2/videophy_2_auto",
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/hbXNov--videophy_2_auto",
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/VideoPhy2/videophy_2_auto",
                    ),
                    hf_repo_id="hbXNov/videophy_2_auto",
                    min_file_count=3,
                    required_files=("config.json",),
                    note="VideoPhy2 AutoEval Mplug-Owl checkpoint used by the in-tree SA/PC/rule judge.",
                ),
            ),
            gpu_required=True,
        ),
        "videocon_physics_model": BaseModelCapability(
            id="videocon_physics_model",
            family="mllm",
            canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.videocon_physics",
            canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/videocon_physics",
            package_imports=("torch", "transformers", "pandas", "numpy", "decord", "PIL", "einops"),
            install_packages=("torch", "transformers", "pandas", "numpy", "decord", "Pillow", "einops"),
            asset_env=("WORLDFOUNDRY_VIDEOPHY_VIDEOCON_CKPT",),
            assets=(
                BaseModelAsset(
                    id="videocon_physics_checkpoint_dir",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_VIDEOPHY_VIDEOCON_CKPT",),
                    local_path="${WORLDFOUNDRY_HFD_ROOT}/videophysics--videocon_physics",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/videophysics/videocon_physics",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/videophysics--videocon_physics",
                        "${WORLDFOUNDRY_CKPT_DIR}/VideoPhy/videocon_physics",
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/hfd/videophysics--videocon_physics",
                        "${WORLDFOUNDRY_WORKSPACE_ROOT}/ckpt/VideoPhy/videocon_physics",
                    ),
                    hf_repo_id="videophysics/videocon_physics",
                    min_file_count=3,
                    required_files=("config.json",),
                    note="VideoCon-Physics checkpoint used by the in-tree VideoPhy SA/PC AutoEvaluator.",
                ),
            ),
            gpu_required=True,
        ),
        "videoverse_dataset_assets": _benchmark_dataset_capability(
            "videoverse_dataset_assets",
            (
                {
                    "asset_id": "videoverse_dataset_dir",
                    "env": "WORLDFOUNDRY_VIDEOVERSE_DATASET_ROOT",
                    "repo_id": "NNaptmn/VideoVerse",
                    "local_name": "NNaptmn__VideoVerse",
                    "alternate_local_names": ("NNaptmn/VideoVerse",),
                    "note": "VideoVerse public HF dataset for video-world evaluation normalization.",
                },
            ),
        ),
        "worldmodelbench_dataset_assets": _benchmark_dataset_capability(
            "worldmodelbench_dataset_assets",
            (
                {
                    "asset_id": "worldmodelbench_dataset_dir",
                    "env": "WORLDFOUNDRY_WORLDMODELBENCH_DATASET_ROOT",
                    "repo_id": "Efficient-Large-Model/worldmodelbench",
                    "local_name": "Efficient-Large-Model__worldmodelbench",
                    "alternate_local_names": (
                        "Efficient-Large-Model/worldmodelbench",
                        "datasets--Efficient-Large-Model--worldmodelbench",
                        "worldmodelbench",
                    ),
                    "note": "WorldModelBench public prompt/reference dataset for world-model evaluation.",
                },
            ),
        ),
        "worldmodelbench_vila_judge_model": BaseModelCapability(
            id="worldmodelbench_vila_judge_model",
            family="mllm",
            canonical_owner="worldfoundry.base_models.llm_mllm_core.mllm.vila",
            canonical_path="worldfoundry/base_models/llm_mllm_core/mllm/vila",
            package_imports=("torch", "transformers", "huggingface_hub", "PIL", "cv2"),
            install_packages=("torch", "transformers", "huggingface_hub", "Pillow", "opencv-python"),
            assets=(
                BaseModelAsset(
                    id="worldmodelbench_vila_judge_dir",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_WORLDMODELBENCH_JUDGE",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/worldmodelbench/Efficient-Large-Model/vila-ewm-qwen2-1.5b",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/Efficient-Large-Model--vila-ewm-qwen2-1.5b",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/Efficient-Large-Model--vila-ewm-qwen2-1.5b",
                        "${WORLDFOUNDRY_CKPT_DIR}/Efficient-Large-Model__vila-ewm-qwen2-1.5b",
                    ),
                    hf_repo_id="Efficient-Large-Model/vila-ewm-qwen2-1.5b",
                    min_file_count=1,
                    required_files=("config.json",),
                    required=False,
                    note="Official WorldModelBench VILA judge checkpoint loaded through the in-tree VILA runtime.",
                ),
            ),
            gpu_required=True,
        ),
        "worldscore_dataset_assets": _benchmark_dataset_capability(
            "worldscore_dataset_assets",
            (
                {
                    "asset_id": "worldscore_hf_dataset_dir",
                    "env": "WORLDFOUNDRY_WORLDSCORE_HF_DATASET_ROOT",
                    "repo_id": "Howieeeee/WorldScore",
                    "revision": "42c4e267e1cc0529af1b0284ce018e04d43906e3",
                    "local_name": "Howieeeee__WorldScore",
                    "alternate_local_names": ("Howieeeee/WorldScore",),
                    "note": "WorldScore public HF dataset archive used to stage the official prompt suite.",
                },
            ),
        ),
    }
)


BASE_MODEL_CAPABILITIES.update(
    {
        "bridgedata_v2_source_assets": _source_repo_capability(
            capability_id="bridgedata_v2_source_assets",
            env="WORLDFOUNDRY_BRIDGEDATA_V2_ROOT",
            repo_url="https://github.com/rail-berkeley/bridge_data_v2.git",
            local_name="rail-berkeley--bridge_data_v2",
            note="BridgeData V2 source checkout used to stage real-robot offline evaluation assets.",
        ),
        "bridgedata_v2_dataset_assets": _external_dir_capability(
            capability_id="bridgedata_v2_dataset_assets",
            env="WORLDFOUNDRY_BRIDGEDATA_V2_DATASET_ROOT",
            local_name="bridgedata-v2",
            role="external_benchmark_dataset",
            manual_action=(
                "Download the official BridgeData V2 raw or TFDS files from "
                "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/ "
                "into $WORLDFOUNDRY_BRIDGEDATA_V2_DATASET_ROOT, then rerun readiness. "
                "The release is an HTTP directory, so choose the demos/scripted zips or tfds subset "
                "needed for your evaluation instead of blindly mirroring the whole dataset."
            ),
            note="BridgeData V2 dataset root; no single public HF mirror is declared in the benchmark manifest.",
        ),
        "calvin_source_assets": _source_repo_capability(
            capability_id="calvin_source_assets",
            env="WORLDFOUNDRY_CALVIN_ROOT",
            repo_url="https://github.com/mees/calvin.git",
            local_name="mees--calvin",
            note="CALVIN source checkout for long-horizon sequence evaluation.",
        ),
        "calvin_dataset_assets": _external_dir_capability(
            capability_id="calvin_dataset_assets",
            env="WORLDFOUNDRY_CALVIN_DATASET_ROOT",
            local_name="calvin",
            alternate_paths=("${WORLDFOUNDRY_WORKSPACE_ROOT}/data/worldfoundry/datasets/calvin",),
            role="external_benchmark_dataset",
            required_any_paths=("task_D_D", "task_ABC_D", "task_ABCD_D"),
            manual_action=(
                "Stage the official CALVIN dataset split directly under "
                "$WORLDFOUNDRY_CALVIN_DATASET_ROOT or $WORLDFOUNDRY_DATA_DIR/datasets/calvin. "
                "WorldFoundry does not run the upstream calvin repository download_data.sh script during readiness."
            ),
            note="CALVIN dataset split root; users must stage the official dataset before full rollouts.",
        ),
        "libero_source_assets": _source_repo_capability(
            capability_id="libero_source_assets",
            env="WORLDFOUNDRY_LIBERO_ROOT",
            repo_url="https://github.com/Lifelong-Robot-Learning/LIBERO.git",
            local_name="Lifelong-Robot-Learning--LIBERO",
            revision="8f1084e3132a39270c3a13ebe37270a43ece2a01",
            required_files=("README.md", "setup.py"),
            required_paths=("libero",),
            note="Official LIBERO source checkout for simulator-backed language-conditioned manipulation rollouts.",
        ),
        "maniskill_source_assets": _source_repo_capability(
            capability_id="maniskill_source_assets",
            env="WORLDFOUNDRY_MANISKILL_ROOT",
            repo_url="https://github.com/haosulab/ManiSkill.git",
            local_name="haosulab--ManiSkill",
            role="simulator_source_repo",
            note="ManiSkill simulator source checkout; SAPIEN/runtime packages remain environment-specific.",
        ),
        "metaworld_source_assets": _source_repo_capability(
            capability_id="metaworld_source_assets",
            env="WORLDFOUNDRY_METAWORLD_ROOT",
            repo_url="https://github.com/Farama-Foundation/Metaworld.git",
            local_name="Farama-Foundation--Metaworld",
            role="simulator_source_repo",
            note="Meta-World source checkout for simulator task-suite mapping and rollout result normalization.",
        ),
        "rlbench_source_assets": _source_repo_capability(
            capability_id="rlbench_source_assets",
            env="WORLDFOUNDRY_RLBENCH_ROOT",
            repo_url="https://github.com/stepjam/RLBench.git",
            local_name="stepjam--RLBench",
            role="simulator_source_repo",
            note="RLBench source checkout; full evaluation still requires CoppeliaSim/PyRep and task assets.",
        ),
        "robocasa_source_assets": _source_repo_capability(
            capability_id="robocasa_source_assets",
            env="WORLDFOUNDRY_ROBOCASA_ROOT",
            repo_url="https://github.com/robocasa/robocasa.git",
            local_name="robocasa--robocasa",
            role="simulator_source_repo",
            note="RoboCasa source checkout; scene assets and policy rollouts remain explicit runtime assets.",
        ),
        "simpler_env_source_assets": _source_repo_capability(
            capability_id="simpler_env_source_assets",
            env="WORLDFOUNDRY_SIMPLERENV_ROOT",
            repo_url="https://github.com/simpler-env/SimplerEnv.git",
            local_name="simpler-env--SimplerEnv",
            role="simulator_source_repo",
            note="SimplerEnv source checkout for simulator-backed VLA policy rollouts.",
        ),
        "iworld_bench_source_assets": _source_repo_capability(
            capability_id="iworld_bench_source_assets",
            env="WORLDFOUNDRY_IWORLD_BENCH_ROOT",
            repo_url="https://github.com/EmbodiedCity/iWorld-Bench.git",
            local_name="EmbodiedCity--iWorld-Bench",
            revision="666cdea0e98ec77c2666074abdc3c01ce4f1eb44",
            required_files=("run_iworldbench_evaluation.py", "unified_video_metrics.py", "README_metrics.md"),
            note="Released iWorld-Bench official evaluator and camera trajectory resources.",
        ),
        "t2vworldbench_source_assets": _source_repo_capability(
            capability_id="t2vworldbench_source_assets",
            env="WORLDFOUNDRY_T2VWORLDBENCH_ROOT",
            repo_url="https://github.com/magiclinux/world_knowledge.git",
            local_name="magiclinux--world_knowledge",
            required_files=("prompt/prompt.csv",),
            note="T2VWorldBench prompt-suite repository.",
        ),
        "vbench_source_assets": BaseModelCapability(
            id="vbench_source_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners.vbench",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/vbench/runtime",
            asset_env=(),
            assets=(
                BaseModelAsset(
                    id="vbench_in_tree_official_source",
                    kind="dir",
                    role="benchmark_source",
                    env=(),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/evaluation/tasks/execution/runners/vbench/runtime",
                    min_file_count=1,
                    required_files=("vbench/VBench_full_info.json",),
                    required_paths=("vbench", "prompts"),
                    note="In-tree VBench official evaluator source.",
                ),
            ),
            gpu_required=False,
        ),
        "vbench_plus_plus_source_assets": BaseModelCapability(
            id="vbench_plus_plus_source_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/vbench_plus_plus/runtime",
            asset_env=(),
            assets=(
                BaseModelAsset(
                    id="vbench_plus_plus_in_tree_runtime",
                    kind="dir",
                    role="benchmark_source",
                    env=(),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/evaluation/tasks/execution/runners/vbench_plus_plus/runtime",
                    min_file_count=1,
                    required_files=(
                        "vbench2_beta_i2v/vbench2_i2v_full_info.json",
                        "vbench2_beta_long/VBench_full_info.json",
                        "vbench2_beta_trustworthiness/vbench2_trustworthy.json",
                    ),
                    required_paths=("entrypoints",),
                    note="In-tree VBench++ I2V, long-video, and trustworthiness runtime source.",
                ),
            ),
            gpu_required=False,
        ),
        "vbench2_source_assets": BaseModelCapability(
            id="vbench2_source_assets",
            family="benchmark_data_asset",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners.vbench_2_0",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/vbench_2_0/runtime",
            asset_env=(),
            assets=(
                BaseModelAsset(
                    id="vbench2_in_tree_runtime",
                    kind="dir",
                    role="benchmark_source",
                    env=(),
                    local_path="${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/evaluation/tasks/execution/runners/vbench_2_0/runtime",
                    min_file_count=1,
                    required_files=("vbench2/VBench2_full_info.json",),
                    required_paths=("entrypoints", "vbench2"),
                    note="In-tree VBench-2.0 official evaluator source.",
                ),
            ),
            gpu_required=False,
        ),
        "vbench_metric_checkpoint_assets": BaseModelCapability(
            id="vbench_metric_checkpoint_assets",
            family="video_metric_model",
            canonical_owner="worldfoundry.evaluation.tasks.execution.runners.vbench",
            canonical_path="worldfoundry/evaluation/tasks/execution/runners/vbench/runtime",
            package_imports=("torch", "torchvision", "transformers", "open_clip", "cv2", "numpy"),
            install_packages=("torch", "torchvision", "transformers", "open-clip-torch", "opencv-python", "numpy"),
            asset_env=(
                "WORLDFOUNDRY_VBENCH_CACHE_DIR",
                "WORLDFOUNDRY_VBENCH2_CACHE_DIR",
                "WORLDFOUNDRY_VBENCH_AMT_S_CKPT",
                "WORLDFOUNDRY_VBENCH_UMT_CKPT",
                "WORLDFOUNDRY_VBENCH_VICLIP_CKPT",
                "WORLDFOUNDRY_VBENCH_GRIT_CKPT",
                "WORLDFOUNDRY_VBENCH_TAG2TEXT_CKPT",
                "WORLDFOUNDRY_VBENCH_DINO_VITB16_CKPT",
                "WORLDFOUNDRY_VBENCH_CLIP_VIT_B32_PT",
                "WORLDFOUNDRY_VBENCH_CLIP_VIT_L14_PT",
                "WORLDFOUNDRY_VBENCH_AESTHETIC_LINEAR_CKPT",
                "WORLDFOUNDRY_VBENCH_MUSIQ_SPAQ_CKPT",
                "WORLDFOUNDRY_VBENCH_RETINAFACE_CKPT",
                "WORLDFOUNDRY_VBENCH_SD_SAFETY_CHECKER_DIR",
                "WORLDFOUNDRY_VBENCH_Q16_PROMPTS",
            ),
            asset_env_policy="all",
            assets=(
                BaseModelAsset(
                    id="vbench_amt_s_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_AMT_S_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/amt_model/amt-s.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/amt_model/amt-s.pth",),
                    hf_repo_id="lalala125/AMT",
                    hf_filename="amt-s.pth",
                    min_size_bytes=10_000_000,
                    note="AMT motion-smoothness checkpoint used by VBench.",
                ),
                BaseModelAsset(
                    id="vbench_umt_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_UMT_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",),
                    hf_repo_id="OpenGVLab/VBench_Used_Models",
                    hf_filename="l16_ptk710_ftk710_ftk400_f16_res224.pth",
                    min_size_bytes=200_000_000,
                    note="UMT action-recognition checkpoint used by VBench human-action metrics.",
                ),
                BaseModelAsset(
                    id="vbench_viclip_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_VICLIP_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/ViCLIP/ViClip-InternVid-10M-FLT.pth",),
                    hf_repo_id="OpenGVLab/VBench_Used_Models",
                    hf_filename="ViClip-InternVid-10M-FLT.pth",
                    min_size_bytes=300_000_000,
                    note="ViCLIP checkpoint used by VBench overall-consistency metrics.",
                ),
                BaseModelAsset(
                    id="vbench_grit_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_GRIT_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/grit_model/grit_b_densecap_objectdet.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/grit_model/grit_b_densecap_objectdet.pth",),
                    hf_repo_id="OpenGVLab/VBench_Used_Models",
                    hf_filename="grit_b_densecap_objectdet.pth",
                    min_size_bytes=400_000_000,
                    note="GRiT dense-caption/object-detection checkpoint used by VBench scene/object metrics.",
                ),
                BaseModelAsset(
                    id="vbench_tag2text_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_TAG2TEXT_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/caption_model/tag2text_swin_14m.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/caption_model/tag2text_swin_14m.pth",),
                    hf_repo_id="spaces/xinyu1205/recognize-anything",
                    hf_repo_type="space",
                    hf_filename="tag2text_swin_14m.pth",
                    min_size_bytes=600_000_000,
                    note="Tag2Text checkpoint used by VBench multiple-object/color metrics.",
                ),
                BaseModelAsset(
                    id="vbench_dino_vitb16_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_DINO_VITB16_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/dino_model/dino_vitbase16_pretrain.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/dino_model/dino_vitbase16_pretrain.pth",),
                    download_url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
                    min_size_bytes=300_000_000,
                    note="DINO ViT-B/16 checkpoint used by VBench subject/background consistency metrics.",
                ),
                BaseModelAsset(
                    id="vbench_dino_source",
                    kind="dir",
                    role="source_asset",
                    env=(),
                    local_path=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/base_models/three_dimensions/"
                        "point_clouds/pixelsplat/third_party/dino"
                    ),
                    min_file_count=1,
                    required_files=("hubconf.py", "vision_transformer.py"),
                    manual_action=(
                        "Restore the checked-in WorldFoundry DINO torch.hub shim under "
                        "worldfoundry/base_models/three_dimensions/point_clouds/pixelsplat/third_party/dino; "
                        "do not clone facebookresearch/dino for VBench readiness."
                    ),
                    note="In-tree DINO torch.hub shim used by the VBench subject/background consistency loaders.",
                ),
                BaseModelAsset(
                    id="vbench_clip_vit_b32_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_CLIP_VIT_B32_PT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/clip_model/ViT-B-32.pt",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/clip_model/ViT-B-32.pt",),
                    download_url="https://openaipublic.azureedge.net/clip/models/40d365715913c9a29492c0b13cf0c8027c5ca8149563f1a35cf5f5456eee0fc6/ViT-B-32.pt",
                    min_size_bytes=300_000_000,
                    note="OpenAI CLIP ViT-B/32 checkpoint used by VBench semantic metrics.",
                ),
                BaseModelAsset(
                    id="vbench_clip_vit_l14_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_CLIP_VIT_L14_PT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/clip_model/ViT-L-14.pt",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/clip_model/ViT-L-14.pt",),
                    download_url="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba65e8d4db8e4e43c80385fc7dcdbf8304dfe3b6a9bf1cc7/ViT-L-14.pt",
                    min_size_bytes=800_000_000,
                    note="OpenAI CLIP ViT-L/14 checkpoint used by VBench/VBench++ image-text metrics.",
                ),
                BaseModelAsset(
                    id="vbench_aesthetic_linear_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_AESTHETIC_LINEAR_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",),
                    download_url="https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
                    min_size_bytes=1_000,
                    note="LAION aesthetic predictor linear head used by VBench aesthetic quality.",
                ),
                BaseModelAsset(
                    id="vbench_musiq_spaq_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_MUSIQ_SPAQ_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",),
                    download_url="https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
                    min_size_bytes=100_000_000,
                    note="MUSIQ-SPAQ checkpoint used by VBench imaging quality.",
                ),
                BaseModelAsset(
                    id="vbench_retinaface_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH_RETINAFACE_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/retina_face_model/retinaface_resnet50_2020-07-20-f168fae3c.zip",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/retina_face_model/retinaface_resnet50_2020-07-20-f168fae3c.zip",),
                    download_url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",
                    min_size_bytes=100_000_000,
                    required=False,
                    note="RetinaFace checkpoint used by VBench++ bias dimensions.",
                ),
                BaseModelAsset(
                    id="vbench_sd_safety_checker_model",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_VBENCH_SD_SAFETY_CHECKER_DIR",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/sd_safety_checker",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/CompVis--stable-diffusion-safety-checker",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/CompVis--stable-diffusion-safety-checker",
                    ),
                    hf_repo_id="CompVis/stable-diffusion-safety-checker",
                    min_file_count=1,
                    required_files=("config.json", "preprocessor_config.json", "pytorch_model.bin"),
                    required=False,
                    note="Stable Diffusion safety checker used by VBench++ trustworthiness safety.",
                ),
                BaseModelAsset(
                    id="vbench_q16_prompts",
                    kind="file",
                    role="prompt_asset",
                    env=("WORLDFOUNDRY_VBENCH_Q16_PROMPTS",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench/q16/prompts.p",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench/q16/prompts.p",),
                    download_url="https://raw.githubusercontent.com/ml-research/Q16/main/data/ViT-B-32/prompts.p",
                    min_size_bytes=1_000,
                    required=False,
                    note="Q16 prompt asset used by VBench++ trustworthiness safety.",
                ),
                BaseModelAsset(
                    id="vbench2_llava_video_qwen2_model",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_VBENCH2_LLAVA_VIDEO_QWEN2_DIR",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/lmms-lab--LLaVA-Video-7B-Qwen2",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/lmms-lab--LLaVA-Video-7B-Qwen2",
                    ),
                    hf_repo_id="lmms-lab/LLaVA-Video-7B-Qwen2",
                    min_file_count=1,
                    required_files=("config.json",),
                    required=False,
                    note="VBench-2.0 text/video reasoning model; required only for dimensions that call LLaVA-Video.",
                ),
                BaseModelAsset(
                    id="vbench2_qwen2_5_7b_model",
                    kind="dir",
                    role="model_dir",
                    env=("WORLDFOUNDRY_VBENCH2_QWEN2_5_7B_DIR",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/Qwen/Qwen2.5-7B-Instruct",
                    alternate_paths=(
                        "${WORLDFOUNDRY_HFD_ROOT}/Qwen--Qwen2.5-7B-Instruct",
                        "${WORLDFOUNDRY_CKPT_DIR}/hfd/Qwen--Qwen2.5-7B-Instruct",
                    ),
                    hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
                    min_file_count=1,
                    required_files=("config.json",),
                    required=False,
                    note="Qwen2.5 instruction model used by selected VBench-2.0 LLM judging paths.",
                ),
                BaseModelAsset(
                    id="vbench2_arcface_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH2_ARCFACE_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/arcface/resnet18_110.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench2/arcface/resnet18_110.pth",),
                    required=False,
                    manual_action=(
                        "Stage VBench-2.0 ArcFace resnet18_110.pth from the official VBench-2.0 asset release "
                        "under $WORLDFOUNDRY_VBENCH2_CACHE_DIR or set WORLDFOUNDRY_VBENCH2_ARCFACE_CKPT."
                    ),
                    note="ArcFace checkpoint used by VBench-2.0 human identity dimensions.",
                ),
                BaseModelAsset(
                    id="vbench2_yolo_world_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH2_YOLO_WORLD_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench2/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth",),
                    required=False,
                    manual_action=(
                        "Stage the official VBench-2.0 YOLO-World checkpoint under $WORLDFOUNDRY_VBENCH2_CACHE_DIR "
                        "or set WORLDFOUNDRY_VBENCH2_YOLO_WORLD_CKPT."
                    ),
                    note="YOLO-World checkpoint used by VBench-2.0 object and layout dimensions.",
                ),
                BaseModelAsset(
                    id="vbench2_human_anomaly_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH2_HUMAN_ANOMALY_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/anomaly_detector/human.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench2/anomaly_detector/human.pth",),
                    required=False,
                    manual_action=(
                        "Stage VBench-2.0 anomaly detector weights from the official asset release "
                        "or set WORLDFOUNDRY_VBENCH2_HUMAN_ANOMALY_CKPT."
                    ),
                    note="Human anomaly detector used by VBench-2.0 human fidelity metrics.",
                ),
                BaseModelAsset(
                    id="vbench2_face_anomaly_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH2_FACE_ANOMALY_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/anomaly_detector/face.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench2/anomaly_detector/face.pth",),
                    required=False,
                    manual_action=(
                        "Stage VBench-2.0 face anomaly detector weights from the official asset release "
                        "or set WORLDFOUNDRY_VBENCH2_FACE_ANOMALY_CKPT."
                    ),
                    note="Face anomaly detector used by VBench-2.0 human fidelity metrics.",
                ),
                BaseModelAsset(
                    id="vbench2_hand_anomaly_checkpoint",
                    kind="file",
                    role="checkpoint",
                    env=("WORLDFOUNDRY_VBENCH2_HAND_ANOMALY_CKPT",),
                    local_path="${WORLDFOUNDRY_CACHE_DIR}/models/vbench2/anomaly_detector/hand.pth",
                    alternate_paths=("${WORLDFOUNDRY_CKPT_DIR}/VBench2/anomaly_detector/hand.pth",),
                    required=False,
                    manual_action=(
                        "Stage VBench-2.0 hand anomaly detector weights from the official asset release "
                        "or set WORLDFOUNDRY_VBENCH2_HAND_ANOMALY_CKPT."
                    ),
                    note="Hand anomaly detector used by VBench-2.0 human fidelity metrics.",
                ),
            ),
            gpu_required=True,
        ),
        "vmbench_source_assets": _source_repo_capability(
            capability_id="vmbench_source_assets",
            env="WORLDFOUNDRY_VMBENCH_ROOT",
            repo_url="https://github.com/AMAP-ML/VMBench.git",
            local_name="AMAP-ML--VMBench",
            required_files=("prompts/prompts.json",),
            note="VMBench source checkout and prompt metadata.",
        ),
        "mirabench_source_assets": _source_repo_capability(
            capability_id="mirabench_source_assets",
            env="WORLDFOUNDRY_MIRABENCH_ROOT",
            repo_url="https://github.com/mira-space/MiraData.git",
            local_name="mira-space--MiraData",
            revision="7ad05795dd74acdc6b364222a9fa885a94b0c1a2",
            note="MiraBench/MiraData source checkout for preference and video-quality evaluation surfaces.",
        ),
        "devil_dynamics_source_assets": _source_repo_capability(
            capability_id="devil_dynamics_source_assets",
            env="WORLDFOUNDRY_DEVIL_DYNAMICS_ROOT",
            repo_url="https://github.com/MingXiangL/DEVIL.git",
            local_name="MingXiangL--DEVIL",
            revision="48556c329e6a6f92a587021bc97022494faec69c",
            note="DEVIL Dynamics source checkout for video dynamics evaluation prompts and metric code.",
        ),
        "phygenbench_source_assets": _source_repo_capability(
            capability_id="phygenbench_source_assets",
            env="WORLDFOUNDRY_PHYGENBENCH_ROOT",
            repo_url="https://github.com/OpenGVLab/PhyGenBench.git",
            local_name="OpenGVLab--PhyGenBench",
            revision="f8642cb796f3bcb01f0b7c1b2ec53b75d357c739",
            note="PhyGenBench source checkout for physical consistency evaluation assets.",
        ),
        "physvidbench_source_assets": _source_repo_capability(
            capability_id="physvidbench_source_assets",
            env="WORLDFOUNDRY_PHYSVIDBENCH_ROOT",
            repo_url="https://github.com/ensanli/PhysVidBenchCode.git",
            local_name="ensanli--PhysVidBenchCode",
            revision="b1d20f121ed3ebe5d18337d884fdecfad065096b",
            required_files=("prompts_questions.csv", "multi_ask.py"),
            required_paths=("captions",),
            note="PhysVidBench official code checkout with prompts_questions.csv and AuroraCap caption tracks.",
        ),
        "visual_chronometer_source_assets": _source_repo_capability(
            capability_id="visual_chronometer_source_assets",
            env="WORLDFOUNDRY_VISUAL_CHRONOMETER_ROOT",
            repo_url="https://github.com/taco-group/Visual_Chronometer.git",
            local_name="taco-group--Visual_Chronometer",
            revision="0dc1fd68641f476f18e76c9d1d0931278c02a386",
            required_paths=("inference/predict.py", "inference/configs/config_fps.yaml"),
            note="Visual Chronometer source assets are vendored in-tree for PhyFPS inference runtime.",
        ),
        "videophy_source_assets": _source_repo_capability(
            capability_id="videophy_source_assets",
            env="WORLDFOUNDRY_VIDEOPHY_ROOT",
            repo_url="https://github.com/Hritikbansal/videophy.git",
            local_name="Hritikbansal--videophy",
            revision="01d366c979e4a82875852adbf427ccc036e7373a",
            note="VideoPhy source checkout for physical commonsense and result normalization assets.",
        ),
        "physics_iq_source_assets": _source_repo_capability(
            capability_id="physics_iq_source_assets",
            env="WORLDFOUNDRY_PHYSICS_IQ_ROOT",
            repo_url="https://github.com/google-deepmind/physics-IQ-benchmark.git",
            local_name="google-deepmind--physics-IQ-benchmark",
            revision="52e52a14d3a7284ae24d8aff98fe982ac7a60971",
            note="Physics-IQ source checkout for official task definitions and normalization surfaces.",
        ),
        "t2v_safety_bench_source_assets": _source_repo_capability(
            capability_id="t2v_safety_bench_source_assets",
            env="WORLDFOUNDRY_T2V_SAFETY_BENCH_ROOT",
            repo_url="https://github.com/yibo-miao/T2VSafetyBench.git",
            local_name="yibo-miao--T2VSafetyBench",
            revision="97a841699fc25bc61662e6453bd6a2d9d187a9db",
            note="T2VSafetyBench source checkout for prompt and safety-rule assets.",
        ),
        "ipv_bench_source_assets": _source_repo_capability(
            capability_id="ipv_bench_source_assets",
            env="WORLDFOUNDRY_IPV_BENCH_ROOT",
            repo_url="https://github.com/showlab/Impossible-Videos.git",
            local_name="showlab--Impossible-Videos",
            revision="b959c423093d6e46a7e7008e205b9a16104e3f70",
            note="Impossible-Videos/IPV-Bench source checkout for counterfactual video prompts and assets.",
        ),
        "videoscience_bench_source_assets": _source_repo_capability(
            capability_id="videoscience_bench_source_assets",
            env="WORLDFOUNDRY_VIDEOSCIENCE_BENCH_ROOT",
            repo_url="https://github.com/hao-ai-lab/VideoScience.git",
            local_name="hao-ai-lab--VideoScience",
            revision="0b3bb2da74388991bf842bca1096a6a25d3c9311",
            note="VideoScience-Bench source checkout for scientific reasoning prompts and rubrics.",
        ),
        "phyeduvideo_source_assets": _source_repo_capability(
            capability_id="phyeduvideo_source_assets",
            env="WORLDFOUNDRY_PHYEDUVIDEO_ROOT",
            repo_url="https://github.com/meghamariamkm/PhyEduVideo.git",
            local_name="meghamariamkm--PhyEduVideo",
            revision="5a36a13818552095c9ed12c6562f236d5b5011bd",
            note="PhyEduVideo source checkout for physics-education prompts and rubrics.",
        ),
        "worldarena_source_assets": _source_repo_capability(
            capability_id="worldarena_source_assets",
            env="WORLDFOUNDRY_WORLDARENA_ROOT",
            repo_url="https://github.com/tsinghua-fib-lab/WorldArena.git",
            local_name="tsinghua-fib-lab--WorldArena",
            revision="329f6c7ce6c8019715aa8a2fff7b71112e4001d4",
            note="WorldArena source checkout for embodied world-model task and data-engine assets.",
        ),
        "world_in_world_source_assets": _external_dir_capability(
            capability_id="world_in_world_source_assets",
            env="WORLDFOUNDRY_WORLD_IN_WORLD_ASSETS_ROOT",
            local_name="world-in-world",
            alternate_paths=("worldfoundry/data/benchmarks/assets/world-in-world",),
            role="external_benchmark_asset",
            required=False,
            note="Optional World-in-World asset override; bundled prompt assets are used by default.",
        ),
    }
)


DEPTH_MODEL_CAPABILITY_IDS = (
    "depth_anything_v3",
    "midas_v21_small_256",
    "moge_vitl",
    "moge_v2_vitl_normal",
    "unidepth_v2_vitl14",
    "unik3d_vitl",
)
CORE_DEPTH_MODEL_CAPABILITY_IDS = ("depth_anything_v3", "moge_v2_vitl_normal", "unidepth_v2_vitl14")
SLAM_GEOMETRY_MODEL_CAPABILITY_IDS = ("droid_slam", "vggt_1b")
DETECTION_MODEL_CAPABILITY_IDS = ("grounding_dino",)
SEGMENTATION_MODEL_CAPABILITY_IDS = ("mobile_sam", "repvit_sam", "sam_v1", "sam2", "sam3")
SEGMENTATION_TRACKING_MODEL_CAPABILITY_IDS = (
    *SEGMENTATION_MODEL_CAPABILITY_IDS,
    "sam_vit_b",
    "aot_deaot_l",
)
MOTION_MODEL_CAPABILITY_IDS = ("raft", "sea_raft", "flowformerplusplus", "vfimamba")
CORE_MOTION_MODEL_CAPABILITY_IDS = ("raft",)
DEPTH_SLAM_DATA_CAPABILITY_IDS = ("tum_rgbd_slam_sample_assets", "kitti_depth_slam_assets")
DETECTION_SEGMENTATION_DATA_CAPABILITY_IDS = (
    "coco2017_detection_segmentation_assets",
    "davis2017_video_segmentation_assets",
)
SPATIAL_PERCEPTION_DATA_CAPABILITY_IDS = (
    *DETECTION_SEGMENTATION_DATA_CAPABILITY_IDS,
    *DEPTH_SLAM_DATA_CAPABILITY_IDS,
)
SPATIAL_PERCEPTION_CORE_CAPABILITY_IDS = (
    *CORE_DEPTH_MODEL_CAPABILITY_IDS,
    *SLAM_GEOMETRY_MODEL_CAPABILITY_IDS,
    *DETECTION_MODEL_CAPABILITY_IDS,
    "sam_v1",
    "sam2",
    *CORE_MOTION_MODEL_CAPABILITY_IDS,
)
SPATIAL_PERCEPTION_HEAVY_CAPABILITY_IDS = (
    *DEPTH_MODEL_CAPABILITY_IDS,
    *SLAM_GEOMETRY_MODEL_CAPABILITY_IDS,
    *DETECTION_MODEL_CAPABILITY_IDS,
    *SEGMENTATION_TRACKING_MODEL_CAPABILITY_IDS,
    *MOTION_MODEL_CAPABILITY_IDS,
)


BASE_MODEL_STACKS: dict[str, BaseModelStack] = {
    "depth_stack": BaseModelStack(
        id="depth_stack",
        family="depth",
        capability_ids=DEPTH_MODEL_CAPABILITY_IDS,
        note="Shared monocular and metric-depth assets used by video-world and 3D perception benchmarks.",
    ),
    "geometry_stack": BaseModelStack(
        id="geometry_stack",
        family="geometry",
        capability_ids=(
            *SLAM_GEOMETRY_MODEL_CAPABILITY_IDS,
            "moge_v2_vitl_normal",
            "unidepth_v2_vitl14",
        ),
        note="Reusable SLAM, camera-pose, multi-view geometry, and metric-depth assets.",
    ),
    "slam_stack": BaseModelStack(
        id="slam_stack",
        family="geometry",
        capability_ids=SLAM_GEOMETRY_MODEL_CAPABILITY_IDS,
        note="Camera/trajectory and multi-view geometry assets.",
    ),
    "camera_geometry_stack": BaseModelStack(
        id="camera_geometry_stack",
        family="geometry",
        capability_ids=(
            "vggt_1b",
            "moge_v2_vitl_normal",
            "unidepth_v2_vitl14",
            *CORE_MOTION_MODEL_CAPABILITY_IDS,
        ),
        note="Driver-compatible camera, metric-depth, and motion assets for camera-control/video-motion metrics.",
    ),
    "detection_stack": BaseModelStack(
        id="detection_stack",
        family="open_vocab_detection",
        capability_ids=DETECTION_MODEL_CAPABILITY_IDS,
        note="Open-vocabulary object detection assets.",
    ),
    "segmentation_stack": BaseModelStack(
        id="segmentation_stack",
        family="segmentation",
        capability_ids=SEGMENTATION_MODEL_CAPABILITY_IDS,
        note="SAM-family image and video segmentation assets.",
    ),
    "segmentation_tracking_stack": BaseModelStack(
        id="segmentation_tracking_stack",
        family="video_object_segmentation",
        capability_ids=SEGMENTATION_TRACKING_MODEL_CAPABILITY_IDS,
        note="SAM-family segmentation plus DeAOT tracking assets for object-level video metrics.",
    ),
    "detection_segmentation_stack": BaseModelStack(
        id="detection_segmentation_stack",
        family="open_vocab_perception",
        capability_ids=(*DETECTION_MODEL_CAPABILITY_IDS, *SEGMENTATION_TRACKING_MODEL_CAPABILITY_IDS),
        note="Open-vocabulary detection plus image/video segmentation and tracking assets.",
    ),
    "spatial_perception_core_stack": BaseModelStack(
        id="spatial_perception_core_stack",
        family="video_world_perception",
        capability_ids=SPATIAL_PERCEPTION_CORE_CAPABILITY_IDS,
        note="Default depth, SLAM/geometry, detection, segmentation, and motion stack for benchmark validation runs.",
    ),
    "spatial_perception_heavy_stack": BaseModelStack(
        id="spatial_perception_heavy_stack",
        family="video_world_perception",
        capability_ids=SPATIAL_PERCEPTION_HEAVY_CAPABILITY_IDS,
        note="Heavy depth, SLAM, multi-view geometry, detection, segmentation, tracking, and optical-flow assets.",
    ),
    "worldscore_spatial_metric_stack": BaseModelStack(
        id="worldscore_spatial_metric_stack",
        family="video_world_perception",
        capability_ids=(
            "worldscore_official_assets",
            "droid_slam",
            "grounding_dino",
            "raft",
            "sam_v1",
            "sam2",
            "sea_raft",
            "flowformerplusplus",
            "vfimamba",
        ),
        note="WorldScore official dataset/checkpoint assets plus the reusable spatial metric models.",
    ),
    "grounded_depth_segmentation_stack": BaseModelStack(
        id="grounded_depth_segmentation_stack",
        family="grounded_video_perception",
        capability_ids=("depth_anything_v3", "grounding_dino", "sam_v1", "sam2"),
        note="Depth plus GroundingDINO/SAM segmentation assets used by compositional T2V metric runners.",
    ),
    "grounded_video_quality_stack": BaseModelStack(
        id="grounded_video_quality_stack",
        family="video_quality",
        capability_ids=("clip_vit_b32", "grounding_dino", "raft", "sam_v1", "sam2"),
        note="CLIP, flow, detection, and segmentation assets shared by object-aware video quality metrics.",
    ),
    "video_quality_motion_stack": BaseModelStack(
        id="video_quality_motion_stack",
        family="video_quality",
        capability_ids=("clip_vit_b32", "raft"),
        note="Compact CLIP and flow stack for lightweight video quality normalizers.",
    ),
    "video_quality_dino_motion_stack": BaseModelStack(
        id="video_quality_dino_motion_stack",
        family="video_quality",
        capability_ids=("clip_vit_b32", "dinov2_base", "raft"),
        note="CLIP, DINOv2, and flow stack for video quality metrics with DINO-style embeddings.",
    ),
    "vbench_perception_metric_stack": BaseModelStack(
        id="vbench_perception_metric_stack",
        family="video_quality",
        capability_ids=(
            "clip_vit_b32",
            "dinov2_base",
            "grounding_dino",
            "sam_v1",
            "sam2",
            "sam3",
            "raft",
            "vbench_metric_checkpoint_assets",
        ),
        note="Shared VBench-family quality, detection, segmentation, embedding, motion, and metric checkpoint assets.",
    ),
    "wbench_metric_stack": BaseModelStack(
        id="wbench_metric_stack",
        family="video_world",
        capability_ids=(
            "wbench_quality_metric_assets",
            "dinov2_base",
            "sam2",
            "depth_anything_v3",
            "raft",
            "wbench_transnetv2",
            "wbench_hpsv3",
            "wbench_pavrm_qwen3vl",
            "wbench_dreamsim",
            "wbench_megasam",
            "unidepth_v2_vitl14",
        ),
        note="WBench in-tree metric stack: segmentation, depth, flow, quality, reward, DreamSim, and MegaSAM pose assets.",
    ),
    "videoscore_reward_metric_stack": BaseModelStack(
        id="videoscore_reward_metric_stack",
        family="video_reward_model",
        capability_ids=("videoscore_reward_model_v1_1", "videoscore_bench_dataset_assets"),
        note="VideoScore reward-model checkpoint plus VideoScore-Bench data assets for bounded official scorer runs.",
    ),
    "videophy_metric_stack": BaseModelStack(
        id="videophy_metric_stack",
        family="video_metric_model",
        capability_ids=("videophy_source_assets", "videocon_physics_model"),
        note="VideoPhy prompt assets plus the in-tree VideoCon-Physics AutoEvaluator checkpoint.",
    ),
    "videophy2_autoeval_metric_stack": BaseModelStack(
        id="videophy2_autoeval_metric_stack",
        family="video_metric_model",
        capability_ids=("videophy2_dataset_assets", "videophy2_autoeval_model"),
        note="VideoPhy2 prompt assets plus the in-tree AutoEval MLLM checkpoint.",
    ),
    "genai_bench_metric_stack": BaseModelStack(
        id="genai_bench_metric_stack",
        family="video_metric_model",
        capability_ids=(
            "t2v_metrics_source_assets",
            "t2v_metrics_clip_flant5_xxl",
            "genai_bench_dataset_assets",
            "t2v_metrics_blip2_itm",
            "t2v_metrics_internvideo2_itm",
        ),
        note="In-tree VQAScore backend plus GenAI-Bench dataset assets for mock or GPU scorer runs.",
    ),
    "t2v_metrics_metric_stack": BaseModelStack(
        id="t2v_metrics_metric_stack",
        family="video_metric_model",
        capability_ids=(
            "t2v_metrics_source_assets",
            "t2v_metrics_clip_flant5_xxl",
            "t2v_metrics_blip2_itm",
            "t2v_metrics_internvideo2_itm",
        ),
        note="Shared in-tree VQAScore/ITMScore model stack for GenAI-Bench, PhyGenBench, and CameraBench score generation.",
    ),
    "jedi_metric_stack": BaseModelStack(
        id="jedi_metric_stack",
        family="video_metric_model",
        capability_ids=(
            "jedi_source_assets",
            "jedi_vjepa_vith16",
            "jedi_vjepa_ssv2_probe",
        ),
        note="In-tree VideoJEDi package plus V-JEPA checkpoints for mock or GPU distribution-quality scoring.",
    ),
    "visual_chronometer_fps_metric_stack": BaseModelStack(
        id="visual_chronometer_fps_metric_stack",
        family="video_metric_model",
        capability_ids=("visual_chronometer_fps_predictor", "visual_chronometer_source_assets"),
        note="Visual Chronometer PhyFPS predictor checkpoint plus official inference source checkout.",
    ),
    "official_metric_asset_stack": BaseModelStack(
        id="official_metric_asset_stack",
        family="benchmark_data_asset",
        capability_ids=(
            "worldscore_official_assets",
            "worldscore_dataset_assets",
            "videoscore_bench_dataset_assets",
            "chronomagic_dataset_assets",
            "aigcbench_dataset_assets",
            "camerabench_dataset_assets",
            "ewmbench_dataset_assets",
            "evalcrafter_dataset_assets",
            "t2v_compbench_dataset_assets",
            "fetv_dataset_assets",
            "genai_bench_dataset_assets",
            "t2v_metrics_source_assets",
            "t2v_metrics_clip_flant5_xxl",
            "t2v_metrics_blip2_itm",
            "t2v_metrics_internvideo2_itm",
            "jedi_source_assets",
            "jedi_vjepa_vith16",
            "jedi_vjepa_ssv2_probe",
            "iworld_bench_source_assets",
            "iworld_bench_dataset_assets",
            "libero_dataset_assets",
            "likephys_dataset_assets",
            "phyground_dataset_assets",
            "physvidbench_dataset_assets",
            "rbench_dataset_assets",
            "robotwin_dataset_assets",
            "vbench2_dataset_assets",
            "video_bench_dataset_assets",
            "videophy2_dataset_assets",
            "videophy2_autoeval_model",
            "videoverse_dataset_assets",
            "worldbench_dataset_assets",
            "worldmodelbench_dataset_assets",
            "bridgedata_v2_source_assets",
            "bridgedata_v2_dataset_assets",
            "calvin_source_assets",
            "calvin_dataset_assets",
            "maniskill_source_assets",
            "metaworld_source_assets",
            "rlbench_source_assets",
            "robocasa_source_assets",
            "simpler_env_source_assets",
            "t2vworldbench_source_assets",
            "vbench_source_assets",
            "vbench_plus_plus_source_assets",
            "vbench2_source_assets",
            "vmbench_source_assets",
            "mirabench_source_assets",
            "devil_dynamics_source_assets",
            "phygenbench_source_assets",
            "physvidbench_source_assets",
            "videophy_source_assets",
            "videocon_physics_model",
            "physics_iq_source_assets",
            "t2v_safety_bench_source_assets",
            "ipv_bench_source_assets",
            "videoscience_bench_source_assets",
            "phyeduvideo_source_assets",
            "visual_chronometer_source_assets",
            "worldarena_source_assets",
            "world_in_world_source_assets",
        ),
        note="Reusable benchmark data/checkpoint bundles staged through the base-model asset planner.",
    ),
    "benchmark_dataset_asset_stack": BaseModelStack(
        id="benchmark_dataset_asset_stack",
        family="benchmark_data_asset",
        capability_ids=(
            "aigcbench_dataset_assets",
            "camerabench_dataset_assets",
            "chronomagic_dataset_assets",
            "ewmbench_dataset_assets",
            "evalcrafter_dataset_assets",
            "t2v_compbench_dataset_assets",
            "fetv_dataset_assets",
            "genai_bench_dataset_assets",
            "t2v_metrics_source_assets",
            "t2v_metrics_clip_flant5_xxl",
            "t2v_metrics_blip2_itm",
            "t2v_metrics_internvideo2_itm",
            "iworld_bench_source_assets",
            "iworld_bench_dataset_assets",
            "libero_dataset_assets",
            "likephys_dataset_assets",
            "phyground_dataset_assets",
            "physvidbench_dataset_assets",
            "rbench_dataset_assets",
            "robotwin_dataset_assets",
            "vbench2_dataset_assets",
            "video_bench_dataset_assets",
            "videophy2_dataset_assets",
            "videophy2_autoeval_model",
            "videoverse_dataset_assets",
            "worldbench_dataset_assets",
            "worldmodelbench_dataset_assets",
            "worldscore_dataset_assets",
            "videoscore_bench_dataset_assets",
            "bridgedata_v2_source_assets",
            "bridgedata_v2_dataset_assets",
            "calvin_source_assets",
            "calvin_dataset_assets",
            "maniskill_source_assets",
            "metaworld_source_assets",
            "rlbench_source_assets",
            "robocasa_source_assets",
            "simpler_env_source_assets",
            "t2vworldbench_source_assets",
            "vbench_source_assets",
            "vbench_plus_plus_source_assets",
            "vbench2_source_assets",
            "vmbench_source_assets",
            "mirabench_source_assets",
            "devil_dynamics_source_assets",
            "phygenbench_source_assets",
            "physvidbench_source_assets",
            "videophy_source_assets",
            "videocon_physics_model",
            "physics_iq_source_assets",
            "t2v_safety_bench_source_assets",
            "ipv_bench_source_assets",
            "videoscience_bench_source_assets",
            "phyeduvideo_source_assets",
            "visual_chronometer_source_assets",
            "worldarena_source_assets",
            "world_in_world_source_assets",
        ),
        note="Benchmark datasets, source repositories, and external staging roots with env exports and download commands where available.",
    ),
    "depth_eval_data_stack": BaseModelStack(
        id="depth_eval_data_stack",
        family="perception_data_asset",
        capability_ids=DEPTH_SLAM_DATA_CAPABILITY_IDS,
        note="Depth-oriented RGB-D/KITTI data assets staged through the base-model asset planner.",
    ),
    "slam_eval_data_stack": BaseModelStack(
        id="slam_eval_data_stack",
        family="perception_data_asset",
        capability_ids=DEPTH_SLAM_DATA_CAPABILITY_IDS,
        note="SLAM/trajectory data assets for reusable camera-pose and trajectory sanity checks.",
    ),
    "detection_segmentation_eval_data_stack": BaseModelStack(
        id="detection_segmentation_eval_data_stack",
        family="perception_data_asset",
        capability_ids=DETECTION_SEGMENTATION_DATA_CAPABILITY_IDS,
        note="Detection, image segmentation, and video object segmentation evaluation data assets.",
    ),
    "spatial_perception_data_stack": BaseModelStack(
        id="spatial_perception_data_stack",
        family="perception_data_asset",
        capability_ids=SPATIAL_PERCEPTION_DATA_CAPABILITY_IDS,
        note="Common depth, SLAM, detection, and segmentation datasets kept at the base-model layer.",
    ),
    "spatial_perception_full_eval_stack": BaseModelStack(
        id="spatial_perception_full_eval_stack",
        family="video_world_perception",
        capability_ids=(
            *SPATIAL_PERCEPTION_HEAVY_CAPABILITY_IDS,
            *SPATIAL_PERCEPTION_DATA_CAPABILITY_IDS,
        ),
        note="Heavy spatial perception model stack plus reusable eval data assets for end-to-end validation.",
    ),
    "motion_stack": BaseModelStack(
        id="motion_stack",
        family="motion",
        capability_ids=MOTION_MODEL_CAPABILITY_IDS,
        note="Optical-flow and motion-estimation assets shared by temporal quality metrics.",
    ),
    "video_world_perception_stack": BaseModelStack(
        id="video_world_perception_stack",
        family="video_world",
        capability_ids=(
            "clip_vit_b32",
            "dinov2_base",
            "siglip_so400m",
            "depth_anything_v3",
            "droid_slam",
            *DETECTION_MODEL_CAPABILITY_IDS,
            "sam_v1",
            "sam2",
            "raft",
            "sea_raft",
        ),
        note="Default shared perception stack for video-world benchmark runners.",
    ),
}


BENCHMARK_DATA_ASSET_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "aigcbench": ("aigcbench_dataset_assets",),
    "bridgedata-v2": ("bridgedata_v2_source_assets", "bridgedata_v2_dataset_assets"),
    "camerabench": ("camerabench_dataset_assets", "t2v_metrics_source_assets", "t2v_metrics_blip2_itm"),
    "calvin": ("calvin_source_assets", "calvin_dataset_assets"),
    "chronomagic-bench": ("chronomagic_dataset_assets",),
    "devil-dynamics": ("devil_dynamics_source_assets",),
    "ewmbench": ("ewmbench_dataset_assets",),
    "evalcrafter": ("evalcrafter_dataset_assets", "jedi_source_assets"),
    "fetv": (
        "fetv_dataset_assets",
        "fetv_blip_retrieval_source",
        "fetv_stylegan_v_fvd_source",
        "jedi_source_assets",
    ),
    "genai-bench": (
        "genai_bench_dataset_assets",
        "t2v_metrics_source_assets",
        "t2v_metrics_clip_flant5_xxl",
    ),
    "ipv-bench": ("ipv_bench_source_assets",),
    "iworld-bench": ("iworld_bench_source_assets", "iworld_bench_dataset_assets"),
    "libero": ("libero_source_assets", "libero_dataset_assets"),
    "likephys": ("likephys_dataset_assets",),
    "maniskill": ("maniskill_source_assets",),
    "metaworld": ("metaworld_source_assets",),
    "mirabench": ("mirabench_source_assets",),
    "phyground": ("phyground_dataset_assets",),
    "phyeduvideo": ("phyeduvideo_source_assets",),
    "phygenbench": ("phygenbench_source_assets", "t2v_metrics_source_assets", "t2v_metrics_clip_flant5_xxl"),
    "phyfps-bench-gen": ("visual_chronometer_source_assets",),
    "physics-iq": ("physics_iq_source_assets",),
    "physvidbench": ("physvidbench_dataset_assets", "physvidbench_source_assets"),
    "rbench": ("rbench_dataset_assets",),
    "rlbench": ("rlbench_source_assets",),
    "robocasa": ("robocasa_source_assets",),
    "robotwin": ("robotwin_dataset_assets",),
    "simpler-env": ("simpler_env_source_assets",),
    "t2v-safety-bench": ("t2v_safety_bench_source_assets",),
    "t2v-compbench": ("t2v_compbench_dataset_assets",),
    "t2vworldbench": ("t2vworldbench_source_assets",),
    "vbench": ("vbench_source_assets", "vbench_metric_checkpoint_assets"),
    "vbench-2.0": ("vbench2_source_assets", "vbench_metric_checkpoint_assets", "vbench2_dataset_assets"),
    "vbench-plus-plus": ("vbench_source_assets", "vbench_plus_plus_source_assets", "vbench_metric_checkpoint_assets"),
    "video-bench": ("video_bench_dataset_assets",),
    "videoscience-bench": ("videoscience_bench_source_assets",),
    "videoverse": ("videoverse_dataset_assets",),
    "visual-chronometer": ("visual_chronometer_source_assets",),
    "videophy": ("videophy_source_assets", "videocon_physics_model"),
    "videophy2": ("videophy2_dataset_assets", "videophy2_autoeval_model"),
    "videoscore": ("videoscore_bench_dataset_assets",),
    "vmbench": ("vmbench_source_assets",),
    "wbench": ("wbench_dataset_assets",),
    "world-in-world": ("world_in_world_source_assets",),
    "worldarena": ("worldarena_source_assets",),
    "worldbench": ("worldbench_dataset_assets",),
    "worldmodelbench": ("worldmodelbench_dataset_assets", "worldmodelbench_vila_judge_model"),
    "worldscore": ("worldscore_official_assets", "worldscore_dataset_assets"),
}


def benchmark_data_asset_capability_ids(benchmark_id: str) -> tuple[str, ...]:
    """Return benchmark-owned data asset capabilities staged with base-model assets."""

    return BENCHMARK_DATA_ASSET_CAPABILITIES.get(str(benchmark_id), ())


def benchmark_base_model_dependency_ids(
    benchmark_id: str,
    base_model_dependencies: tuple[str, ...] | list[str] = (),
    optional_base_model_dependencies: tuple[str, ...] | list[str] = (),
    *,
    include_optional: bool = False,
    include_data_assets: bool = True,
) -> tuple[str, ...]:
    """Return base-model stack/capability ids needed to prepare a benchmark."""

    ids: list[str] = []
    ids.extend(str(item) for item in base_model_dependencies if str(item).strip())
    if include_optional:
        ids.extend(str(item) for item in optional_base_model_dependencies if str(item).strip())
    if include_data_assets:
        ids.extend(benchmark_data_asset_capability_ids(benchmark_id))
    return tuple(dict.fromkeys(ids))


def _append_group(groups: dict[str, list[str]], key: str, value: str) -> None:
    """Helper function to append group.

    Args:
        groups: The groups.
        key: The key.
        value: The value.

    Returns:
        The return value.
    """
    group = groups.setdefault(key, [])
    if value not in group:
        group.append(value)


def _count_asset_roles(capabilities: Mapping[str, BaseModelCapability]) -> dict[str, int]:
    """Helper function to count asset roles.

    Args:
        capabilities: The capabilities.

    Returns:
        The return value.
    """
    counts: dict[str, int] = {}
    for capability in capabilities.values():
        for asset in capability.assets:
            counts[asset.role] = counts.get(asset.role, 0) + 1
    return dict(sorted(counts.items()))


def _capabilities_by_family() -> dict[str, list[str]]:
    """Helper function to capabilities by family.

    Returns:
        The return value.
    """
    groups: dict[str, list[str]] = {}
    for capability_id, capability in BASE_MODEL_CAPABILITIES.items():
        _append_group(groups, capability.family, capability_id)
    return {family: sorted(ids) for family, ids in sorted(groups.items())}


def _stacks_by_family() -> dict[str, list[str]]:
    """Helper function to stacks by family.

    Returns:
        The return value.
    """
    groups: dict[str, list[str]] = {}
    for stack_id, stack in BASE_MODEL_STACKS.items():
        _append_group(groups, stack.family, stack_id)
    return {family: sorted(ids) for family, ids in sorted(groups.items())}


def base_model_inventory() -> dict[str, Any]:
    """Base model inventory.

    Returns:
        The return value.
    """
    asset_count = sum(len(capability.assets) for capability in BASE_MODEL_CAPABILITIES.values())
    return {
        "schema_version": "worldfoundry-base-model-inventory-v1",
        "capability_count": len(BASE_MODEL_CAPABILITIES),
        "stack_count": len(BASE_MODEL_STACKS),
        "asset_count": asset_count,
        "required_asset_count": sum(
            1
            for capability in BASE_MODEL_CAPABILITIES.values()
            for asset in capability.assets
            if asset.required
        ),
        "capabilities_by_family": _capabilities_by_family(),
        "stacks_by_family": _stacks_by_family(),
        "asset_counts_by_role": _count_asset_roles(BASE_MODEL_CAPABILITIES),
        "benchmark_data_asset_count": len(BENCHMARK_DATA_ASSET_CAPABILITIES),
        "benchmark_data_assets": {
            benchmark_id: list(capability_ids)
            for benchmark_id, capability_ids in sorted(BENCHMARK_DATA_ASSET_CAPABILITIES.items())
        },
        "capabilities": [
            BASE_MODEL_CAPABILITIES[capability_id].to_dict()
            for capability_id in sorted(BASE_MODEL_CAPABILITIES)
        ],
        "stacks": [
            BASE_MODEL_STACKS[stack_id].to_dict()
            for stack_id in sorted(BASE_MODEL_STACKS)
        ],
    }


def _base_model_plan_summary(
    checks: list[dict[str, Any]],
    *,
    download_commands: list[str],
    export_commands: list[str],
    pip_install_packages: list[str],
    manual_actions: list[str],
) -> dict[str, Any]:
    """Helper function to base model plan summary.

    Args:
        checks: The checks.

    Returns:
        The return value.
    """
    family_counts: dict[str, dict[str, int]] = {}
    asset_counts_by_role: dict[str, int] = {}
    missing_required_asset_ids: list[str] = []
    ready_asset_ids: list[str] = []
    missing_capability_ids: list[str] = []

    for check in checks:
        family = str(check.get("family") or "unknown")
        family_row = family_counts.setdefault(
            family,
            {
                "capability_count": 0,
                "ok_count": 0,
                "missing_count": 0,
                "asset_count": 0,
                "missing_required_asset_count": 0,
            },
        )
        family_row["capability_count"] += 1
        if check.get("ok"):
            family_row["ok_count"] += 1
        else:
            family_row["missing_count"] += 1
            missing_capability_ids.append(str(check.get("id") or ""))

        for asset in check.get("asset_status") or ():
            if not isinstance(asset, Mapping):
                continue
            role = str(asset.get("role") or "asset")
            asset_counts_by_role[role] = asset_counts_by_role.get(role, 0) + 1
            family_row["asset_count"] += 1
            if asset.get("ready"):
                ready_asset_ids.append(str(asset.get("id") or ""))
            elif asset.get("required"):
                asset_id = str(asset.get("id") or "")
                missing_required_asset_ids.append(asset_id)
                family_row["missing_required_asset_count"] += 1

    return {
        "capability_count": len(checks),
        "ok_capability_count": sum(1 for check in checks if check.get("ok")),
        "missing_capability_count": sum(1 for check in checks if not check.get("ok")),
        "missing_capability_ids": [item for item in missing_capability_ids if item],
        "asset_count": sum(len(check.get("asset_status") or ()) for check in checks),
        "ready_asset_count": len([item for item in ready_asset_ids if item]),
        "missing_required_asset_count": len([item for item in missing_required_asset_ids if item]),
        "missing_required_asset_ids": [item for item in missing_required_asset_ids if item],
        "asset_counts_by_role": dict(sorted(asset_counts_by_role.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "download_command_count": len(download_commands),
        "export_command_count": len(export_commands),
        "pip_install_package_count": len(pip_install_packages),
        "manual_action_count": len(manual_actions),
    }


def resolve_base_model_capability_ids(capability_or_stack_ids: list[str] | tuple[str, ...]) -> list[str]:
    """Resolve base model capability ids.

    Args:
        capability_or_stack_ids: The capability or stack ids.

    Returns:
        The return value.
    """
    resolved: list[str] = []
    for raw_id in capability_or_stack_ids:
        item_id = str(raw_id)
        if item_id in BASE_MODEL_CAPABILITIES:
            candidate_ids = (item_id,)
        elif item_id in BASE_MODEL_STACKS:
            candidate_ids = BASE_MODEL_STACKS[item_id].capability_ids
        else:
            available = ", ".join(sorted([*BASE_MODEL_CAPABILITIES, *BASE_MODEL_STACKS]))
            raise KeyError(f"unknown base-model capability or stack {item_id!r}; available: {available}")
        for capability_id in candidate_ids:
            if capability_id not in resolved:
                resolved.append(capability_id)
    return resolved


def get_base_model_capability(capability_id: str) -> BaseModelCapability:
    """Get base model capability.

    Args:
        capability_id: The capability id.

    Returns:
        The return value.
    """
    try:
        return BASE_MODEL_CAPABILITIES[capability_id]
    except KeyError as exc:
        available = ", ".join(sorted(BASE_MODEL_CAPABILITIES))
        raise KeyError(f"unknown base-model capability {capability_id!r}; available: {available}") from exc


def _vbench_metric_asset_status(asset_id: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    capability = get_base_model_capability("vbench_metric_checkpoint_assets")
    for asset in capability.assets:
        if asset.id == asset_id:
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
    raise KeyError(f"unknown VBench asset id: {asset_id}")


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


def check_base_model_capability(
    capability_id: str,
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
    check_imports: bool = True,
) -> dict[str, Any]:
    """Check base model capability.

    Args:
        capability_id: The capability id.
        env: The env.
        python_executable: The python executable.
        check_imports: The check imports.

    Returns:
        The return value.
    """
    environ = os.environ if env is None else env
    capability = get_base_model_capability(capability_id)
    imports = []
    for name in capability.package_imports:
        imports.append(
            {
                "name": name,
                "available": _import_available(name, python_executable) if check_imports else None,
                "python_executable": None if python_executable is None else str(python_executable),
                "skipped": not check_imports,
            }
        )
    asset_env = [
        {
            "name": name,
            "present": bool(environ.get(name)),
        }
        for name in capability.asset_env
    ]
    assets = [asset.check(environ) for asset in capability.assets]
    owner_path = capability.owner_path()
    if assets:
        required_assets_ready = all(asset["ready"] for asset in assets if asset["required"])
    elif not asset_env:
        required_assets_ready = True
    elif capability.asset_env_policy == "all":
        required_assets_ready = all(item["present"] for item in asset_env)
    else:
        required_assets_ready = any(item["present"] for item in asset_env)
    package_by_import = capability.pip_package_by_import()
    missing_packages = []
    missing_non_pip_imports = []
    for import_status in imports:
        if import_status.get("skipped"):
            continue
        if import_status["available"]:
            continue
        import_name = str(import_status["name"])
        if import_name in capability.non_pip_imports:
            missing_non_pip_imports.append(import_name)
            continue
        package = package_by_import.get(import_name, import_name)
        if package:
            missing_packages.append(package)
    manual_install_actions = list(capability.manual_install_actions) if missing_non_pip_imports else []
    manual_asset_actions = [
        asset.get("manual_action") or f"stage {asset['id']} at one of: {', '.join(asset['candidate_paths'])}"
        for asset in assets
        if asset["required"] and not asset["ready"] and not asset["download_command_text"]
    ]
    repair_hints = {
        "pip_install_packages": missing_packages,
        "pip_command": None if not missing_packages else "python -m pip install " + " ".join(missing_packages),
        "missing_non_pip_imports": missing_non_pip_imports,
        "manual_install_actions": manual_install_actions,
        "download_command_argvs": [
            asset["download_command"]
            for asset in assets
            if asset["required"] and not asset["ready"] and asset["download_command"]
        ],
        "download_commands": [
            asset["download_command_text"]
            for asset in assets
            if asset["required"] and not asset["ready"] and asset["download_command_text"]
        ],
        "export_commands": [
            command
            for asset in assets
            if asset["required"] and not asset["env_ready"]
            for command in asset["export_commands"]
        ],
        "manual_actions": [*manual_install_actions, *manual_asset_actions],
    }
    imports_ok = True if not check_imports else all(item["available"] for item in imports)
    ok = owner_path.exists() and imports_ok and required_assets_ready
    payload = capability.to_dict()
    payload.update(
        {
            "owner_path": str(owner_path),
            "owner_path_exists": owner_path.exists(),
            "imports": imports,
            "import_checks_skipped": not check_imports,
            "asset_env_status": asset_env,
            "asset_status": assets,
            "required_assets_ready": required_assets_ready,
            "repair_hints": repair_hints,
            "ok": ok,
        }
    )
    return payload


def check_base_model_dependencies(
    capability_ids: list[str] | tuple[str, ...],
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
    check_imports: bool = True,
) -> dict[str, Any]:
    """Check base model dependencies.

    Args:
        capability_ids: The capability ids.
        env: The env.
        python_executable: The python executable.
        check_imports: The check imports.

    Returns:
        The return value.
    """
    resolved_ids = resolve_base_model_capability_ids(capability_ids)
    stack_ids = [str(capability_id) for capability_id in capability_ids if str(capability_id) in BASE_MODEL_STACKS]
    checks = [
        check_base_model_capability(
            str(capability_id),
            env=env,
            python_executable=python_executable,
            check_imports=check_imports,
        )
        for capability_id in resolved_ids
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "python_executable": None if python_executable is None else str(python_executable),
        "import_checks_skipped": not check_imports,
        "requested_ids": [str(capability_id) for capability_id in capability_ids],
        "stack_ids": stack_ids,
        "capability_ids": [check["id"] for check in checks],
        "checks": checks,
    }


def base_model_materialization_plan(
    capability_ids: list[str] | tuple[str, ...] | None = None,
    env: Mapping[str, str] | None = None,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Base model materialization plan.

    Args:
        capability_ids: The capability ids.
        env: The env.
        python_executable: The python executable.

    Returns:
        The return value.
    """
    requested_ids = sorted(BASE_MODEL_CAPABILITIES) if capability_ids is None else list(capability_ids)
    selected_ids = resolve_base_model_capability_ids(requested_ids)
    stack_ids = [str(capability_id) for capability_id in requested_ids if str(capability_id) in BASE_MODEL_STACKS]
    checks = [
        check_base_model_capability(capability_id, env=env, python_executable=python_executable)
        for capability_id in selected_ids
    ]
    download_commands = [
        command
        for check in checks
        for command in check["repair_hints"]["download_commands"]
    ]
    download_command_argvs = [
        command
        for check in checks
        for command in check["repair_hints"]["download_command_argvs"]
    ]
    export_commands = [
        command
        for check in checks
        for command in check["repair_hints"]["export_commands"]
    ]
    pip_install_packages = sorted(
        {
            package
            for check in checks
            for package in check["repair_hints"]["pip_install_packages"]
        }
    )
    manual_actions = [
        action
        for check in checks
        for action in check["repair_hints"]["manual_actions"]
    ]
    return {
        "schema_version": "worldfoundry-base-model-assets-v1",
        "ok": all(check["ok"] for check in checks),
        "python_executable": None if python_executable is None else str(python_executable),
        "requested_ids": requested_ids,
        "stack_ids": stack_ids,
        "stacks": [BASE_MODEL_STACKS[stack_id].to_dict() for stack_id in stack_ids],
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
        "download_command_argvs": download_command_argvs,
        "export_commands": export_commands,
        "pip_install_packages": pip_install_packages,
        "manual_actions": manual_actions,
    }
