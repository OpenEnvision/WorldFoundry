"""Docker execution support for embodied evaluation configs."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from .image_refs import resolve_docker_image, resolve_docker_platform

ACCEPTED_LICENSES_ENV = "WORLDFOUNDRY_ACCEPTED_LICENSES"


def inside_docker() -> bool:
    return Path("/.dockerenv").exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _docker_available(docker: str) -> None:
    if subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        raise RuntimeError("Docker daemon is not reachable")


def _ensure_image(
    docker: str,
    image: str,
    *,
    source_image: str | None = None,
    platform: str | None = None,
    pull: bool = False,
) -> None:
    exists = subprocess.run([docker, "image", "inspect", image], capture_output=True).returncode == 0
    if exists or not pull:
        return
    pull_cmd = [docker, "pull"]
    if platform:
        pull_cmd.extend(["--platform", platform])
    if source_image and source_image != image:
        source_exists = subprocess.run([docker, "image", "inspect", source_image], capture_output=True).returncode == 0
        if not source_exists:
            rc = subprocess.call([*pull_cmd, source_image])
            if rc != 0:
                raise RuntimeError(f"docker pull failed for {source_image} with exit code {rc}")
        rc = subprocess.call([docker, "tag", source_image, image])
        if rc != 0:
            raise RuntimeError(f"docker tag failed for {source_image} -> {image} with exit code {rc}")
        return
    rc = subprocess.call([*pull_cmd, image])
    if rc != 0:
        raise RuntimeError(f"docker pull failed for {image} with exit code {rc}")


def _gpu_flags(gpus: Any) -> list[str]:
    if gpus in (None, "", False):
        return []
    return ["--gpus", str(gpus)]


def _inner_run_args(
    *,
    shard_id: int | None = None,
    num_shards: int | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
) -> list[str]:
    args = ["--no-docker", "--config", "/tmp/eval_config.yaml"]
    if shard_id is not None:
        args.extend(["--shard-id", str(shard_id), "--num-shards", str(num_shards)])
    if eval_id:
        args.extend(["--eval-id", eval_id])
    if no_save:
        args.append("--no-save")
    return args


def _default_container_command(
    docker_cfg: Mapping[str, Any],
    inner_args: list[str],
) -> list[str]:
    configured = docker_cfg.get("command")
    if configured:
        command = list(configured)
        command.extend(inner_args)
        return command

    python_env = docker_cfg.get("python_env")
    if python_env:
        module_cmd = ["python", "-m", "worldfoundry.cli.main", "embodied", "run", *inner_args]
        if docker_cfg.get("xvfb"):
            shell = (
                "Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset & "
                "export DISPLAY=:99; "
                'eval "$(conda shell.bash hook 2>/dev/null)"; '
                f"conda activate {python_env}; "
                'exec "$@"'
            )
            return ["/bin/bash", "-lc", shell, "worldfoundry-embodied", *module_cmd]
        return ["conda", "run", "--no-capture-output", "-n", str(python_env), *module_cmd]

    return ["python", "-m", "worldfoundry.cli.main", "embodied", "run", *inner_args]


def write_docker_config(config: Mapping[str, Any], output_dir: Path) -> Path:
    """Write a container-remapped config file and return its temp path."""
    docker_config = dict(config)
    docker_config["output_dir"] = "/workspace/results"
    fd, temp_path = tempfile.mkstemp(prefix="wf-embodied-docker-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(docker_config, handle, sort_keys=False)
    except Exception:
        os.close(fd)
        raise
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(temp_path)


def build_docker_run_command(
    config: Mapping[str, Any],
    *,
    docker_config_path: Path,
    output_dir: Path,
    shard_id: int | None = None,
    num_shards: int | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
) -> list[str]:
    """Build the ``docker run`` command for an embodied eval."""
    docker_cfg = dict(config.get("docker") or {})
    image = resolve_docker_image(docker_cfg)
    platform = resolve_docker_platform(docker_cfg)

    repo_root = _repo_root()
    command = _default_container_command(
        docker_cfg,
        _inner_run_args(shard_id=shard_id, num_shards=num_shards, eval_id=eval_id, no_save=no_save),
    )

    cmd = [
        shutil.which("docker") or "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{output_dir}:/workspace/results",
        "-v",
        f"{docker_config_path}:/tmp/eval_config.yaml:ro",
        "-v",
        f"{repo_root}:/workspace/WorldFoundry",
        "-w",
        "/workspace/WorldFoundry",
        "-e",
        "PYTHONPATH=/workspace/WorldFoundry:/workspace/WorldFoundry/src",
        "-e",
        "WORLDFOUNDRY_REPO_ROOT=/workspace/WorldFoundry",
        "-e",
        f"WORLDFOUNDRY_HOST_OUTPUT_DIR={output_dir}",
    ]

    entrypoint = docker_cfg.get("entrypoint")
    if entrypoint is None and (docker_cfg.get("python_env") or docker_cfg.get("command")):
        entrypoint = ""
    if entrypoint is not None:
        cmd.extend(["--entrypoint", str(entrypoint)])

    if docker_cfg.get("name"):
        cmd.extend(["--name", str(docker_cfg["name"])])
    if docker_cfg.get("user") == "host" and hasattr(os, "getuid"):
        cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    elif docker_cfg.get("user"):
        cmd.extend(["--user", str(docker_cfg["user"])])
    for volume in docker_cfg.get("volumes") or ():
        cmd.extend(["-v", os.path.expandvars(str(volume))])
    for env_value in docker_cfg.get("env") or ():
        cmd.extend(["-e", os.path.expandvars(str(env_value))])
    accepted_licenses = os.environ.get(ACCEPTED_LICENSES_ENV, "").strip()
    if accepted_licenses:
        cmd.extend(["-e", f"{ACCEPTED_LICENSES_ENV}={accepted_licenses}"])
    if docker_cfg.get("cpus"):
        cmd.extend(["--cpus", str(docker_cfg["cpus"])])
    if docker_cfg.get("runtime"):
        cmd.extend(["--runtime", str(docker_cfg["runtime"])])
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend(_gpu_flags(docker_cfg.get("gpus", "all")))
    cmd.append(str(image))
    cmd.extend(command)
    return cmd


def run_embodied_via_docker(
    config: Mapping[str, Any],
    *,
    shard_id: int | None = None,
    num_shards: int | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
    pull: bool = False,
) -> int:
    """Execute an embodied eval inside Docker."""
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("'docker' executable not found")
    _docker_available(docker)
    docker_cfg = dict(config.get("docker") or {})
    image = resolve_docker_image(docker_cfg)
    platform = resolve_docker_platform(docker_cfg)
    source_image = docker_cfg.get("source_image")
    _ensure_image(
        docker,
        image,
        source_image=str(source_image) if source_image else None,
        platform=platform,
        pull=pull,
    )

    output_dir = Path(config.get("output_dir", "./results")).resolve()
    docker_config_path = write_docker_config(config, output_dir)
    try:
        cmd = build_docker_run_command(
            config,
            docker_config_path=docker_config_path,
            output_dir=output_dir,
            shard_id=shard_id,
            num_shards=num_shards,
            eval_id=eval_id,
            no_save=no_save,
        )
        return subprocess.call(cmd)
    finally:
        docker_config_path.unlink(missing_ok=True)


__all__ = [
    "build_docker_run_command",
    "inside_docker",
    "run_embodied_via_docker",
    "write_docker_config",
]
