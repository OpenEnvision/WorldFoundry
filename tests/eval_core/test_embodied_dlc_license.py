from __future__ import annotations

import importlib
import os
from pathlib import Path

cli_main = importlib.import_module("worldfoundry.cli.main")
cli_embodied = importlib.import_module("worldfoundry.cli.embodied")


def _load_docker_runner():
    return importlib.import_module("worldfoundry.evals.tasks.embodied.docker_runner")


def test_embodied_accept_license_flags_merge_with_existing_environment(monkeypatch) -> None:
    args = cli_main._build_parser().parse_args(
        [
            "embodied",
            "run",
            "--config",
            "unused.yaml",
            "--accept-license",
            "existing-license",
            "--accept-license",
            "behavior-dataset-tos",
        ]
    )
    monkeypatch.setenv("WORLDFOUNDRY_ACCEPTED_LICENSES", "existing-license")

    cli_embodied._accept_embodied_licenses(args.accept_license)

    assert args.func is cli_embodied._handle_embodied_run
    assert os.environ["WORLDFOUNDRY_ACCEPTED_LICENSES"] == ("existing-license,behavior-dataset-tos")


def test_docker_command_forwards_accepted_licenses(tmp_path: Path, monkeypatch) -> None:
    docker_runner = _load_docker_runner()
    config = {"docker": {"image": "example/worldfoundry:latest"}}
    docker_config = docker_runner.write_docker_config(config, tmp_path / "out")
    monkeypatch.setenv(
        "WORLDFOUNDRY_ACCEPTED_LICENSES",
        "behavior-dataset-tos,sim-license",
    )

    cmd = docker_runner.build_docker_run_command(
        config,
        docker_config_path=docker_config,
        output_dir=tmp_path / "out",
    )

    assert "WORLDFOUNDRY_ACCEPTED_LICENSES=behavior-dataset-tos,sim-license" in cmd


def test_docker_command_does_not_inject_an_empty_license_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    docker_runner = _load_docker_runner()
    config = {"docker": {"image": "example/worldfoundry:latest"}}
    docker_config = docker_runner.write_docker_config(config, tmp_path / "out")
    monkeypatch.delenv("WORLDFOUNDRY_ACCEPTED_LICENSES", raising=False)

    cmd = docker_runner.build_docker_run_command(
        config,
        docker_config_path=docker_config,
        output_dir=tmp_path / "out",
    )

    assert not any(item.startswith("WORLDFOUNDRY_ACCEPTED_LICENSES=") for item in cmd)
