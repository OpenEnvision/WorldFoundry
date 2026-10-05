from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
cli_main = importlib.import_module("worldfoundry.cli.main")


def _load_docker_runner():
    path = REPO_ROOT / "worldfoundry/evaluation/tasks/embodied/docker_runner.py"
    spec = importlib.util.spec_from_file_location("_worldfoundry_embodied_docker_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    cli_main._accept_embodied_licenses(args.accept_license)

    assert args.func is cli_main._handle_embodied_run
    assert os.environ["WORLDFOUNDRY_ACCEPTED_LICENSES"] == (
        "existing-license,behavior-dataset-tos"
    )


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


def test_dlc_workers_derive_shards_and_rank_zero_merges(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts/embodied/run_dlc_embodied_eval.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    merge_log = tmp_path / "merge.log"
    fake_python = bin_dir / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-m\" ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"${WF_DLC_TEST_LOG}\"\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = bin_dir / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_nvidia_smi.chmod(0o755)
    config = tmp_path / "eval.yaml"
    config.write_text("id: dlc-test\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    base_env = os.environ.copy()
    for name in (
        "WF_EMBODIED_SHARD_ID",
        "WF_EMBODIED_NUM_SHARDS",
        "WF_EMBODIED_CONDA_ENV",
    ):
        base_env.pop(name, None)
    base_env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{base_env.get('PATH', '')}",
            "WORLDFOUNDRY_REPO_ROOT": str(REPO_ROOT),
            "WF_EMBODIED_PYTHON": str(fake_python),
            "WF_EMBODIED_EVAL_ID": "eval-test",
            "WF_EMBODIED_MERGE_TIMEOUT": "5",
            "WF_EMBODIED_MERGE_POLL_SECONDS": "0.05",
            "WF_DLC_TEST_LOG": str(merge_log),
            "WORLD_SIZE": "2",
        }
    )

    processes = []
    for rank in (0, 1):
        rank_env = dict(base_env, RANK=str(rank))
        processes.append(
            subprocess.Popen(
                ["/bin/bash", str(script), str(config), str(output_dir)],
                env=rank_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    outputs = []
    try:
        for process in processes:
            outputs.append(process.communicate(timeout=10))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()

    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert "WF_EMBODIED_SHARD_ID=0" in outputs[0][0]
    assert "WF_EMBODIED_SHARD_ID=1" in outputs[1][0]
    status_dir = output_dir / ".dlc-status-eval-test"
    assert (status_dir / "rank0.status").read_text().strip() == "0"
    assert (status_dir / "rank1.status").read_text().strip() == "0"
    merge_command = merge_log.read_text(encoding="utf-8")
    assert merge_command.count("embodied merge") == 1
    assert f"--output-dir {output_dir}" in merge_command
    assert "--eval-id eval-test" in merge_command


def test_dlc_multiworker_configuration_fails_without_rank_environment(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts/embodied/run_dlc_embodied_eval.sh"
    config = tmp_path / "eval.yaml"
    config.write_text("id: dlc-test\n", encoding="utf-8")
    env = os.environ.copy()
    for name in ("RANK", "WORLD_SIZE", "WF_EMBODIED_SHARD_ID", "WF_EMBODIED_NUM_SHARDS"):
        env.pop(name, None)
    env["WF_EMBODIED_EXPECTED_NUM_SHARDS"] = "2"

    result = subprocess.run(
        ["/bin/bash", str(script), str(config), str(tmp_path / "output")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "configured 2 workers but did not provide RANK/WORLD_SIZE" in result.stderr
