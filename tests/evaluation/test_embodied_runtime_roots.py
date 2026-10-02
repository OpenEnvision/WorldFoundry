from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.embodied.simulators.calvin.benchmark import CALVINBenchmark
from worldfoundry.evaluation.tasks.embodied.simulators.robocerebra.benchmark import RoboCerebraBenchmark
from worldfoundry.evaluation.tasks.embodied.simulators.robotwin import benchmark as robotwin_module


def test_calvin_runtime_roots_follow_environment_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "calvin-source"
    explicit_root = tmp_path / "explicit-source"
    env_dataset = tmp_path / "calvin-dataset"
    explicit_dataset = tmp_path / "explicit-dataset"
    monkeypatch.setenv("WORLDFOUNDRY_CALVIN_ROOT", str(env_root))
    monkeypatch.setenv("WORLDFOUNDRY_CALVIN_DATASET_ROOT", str(env_dataset))

    configured = CALVINBenchmark()
    explicit = CALVINBenchmark(dataset_path=explicit_dataset, calvin_root=explicit_root)

    assert configured.calvin_root == env_root
    assert configured.dataset_path == str(env_dataset)
    assert explicit.calvin_root == explicit_root
    assert explicit.dataset_path == str(explicit_dataset)


def test_calvin_source_checkout_exposes_agent_and_environment_packages(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.embodied.simulators.calvin import benchmark as calvin_module

    root = tmp_path / "calvin"
    agent_package = root / "calvin_models" / "calvin_agent"
    env_package = root / "calvin_env" / "calvin_env"
    agent_package.mkdir(parents=True)
    env_package.mkdir(parents=True)
    (agent_package / "__init__.py").write_text("SOURCE = 'agent'\n", encoding="utf-8")
    (env_package / "__init__.py").write_text("SOURCE = 'env'\n", encoding="utf-8")
    previous_path = list(sys.path)
    try:
        inserted = calvin_module._prepend_calvin_source_paths(root)
        assert str(root / "calvin_models") in inserted
        assert str(root / "calvin_env") in inserted
        assert importlib.import_module("calvin_agent").SOURCE == "agent"
        assert importlib.import_module("calvin_env").SOURCE == "env"
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("calvin_agent", None)
        sys.modules.pop("calvin_env", None)


def test_calvin_dataset_root_resolves_documented_task_split_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "calvin-dataset"
    validation = dataset_root / "task_D_D" / "validation"
    validation.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CALVIN_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("WORLDFOUNDRY_CALVIN_SPLIT", "validation")

    assert CALVINBenchmark().dataset_path == str(validation)

    validation.rmdir()
    assert CALVINBenchmark().dataset_path == str(dataset_root / "task_D_D")


def test_calvin_dataset_requires_explicit_path_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_CALVIN_DATASET_ROOT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_CALVIN_SPLIT", raising=False)

    with pytest.raises(ValueError, match="WORLDFOUNDRY_CALVIN_DATASET_ROOT"):
        CALVINBenchmark()


def test_robotwin_root_follows_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "RoboTwin"
    monkeypatch.setenv("WORLDFOUNDRY_ROBOTWIN_ROOT", str(root))

    benchmark = robotwin_module.RoboTwinBenchmark(task_name="example_task")

    assert benchmark.robotwin_root == root


def test_robocerebra_root_follows_environment_and_has_no_machine_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "RoboCerebra"
    monkeypatch.setenv("WORLDFOUNDRY_ROBOCEREBRA_ROOT", str(root))
    assert RoboCerebraBenchmark().robocerebra_root == root

    explicit = tmp_path / "explicit"
    assert RoboCerebraBenchmark(robocerebra_root=explicit).robocerebra_root == explicit

    monkeypatch.delenv("WORLDFOUNDRY_ROBOCEREBRA_ROOT")
    with pytest.raises(ValueError, match="WORLDFOUNDRY_ROBOCEREBRA_ROOT"):
        RoboCerebraBenchmark()


def test_robotwin_initialization_restores_cwd_after_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "RoboTwin"
    configs = root / "configs"
    robot = root / "robots" / "arm"
    (root / "task_config").mkdir(parents=True)
    configs.mkdir(parents=True)
    robot.mkdir(parents=True)
    (root / "task_config" / "demo_clean.yml").write_text(
        "embodiment: [arm]\ncamera:\n  head_camera_type: default\ndata_type:\n  pointcloud: false\n",
        encoding="utf-8",
    )
    (configs / "_embodiment_config.yml").write_text(
        f"arm:\n  file_path: {robot}\n",
        encoding="utf-8",
    )
    (configs / "_camera_config.yml").write_text(
        "default:\n  h: 128\n  w: 128\n",
        encoding="utf-8",
    )
    (robot / "config.yml").write_text("name: arm\n", encoding="utf-8")

    envs = types.ModuleType("envs")
    envs.CONFIGS_PATH = str(configs)
    monkeypatch.setitem(sys.modules, "envs", envs)

    real_import_module = importlib.import_module

    def fail_task_import(name: str, package: str | None = None):
        if name == "envs.example_task":
            raise RuntimeError("simulated import failure")
        return real_import_module(name, package)

    monkeypatch.setattr(robotwin_module.importlib, "import_module", fail_task_import)
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)
    benchmark = robotwin_module.RoboTwinBenchmark(
        task_name="example_task",
        robotwin_root=root,
    )

    with pytest.raises(RuntimeError, match="simulated import failure"):
        benchmark._init_robotwin()

    assert Path.cwd() == caller_cwd
    assert benchmark._args is None


def test_robotwin_runtime_calls_use_root_and_restore_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "RoboTwin"
    root.mkdir()
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)
    observed_cwds: list[Path] = []

    class FakeRoboTwinEnv:
        eval_success = False
        take_action_cnt = 0
        step_lim = 10

        def __init__(self) -> None:
            observed_cwds.append(Path.cwd())

        def setup_demo(self, **_kwargs) -> None:
            observed_cwds.append(Path.cwd())

        def set_instruction(self, **_kwargs) -> None:
            observed_cwds.append(Path.cwd())

        def get_obs(self):
            observed_cwds.append(Path.cwd())
            return {}

        def take_action(self, _action, **_kwargs) -> None:
            observed_cwds.append(Path.cwd())
            self.take_action_cnt += 1

        def close_env(self, **_kwargs) -> None:
            observed_cwds.append(Path.cwd())

    benchmark = robotwin_module.RoboTwinBenchmark(
        task_name="example_task",
        robotwin_root=root,
        fast_init=False,
        fast_render=False,
    )
    benchmark._args = {}
    benchmark._env_class = FakeRoboTwinEnv

    benchmark.reset({"seed": 1, "instruction": "move", "episode_idx": 0})
    assert Path.cwd() == caller_cwd
    benchmark.step({"action": [0.0] * 14})
    assert Path.cwd() == caller_cwd
    benchmark.cleanup()
    assert Path.cwd() == caller_cwd
    assert observed_cwds
    assert set(observed_cwds) == {root}
