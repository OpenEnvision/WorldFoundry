from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from worldfoundry.evaluation.tasks.embodied.simulators.calvin.benchmark import (
    CALVINBenchmark,
    _configure_calvin_import_path,
)
from worldfoundry.evaluation.tasks.embodied.simulators.robotwin.benchmark import (
    RoboTwinBenchmark,
    _configured_robotwin_root,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_prepare_module():
    path = REPO_ROOT / "scripts" / "embodied" / "prepare_official_assets.py"
    spec = importlib.util.spec_from_file_location("_test_prepare_official_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_assets = _load_prepare_module()


class EmbodiedInfraRound6Tests(unittest.TestCase):
    def test_pinned_git_repo_fetches_revision_before_detached_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = prepare_assets.PrepareItem(
                "benchmark",
                "git_repo",
                "demo",
                "official_repo",
                root / "repo",
                source="https://github.com/example/demo",
                revision="a" * 40,
            )
            args = Namespace(skip_existing=False, plan_only=False, timeout_seconds=30)
            commands: list[list[str]] = []

            def fake_run(command, *, env, log_path, timeout):
                del env, log_path, timeout
                commands.append(command)
                if "clone" in command:
                    (item.local_path / ".git").mkdir(parents=True)
                return 0, 0.01

            with (
                mock.patch.object(prepare_assets, "tool_path", return_value="/usr/bin/git"),
                mock.patch.object(prepare_assets, "run_command", side_effect=fake_run),
                mock.patch.object(prepare_assets, "git_head_revision", return_value="a" * 40),
            ):
                row = prepare_assets.prepare_git_item(item, args, {}, root / "logs")

        self.assertEqual(row["status"], "ready")
        self.assertEqual(commands[0][1:3], ["clone", "--depth"])
        self.assertEqual(
            commands[1],
            ["/usr/bin/git", "-C", str(item.local_path), "fetch", "--depth", "1", "origin", "a" * 40],
        )
        self.assertEqual(
            commands[2],
            ["/usr/bin/git", "-C", str(item.local_path), "checkout", "--detach", "FETCH_HEAD"],
        )

    def test_catalog_evidenced_repo_revisions_cover_shared_active_repos(self) -> None:
        revisions = prepare_assets.template_repo_revisions(prepare_assets.load_template_benchmark_assets())
        expected = {
            "https://github.com/Lifelong-Robot-Learning/LIBERO": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
            "https://github.com/simpler-env/SimplerEnv": "06accaca93535902d408da4855f21cece12bceb7",
            "https://github.com/robocasa/robocasa": "56e355ccc64389dfc1b8a61a33b9127b975ba681",
            "https://github.com/mees/calvin": "fa03f01f19c65920e18cf37398a9ce859274af76",
            "https://github.com/haosulab/ManiSkill": "a4a4f9272ad64b1564035874b605ceb687b63ed8",
            "https://github.com/stepjam/RLBench": "02720bba4c73fe02eb75df946b8791b806028a9d",
            "https://github.com/RoboTwin-Platform/RoboTwin": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4",
        }
        self.assertEqual({url: revisions[url] for url in expected}, expected)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = prepare_assets.target_env(root / "data", root / "ckpt", root / "hfd", root / "datasets")
            items = prepare_assets.discover_benchmark_items(env, {"libero-plus", "maniskill2", "robotwin"})
        by_owner = {item.owner_id: item for item in items if item.kind == "git_repo"}
        self.assertEqual(by_owner["libero-plus"].revision, expected["https://github.com/Lifelong-Robot-Learning/LIBERO"])
        self.assertEqual(by_owner["maniskill2"].revision, expected["https://github.com/haosulab/ManiSkill"])
        self.assertEqual(by_owner["robotwin"].revision, expected["https://github.com/RoboTwin-Platform/RoboTwin"])

    def test_pinned_git_repo_reports_revision_fetch_timeout_without_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = prepare_assets.PrepareItem(
                "benchmark",
                "git_repo",
                "demo",
                "official_repo",
                root / "repo",
                source="https://github.com/example/demo",
                revision="b" * 40,
            )
            args = Namespace(skip_existing=False, plan_only=False, timeout_seconds=7)
            commands: list[list[str]] = []

            def fake_run(command, *, env, log_path, timeout):
                del env, log_path, timeout
                commands.append(command)
                if "clone" in command:
                    (item.local_path / ".git").mkdir(parents=True)
                    return 0, 0.01
                raise prepare_assets.subprocess.TimeoutExpired(command, 7)

            with (
                mock.patch.object(prepare_assets, "tool_path", return_value="/usr/bin/git"),
                mock.patch.object(prepare_assets, "run_command", side_effect=fake_run),
            ):
                row = prepare_assets.prepare_git_item(item, args, {}, root / "logs")

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["reason"], "revision fetch timeout after 7s")
        self.assertEqual(len(commands), 2)
        self.assertIn("fetch", commands[-1])

    def test_robotwin_root_env_is_used_and_cwd_is_restored_on_init_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            original_cwd = Path.cwd()
            with (
                mock.patch.dict(os.environ, {"WORLDFOUNDRY_ROBOTWIN_ROOT": str(root)}),
                mock.patch.object(sys, "path", list(sys.path)),
            ):
                benchmark = RoboTwinBenchmark("grab_roller", skip_expert_check=True)
                with self.assertRaises(FileNotFoundError):
                    benchmark._init_robotwin()
                self.assertEqual(_configured_robotwin_root(), str(root))
                self.assertEqual(Path.cwd(), original_cwd)

    def test_calvin_roots_are_environment_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "WORLDFOUNDRY_CALVIN_ROOT": str(root / "source"),
                        "WORLDFOUNDRY_CALVIN_DATASET_ROOT": str(root / "dataset"),
                    },
                ),
                mock.patch.object(sys, "path", list(sys.path)),
            ):
                benchmark = CALVINBenchmark()
                configured_root = _configure_calvin_import_path()

            self.assertEqual(benchmark.dataset_path, str(root / "dataset" / "validation"))
            self.assertEqual(configured_root, str((root / "source").resolve()))


if __name__ == "__main__":
    unittest.main()
