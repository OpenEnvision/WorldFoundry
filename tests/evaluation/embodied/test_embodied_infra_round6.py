from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from worldfoundry.evals.tasks.embodied.simulators.calvin.benchmark import CALVINBenchmark
from worldfoundry.evals.tasks.embodied.simulators.robotwin.benchmark import RoboTwinBenchmark


class EmbodiedInfraRound6Tests(unittest.TestCase):
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
                self.assertEqual(benchmark.robotwin_root, root)
                self.assertEqual(Path.cwd(), original_cwd)

    def test_calvin_roots_are_environment_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "dataset" / "validation").mkdir(parents=True)
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

            self.assertEqual(benchmark.dataset_path, str(root / "dataset" / "validation"))
            self.assertEqual(benchmark.calvin_root, root / "source")


if __name__ == "__main__":
    unittest.main()
