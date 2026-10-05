from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
from worldfoundry.evaluation.tasks.embodied.docker_runner import build_docker_run_command, write_docker_config
from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests
from worldfoundry.evaluation.tasks.embodied.model_server.protocol import Message, MessageType, pack_message, unpack_message
from worldfoundry.evaluation.tasks.embodied.merge_results import merge_embodied_results
from worldfoundry.evaluation.tasks.embodied.orchestrator import run_embodied_eval_sync
from worldfoundry.evaluation.tasks.embodied.policy_adapter import normalize_action_payload
from worldfoundry.evaluation.tasks.embodied.rollout_runner import EmbodiedClosedLoopRunner
from worldfoundry.evaluation.tasks.embodied.simulators import specs


@dataclass
class _StepResult:
    obs: dict


class _FakeBenchmark:
    constructed_kwargs: dict | None = None
    reset_task: dict | None = None
    seen_actions: list[dict] = []

    def __init__(self, *, suite: str = "fake_suite", seed: int = 7) -> None:
        self.suite = suite
        self.seed = seed
        self.steps = 0
        type(self).constructed_kwargs = {"suite": suite, "seed": seed}
        type(self).seen_actions = []

    def get_tasks(self):
        return (
            {"name": "pick up the cube", "suite": self.suite, "task_id": 0, "task_obj": object()},
            {"name": "open the drawer", "suite": self.suite, "task_id": 1, "task_obj": object()},
        )

    def reset(self, task):
        type(self).reset_task = dict(task)
        self.steps = 0
        return {"pixels": 1}

    def make_obs(self, raw_obs, task):
        return {"images": {"agentview": [[1]]}, "task_description": task["name"], "raw": raw_obs}

    def step(self, action):
        type(self).seen_actions.append(action)
        self.steps += 1
        return _StepResult(obs={"pixels": self.steps})

    def check_done(self, step_result):
        return self.steps >= 1

    def get_step_result(self, step_result):
        return {"success": True, "steps": self.steps}

    def get_metadata(self):
        return {"max_steps": 3}

    def get_action_spec(self):
        return {"position": specs.POSITION_DELTA, "rotation": specs.ROTATION_AA, "gripper": specs.GRIPPER_CLOSE_POS}

    def get_observation_spec(self):
        return {"agentview": specs.IMAGE_RGB, "language": specs.LANGUAGE}

    def cleanup(self):
        return None


@contextmanager
def _fake_simulator_registry():
    import worldfoundry.evaluation.tasks.embodied.rollout_runner as rollout_runner
    import worldfoundry.evaluation.tasks.embodied.simulators.registry as registry

    entry = registry.SimulatorEntry("fake", "tests.fake", "FakeBenchmark")
    with (
        mock.patch.object(rollout_runner, "get_simulator_entry", lambda benchmark_id: entry if benchmark_id == "fake" else None),
        mock.patch.object(rollout_runner, "resolve_simulator_class", lambda benchmark_id: _FakeBenchmark),
    ):
        yield


class EmbodiedWiringTests(unittest.TestCase):
    def test_normalize_structured_action_payload(self) -> None:
        self.assertEqual(
            normalize_action_payload({"position": [1, 2, 3], "rotation": [0, 0, 1], "gripper": -1}),
            {"actions": [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, -1.0]},
        )

    def test_rollout_runner_uses_adapter_policy_and_resolves_task(self) -> None:
        class Policy:
            def predict(self, obs, instruction):
                return {"position": [0.1, 0.2, 0.3], "rotation": [0, 0, 0], "gripper": 1}

            def get_action_spec(self):
                return {"position": specs.POSITION_DELTA, "rotation": specs.ROTATION_AA, "gripper": specs.GRIPPER_CLOSE_POS}

            def get_observation_spec(self):
                return {"agentview": specs.IMAGE_RGB, "language": specs.LANGUAGE}

            def cleanup(self):
                return None

        with _fake_simulator_registry():
            runner = EmbodiedClosedLoopRunner(
                "test-policy",
                "fake",
                policy_runner=Policy(),
                benchmark_kwargs={"suite": "fake_suite"},
            )
            result = runner.generate(
                [
                    GenerationRequest(
                        sample_id="sample-1",
                        task_name="fake_suite/task_001",
                        inputs={"task_id": 1, "episode_idx": 2, "seed": 123},
                    )
                ]
            )[0]

        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata["policy_source"], "provided_policy")
        self.assertIs(result.metadata["official_runtime_executed"], True)
        self.assertEqual(_FakeBenchmark.constructed_kwargs, {"suite": "fake_suite", "seed": 7})
        self.assertEqual(_FakeBenchmark.reset_task["task_id"], 1)
        self.assertEqual(_FakeBenchmark.reset_task["episode_idx"], 2)
        self.assertEqual(_FakeBenchmark.seen_actions, [{"actions": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]}])

    def test_protocol_roundtrip_msgpack(self) -> None:
        original = Message(MessageType.ACTION, {"actions": [0.0, 1.0]}, seq=42)
        restored = unpack_message(pack_message(original))
        self.assertEqual(restored.type, MessageType.ACTION)
        self.assertEqual(restored.payload, {"actions": [0.0, 1.0]})
        self.assertEqual(restored.seq, 42)

    def test_docker_command_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = {
                "output_dir": str(tmp_path / "out"),
                "docker": {"image": "example/worldfoundry:latest", "gpus": "all"},
            }
            docker_config = write_docker_config(config, tmp_path / "out")
            cmd = build_docker_run_command(config, docker_config_path=docker_config, output_dir=tmp_path / "out")
        self.assertIn("example/worldfoundry:latest", cmd)
        self.assertIn("python", cmd)
        self.assertIn("worldfoundry.cli.main", cmd)
        self.assertIn("embodied", cmd)
        self.assertIn("--no-docker", cmd)
        self.assertIn(f"{tmp_path / 'out'}:/workspace/results", cmd)

    def test_runtime_profile_supplies_worldfoundry_docker_image(self) -> None:
        config = load_canonical_embodied_config(
            Path("worldfoundry/data/benchmarks/eval_configs/embodied/libero/spatial.yaml")
        )
        self.assertEqual(config["docker"]["image"], "ghcr.io/openenvision/worldfoundry-embodied-libero:latest")
        self.assertEqual(
            config["docker"]["source_image"],
            "ghcr.io/allenai/vla-evaluation-harness/"
            "libero@sha256:d0c45bc5a3720d569180e6b8dd92510da895f16c3cc509ccc76e4b4ffbb9e0f0",
        )
        self.assertEqual(config["docker"]["platform"], "linux/amd64")

    def test_docker_command_uses_profile_python_env_entrypoint_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = {
                "output_dir": str(tmp_path / "out"),
                "docker": {
                    "image": "ghcr.io/openenvision/worldfoundry-embodied-libero:latest",
                    "python_env": "libero",
                    "gpus": "all",
                },
            }
            docker_config = write_docker_config(config, tmp_path / "out")
            cmd = build_docker_run_command(config, docker_config_path=docker_config, output_dir=tmp_path / "out")
        self.assertIn("--entrypoint", cmd)
        self.assertIn("", cmd)
        self.assertIn("conda", cmd)
        self.assertIn("libero", cmd)
        self.assertIn("worldfoundry.cli.main", cmd)

    def test_materializer_uses_native_task_name(self) -> None:
        requests = materialize_embodied_rollout_requests(
            {
                "benchmark_id": "robotwin",
                "params": {"task_name": "grab_roller", "seed": 0},
                "episodes_per_task": 1,
                "max_tasks": 1,
            }
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].task_name, "grab_roller")
        self.assertEqual(requests[0].inputs["task_name"], "grab_roller")

    def test_async_orchestrator_zero_policy_no_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _fake_simulator_registry():
            result = run_embodied_eval_sync(
                {
                    "id": "fake_eval",
                    "output_dir": str(Path(tmp) / "run"),
                    "model": {"id": "zero", "parameters": {"zero_policy": True}},
                    "benchmarks": [
                        {
                            "id": "fake",
                            "benchmark_id": "fake",
                            "params": {"suite": "fake_suite"},
                            "episodes_per_task": 1,
                            "max_tasks": 1,
                        }
                    ],
                },
                no_save=True,
            )
        self.assertEqual(result.evaluate_result.sample_count, 1)
        self.assertEqual(result.evaluate_result.failed_sample_count, 0)

    def test_async_orchestrator_writes_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _fake_simulator_registry():
            result = run_embodied_eval_sync(
                {
                    "id": "fake_scorecard",
                    "output_dir": str(Path(tmp) / "run"),
                    "model": {"id": "zero", "parameters": {"zero_policy": True}},
                    "benchmarks": [
                        {
                            "id": "fake",
                            "benchmark_id": "fake",
                            "params": {"suite": "fake_suite"},
                            "episodes_per_task": 1,
                            "max_tasks": 1,
                        }
                    ],
                }
            )
            self.assertTrue(result.evaluate_result.scorecard_path.is_file())
            self.assertTrue((Path(tmp) / "run" / "results.jsonl").is_file())
        self.assertEqual(result.evaluate_result.sample_count, 1)
        self.assertEqual(result.evaluate_result.failed_sample_count, 0)

    def test_merge_sharded_outputs_writes_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard0of1"
            shard.mkdir(parents=True)
            request = GenerationRequest(sample_id="s1", task_name="fake/task_000").to_dict()
            result = {
                "sample_id": "s1",
                "model_id": "zero",
                "status": "success",
                "metadata": {
                    "vla_va_wam": {"metrics": {"task_success": 1.0, "success_rate": 1.0}},
                    "task_spec": {"suite": "fake", "request_task_name": "fake/task_000"},
                },
            }
            (shard / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")
            (shard / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
            merged = merge_embodied_results(root, config={"id": "fake_merge", "model": {"id": "zero"}})
            self.assertTrue(merged.scorecard_path.is_file())
            self.assertEqual(merged.sample_count, 1)
            self.assertEqual(merged.failed_sample_count, 0)


if __name__ == "__main__":
    unittest.main()
