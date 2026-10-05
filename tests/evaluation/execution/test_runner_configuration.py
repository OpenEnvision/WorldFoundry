"""Pipeline configuration must follow directly supplied runners into generation reuse."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from worldfoundry.evaluation.api import GenerationRequest, WorldModelConfig
from worldfoundry.evaluation.models.pipelines.lifecycle import PipelineRuntimeProfile
from worldfoundry.evaluation.models.pipelines.loading import PipelineRunnerSpec
from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import run_evaluate
from worldfoundry.evaluation.utils import model_runner_fingerprint


class RecordingPipeline:
    def __init__(self):
        self.calls = []

    def run_pipeline_invocation(self, invocation):
        scale = invocation.pipeline_kwargs["guidance_scale"]
        self.calls.append(scale)
        invocation.output_path.write_text(str(scale))
        return {"artifact_path": str(invocation.output_path)}


class RunnerConfigurationTests(unittest.TestCase):
    def test_changed_defaults_invalidate_resume_and_shared_cache(self):
        for strategy in ("resume", "cache"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pipeline = RecordingPipeline()
                runner = WorldFoundryPipelineRunner(
                    "review", pipeline, pipeline_target="review:Pipeline", output_dir=root / "artifacts",
                    generation_defaults={"guidance_scale": 1},
                )
                profile = PipelineRuntimeProfile("sample.txt", "text", "review")
                with patch.object(runner, "_runtime_profile", return_value=profile), patch(
                    "worldfoundry.core.worldfoundry_inference_context", side_effect=nullcontext,
                ):
                    for index, scale in enumerate((1, 1, 9)):
                        runner.generation_defaults["guidance_scale"] = scale
                        output = root / ("run" if strategy == "resume" else f"run-{index}")
                        run_evaluate(
                            mode="model", runner=runner, cleanup_runner=False, output_dir=output,
                            requests=[GenerationRequest(sample_id="one", task_name="review")],
                            metrics=["artifact_count"], resume=strategy == "resume",
                            generation_cache_dir=root / "cache",
                            generation_cache_mode="read-write" if strategy == "cache" else "off",
                        )
                        row = json.loads((output / "results.jsonl").read_text())
                        artifact = next(iter(row["artifacts"].values()))
                        self.assertEqual(Path(artifact["uri"]).read_text(), str(scale))
                self.assertEqual(pipeline.calls, [1, 9])

    def test_from_config_retains_weights_and_live_runtime_parameters(self):
        identities = []
        for weights in ("weights-a", "weights-b"):
            config = WorldModelConfig("review", "worldfoundry.pipeline", parameters={"model_path": weights})
            spec = PipelineRunnerSpec("review", "review:Pipeline", "review", weights, weights, None, "cpu")
            with patch("worldfoundry.evaluation.models.runners.pipeline.load_pipeline_from_config",
                       return_value=(spec, RecordingPipeline())):
                runner = WorldFoundryPipelineRunner.from_config(config)
            identities.append(model_runner_fingerprint(runner))
        self.assertNotEqual(*identities)
        for attribute in ("pipeline_target", "runtime_profile_id"):
            before = model_runner_fingerprint(runner)
            setattr(runner, attribute, "changed")
            self.assertNotEqual(before, model_runner_fingerprint(runner))


if __name__ == "__main__":
    unittest.main()
