"""WorldFoundry public pipeline for HyperFlow video and audio generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.hyperflow.runtime import HyperFlowRuntime


class HyperFlowPipeline(PipelineABC):
    MODEL_ID = "hyperflow"
    MODEL_PATH_OPTION = "model_path"

    def __init__(self, runtime: HyperFlowRuntime):
        super().__init__(model_id=self.MODEL_ID, device=runtime.device)
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", model_id=None, **kwargs):
        options = cls._runtime_options(model_path, required_components, kwargs)
        cls._strip_framework_loading_options(options)
        for name in ("runtime_profile", "pipeline_binding", "variant_id"):
            options.pop(name, None)
        env_dir = options.pop("python_env_dir", options.pop("conda_dir", None))
        if env_dir and "python_executable" not in options:
            options["python_executable"] = Path(env_dir).expanduser() / "bin" / "python"
        return cls(HyperFlowRuntime(device=device, **options))

    def preflight(self, workflow="t2va"):
        return self.runtime.preflight(workflow)

    def __call__(
        self,
        prompt=None,
        images=None,
        video=None,
        interactions=None,
        output_path=None,
        output_dir=None,
        return_dict=False,
        execute=True,
        timeout_seconds=7200,
        operator_kwargs=None,
        **kwargs: Any,
    ):
        if not execute:
            raise ValueError("HyperFlow requires execute=True; use runtime.build_plan for inspection.")
        if video is not None or interactions is not None:
            raise ValueError("Use ordered references for media conditioning; HyperFlow does not accept interactions.")
        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        request["prompt"] = prompt if prompt is not None else request.get("prompt")
        if images is not None:
            request["images"] = images
        destination = output_path or Path(output_dir or ".") / "hyperflow.mp4"
        plan = self.runtime.build_plan(output_path=destination, **request)
        result = self.runtime.run(plan, timeout_seconds=timeout_seconds)
        return result if return_dict else result["video"]
