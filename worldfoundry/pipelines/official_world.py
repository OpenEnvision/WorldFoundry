"""Shared pipeline lifecycle for official world-model inference adapters."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from worldfoundry.operators.interactive_world_operator import InteractiveWorldOperator
from worldfoundry.pipelines.pipeline_utils import PipelineABC


class OfficialWorldPipeline(PipelineABC):
    RUNTIME_CLS = None
    MODEL_ID = ""
    OPERATOR_CLS = InteractiveWorldOperator

    def __init__(self, runtime):
        self.runtime = runtime
        self.history = []
        self.operator = self.OPERATOR_CLS()

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", **kwargs):
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_dir"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        for key in (
            "model_id",
            "required_components",
            "profile_id",
            "runtime_profile",
            "pipeline_binding",
            "runner",
            "runner_target",
            "profile_path",
            "manifest_path",
            "variant_id",
            "repo_id",
            "install_profile",
            "lazy",
        ):
            options.pop(key, None)
        env_dir = options.pop("python_env_dir", None)
        if env_dir is not None:
            options.setdefault("python_executable", str(Path(env_dir) / "bin/python"))
        return cls(cls.RUNTIME_CLS(device=device, **options))

    def preflight(self):
        return self.runtime.preflight()

    def process(self, prompt=None, images=None, video=None, interactions=None, operator_kwargs=None, **kwargs):
        return self.operator.process_perception(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )

    def __call__(
        self,
        prompt=None,
        images=None,
        video=None,
        interactions=None,
        *,
        output_path=None,
        output_dir=None,
        execute=False,
        timeout_seconds=7200,
        return_dict=False,
        operator_kwargs=None,
        **kwargs,
    ):
        if not execute:
            raise RuntimeError(f"{self.MODEL_ID} requires execute=True and a prepared official inference environment.")
        request = self.process(prompt, images, video, interactions, operator_kwargs, **kwargs)
        # These routes may produce multiple samples. Keep the upstream artifact structure intact.
        requested_file = Path(output_path).expanduser().resolve() if output_path and Path(output_path).suffix else None
        root = (
            Path(
                output_dir
                or (requested_file.parent / requested_file.stem if requested_file else output_path)
                or Path("tmp") / self.MODEL_ID
            )
            .expanduser()
            .resolve()
        )
        run_dir = root / uuid.uuid4().hex
        plan = self.runtime.build_plan(request=request, output_dir=run_dir)
        result = self.runtime.run_plan(plan, timeout_seconds=timeout_seconds)
        if requested_file is not None and result["status"] == "success" and len(result["artifact_files"]) == 1:
            requested_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result["artifact_files"][0], requested_file)
            result["artifact_path"] = str(requested_file)
            result["generated_video_path"] = str(requested_file)
            result["source_artifact_files"] = result["artifact_files"]
            result["artifact_files"] = [str(requested_file)]
            Path(result["metadata_path"]).write_text(json.dumps(result, indent=2) + "\n")
        self.history.append(result)
        if result["status"] != "success" and not return_dict:
            raise RuntimeError(f"{result['error']}; see {result['stderr_path']}")
        return result if return_dict else result["artifact_files"]
