"""Public text/image, keyboard and prompt-switching Zing pipeline."""

import inspect
from collections.abc import Mapping

from worldfoundry.operators.interactive_world_operator import InteractiveWorldOperator
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.zing import ZingSynthesis
from worldfoundry.synthesis.visual_generation.zing.runtime import ZingRuntime


class ZingPipeline(PipelineABC):
    MODEL_ID = "zing"
    OPERATOR_CLS = InteractiveWorldOperator
    SYNTHESIS_CLS = ZingSynthesis

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", lazy=True, **kwargs):
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        # The runner attaches metadata alongside model assets.
        for name in (
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
        ):
            options.pop(name, None)
        unknown = options.keys() - inspect.signature(ZingRuntime).parameters.keys()
        if unknown:
            raise TypeError(f"Unknown Zing load options: {sorted(unknown)}")
        synthesis = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=None if isinstance(model_path, Mapping) else model_path,
            generator_overrides=options,
            device=device,
            lazy=lazy,
        )
        return cls(model_id=cls.MODEL_ID, synthesis_model=synthesis, device=device, operator=cls.OPERATOR_CLS())

    @staticmethod
    def process(prompt, images=None, **kwargs):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Zing requires a non-empty prompt.")
        return {"prompt": prompt, "images": images, **kwargs}

    def __call__(
        self, prompt, images=None, *, output_path=None, fps=None, return_dict=False, operator_kwargs=None, **kwargs
    ):
        request = self.operator.process_perception(
            prompt=prompt, images=images, operator_kwargs=operator_kwargs, **kwargs
        )
        if self.synthesis_model is None:
            raise RuntimeError("Call ZingPipeline.from_pretrained first.")
        self.process(prompt, images)
        return self.synthesis_model.predict(output_path=output_path, fps=fps, return_dict=return_dict, **request)

    def close(self):
        if self.synthesis_model is not None:
            self.synthesis_model.close()
