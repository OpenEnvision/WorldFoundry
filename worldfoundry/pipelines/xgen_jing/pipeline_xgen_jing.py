"""Public JING-Flash-v1 pipeline with the shared WorldFoundry invocation contract."""
from pathlib import Path

from worldfoundry.pipelines.pipeline_utils import PipelineABC


class XGENJINGPipeline(PipelineABC):
    MODEL_ID = "xgen-jing"
    MODEL_PATH_OPTION = "checkpoint_path"

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", model_id=None, **kwargs):
        from worldfoundry.core.io.paths import hfd_root_path
        from worldfoundry.synthesis.visual_generation.xgen_jing.runtime import JINGRuntime

        options = cls._runtime_options(model_path, required_components, kwargs)
        cls._strip_framework_loading_options(options)
        for name in ("runtime_profile", "variant_id", "pipeline_binding"):
            options.pop(name, None)
        checkpoint = options.pop("checkpoint_path", None) or hfd_root_path("XGENlabs--XGEN-JING")
        base = options.pop("base_model_path", None)
        if base is None:
            candidates = (hfd_root_path("MiniMaxAI--MiniMax-H3"), hfd_root_path("minimax-h3"))
            base = next((candidate for candidate in candidates if Path(candidate).is_dir()), candidates[0])
        cpu_offload = options.pop("cpu_offload", True)
        if options:
            raise TypeError(f"Unsupported JING loading options: {sorted(options)}")
        if model_id not in (None, "xgen-jing", "xgen-jing-flash-v1"):
            raise ValueError("Only the released JING-Flash-v1 bidirectional checkpoint is supported")
        pipeline = cls(model_id=model_id or cls.MODEL_ID, device=device)
        pipeline.runtime = JINGRuntime(checkpoint, base, device=device, cpu_offload=cpu_offload)
        return pipeline

    def __call__(self, prompt="", images=None, video=None, output_path=None, return_dict=False, **kwargs):
        kwargs.pop("operator_kwargs", None)
        if video is not None:
            raise ValueError("JING-Flash accepts ordered reference images; causal history is not released")
        if output_path is None:
            raise ValueError("output_path is required")
        references = kwargs.pop("reference_images", None)
        if references is not None and images is not None:
            raise ValueError("Use images or reference_images, not both")
        references = images if references is None else references
        if references is None:
            references = []
        elif not isinstance(references, (list, tuple)):
            references = [references]
        result = self.runtime.generate(prompt=prompt, reference_images=references, output_path=output_path, **kwargs)
        return result if return_dict else result["artifact_path"]
