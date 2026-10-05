"""BiWM's public camera-controlled Wan inference pipeline."""
from __future__ import annotations

from worldfoundry.pipelines.pipeline_utils import PipelineABC


class BiWMPipeline(PipelineABC):
    MODEL_ID = "biwm"
    MODEL_PATH_OPTION = "generator_ckpt"

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None,
                        device="cuda", model_id=None, **kwargs):
        from worldfoundry.synthesis.visual_generation.biwm.runtime import checkpoint_files
        options = cls._runtime_options(model_path, required_components, kwargs)
        cls._strip_framework_loading_options(options)
        for name in ("runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(name, None)
        generator = options.pop("generator_ckpt", None)
        base = options.pop("wan_base", None)
        stage = options.pop("stage", 2)
        if not generator or not base:
            raise ValueError("BiWM requires generator_ckpt (trained BiWM weights) and wan_base (native Wan components)")
        if options:
            raise TypeError(f"Unsupported BiWM loading options: {sorted(options)}")
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        checkpoint, base, _, version = checkpoint_files(generator, base)
        expected = {"biwm-wan21": "2.1", "biwm-wan22": "2.2"}.get(model_id)
        if expected and version != expected:
            raise ValueError(f"{model_id} requires Wan{expected} base components")
        pipeline = cls(model_id=model_id or cls.MODEL_ID, device=device)
        pipeline.generator_ckpt, pipeline.wan_base, pipeline.stage = checkpoint, base, stage
        pipeline._runtime = None
        return pipeline

    def __call__(self, prompt="", images=None, video=None, actions=None,
                 output_path=None, return_dict=False, **kwargs):
        from worldfoundry.synthesis.visual_generation.biwm.runtime import BiWMRuntime
        kwargs.pop("operator_kwargs", None)
        if video is not None:
            raise ValueError("Use one initial image for BiWM I2V")
        if output_path is None:
            raise ValueError("output_path is required")
        if isinstance(images, (list, tuple)):
            if len(images) != 1:
                raise ValueError("BiWM I2V accepts exactly one image")
            images = images[0]
        if self.stage == 1 and images is not None:
            raise ValueError("The released stage-1 inference supports T2V only")
        common = {"action_label", "height", "width", "seed", "fps", "sigma_shift"}
        stage_options = ({"num_frames", "num_inference_steps", "guidance_scale", "negative_prompt"}
                         if self.stage == 1 else
                         {"num_chunks", "chunk_size", "max_chunks", "sink_chunks", "sigmas"})
        unknown = set(kwargs) - common - stage_options
        if unknown:
            raise TypeError(f"Unsupported BiWM stage-{self.stage} options: {sorted(unknown)}")
        if self._runtime is None:
            self._runtime = BiWMRuntime(self.generator_ckpt, self.wan_base, device=self.device)
        result = self._runtime.generate(prompt=prompt, image=images, actions=actions,
                                        output_path=output_path, stage=self.stage, **kwargs)
        result.update(status="success", model_id=self.model_id, artifact_path=result["video_path"])
        return result if return_dict else result["video_path"]
