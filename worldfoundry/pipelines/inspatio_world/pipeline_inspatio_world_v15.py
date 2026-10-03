"""Unified single-image, four-image and video route for InSpatio World 1.5."""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_v15_synthesis import (
    InspatioWorldV15Synthesis,
)
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_runtime import DEFAULT_CHECKPOINT_REPO

from ..pipeline_utils import PipelineABC


class InspatioWorldV15Pipeline(PipelineABC):
    MODEL_ID = "inspatio-world-1p5"

    @classmethod
    def from_pretrained(cls, model_path=DEFAULT_CHECKPOINT_REPO, required_components=None,
                        device="cuda", weight_dtype=None, **kwargs):
        options = {**(required_components or {}), **kwargs}
        cls._strip_framework_loading_options(options)
        synthesis = InspatioWorldV15Synthesis.from_pretrained(
            model_path or DEFAULT_CHECKPOINT_REPO, device=device, weight_dtype=weight_dtype, **options,
        )
        return cls(synthesis_model=synthesis, device=device)

    def __call__(self, images=None, videos=None, prompt="", scene_dir=None, traj_txt_path=None,
                 output_dir=None, output_path=None, seed=0, return_dict=True, **kwargs):
        if self.synthesis_model is None:
            raise RuntimeError("Load InSpatio World 1.5 with from_pretrained() first")
        result = self.synthesis_model.predict(
            images=images, videos=videos, prompt=prompt, scene_dir=scene_dir,
            traj_txt_path=traj_txt_path, output_dir=output_dir, output_path=output_path,
            seed=seed, **kwargs,
        )
        return result if return_dict else result["video_path"]

    def run_pipeline_invocation(self, invocation):
        options = dict(getattr(invocation, "pipeline_kwargs", {}) or {})
        images = options.pop("images", getattr(invocation, "image", None))
        videos = options.pop("videos", getattr(invocation, "video", None))
        if options.get("scene_dir") is not None:
            images = videos = None
        options.pop("return_dict", None)
        options.pop("output_path", None)
        prompt = options.pop("prompt", getattr(invocation, "prompt", "") or "")
        result = self(images=images, videos=videos, prompt=prompt,
                      output_path=str(invocation.output_path), return_dict=True, **options)
        return {**result, "status": "succeeded"}
