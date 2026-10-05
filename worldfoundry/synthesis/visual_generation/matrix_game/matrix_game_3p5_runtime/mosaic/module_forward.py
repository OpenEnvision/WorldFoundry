import torch

from .inference import run_mosaic_segment_inference


class _ForwardMixin:
    @torch.no_grad()
    def inference_step(self, **kwargs):
        kwargs.setdefault("enable_mosaic", self.enable_mosaic)
        kwargs.setdefault("only_prope", self.only_prope)
        kwargs.setdefault("mosaic_view_change_prope", self.mosaic_view_change_prope)
        kwargs.setdefault("allow_empty_prompt", self.allow_no_prompt)
        return run_mosaic_segment_inference(pipeline=self.pipeline, **kwargs)
