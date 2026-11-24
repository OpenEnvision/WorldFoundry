"""Teacher-forced W6 AnyFlow v1.5 with H3's native data-ward velocity."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .layout import build_stage1_layout
from .stage0p5 import H3Stage0p5Core


class H3Stage1Core(H3Stage0p5Core):
    """Evaluate detached targets before the trainable AnyFlow prediction."""

    def __init__(
        self,
        model: Any,
        silence_latents: Any,
        device: Any,
        train: Any,
        validation_silence: Any = None,
    ) -> None:
        super().__init__(model, silence_latents, device)
        self.train = train
        self.validation_silence = validation_silence
        self._masks: OrderedDict[tuple[int, int, int], Any] = OrderedDict()

    def layout(self, tags: Any) -> Any:
        key = tuple(tags.detach().cpu().tolist())
        if key not in self._layouts:
            self._layouts[key] = build_stage1_layout(
                key,
                45,
                48,
                84,
                self.silence.shape[-1],
            ).to(self.device)
            while len(self._layouts) > 4:
                self._layouts.pop(next(iter(self._layouts)))
        return self._layouts[key]

    def _camera(self, batch: Any, layout: Any) -> tuple[Any, Any, Any]:
        import torch

        views = batch.camera_viewmats.to(self.device, torch.float32).unsqueeze(0)
        intrinsics = batch.camera_K.to(self.device, torch.float32).unsqueeze(0)
        if views.shape[1] == 47:
            return views[:, :45], intrinsics[:, :45], layout.camera_frame_ids
        if views.shape[1] != 47 * 1008:
            raise ValueError("Stage1 camera must describe the complete 47-latent artifact")
        views, intrinsics = views[:, : 45 * 1008], intrinsics[:, : 45 * 1008]
        return (
            torch.cat((views[:, :1008], views, views), dim=1),
            torch.cat((intrinsics[:, :1008], intrinsics, intrinsics), dim=1),
            None,
        )

    @staticmethod
    def _sp_enabled() -> bool:
        from .distributed import is_sequence_parallel_enabled

        return is_sequence_parallel_enabled()

    def generate(self, batch: Any, *, noise_seed: int, num_inference_steps: int) -> Any:
        from .stage1_sampling import generate_stage1

        if num_inference_steps != 4 or self.validation_silence is None:
            raise ValueError("H3 Stage1 validation needs NFE4 and the encoded 170-frame silence")
        return generate_stage1(self.model, batch, self.validation_silence, device=self.device, seed=noise_seed)
