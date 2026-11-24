# Inference-only LAPA source retained in-tree.

import numpy as np
from PIL import Image

from .sampler import DeltaSampler


class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


class LAPAInference:
    def __init__(
        self,
        image_size: int = 256,
        **kwargs,
    ) -> None:
        flags = FLAGSClass(kwargs)

        self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs["tokens_per_delta"]
        self.task_description = None

    def inference(
        self, image: np.ndarray, task_description: str | None = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        assert image.dtype == np.uint8
        image = Image.fromarray(image)
        prompts = [{"image": [image], "question": task_description}]

        latent_output = self.model(prompts)
        latent_action = latent_output[0]

        return latent_action
