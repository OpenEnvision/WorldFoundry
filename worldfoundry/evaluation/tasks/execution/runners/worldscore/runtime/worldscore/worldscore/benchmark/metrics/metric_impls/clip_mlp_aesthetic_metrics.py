"""WorldScore's CLIP/LAION aesthetic metric using the shared MLP head."""

from collections.abc import Sequence

import torch
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.evaluation.tasks.execution.runners.worldscore.runtime.worldscore.worldscore.benchmark.metrics.base_metrics import (
    BaseMetric,
)
from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import load_laion_aesthetic_mlp_head


def normalized(a, axis: int = -1, order: int = 2):
    """Retain the official NumPy feature normalization, including zero rows."""
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class CLIPMLPAestheticScoreMetric(BaseMetric):
    """Score CLIP ViT-L/14 features with the improved-aesthetic-predictor MLP."""

    def __init__(self) -> None:
        super().__init__()
        self._model = load_laion_aesthetic_mlp_head(
            "worldscore/benchmark/metrics/checkpoints/sac+logos+ava1-l14-linearMSE.pth"
        ).to(self._device)
        self._model2, self._preprocess = clip.load("ViT-L/14", device=self._device)

    def _compute_scores(self, rendered_images: Sequence[str | Image.Image]) -> float:
        scores = []
        for image in rendered_images:
            if isinstance(image, str):
                image = Image.open(image)
            tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model2.encode_image(tensor)
                features = normalized(image_features.cpu().detach().numpy())
                prediction = self._model(torch.from_numpy(features).type(torch.FloatTensor).to(self._device))
            scores.append(prediction.detach()[0].item())
        return sum(scores) / len(scores)
