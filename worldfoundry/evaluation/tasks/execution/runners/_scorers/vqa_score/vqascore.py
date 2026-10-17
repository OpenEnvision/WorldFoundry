from typing import List

from worldfoundry.base_models.perception_core.video_text.vqa_score.constants import HF_CACHE_DIR
from worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models import (
    get_vqascore_model,
    list_all_vqascore_models,
)

from ..score import Score


class VQAScore(Score):
    def prepare_scoremodel(self, model="clip-flant5-xxl", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
        return get_vqascore_model(model, device=device, cache_dir=cache_dir, **kwargs)

    def list_all_models(self) -> List[str]:
        return list_all_vqascore_models()
