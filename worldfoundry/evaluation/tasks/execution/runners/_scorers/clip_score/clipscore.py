from typing import List

from worldfoundry.base_models.perception_core.video_text.vqa_score.constants import HF_CACHE_DIR

from ..score import Score
from .models.clipscore_models import get_clipscore_model, list_all_clipscore_models


class CLIPScore(Score):
    def prepare_scoremodel(self, model="openai:ViT-L/14", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
        return get_clipscore_model(model, device=device, cache_dir=cache_dir, **kwargs)

    def list_all_models(self) -> List[str]:
        return list_all_clipscore_models()
