from typing import List

from worldfoundry.base_models.perception_core.video_text.vqa_score.constants import HF_CACHE_DIR

from ..score import Score
from .models.itmscore_models import get_itmscore_model, list_all_itmscore_models


class ITMScore(Score):
    def prepare_scoremodel(self, model="blip2-itm", device="cuda", cache_dir=HF_CACHE_DIR):
        return get_itmscore_model(model, device=device, cache_dir=cache_dir)

    def list_all_models(self) -> List[str]:
        return list_all_itmscore_models()
