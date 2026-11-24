import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging

import vbench2.hack_registry  # noqa: F401 - Patches mmengine's Registry so duplicate registrations are logged instead of raising.
from vbench2.utils import load_dimension_info

from worldfoundry.base_models.perception_core.human_anatomy.vit_detector import compute_abnormality

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_human_anatomy(json_dir, device, submodules_dict, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension="human_anatomy", lang="en")
    all_results, video_results = compute_abnormality(video_list, device, submodules_dict, **kwargs)
    all_results = sum([x["video_results"] for x in video_results]) / len(video_results)
    return all_results, video_results
