import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging

import vbench2.hack_registry  # noqa: F401 - Patches mmengine's Registry so duplicate registrations are logged instead of raising.
from vbench2.utils import load_dimension_info

from worldfoundry.base_models.llm_mllm_core.mllm.instance_anomaly_detector import compute_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_instance_preservation(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="instance_preservation", lang="en")
    all_results, video_results = compute_anomaly(prompt_dict_ls, device, submodules_dict)
    return all_results, video_results
