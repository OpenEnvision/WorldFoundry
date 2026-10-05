"""VBench adapter for the shared in-tree AMT smoothness runtime."""

import numpy as np
from tqdm import tqdm
from vbench.utils import load_dimension_info

from worldfoundry.base_models.perception_core.frame_interpolation.amt.motion_smoothness import (
    FrameProcess,
    MotionSmoothness,
)

from worldfoundry.core.distributed.collectives.evaluation import (
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)


def motion_smoothness(motion, video_list):
    scores = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        score_per_video = motion.motion_score(video_path)
        video_results.append({"video_path": video_path, "video_results": score_per_video})
        scores.append(score_per_video)
    return float(np.mean(scores)) if scores else 0.0, video_results


def compute_motion_smoothness(json_dir, device, submodules_list, **kwargs):
    del kwargs
    motion = MotionSmoothness(submodules_list["config"], submodules_list["ckpt"], device)
    video_list, _ = load_dimension_info(json_dir, dimension="motion_smoothness", lang="en")
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = motion_smoothness(motion, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = (
            sum(float(row["video_results"]) for row in video_results) / len(video_results)
            if video_results
            else 0.0
        )
    return all_results, video_results


__all__ = [
    "FrameProcess",
    "MotionSmoothness",
    "compute_motion_smoothness",
    "motion_smoothness",
]
