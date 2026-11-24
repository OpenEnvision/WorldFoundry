import os

from easydict import EasyDict as edict
from vbench.temporal_flickering import compute_temporal_flickering
from vbench.utils import load_dimension_info

# isort: split
from vbench2_beta_long.utils import build_filtered_info_json, reorganize_clips_results

# isort: split
from vbench2_beta_long.static_filter import static_filter

# isort: split
from worldfoundry.base_models.perception_core.optical_flow.raft import checkpoint_path as raft_checkpoint_path


def compute_long_temporal_flickering(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension="temporal_flickering", lang="en")
    base_video_path = os.path.dirname(video_list[0]).split("split_clip")[0]
    video_clips_path = os.path.join(base_video_path, "split_clip")

    output_path = os.path.join(base_video_path, "temporal_filtered_cilps")
    os.makedirs(output_path, exist_ok=True)

    input_path = video_clips_path

    if kwargs["static_filter_flag"]:
        filter_static_clips(input_path, output_path, base_video_path)
        json_dir = build_filtered_info_json(
            videos_path=output_path, output_path=output_path, name="filtered_temporal_flickering"
        )

    all_results, detailed_results = compute_temporal_flickering(json_dir, device, submodules_list)

    return reorganize_clips_results(detailed_results)


def filter_static_clips(clip_video_path, output_dir, base_video_path):
    args_new = edict(
        {
            "model": str(raft_checkpoint_path()),
            "videos_path": "",
            "base_video_path": "",
            "result_path": "./filter_results",
            "store_name": "filtered_static_video.json",
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
            "filter_scope": "temporal_flickering",
        }
    )
    args_new.videos_path = clip_video_path
    args_new.base_video_path = base_video_path
    args_new.result_path = output_dir
    static_filter(args_new)
