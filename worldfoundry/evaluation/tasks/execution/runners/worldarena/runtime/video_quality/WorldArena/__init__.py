import os

from .utils import init_submodules, save_json
import importlib
from itertools import chain
from pathlib import Path
import importlib.util
import json

from worldfoundry.core.distributed.collectives.evaluation import get_rank, print0

import re
from collections import defaultdict


# Empirical bounds for normalization (higher is better unless noted)
_EMPIRICAL_BOUNDS = {
    "photometric_smoothness": {"min": 0.1257, "max": 6.7899, "invert": False},
    "motion_smoothness": {"min": 0.0, "max": 2.6413, "invert": False},
    "trajectory_accuracy": {"min": 0.0, "max": 40.8540, "invert": False},
    "flow_score": {"min": 0.0531, "max": 8.9414, "invert": False},
    "depth_accuracy": {"min": 0.2228, "max": 4.3711, "invert": True},
}


def _normalize_value(metric: str, value):
    bounds = _EMPIRICAL_BOUNDS.get(metric)
    if bounds is None:
        return value
    vmin, vmax = bounds["min"], bounds["max"]
    if vmax <= vmin:
        return value
    norm = (value - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))
    if bounds.get("invert", False):
        norm = 1.0 - norm
    return norm


def _add_normalized_scores(metric: str, results):
    """Add per-video normalized scores when results follow [avg, list_of_dict] structure."""
    if not (isinstance(results, (list, tuple)) and len(results) >= 2 and isinstance(results[1], list)):
        return results

    avg_val, video_list = results[0], results[1]
    norm_video_list = []
    for item in video_list:
        if not isinstance(item, dict) or "video_results" not in item:
            norm_video_list.append(item)
            continue
        norm_item = dict(item)
        norm_item["video_results_normalized"] = _normalize_value(metric, item["video_results"])
        norm_video_list.append(norm_item)

    # Keep the original aggregate; only enrich per-video entries.
    return [avg_val, norm_video_list]


def _to_standard_results(metric: str, results, data_base: str):
    """Convert nested dict metrics to [avg, video_list] format with video paths."""
    if not isinstance(results, dict):
        return results

    if metric not in {"trajectory_accuracy", "semantic_alignment", "action_following"}:
        return results

    video_entries = []

    for task_id, episodes in results.items():
        if not isinstance(episodes, dict):
            continue
        for episode_id, groups in episodes.items():
            # action_following stores a single float per episode; other metrics store per-gid dicts
            if not isinstance(groups, dict):
                try:
                    val = float(groups)
                except (TypeError, ValueError):
                    val = 0.0
                video_path = os.path.join(data_base, task_id, episode_id)
                video_entries.append({
                    "video_path": video_path,
                    "video_results": val,
                })
                continue

            for gid, metrics in groups.items():
                if isinstance(metrics, dict):
                    if metric == "trajectory_accuracy":
                        raw_val = metrics.get("ndtw", 0.0)
                    else:  # semantic_alignment
                        raw_val = metrics.get("CLIPScore", 0.0)
                else:
                    raw_val = metrics

                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = 0.0

                # action_following keeps average per episode already; we keep gid if present
                video_path = os.path.join(data_base, task_id, episode_id, gid, "video")
                video_entries.append({
                    "video_path": video_path,
                    "video_results": val,
                })

    if not video_entries:
        return results

    avg_val = sum(item["video_results"] for item in video_entries) / len(video_entries)
    return [avg_val, video_entries]




class WorldArenaBenchmark(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path             
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return [
            'action_following',
            'trajectory_accuracy',
            'semantic_alignment',
            'depth_accuracy',
            'aesthetic_quality',
            'background_consistency',
            'dynamic_degree',
            'flow_score',
            'photometric_smoothness',
            'motion_smoothness',
            'image_quality',
            'subject_consistency'
        ]        


    def build_full_info_json(self, data_base, data_name, dimension_list, **kwargs):

        task_names = sorted(os.listdir(data_base))

        cur_full_info_list = []
        for task_id in task_names:
            task_path = os.path.join(data_base, task_id)
            for episode_id in sorted(os.listdir(task_path)):
                if episode_id.endswith(('.png', '.json')): 
                    continue
                episode_path = os.path.join(task_path, episode_id)
                for gid in sorted(os.listdir(episode_path)):
                    gid_path = os.path.join(episode_path, gid)
                    video_path = os.path.join(gid_path, "video")

                    cur_full_info_list.append({
                        "dimension": dimension_list, 
                        "video_list": [video_path]
                    })
        
        cur_full_info_path = os.path.join(self.output_path, data_name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')

        return cur_full_info_path

    def build_full_gt_info_json(self, data_base, data_name, **kwargs):
        task_names = sorted(os.listdir(data_base))

        cur_full_info_list = []
        for task_id in task_names:
            task_path = os.path.join(data_base, task_id)
            for episode_id in sorted(os.listdir(task_path)):
                if episode_id.endswith(('.png', '.json')): 
                    continue
                episode_path = os.path.join(task_path, episode_id)

                video_path = os.path.join(episode_path, "video")

                cur_full_info_list.append({
                    "video_list": [video_path]
                })
        
        cur_full_info_path = os.path.join(self.output_path, data_name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation gt data saved to {cur_full_info_path}')

        return cur_full_info_path        

    def get_evaluator(self, model_ckpt=None, model_code=None, device=None):
        from .action_following import compute_action_following
        return lambda full_info_path, submodules, **kwargs: compute_action_following(full_info_path, submodules, **kwargs)


    def evaluate(
        self,
        data_base,
        data_name,
        dimension_list=None,
        local=False,
        gt_path=None,
        overwrite=False,
        data_base_action=None,
        gt_path_action=None,
        **kwargs,
    ):

        json_path = os.path.join(self.output_path, f"{data_name}_results.json")

        if overwrite and os.path.exists(json_path):
            print0(f"[overwrite] Removing existing results at {json_path}")
            try:
                os.remove(json_path)
            except OSError as exc:
                print0(f"[overwrite] Warning: failed to remove {json_path}: {exc}")

        if (not os.path.exists(json_path)) or overwrite:

            results_dict = {}
            
            if dimension_list is None:
                dimension_list = self.build_full_dimension_list()

            if "psnr" in dimension_list and "ssim" in dimension_list:
                dimension_list.pop(dimension_list.index("psnr"))
                dimension_list.pop(dimension_list.index("ssim"))
                dimension_list.append("psnr_ssim")

            print(dimension_list)

            submodules_dict = init_submodules(dimension_list, local=local, **kwargs)

            for dimension in dimension_list:
                
                print0(f"Evaluating: {dimension}")

                # choose dataset roots per dimension (action_following can use dedicated roots)
                if dimension == 'action_following':
                    cur_data_base = data_base_action or data_base
                    cur_gt_path = gt_path_action or gt_path
                    cur_data_name = f"{data_name}_action_following"

                    if not cur_data_base:
                        raise ValueError("action_following requires data_action_following.val_base in config.yaml")
                    if not os.path.exists(cur_data_base):
                        raise FileNotFoundError(
                            f"action_following data base not found: {cur_data_base}. "
                            "Please run preprocess_datasets_diversity with output_base matching this path."
                        )
                    if not cur_gt_path:
                        raise ValueError("action_following requires data_action_following.gt_path in config.yaml")
                    if not os.path.exists(cur_gt_path):
                        raise FileNotFoundError(
                            f"action_following gt path not found: {cur_gt_path}. "
                            "Please generate gt_dataset_action_following accordingly."
                        )
                else:
                    cur_data_base = data_base
                    cur_gt_path = gt_path
                    cur_data_name = data_name

                # build per-dimension full info to honor per-metric data roots
                cur_full_info_path = self.build_full_info_json(cur_data_base, cur_data_name, [dimension], **kwargs)

                if dimension == 'trajectory_accuracy':
                    from .trajectory_accuracy import compute_trajectory_accuracy

                    results = compute_trajectory_accuracy(
                        gt_path=cur_gt_path, data_base=cur_data_base
                    )

                elif dimension == 'semantic_alignment':  
                    from .caption import caption_reference
                    from .semantic_alignment import compute_semantic_alignment

                    submodules_list = submodules_dict[dimension] 
                    caption_model = submodules_list['caption_model'] 
                    semantics_model = submodules_list['clip_model'] 
                    caption = caption_reference(
                                        model_name=data_name,
                                        model_path = caption_model,
                                        video_folder_root = cur_full_info_path,
                                        save_path = self.output_path,
                                        **kwargs
                                        )
                    caption_json = os.path.join(self.output_path, f"{data_name}_caption_responses.json")
                    with open(caption_json, 'r') as f:
                        data = json.load(f)

                    result = {}
                    for sample_id, info in data.items():
                        if "Overall_Constraints" in info:
                            result[sample_id] = info["Overall_Constraints"]
                        else:
                            print(f"Warning: No 'Overall_Constraints' found in {sample_id}")
                    results_dict['logics'] = result

                    gt_caption_json = os.path.join(self.output_path, f"gt_caption_responses.json")
                    if not os.path.isfile(gt_caption_json):
                        gt_full_info_path = self.build_full_gt_info_json(gt_path, 'gt', **kwargs)
                        gt_caption = caption_reference(
                                            model_name='gt',
                                            model_path = caption_model,
                                            video_folder_root = gt_full_info_path,
                                            save_path = self.output_path,
                                            **kwargs
                                            )                                                                     
                    
                    
                    results = compute_semantic_alignment(caption_json, gt_caption_json, semantics_model)
                
                
                elif dimension == 'action_following':
                    from .action_following import compute_action_following

                    submodules_list = submodules_dict[dimension]

                    results = compute_action_following(cur_full_info_path, submodules_list, **kwargs)

                elif dimension == 'psnr_ssim':
                    from .basic_metrics import compute_basic_metrics
                    
                    results = compute_basic_metrics(
                        gt_path=gt_path, pd_path=data_base, metric_names=["psnr", "ssim"]
                    )
                
                elif dimension == 'psnr':
                    from .basic_metrics import compute_basic_metrics
                    
                    results = compute_basic_metrics(
                        gt_path=gt_path, pd_path=data_base, metric_names=["psnr"]
                    )

                elif dimension == 'ssim':
                    from .basic_metrics import compute_basic_metrics
        
                    results = compute_basic_metrics(
                        gt_path=gt_path, pd_path=data_base, metric_names=["ssim"]
                    )

                elif dimension == 'depth_accuracy':
                    from .depth_accuracy import compute_depth_accuracy

                    submodules_list = submodules_dict[dimension]
                    results = compute_depth_accuracy(
                        cur_full_info_path, submodules_list, gt_path=cur_gt_path
                    )
                elif dimension == 'aesthetic_quality':
                    from .aesthetic_quality import compute_aesthetic_quality

                    submodules_list = submodules_dict[dimension]
                    results = compute_aesthetic_quality(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                elif dimension == 'background_consistency':
                    from .background_consistency import compute_background_consistency

                    submodules_list = submodules_dict[dimension]
                    results = compute_background_consistency(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                elif dimension == 'dynamic_degree':
                    from .dynamic_degree import compute_dynamic_degree

                    submodules_list = submodules_dict[dimension]
                    results = compute_dynamic_degree(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                elif dimension == 'image_quality':
                    from .imaging_quality import compute_imaging_quality

                    submodules_list = submodules_dict[dimension]
                    results = compute_imaging_quality(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                elif dimension == 'subject_consistency':
                    from .subject_consistency import compute_subject_consistency

                    submodules_list = submodules_dict[dimension]
                    results = compute_subject_consistency(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                elif dimension == 'flow_score':
                    from .flow_score import compute_flow_score

                    submodules_list = submodules_dict[dimension]
                    results = compute_flow_score(
                        cur_full_info_path, submodules_list, **kwargs
                    )

                elif dimension == 'photometric_smoothness':
                    from .flow_aepe_metrics import compute_photometric_smoothness

                    submodules_list = submodules_dict[dimension]
                    results = compute_photometric_smoothness(
                        cur_full_info_path, submodules_list, **kwargs
                    )

                elif dimension == 'motion_smoothness':
                    from .motion_smoothness_metrics import compute_motion_smoothness

                    submodules_list = submodules_dict[dimension]
                    results = compute_motion_smoothness(
                        cur_full_info_path, submodules_list, **kwargs
                    )
                    
                else:
                    raise ValueError(f"[Error] Unsupported evaluation dimension: {dimension}")

                # Standardize structure and attach normalized per-video scores
                results = _to_standard_results(dimension, results, data_base)
                results = _add_normalized_scores(dimension, results)

                if dimension == "psnr_ssim":
                    results_dict["psnr"] = results["psnr"]
                    results_dict["ssim"] = results["ssim"]
                else:
                    results_dict[dimension] = results
            

            results_json = os.path.join(self.output_path,f'{data_name}_results.json')    
            with open(results_json, "w") as f:
                json.dump(results_dict, f, indent=2)

        else:

            with open(json_path, "r") as f:
                results_dict = json.load(f)








