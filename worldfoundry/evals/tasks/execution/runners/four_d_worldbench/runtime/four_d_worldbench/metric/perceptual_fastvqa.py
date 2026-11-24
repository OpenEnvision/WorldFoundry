import json
import os
import sys

import numpy as np
import torch
import yaml

# Follow the benchmark's module style
try:
    from metric.utils import load_dimension_info
except ImportError:
    from .utils import load_dimension_info


def _ensure_fastvqa_importable() -> str:
    """Add the in-tree FAST-VQA runtime to sys.path and return its root."""
    from worldfoundry.base_models.perception_core.video_quality.fastvqa import runtime_root

    fastvqa_root = str(runtime_root())
    if fastvqa_root not in sys.path:
        sys.path.insert(0, fastvqa_root)
    return fastvqa_root


def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds.get(model, mean_stds["FAST-VQA"])
    x = (score - mean) / std
    score = 1 / (1 + np.exp(-x))
    return score


mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA": (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}


def _build_model_and_opts(model_name: str, device: str):
    """Create evaluator and return (evaluator, opt_dict)."""
    _ensure_fastvqa_importable()
    from fastvqa.models import DiViDeAddEvaluator

    from worldfoundry.base_models.perception_core.video_quality.fastvqa import checkpoint_path, options_dir

    opts = {
        "FasterVQA": options_dir() / "f3dvqa-b.yml",
        "FasterVQA-MS": options_dir() / "fastervqa-ms.yml",
        "FasterVQA-MT": options_dir() / "fastervqa-mt.yml",
        "FAST-VQA": options_dir() / "fast-b.yml",
        "FAST-VQA-M": options_dir() / "fast-m.yml",
    }
    opt_path = opts.get(model_name, opts["FAST-VQA"])
    with open(opt_path) as f:
        opt = yaml.safe_load(f)
    opt["test_load_path"] = checkpoint_path(model_name)

    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    state = torch.load(opt["test_load_path"], map_location=device)
    evaluator.load_state_dict(state["state_dict"])
    evaluator.eval()
    return evaluator, opt


def _infer_video_score(evaluator, opt: dict, video_path: str, device: str, model_name: str) -> float:
    """Run FAST-VQA/FasterVQA on a single video and return [0,1] score."""
    _ensure_fastvqa_importable()
    import decord
    from fastvqa.datasets import (
        FragmentSampleFrames,
        SampleFrames,
        get_spatial_fragments,
    )

    try:
        video_reader = decord.VideoReader(video_path)
    except Exception as e:
        print(f"[fastvqa] Failed to open video: {video_path} ({e})")
        return None

    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = t_data_opt["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        if t_data_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                fragments_t=sample_args.get("t_frag", 1),
                num_clips=sample_args.get("num_clips", 1),
            )
        else:
            sampler = SampleFrames(
                clip_len=sample_args["clip_len"],
                num_clips=sample_args["num_clips"],
            )

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        sampled_video = get_spatial_fragments(video, **sample_args)
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        sampled_video = sampled_video.reshape(
            sampled_video.shape[0],
            num_clips,
            -1,
            *sampled_video.shape[2:],
        ).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(device)

    with torch.no_grad():
        result = evaluator(vsamples)
    raw = float(result.mean().item())
    score = float(sigmoid_rescale(raw, model=model_name))
    return score


def compute_perceptual_fastvqa(json_dir, device, submodules_dict, **kwargs):
    """
    Benchmark entry point. Returns (avg_score, detailed_results_list).
    """
    model_name = kwargs.get("model", "FasterVQA")
    dataset_json = kwargs.get("dataset_json", "")
    dimension_name = "fastvqa"

    evaluator, opt = _build_model_and_opts(model_name, device)

    # Load videos for this dimension
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension_name, lang="en")

    video_results = []
    scores = []
    for prompt_dict in prompt_dict_ls:
        video_paths = prompt_dict.get("video_list", [])
        for video_path in video_paths:
            score = None
            try:
                score = _infer_video_score(evaluator, opt, video_path, device, model_name)
            except Exception as e:
                print(f"[fastvqa] Error scoring {video_path}: {e}")
            if score is not None:
                scores.append(score)
            video_results.append(
                {
                    "video_path": video_path,
                    "fastvqa_score": None if score is None else float(score),
                }
            )

    avg_score = float(np.mean(scores)) if scores else 0.0

    # Save JSON like other metrics
    output_dir = os.path.dirname(json_dir)
    dim_name = os.path.splitext(os.path.basename(json_dir))[0]
    dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else "dataset"
    suffix = f"{dim_name}__{model_name}__{dataset_base}_results.json" if model_name else f"{dim_name}_results.json"
    output_file = os.path.join(output_dir, suffix)

    detailed_output = {
        "evaluation_summary": {
            "total_videos": len(video_results),
            "average_score": avg_score,
        },
        "video_details": video_results,
    }
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"[fastvqa] Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"[fastvqa] Failed to save results JSON: {e}")

    return avg_score, video_results
