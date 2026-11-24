# caption_metric.py
import json
import os

import numpy as np
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

results_list = {}


def load_clip_text_encoder(clip_model_path: str):
    """Load CLIP text encoder + tokenizer once per process."""
    from transformers import CLIPModel, CLIPTokenizerFast

    model = CLIPModel.from_pretrained(clip_model_path, torch_dtype=torch.float16 if device == "cuda" else None)
    model.to(device)
    model.eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_path)
    return model, tokenizer


def run_clip_text_similarity(model, tokenizer, gen_text: str, gt_text: str) -> float:
    """Compute cosine similarity between two texts using CLIP text encoder."""
    with torch.no_grad():
        inputs_gen = tokenizer(gen_text, return_tensors="pt", truncation=True, max_length=77).to(device)
        inputs_gt = tokenizer(gt_text, return_tensors="pt", truncation=True, max_length=77).to(device)
        gen_out = model.get_text_features(**inputs_gen)
        gt_out = model.get_text_features(**inputs_gt)

        def extract_feat(out):
            if torch.is_tensor(out):
                return out
            if hasattr(out, "text_embeds") and out.text_embeds is not None:
                return out.text_embeds
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # fallback: use cls token
                return out.last_hidden_state[:, 0]
            raise ValueError("Cannot extract text features from CLIP output")

        gen_feat = extract_feat(gen_out)
        gt_feat = extract_feat(gt_out)
        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
        gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
        score = (gen_feat * gt_feat).sum(dim=-1).item()
    return float(score)


def get_strings(json_path, key):
    with open(json_path) as f:
        data = json.load(f)  # dict
    all_idxs = []
    all_strings = []
    for idx, value in data.items():
        all_idxs.append(idx)
        if key == "General":
            if isinstance(value, dict) and "General" in value:
                all_strings.append(value["General"])
            else:
                # Fallback when value itself is already a string
                all_strings.append(str(value))
        elif key == "Events":
            event_strings = ""
            current_events = value["Events"]
            for event in current_events:
                if len(event) < 2:
                    # we discard events with less than 2 words
                    break
                event_strings += event + " "
            all_strings.append(event_strings)
        else:
            raise ValueError("Invalid key")
    return all_idxs, all_strings


def compute_metric_scores(semantics_model, metric_type, list_gen_strings, list_gt_strings):
    if metric_type != "CLIPScore":
        raise ValueError("Invalid metric type")

    clip_model, clip_tokenizer = load_clip_text_encoder(semantics_model)

    scores = [
        run_clip_text_similarity(clip_model, clip_tokenizer, gen_strings, gt_strings)
        for gen_strings, gt_strings in tqdm(
            list(zip(list_gen_strings, list_gt_strings, strict=False)), desc="Compute metric scores"
        )
    ]
    return scores


def evaluate_run(semantics_model, eval_config, dt_json, gt_json):
    metric_type = eval_config["metric_type"]
    key = eval_config["key"]

    # Load generated and ground-truth strings
    gen_idxs, gen_strings = get_strings(dt_json, key)
    gt_idxs, gt_strings = get_strings(gt_json, key)
    expanded_gt_strings = []
    expanded_gt_idxs = []

    for gen_id, _gen_string in zip(gen_idxs, gen_strings, strict=False):
        # Robustly parse generated id. We expect keys like:
        #  - generated_dataset_367_649524_1
        #  - generated_dataset_egodex_dataset_assemble_disassemble_jigsaw_puzzle_1
        # Strategy: strip the leading 'generated_dataset_' and rsplit from the right into
        # (task_name, episode_id, gid). task_name may contain underscores.
        rest = gen_id.split("generated_dataset_")[-1]
        try:
            task_name, episode_id, gid = rest.rsplit("_", 2)
        except ValueError:
            # fallback: try a more permissive split
            parts = rest.split("_")
            if len(parts) >= 3:
                task_name = "_".join(parts[:-2])
                episode_id = parts[-2]
                gid = parts[-1]
            elif len(parts) == 2:
                task_name = parts[0]
                episode_id = parts[1]
                gid = ""
            else:
                task_name = rest
                episode_id = ""
                gid = ""

        # Try GT key variants: with episode id and without (some gt jsons omit episode)
        candidate_with_episode = f"gt_dataset_{task_name}_{episode_id}" if episode_id else None
        candidate_no_episode = f"gt_dataset_{task_name}"

        if candidate_with_episode and candidate_with_episode in gt_idxs:
            gt_idx = candidate_with_episode
        elif candidate_no_episode in gt_idxs:
            gt_idx = candidate_no_episode
        else:
            # Helpful error message listing some available GT keys for debugging
            sample = gt_idxs[:50]
            raise ValueError(
                f"GT key not found for generated id '{gen_id}'.\nTried: {candidate_with_episode}, {candidate_no_episode}.\nAvailable GT keys (first 50): {sample}"
            )

        gt_string = gt_strings[gt_idxs.index(gt_idx)]
        expanded_gt_strings.append(gt_string)
        expanded_gt_idxs.append(gt_idx)

    gt_strings = expanded_gt_strings

    assert len(gen_strings) == len(gt_strings), "Number of generated and ground-truth strings do not match"
    print(f"the number of generated strings: {len(gen_strings)}")
    # Compute metric scores
    scores = compute_metric_scores(semantics_model, metric_type, gen_strings, gt_strings)

    # Save scores to file
    scores_list = [np.around(score, decimals=6).tolist() for score in scores]

    for idx, gen_idx in enumerate(gen_idxs):
        # parse task/episode/gid robustly similar to above
        rest = gen_idx.split("generated_dataset_")[-1]
        try:
            task_name, episode_id, gid = rest.rsplit("_", 2)
        except ValueError:
            parts = rest.split("_")
            if len(parts) >= 3:
                task_name = "_".join(parts[:-2])
                episode_id = parts[-2]
                gid = parts[-1]
            elif len(parts) == 2:
                task_name = parts[0]
                episode_id = parts[1]
                gid = ""
            else:
                task_name = rest
                episode_id = ""
                gid = ""

        task_id = task_name
        episode_id = episode_id

        if task_id not in results_list:
            results_list[task_id] = {}
        if episode_id not in results_list[task_id]:
            results_list[task_id][episode_id] = {}
        if gid not in results_list[task_id][episode_id]:
            results_list[task_id][episode_id][gid] = {}

        results_list[task_id][episode_id][gid][metric_type] = scores_list[idx]

    avg_score = np.mean(scores).tolist()  # noqa: F841 - np.mean warns on empty scores as upstream.

    return scores


def evaluate_runs_single_config(eval_config, json_path, gt_json, semantics_model):
    assert gt_json is not None, "Ground-truth JSON not found"

    scores_list = {}

    if json_path != gt_json:
        dt_dataset_name = os.path.basename(json_path).split("_dataset")[0]
        scores = evaluate_run(semantics_model, eval_config, json_path, gt_json)
        avg_score = np.mean(scores)
        print(f"{dt_dataset_name}: {avg_score:.6f}")
        scores_list[dt_dataset_name] = float(avg_score)

    return scores_list


def evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model):
    eval_results = {}
    for eval_config in tqdm(eval_configs, desc="Eval different configs"):
        exp_name = eval_config["metric_type"]
        print(f"Evaluating {exp_name}...")
        scores_list = evaluate_runs_single_config(eval_config, json_path, gt_path, semantics_model)
        print(f"Scores: {scores_list}")
        eval_results[exp_name] = scores_list
    return eval_results


def compute_semantic_alignment(json_path, gt_path, semantics_model):
    eval_configs = [
        {
            "metric_type": "CLIPScore",
            "key": "General",
        }
    ]

    evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model)
    return results_list
