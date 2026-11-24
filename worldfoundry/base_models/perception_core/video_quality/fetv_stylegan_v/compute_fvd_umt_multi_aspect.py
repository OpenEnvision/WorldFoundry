import argparse
import json
import os
import random
import shutil
import time

import torch

categories = {
    "spatial": [
        "animals",
        "people",
        "plants",
        "illustrations",
        "artifacts",
        "vehicles",
        "buildings & infrastructure",
        "food & beverage",
        "scenery & natural objects",
    ],
    "temporal": ["fluid motions", "light change", "actions", "kinetic motions"],
}


def read_data(data_path):
    datas = []
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        datas.append(json.loads(line))
    return datas


def sample_feats(ids, src_feat_file, tgt_feat_file):
    # sample a subset of features from the source features, based on the ids
    src_feats = torch.load(src_feat_file)
    tgt_feats = src_feats[ids]
    tgt_path = os.path.dirname(tgt_feat_file)
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    torch.save(tgt_feats, tgt_feat_file)


def get_data_under_category(meta_datas, aspect, category, sample_num):
    """
    Return all the data ids that belong to the category under aspect
    """
    ids = []
    for i, data in enumerate(meta_datas):
        if category in data["major content"][aspect]:
            ids.append(i)
    ids = random.sample(ids, sample_num)
    return ids


def compute_fvd(result_path, tmp_gen_path, tmp_real_path, gpus, sample_num):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    gpus = ",".join([str(gpu) for gpu in gpus])
    command = f"CUDA_VISIBLE_DEVICES={gpus} python src/scripts/calc_metrics_for_dataset.py \
        --real_data_path {tmp_gen_path} \
        --fake_data_path {tmp_real_path} \
        --real_feat_path {os.path.join(tmp_gen_path, 'feats.pt')} \
        --fake_feat_path {os.path.join(tmp_real_path, 'feats.pt')} \
        --save_path {result_path} \
        --metrics fvd{sample_num}_16f \
        --mirror 1  \
        --gpus 1  \
        --resolution 256 \
        --verbose 0 \
        --use_cache 0 "
    os.system(command)
    result_file = os.path.join(result_path, f"1/metric-fvd{sample_num}_16f.jsonl")
    with open(result_file) as f:
        results = json.load(f)
    # os.remove(result_file)
    shutil.rmtree(result_path)
    return results["results"][f"fvd{sample_num}_16f"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", default=300, type=int, help="The number of videos per category")
    parser.add_argument("--model", default="modelscope-t2v", type=str)
    parser.add_argument(
        "--gen_meta",
        default="../datas/sampled_prompts_for_fid_fvd/prompts_gen.json",
        type=str,
        help="path to generated video ids and category labels",
    )
    parser.add_argument(
        "--real_meta",
        default="../datas/sampled_prompts_for_fid_fvd/prompts_real.json",
        type=str,
        help="path to real video ids and category labels",
    )
    parser.add_argument("--gpus", nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    result_path = f"../auto_eval_results/fvd_umt_fg_results//{args.model}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for seed in [12, 22, 32, 42]:
        random.seed(seed)

        fvd_scores = {}
        gen_meta = read_data(args.gen_meta)
        real_meta = read_data(args.real_meta)
        for aspect in ["spatial", "temporal"]:
            fvd_scores[aspect] = {}
            for category in categories[aspect]:
                data_ids_gen = get_data_under_category(gen_meta, aspect, category, args.sample_num)
                data_ids_real = get_data_under_category(real_meta, aspect, category, args.sample_num)
                now = time.time()
                tmp_gen_path = f"tmp_gen_videos_{now}"
                tmp_real_path = f"tmp_real_videos_{now}"
                print(
                    f"Aspect={aspect}, Category={category}, Real Video Num={len(data_ids_real)}, Gen Video Num={len(data_ids_gen)}"
                )

                sample_feats(
                    data_ids_gen,
                    os.path.join(f"../datas/videos/{args.model}_feats/{args.model}_feats.pt"),
                    os.path.join(tmp_gen_path, "feats.pt"),
                )
                sample_feats(
                    data_ids_real,
                    os.path.join("../datas/videos/real_videos_feats/real_videos_feats.pt"),
                    os.path.join(tmp_real_path, "feats.pt"),
                )

                fvd_score = compute_fvd(
                    os.path.join(result_path, f"tmp{now}"), tmp_gen_path, tmp_real_path, args.gpus, args.sample_num
                )
                fvd_scores[aspect][category] = fvd_score
                shutil.rmtree(f"tmp_gen_videos_{now}")
                shutil.rmtree(f"tmp_real_videos_{now}")
                with open(os.path.join(result_path, f"{seed}.json"), "w") as f:
                    json.dump(fvd_scores, f)
