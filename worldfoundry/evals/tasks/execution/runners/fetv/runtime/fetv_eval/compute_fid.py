import argparse
import json
import os
import random
import shutil
import time

from cleanfid import fid
from tqdm import tqdm

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


def save_json_file(file_path, json_dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(json_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", default=300, type=int, help="The number of videos per category")
    parser.add_argument("--fg_fid", action="store_true", help="Whether to compute fine-grained FID over categories")
    parser.add_argument("--model", default="text2video-zero", type=str)
    parser.add_argument("--gen_path", default=None, type=str)
    parser.add_argument("--real_path", default="datas/videos/real_videos_fid/16frames_uniform", type=str)
    parser.add_argument(
        "--gen_meta",
        default="datas/sampled_prompts_for_fid_fvd/prompts_gen.json",
        type=str,
        help="path to generated video ids and category labels",
    )
    parser.add_argument(
        "--real_meta",
        default="datas/sampled_prompts_for_fid_fvd/prompts_real.json",
        type=str,
        help="path to real video ids and category labels",
    )
    parser.add_argument(
        "--result_path", default="auto_eval_results", type=str, help="path to save the evaluation results"
    )
    args = parser.parse_args()

    gen_path = args.gen_path or f"datas/videos/{args.model}_fid/16frames_uniform"

    for seed in [12, 22, 32, 42]:
        random.seed(seed)

        if args.fg_fid:  # compute fine-grained FID for different categories of videos
            fid_scores = {}
            gen_meta = read_data(args.gen_meta)
            real_meta = read_data(args.real_meta)
            for aspect in ["spatial", "temporal"]:
                fid_scores[aspect] = {}
                for category in categories[aspect]:
                    data_ids_gen = get_data_under_category(gen_meta, aspect, category, args.sample_num)
                    data_ids_real = get_data_under_category(real_meta, aspect, category, args.sample_num)
                    print(
                        f"Aspect={aspect}, Category={category}, Real Video Num={len(data_ids_real)}, Gen Video Num={len(data_ids_gen)}"
                    )
                    now = time.time()
                    for data_ids, video_type in zip([data_ids_gen, data_ids_real], ["gen", "real"], strict=False):
                        for id in tqdm(data_ids):
                            if video_type == "gen":
                                folder_name = f"video{id}_frames"
                                shutil.copytree(
                                    os.path.join(gen_path, folder_name), f"tmp_{video_type}_videos_{now}/{folder_name}"
                                )
                            else:
                                vid = real_meta[id]["video_id"]
                                folder_name = f"{vid}_frames"
                                shutil.copytree(
                                    os.path.join(args.real_path, folder_name),
                                    f"tmp_{video_type}_videos_{now}/{folder_name}",
                                )

                    fid_score = fid.compute_fid(
                        f"tmp_gen_videos_{now}", f"tmp_real_videos_{now}", mode="clean", num_workers=12
                    )
                    fid_scores[aspect][category] = fid_score
                    print(f"FID={fid_score:.3f}")
                    shutil.rmtree(f"tmp_gen_videos_{now}")
                    shutil.rmtree(f"tmp_real_videos_{now}")
                    result_file = os.path.join(args.result_path, f"fid_fg_results/{args.model}/{seed}.json")
                    save_json_file(result_file, fid_scores)

        else:  # compute FID over all categories with different number of videos
            for sample_num in [32, 64, 128, 256, 300, 512, 1024]:
                gen_fs = random.sample(
                    [fn for fn in os.listdir(gen_path) if fn.endswith("_frames")], sample_num
                )  # sample a subset of the folders of generated videos
                real_fs = random.sample([fn for fn in os.listdir(args.real_path) if fn.endswith("_frames")], sample_num)
                now = time.time()
                for gen_f, real_f in zip(gen_fs, real_fs, strict=False):
                    shutil.copytree(os.path.join(gen_path, gen_f), f"tmp_gen_videos_{now}/{gen_f}")
                    shutil.copytree(os.path.join(args.real_path, real_f), f"tmp_real_videos_{now}/{real_f}")
                fid_score = fid.compute_fid(
                    f"tmp_gen_videos_{now}", f"tmp_real_videos_{now}", mode="clean", num_workers=12
                )
                print(f"FID={fid_score}, video_num={sample_num}, seed={seed}")
                save_file = f"{args.result_path}/fid_results/{args.model}/{seed}/metric-fid{sample_num}_16f.jsonl"
                save_json_file(file_path=save_file, json_dict={"results": {f"fid{sample_num}_16f": fid_score}})
                shutil.rmtree(f"tmp_gen_videos_{now}")
                shutil.rmtree(f"tmp_real_videos_{now}")
