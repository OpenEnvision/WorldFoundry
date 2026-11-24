import argparse
import json
import os
import random
import shutil
import time

from tqdm import tqdm

categories = {
    "objects": [
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
    "temporal": ["fluid motions & deformations", "light change", "actions", "kinetic motions"],
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
        if category in data[aspect]:
            ids.append(i)
    ids = random.sample(ids, sample_num)
    return ids


def compute_fvd(result_file, tmp_gen_path, tmp_real_path, gpus):
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    gpus = ",".join([str(gpu) for gpu in gpus])
    command = f"CUDA_VISIBLE_DEVICES={gpus} python src/scripts/calc_metrics_for_dataset.py \
        --real_data_path {tmp_gen_path} \
        --fake_data_path {tmp_real_path} \
        --save_path {result_file} \
        --mirror 1  \
        --gpus 4  \
        --resolution 256 \
        --verbose 0 \
        --use_cache 0 "
    os.system(command)
    with open(result_file) as f:
        results = json.load(f)
    os.remove(result_file)
    return results["results"]["fvd2048_16f"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", default=300, type=int, help="The number of videos per category")
    parser.add_argument("--gen_video_path", default=None, type=str)
    parser.add_argument("--real_video_path", default=None, type=str)
    parser.add_argument(
        "--gen_meta",
        default="../sampled_prompts_for_fid_fvd/prompts_gen.json",
        type=str,
        help="path to generated video ids and category labels",
    )
    parser.add_argument(
        "--real_meta",
        default="../sampled_prompts_for_fid_fvd/prompts_real.json",
        type=str,
        help="path to real video ids and category labels",
    )
    parser.add_argument("--result_path", default="fvd_results", type=str, help="path to save the results")
    parser.add_argument("--gpus", nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    for seed in [12, 22, 32, 42]:
        random.seed(seed)

        fvd_scores = {}

        gen_meta = read_data(args.gen_meta)
        real_meta = read_data(args.real_meta)
        for aspect in ["objects", "temporal"]:
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
                for data_ids, video_type in zip([data_ids_gen, data_ids_real], ["gen", "real"], strict=False):
                    for id in tqdm(data_ids):
                        if video_type == "gen":
                            folder_name = f"video{id}_frames"
                            shutil.copytree(
                                os.path.join(args.gen_video_path, folder_name),
                                f"tmp_{video_type}_videos_{now}/{folder_name}",
                            )
                        else:
                            vid = real_meta[id]["video_id"]
                            folder_name = f"{vid}_frames"
                            shutil.copytree(
                                os.path.join(args.real_video_path, folder_name),
                                f"tmp_{video_type}_videos_{now}/{folder_name}",
                            )

                fvd_score = compute_fvd(f"fvd_results/tmp{now}.json", tmp_gen_path, tmp_real_path, args.gpus)
                fvd_scores[aspect][category] = fvd_score
                shutil.rmtree(tmp_gen_path)
                shutil.rmtree(tmp_real_path)
                with open(f"fvd_results/fvd_score_{seed}.json", "w") as f:
                    json.dump(fvd_scores, f)
