import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.evals.tasks.execution.runners.chronomagic_bench.runtime.chronomagic_bench.MTScore.configs.config import (
    Config,
    eval_dict_leaf,
)
from worldfoundry.evals.tasks.execution.runners.chronomagic_bench.runtime.chronomagic_bench.MTScore.configs.utils import (
    _frame_from_video,
    retrieve_text,
    setup_internvideo2,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Score Calculation")
    parser.add_argument("--seed", type=int, default=1421538, help="Random seed")
    parser.add_argument("--model_names", nargs="+", default=["test"], help="Name of the models.")
    parser.add_argument("--input_folder", type=str, default="../toy_video", help="Input folder containing videos")
    parser.add_argument("--output_folder", type=str, default="results/all", help="Output folder for saving scores")
    parser.add_argument(
        "--config_path", type=str, default="configs/internvideo2_stage2_config.py", help="Path to config file"
    )
    parser.add_argument(
        "--model_pth", type=str, default="InternVideo2-stage2_1b-224p-f4.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["open", "close"],
        default="open",
        help="Specify the evaluation mode: 'open' for open-source models or 'close' for closed-source models.",
    )
    return parser.parse_args()


def retry_setup(config, max_attempts=3, delay=2):
    attempts = 0
    while attempts < max_attempts:
        try:
            intern_model, tokenizer = setup_internvideo2(config)
            return intern_model, tokenizer
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts == max_attempts:
                raise Exception("All attempts failed.")  # noqa: B904 - Preserve the implicit context.
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)


def calculate_video_score(video_path, text_to_index, text_candidates, intern_model, config):
    video = cv2.VideoCapture(video_path)
    frames = list(_frame_from_video(video))

    texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)

    general_prob = 0.0
    metamorphic_prob = 0.0

    for t, p in zip(texts, probs, strict=False):
        text_index = text_to_index[t]
        if text_index in [0, 1, 2, 3, 4]:
            general_prob += p
        else:
            metamorphic_prob += p

    return general_prob, metamorphic_prob


def calculate_average_score(scores):
    total_videos = len(scores)
    total_general_score = sum(score[0] for score in scores)
    total_metamorphic_score = sum(score[1] for score in scores)
    average_general_score = total_general_score / total_videos
    average_metamorphic_score = total_metamorphic_score / total_videos
    return average_general_score, average_metamorphic_score


def load_existing_scores(filepath):
    with open(filepath) as file:
        scores = json.load(file)
    return scores


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        config = Config.from_file(args.config_path)
    except BaseException:
        config = Config.from_file(args.config_path.replace("configs/", "MTScore/configs/"))

    config = eval_dict_leaf(config)

    config["model"]["vision_encoder"]["pretrained"] = args.model_pth

    intern_model, tokenizer = retry_setup(config, 10, 2)

    text_candidates = [
        "A conventional video, not a time-condensed video.",
        "A usual video, not an accelerated video sequence.",
        "A normal video, not a time-lapse video.",
        "A standard video, not a time-lapse.",
        "An ordinary video, different from a fast-motion video.",
        "A time-lapse video, distinct from a regular recording.",
        "A time-lapse footage, not your typical video.",
        "A fast-motion video, unlike a standard video.",
        "A time-condensed video, not a conventional video.",
        "An accelerated video sequence, not a usual video.",
    ]

    text_to_index = {text: index for index, text in enumerate(text_candidates)}

    for model_name in tqdm(args.model_names, desc="Processing models"):
        if args.eval_type == "open":
            for part in tqdm(["1", "2", "3"], desc="Processing parts", leave=False):
                scores_json_path = os.path.join(args.output_folder, f"{model_name}_{part}_MTScore.json")
                video_folder = os.path.join(args.input_folder, model_name, part)

                if os.path.exists(scores_json_path):
                    existing_scores = load_existing_scores(scores_json_path)
                    video_scores = existing_scores["video_scores"]
                    processed_videos = {score["video_name"] for score in video_scores}
                else:
                    processed_videos = {}
                    video_scores = []

                video_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]

                for video_file in tqdm(video_files, desc=f"Processing {model_name}", unit="video", leave=False):
                    if video_file in processed_videos:
                        print(f"Skipping {video_file} as it already exists.")
                        continue

                    video_path = os.path.join(video_folder, video_file)
                    general_score, metamorphic_score = calculate_video_score(
                        video_path, text_to_index, text_candidates, intern_model, config
                    )
                    video_score = {
                        "video_name": video_file,
                        "general_score": general_score,
                        "metamorphic_score": metamorphic_score,
                    }
                    video_scores.append(video_score)

                if video_scores:
                    average_general_score, average_metamorphic_score = calculate_average_score(
                        [(score["general_score"], score["metamorphic_score"]) for score in video_scores]
                    )

                    scores_data = {
                        "average_general_score": average_general_score,
                        "average_metamorphic_score": average_metamorphic_score,
                        "video_scores": video_scores,
                    }
                    with open(scores_json_path, "w") as f:
                        json.dump(scores_data, f, indent=4)

                    print(f"MTScore has been saved to {scores_json_path}")
                else:
                    print(f"No video scores found for {model_name}")
        elif args.eval_type == "close":
            scores_json_path = os.path.join(args.output_folder, f"{model_name}_1_MTScore.json")
            video_folder = os.path.join(args.input_folder, model_name)

            if os.path.exists(scores_json_path):
                existing_scores = load_existing_scores(scores_json_path)
                video_scores = existing_scores["video_scores"]
                processed_videos = {score["video_name"] for score in video_scores}
            else:
                processed_videos = {}
                video_scores = []

            video_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]

            for video_file in tqdm(video_files, desc=f"Processing {model_name}", unit="video", leave=False):
                if video_file in processed_videos:
                    print(f"Skipping {video_file} as it already exists.")
                    continue

                video_path = os.path.join(video_folder, video_file)
                general_score, metamorphic_score = calculate_video_score(
                    video_path, text_to_index, text_candidates, intern_model, config
                )
                video_score = {
                    "video_name": video_file,
                    "general_score": general_score,
                    "metamorphic_score": metamorphic_score,
                }
                video_scores.append(video_score)

            if video_scores:
                average_general_score, average_metamorphic_score = calculate_average_score(
                    [(score["general_score"], score["metamorphic_score"]) for score in video_scores]
                )

                scores_data = {
                    "average_general_score": average_general_score,
                    "average_metamorphic_score": average_metamorphic_score,
                    "video_scores": video_scores,
                }
                with open(scores_json_path, "w") as f:
                    json.dump(scores_data, f, indent=4)

                print(f"MTScore has been saved to {scores_json_path}")
            else:
                print(f"No video scores found for {model_name}")


if __name__ == "__main__":
    main()
