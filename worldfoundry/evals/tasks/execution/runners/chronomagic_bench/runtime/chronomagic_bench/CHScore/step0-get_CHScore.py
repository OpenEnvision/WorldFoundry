import argparse
import json
import os

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos and calculate coherence scores.")
    parser.add_argument("--model_pth", type=str, default="cotracker2.pth", help="Path to the model checkpoint.")
    parser.add_argument("--input_folder", type=str, default="../toy_video", help="Folder containing input videos.")
    parser.add_argument("--output_folder", type=str, default="results/all", help="Folder to save output results.")
    parser.add_argument("--model_names", nargs="+", default=["test_open"], help="Name of the models.")
    parser.add_argument("--grid_size", type=int, default=30, help="Grid size for the model.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for determining frame cuts.")
    parser.add_argument("--size", type=int, default=None, help="Resize the shortest edge of the frame to this size.")
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["open", "close"],
        default="open",
        help="Specify the evaluation mode: 'open' for open-source models or 'close' for closed-source models.",
    )
    return parser.parse_args()


def read_video_from_path(path, size=None):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None

    frames = []
    for _i, im in enumerate(reader):
        image = Image.fromarray(im)
        if size is not None and min(image.size) > size:
            scale = size / min(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        frames.append(np.array(image))
    return np.stack(frames)


def process_video(video_path, size):
    video_original = read_video_from_path(video_path, size)
    video_tensor = torch.from_numpy(video_original).permute(0, 3, 1, 2)[None].float()
    return video_original, video_tensor


def get_movement_vector(trackers):
    N = trackers.shape[2]
    T = trackers.shape[1]

    first_track = trackers[:, 0, :, :].squeeze().reshape(N, 2)
    last_track = trackers[:, int(T // 2), :, :].squeeze().reshape(N, 2)  # before: T // 2; after: T // 3
    pos_vector = last_track - first_track

    return pos_vector


def get_score(model, video, grid_size=30, threshold=0.1):
    trackers, pred_visibility = model(video, grid_size=grid_size)
    _, frames, point_num = pred_visibility.shape

    # Get the movement vector for each tracking point
    pos_vector = get_movement_vector(trackers)

    norms = torch.norm(pos_vector, dim=1, keepdim=True)
    norms = norms + 1e-8  # Add epsilon to avoid division by zero
    movement_directions = pos_vector / norms
    initial_positions = trackers[0, 0, :, :]

    miss_points = []
    for i in range(frames):
        # Identify missing points at frame i
        miss_points_mask = pred_visibility[0, i, :] == 0

        # Get positions at frame i
        positions = trackers[0, i, :, :]

        # Compute the change in position from the initial frame
        delta_positions = positions - initial_positions

        # Compute scalar projections onto each point's movement direction
        scalar_projections = torch.sum(delta_positions * movement_directions, dim=1)

        # Identify points that disappeared in their own far direction
        far_direction_mask = scalar_projections > 0

        # Adjust the missing points by excluding those in their far direction
        adjusted_miss_points_mask = miss_points_mask & (~far_direction_mask)
        miss_points_i = adjusted_miss_points_mask.sum().float() / point_num
        miss_points.append(miss_points_i)

    miss_points = torch.tensor(miss_points)
    miss_points_ap = torch.abs(miss_points[1:] - miss_points[:-1])  # torch.Size([T - 1])

    frames_to_be_cut = (miss_points_ap > threshold).nonzero(as_tuple=True)[0] + 1
    frames_to_be_cut = frames_to_be_cut.cpu().tolist()

    # AMPR :  Rmissed (average proportion of missed points per frame)
    # MPVR :  Vmissed (the variation in the number of missed points between consecutive frames)
    # FCR  :  Rcut    (ratio of frames that need to be cut)
    # CMPV :  Cmissed (indicate the number of consecutive changes in missed points)
    # MCMPV:  Mmissed (maximum continuous change in missed points)
    # TSI_score: CHScore

    global_stats = {
        "AMPR_score": {"max": 0.9368749856948853, "min": 0.0},
        "MPVR_score": {"max": 0.8997353911399841, "min": 0.0},
        "FCR_score": {"max": 0.625, "min": 0.0},
        "CMPV_score": {"max": 20, "min": 0},
        "MCMPV_score": {"max": 1.0, "min": 0.0},
    }

    weights = {
        "CMPV_score": 0.15,
        "FCR_score": 0.15,
        "AMPR_score": 0.35,
        "MPVR_score": 0.25,
        "MCMPV_score": 0.10,
    }

    # Calculate raw scores
    raw_scores = {
        "AMPR_score": miss_points.mean().item(),
        "MPVR_score": miss_points_ap.std().item(),
        "FCR_score": len(frames_to_be_cut) / frames,
        "CMPV_score": (miss_points_ap > threshold).sum().item(),
        "MCMPV_score": miss_points_ap.max().item(),
    }

    # Normalize scores
    normalized_scores = {}
    for key, value in raw_scores.items():
        min_val = global_stats[key]["min"]
        max_val = global_stats[key]["max"]
        normalized_scores[key] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0

    # Calculate TSI_sum
    TSI_sum = sum(normalized_scores[key] * weights[key] for key in normalized_scores)

    # Add normalized values to the scores dictionary
    scores = {key: normalized_scores[key] for key in normalized_scores}
    scores["TSI_sum"] = TSI_sum

    # Calculate TSI_score
    scores["TSI_score"] = 1 / TSI_sum if TSI_sum != 0 else 0

    return scores


def main(args):
    model = CoTrackerPredictor(checkpoint=os.path.join(args.model_pth))

    if torch.cuda.is_available():
        model = model.cuda()

    for model_name in tqdm(args.model_names, desc="Processing models"):
        if args.eval_type == "open":
            for part in tqdm(["1", "2", "3"], desc="Processing parts", leave=False):
                input_dir = os.path.join(args.input_folder, model_name, part)
                output_file_path = os.path.join(args.output_folder, f"{model_name}_{part}_CHScore.json")

                if not os.path.exists(args.output_folder):
                    os.makedirs(args.output_folder)

                if os.path.exists(output_file_path):
                    with open(output_file_path) as existing_file:
                        all_scores_data = json.load(existing_file).get("all_scores", [])
                else:
                    all_scores_data = []

                processed_video_names = {list(video_scores.keys())[0] for video_scores in all_scores_data}  # noqa: RUF015 - Keeps the eager IndexError for an empty record.

                for video_file in tqdm(os.listdir(input_dir), unit="video", leave=False):
                    if video_file.endswith(".mp4"):
                        video_path = os.path.join(input_dir, video_file)
                        video_name = video_file.split(".mp4")[0]

                        if video_name in processed_video_names:
                            print(f"Skipping {video_name} as it already exists.")
                            continue

                        _, video = process_video(video_path, args.size)

                        if torch.cuda.is_available():
                            video = video.cuda()

                        scores = get_score(model, video, args.grid_size, args.threshold)
                        all_scores_data.append({video_name: scores})

                total_average_score = (
                    sum([list(video_scores.values())[0]["TSI_score"] for video_scores in all_scores_data])  # noqa: RUF015 - Keeps the eager IndexError for an empty record.
                    / len(all_scores_data)
                    if all_scores_data
                    else 0
                )

                merged_data = {"total_average_score": total_average_score, "all_scores": all_scores_data}

                with open(output_file_path, "w") as merged_file:
                    json.dump(merged_data, merged_file, indent=4)

                print(f"CHScore has been saved to {output_file_path}")
        elif args.eval_type == "close":
            input_dir = os.path.join(args.input_folder, model_name)
            output_file_path = os.path.join(args.output_folder, f"{model_name}_1_CHScore.json")

            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)

            if os.path.exists(output_file_path):
                with open(output_file_path) as existing_file:
                    all_scores_data = json.load(existing_file).get("all_scores", [])
            else:
                all_scores_data = []

            processed_video_names = {list(video_scores.keys())[0] for video_scores in all_scores_data}  # noqa: RUF015 - Keeps the eager IndexError for an empty record.

            for video_file in tqdm(os.listdir(input_dir), unit="video", leave=False):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(input_dir, video_file)
                    video_name = video_file.split(".mp4")[0]

                    if video_name in processed_video_names:
                        print(f"Skipping {video_name} as it already exists.")
                        continue

                    _, video = process_video(video_path, args.size)

                    if torch.cuda.is_available():
                        video = video.cuda()

                    scores = get_score(model, video, args.grid_size, args.threshold)
                    all_scores_data.append({video_name: scores})

            total_average_score = (
                sum([list(video_scores.values())[0]["TSI_score"] for video_scores in all_scores_data])  # noqa: RUF015 - Keeps the eager IndexError for an empty record.
                / len(all_scores_data)
                if all_scores_data
                else 0
            )

            merged_data = {"total_average_score": total_average_score, "all_scores": all_scores_data}

            with open(output_file_path, "w") as merged_file:
                json.dump(merged_data, merged_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
