import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from loguru import logger

from fvmd.datasets.video_datasets import VideoDataset, VideoDatasetNP
from fvmd.extract_motion_features import calc_hist
from fvmd.frechet_distance import calculate_fd_given_vectors
from fvmd.keypoint_tracking import track_keypoints


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_dir", type=str, help=("Path to the log directory"))
    parser.add_argument("path", type=str, nargs=2, help=("Paths to the generated images or to .npz statistic files"))
    args = parser.parse_args()
    return args


def build_dataset(gen_path: str, gt_path: str):
    """
    Build the dataset from the given paths
    Args:
        gt_path(str): path to the ground truth video
        gen_path(str): path to the generated video

    Returns:
        gt_dataset(VideoDataset): ground truth dataset
        gen_dataset(VideoDataset): generated dataset
    """
    if gt_path.endswith((".npz", ".npy")):
        gt_videos = np.load(gt_path)
        gen_videos = np.load(gen_path)
        gt_dataset = VideoDatasetNP(gt_videos)
        gen_dataset = VideoDatasetNP(gen_videos)
    else:
        gt_dataset = VideoDataset(gt_path)
        gen_dataset = VideoDataset(gen_path)
    return gt_dataset, gen_dataset


def fvmd(log_dir: str, gen_path: str, gt_path: str):
    # tracking the keypoints of the videos
    gen_dataset, gt_dataset = build_dataset(gen_path, gt_path)
    velo_gen, velo_gt, acc_gen, acc_gt = track_keypoints(
        log_dir=log_dir, gen_dataset=gen_dataset, gt_dataset=gt_dataset, v_stride=1
    )

    # calculate statistics 1d histogram
    B = velo_gen.shape[0]
    gt_v_hist = calc_hist(velo_gt).reshape(B, -1)
    gen_v_hist = calc_hist(velo_gen).reshape(B, -1)
    gt_a_hist = calc_hist(acc_gt).reshape(B, -1)
    gen_a_hist = calc_hist(acc_gen).reshape(B, -1)

    # combine the velocity and acceleration histograms
    gt_hist = np.concatenate((gt_v_hist, gt_a_hist), axis=1)
    gen_hist = np.concatenate((gen_v_hist, gen_a_hist), axis=1)

    # calculate FID
    fvmd_value = calculate_fd_given_vectors(gt_hist, gen_hist)
    return fvmd_value


def main():
    args = parse_args()
    logger.add(os.path.join(args.log_dir, "log.txt"))
    logger.info(f"Computing FVMD between {args.path[0]} and {args.path[1]}")

    fvmd_value = fvmd(log_dir=args.log_dir, gen_path=args.path[0], gt_path=args.path[1])

    # save the results
    logger.info(f"FVMD: {fvmd_value}")
    if args.log_dir is not None:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        json_path = os.path.join(args.log_dir, "fvmd.json")
        with open(json_path, "w") as f:
            json.dump(fvmd_value, f, indent=4)


if __name__ == "__main__":
    main()
