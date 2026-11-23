import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from loguru import logger

from fvmd.frechet_distance import calculate_activation_statistics, calculate_frechet_distance


def cut_subcube(vectors: np.ndarray, cell_size: int = 5, cube_frames: int = 4):
    """
    Cut the whole video sequence into subcubes
    Args:
        vectors (np.ndarray): (B, S, H, W, 2)
        cell_size (int): the height and width of the subcube
        cube_frames (int): the number of frames in a subcube

    Returns:
        vectors (np.ndarray): (B*MS*MH*MW, cube_frames, cell_size, cell_size, 2)
        MS (int): the number of subcubes in the time dimension
        MH (int): the number of subcubes in the height dimension
        MW (int): the number of subcubes in the width dimension
    """
    B, S, H, W, _ = vectors.shape
    MH = H // cell_size
    MW = W // cell_size
    MS = S // cube_frames
    vectors = vectors[:, : MS * cube_frames, : MH * cell_size, : MW * cell_size, :]
    vectors = vectors.reshape(B, MS, cube_frames, MH, cell_size, MW, cell_size, 2)
    vectors = vectors.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    vectors = vectors.reshape(-1, cube_frames, cell_size, cell_size, 2)
    return vectors, MS, MH, MW


def count_subcube_hist(vector_cell: np.ndarray, angle_bins: int = 8, magnitude_bins: int = 256) -> np.ndarray:
    """
    Count the histogram for the subcube
    Args:
        vector_cell (np.ndarray): (S, H, W, 2)
        angle_bins (int): the number of angle bins
        magnitude_bins (int): the number of magnitude bins

    Returns:
        HOG_hist (np.ndarray): (angle_bins,)
    """
    HOG_hist = np.zeros(angle_bins)
    S, H, W, _ = vector_cell.shape

    angle_list = np.arctan2(vector_cell[:, :, :, 0], vector_cell[:, :, :, 1])
    angle_bin_list = (angle_list + np.pi) // (2 * np.pi / angle_bins)
    angle_bin_list = np.clip(angle_bin_list, 0, angle_bins - 1)

    magnitude_list = np.linalg.norm(vector_cell, axis=3)
    magnitude_list = np.clip(magnitude_list, 0, magnitude_bins - 1)
    magnitude_list = magnitude_list + 1
    magnitude_list = np.log2(magnitude_list)
    magnitude_list = np.clip(magnitude_list, 0, int(np.log2(magnitude_bins)))
    magnitude_list = np.ceil(magnitude_list)
    magnitude_list = magnitude_list / np.log2(magnitude_bins)

    for s in range(S):
        for i in range(H):
            for j in range(W):
                HOG_hist[int(angle_bin_list[s, i, j])] += magnitude_list[s, i, j]

    return HOG_hist


def calc_hist(vectors: np.ndarray, cell_size: int = 5, angle_bins: int = 8, cube_frames: int = 4) -> np.ndarray:
    """
    Calculate the histogram for the whole video sequence
    Args:
        vectors (np.ndarray): (B, S, H, W, 2)
        cell_size (int): the height and width of the subcube
        angle_bins (int): the number of angle bins
        cube_frames (int): the number of frames in a subcube

    Returns:
        histos (np.ndarray): (B, MS, MH, MW, angle_bins)
    """
    B, S, N, _ = vectors.shape
    H = np.sqrt(N).round().astype(np.int32)
    W = H
    vectors = vectors.reshape(B, S, H, W, 2)

    vectors, MS, MH, MW = cut_subcube(vectors, cell_size, cube_frames)
    histos = []
    for i in range(vectors.shape[0]):
        histos.append(count_subcube_hist(vectors[i], angle_bins))
    histos = np.stack(histos, axis=0)
    histos = histos.reshape(B, MS, MH, MW, angle_bins)
    return histos


def calculate_fvmd_given_paths(gen_path, gt_path):
    """Calculates the FID of two paths"""
    if not os.path.exists(gen_path):
        raise RuntimeError(f"Invalid path: {gen_path}")
    if not os.path.exists(gt_path):
        raise RuntimeError(f"Invalid path: {gt_path}")

    results = {}

    v_gen = np.load(os.path.join(gen_path, "velocity_gen.npy"))
    v_gt = np.load(os.path.join(gt_path, "velocity_gt.npy"))
    a_gen = np.load(os.path.join(gen_path, "acceleration_gen.npy"))
    a_gt = np.load(os.path.join(gt_path, "acceleration_gt.npy"))

    hist_v_gen = calc_hist(v_gen, cell_size=5, angle_bins=8, cube_frames=4)
    hist_v_gt = calc_hist(v_gt, cell_size=5, angle_bins=8, cube_frames=4)
    hist_a_gen = calc_hist(a_gen, cell_size=5, angle_bins=8, cube_frames=4)
    hist_a_gt = calc_hist(a_gt, cell_size=5, angle_bins=8, cube_frames=4)

    B = hist_v_gen.shape[0]
    hist_va_gen = np.concatenate((hist_v_gen.reshape(B, -1), hist_a_gen.reshape(B, -1)), axis=1)
    hist_va_gt = np.concatenate((hist_v_gt.reshape(B, -1), hist_a_gt.reshape(B, -1)), axis=1)

    m1, s1 = calculate_activation_statistics(hist_v_gen)
    m2, s2 = calculate_activation_statistics(hist_v_gt)
    fvmd_value = calculate_frechet_distance(m1, s1, m2, s2)
    results["velocity_fvmd"] = fvmd_value
    logger.info(f"velocity FVMD: {fvmd_value}")

    m1, s1 = calculate_activation_statistics(hist_a_gen)
    m2, s2 = calculate_activation_statistics(hist_a_gt)
    fvmd_value = calculate_frechet_distance(m1, s1, m2, s2)
    results["acceleration_fvmd"] = fvmd_value
    logger.info(f"acceleration FVMD: {fvmd_value}")

    m1, s1 = calculate_activation_statistics(hist_va_gen)
    m2, s2 = calculate_activation_statistics(hist_va_gt)
    fvmd_value = calculate_frechet_distance(m1, s1, m2, s2)
    results["combine_fvmd"] = fvmd_value
    logger.info(f"combine FVMD: {fvmd_value}")

    return results


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_dir", type=str, help=("Path to the log directory"))
    parser.add_argument("path", type=str, nargs=2, help=("Paths to the generated images or to .npz statistic files"))
    args = parser.parse_args()

    fvmd_values = calculate_fvmd_given_paths(args.path[0], args.path[1])

    if args.log_dir is not None:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        json_path = os.path.join(args.log_dir, "fvmd.json")
        with open(json_path, "w") as f:
            json.dump(fvmd_values, f, indent=4)


if __name__ == "__main__":
    main()
