# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan Animate pose-conditioning renderer used by AAPoseMeta.draw_aapose."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import matplotlib
import numpy as np

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.animate.pose2d import AAPoseMeta

def draw_handpose(canvas, keypoints, hand_score_th=0.6):
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    eps = 0.01

    H, W, C = canvas.shape
    stickwidth = max(int(min(H, W) / 200), 1)

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        if k1[2] < hand_score_th or k2[2] < hand_score_th:
            continue

        x1 = int(k1[0])
        y1 = int(k1[1])
        x2 = int(k2[0])
        y2 = int(k2[1])
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                thickness=stickwidth,
            )

    for keypoint in keypoints:

        if keypoint is None:
            continue
        if keypoint[2] < hand_score_th:
            continue

        x, y = keypoint[0], keypoint[1]
        x = int(x)
        y = int(y)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), stickwidth, (0, 0, 255), thickness=-1)
    return canvas

def draw_aapose_by_meta(img, meta: AAPoseMeta, threshold=0.5, stick_width_norm=200, draw_hand=True, draw_head=True):
    kp2ds = np.concatenate([meta.kps_body, meta.kps_body_p[:, None]], axis=1)
    kp2ds_lhand = np.concatenate([meta.kps_lhand, meta.kps_lhand_p[:, None]], axis=1)
    kp2ds_rhand = np.concatenate([meta.kps_rhand, meta.kps_rhand_p[:, None]], axis=1)
    pose_img = draw_aapose(img, kp2ds, threshold, kp2ds_lhand=kp2ds_lhand, kp2ds_rhand=kp2ds_rhand, stick_width_norm=stick_width_norm, draw_hand=draw_hand, draw_head=draw_head)
    return pose_img

def draw_aapose(
    img,
    kp2ds,
    threshold=0.6,
    data_to_json=None,
    idx=-1,
    kp2ds_lhand=None,
    kp2ds_rhand=None,
    draw_hand=False,
    stick_width_norm=200,
    draw_head=True
):
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """

    # kp2ds_body = (kp2ds.copy()[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] + \
    #              kp2ds.copy()[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
    kp2ds = kp2ds.copy()
    if not draw_head:
        kp2ds[[0,14,15,16,17], 2] = 0
    kp2ds_body = kp2ds

    # kp2ds_lhand = kp2ds.copy()[91:112]
    # kp2ds_rhand = kp2ds.copy()[112:133]

    limbSeq = [
        [2, 3],
        [2, 6],  # shoulders
        [3, 4],
        [4, 5],  # left arm
        [6, 7],
        [7, 8],  # right arm
        [2, 9],
        [9, 10],
        [10, 11],  # right leg
        [2, 12],
        [12, 13],
        [13, 14],  # left leg
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],  # face (nose, eyes, ears)
        [14, 19],
        [11, 20],  # foot
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        # foot
        [200, 200, 0],
        [100, 100, 0],
    ]

    H, W, C = img.shape
    stickwidth = max(int(min(H, W) / stick_width_norm), 1)

    for _idx, ((k1_index, k2_index), color) in enumerate(zip(limbSeq, colors)):
        keypoint1 = kp2ds_body[k1_index - 1]
        keypoint2 = kp2ds_body[k2_index - 1]

        if keypoint1[-1] < threshold or keypoint2[-1] < threshold:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    for _idx, (keypoint, color) in enumerate(zip(kp2ds_body, colors)):
        if keypoint[-1] < threshold:
            continue
        x, y = keypoint[0], keypoint[1]
        # cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
        cv2.circle(img, (int(x), int(y)), stickwidth, color, thickness=-1)

    if draw_hand:
        img = draw_handpose(img, kp2ds_lhand, hand_score_th=threshold)
        img = draw_handpose(img, kp2ds_rhand, hand_score_th=threshold)

    kp2ds_body[:, 0] /= W
    kp2ds_body[:, 1] /= H

    if data_to_json is not None:
        if idx == -1:
            data_to_json.append(
                {
                    "image_id": "frame_{:05d}.jpg".format(len(data_to_json) + 1),
                    "height": H,
                    "width": W,
                    "category_id": 1,
                    "keypoints_body": kp2ds_body.tolist(),
                    "keypoints_left_hand": kp2ds_lhand.tolist(),
                    "keypoints_right_hand": kp2ds_rhand.tolist(),
                }
            )
        else:
            data_to_json[idx] = {
                "image_id": "frame_{:05d}.jpg".format(idx + 1),
                "height": H,
                "width": W,
                "category_id": 1,
                "keypoints_body": kp2ds_body.tolist(),
                "keypoints_left_hand": kp2ds_lhand.tolist(),
                "keypoints_right_hand": kp2ds_rhand.tolist(),
            }
    return img
