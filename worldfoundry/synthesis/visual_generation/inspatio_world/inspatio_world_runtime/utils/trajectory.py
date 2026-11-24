"""Trajectory helpers for InSpatio point-cloud rendering."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


def txt_interpolation(input_list, n: int, mode: str = "smooth"):
    x = np.linspace(0, 1, len(input_list))
    if mode == "smooth":
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == "linear":
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    return f(xnew)


def sphere2pose(x_up_angle, y_left_angle, r, is_zoom: bool = False, is_translation: bool = False):
    angle_y = np.deg2rad(y_left_angle)
    sin_value_y = np.sin(angle_y)
    cos_value_y = np.cos(angle_y)
    rot_mat_y = np.array(
        [
            [cos_value_y, 0, sin_value_y],
            [0, 1, 0],
            [-sin_value_y, 0, cos_value_y],
        ]
    )
    angle_x = np.deg2rad(x_up_angle)
    sin_value_x = np.sin(angle_x)
    cos_value_x = np.cos(angle_x)
    rot_mat_x = np.array(
        [
            [1, 0, 0],
            [0, cos_value_x, sin_value_x],
            [0, -sin_value_x, cos_value_x],
        ]
    )

    rotation = rot_mat_y @ rot_mat_x
    translation = np.array([-r * cos_value_x * sin_value_y, -r * sin_value_x, r - r * cos_value_x * cos_value_y])
    if is_translation:
        translation = np.array([0, 0, 0])
    if is_zoom:
        translation = np.array([0, 0, r])

    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    return c2w


def generate_traj_txt(x_up_angles, y_left_angles, r, r_zoom, frame: int, is_translation: bool = False):
    if len(x_up_angles) > 3:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode="smooth")
    else:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode="linear")

    if len(y_left_angles) > 3:
        y_left_angles = txt_interpolation(y_left_angles, frame, mode="smooth")
    else:
        y_left_angles = txt_interpolation(y_left_angles, frame, mode="linear")

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode="smooth")
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode="linear")

    if len(r_zoom) > 3:
        r_zooms = txt_interpolation(r_zoom, frame, mode="smooth")
        r_zooms[0] = r_zoom[0]
        r_zooms[-1] = r_zoom[-1]
    else:
        r_zooms = txt_interpolation(r_zoom, frame, mode="linear")

    c2ws_list = []
    is_zoom = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
    is_not_y = all(y == 0 for y in y_left_angles)

    for x_up_angle, y_left_angle, r_val, r_zoom_val in zip(x_up_angles, y_left_angles, rs, r_zooms, strict=False):
        radius = r_zoom_val if is_not_y else r_val
        c2ws_list.append(
            sphere2pose(
                np.float32(x_up_angle),
                np.float32(y_left_angle),
                np.float32(radius),
                is_zoom=is_zoom,
                is_translation=is_translation,
            )
        )

    return np.stack(c2ws_list, axis=0)


__all__ = ["generate_traj_txt", "sphere2pose", "txt_interpolation"]
