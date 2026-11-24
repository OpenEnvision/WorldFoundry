import os
import re

import decord
import numpy as np
import requests
import torch
from einops import rearrange
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def read_frames(path):
    vr = decord.VideoReader(
        uri=path,
        height=-1,
        width=-1,
    )
    frames = vr.get_batch(range(len(vr)))
    frames = rearrange(frames, "T H W C -> C T H W").contiguous()  # > C T H W
    frames = frames.float() / 255.0
    frames = frames.permute(1, 0, 2, 3)  # t c h w
    return frames


regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def download_url(input_path):
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def recursively_find(root_dir: str, ext: list | None = None, relative_path: str | None = None):
    all_fnames = [os.path.join(root, fname) for root, _dirs, files in os.walk(root_dir) for fname in files]
    if relative_path is not None:
        all_fnames = [os.path.relpath(fname, relative_path) for fname in all_fnames]
    else:
        all_fnames = [os.path.abspath(fname) for fname in all_fnames]
    if ext is None:
        return all_fnames
    else:
        return [fname for fname in all_fnames if os.path.splitext(fname)[-1] in ext]


def get_crop_bbox(ori_h, ori_w, tgt_h, tgt_w):
    tgt_ar = tgt_h / tgt_w
    ori_ar = ori_h / ori_w
    if abs(ori_ar - tgt_ar) < 0.01:
        return 0, ori_h, 0, ori_w
    if ori_ar > tgt_ar:
        crop_h = int(tgt_ar * ori_w)
        y0 = (ori_h - crop_h) // 2
        y1 = y0 + crop_h
        return y0, y1, 0, ori_w
    else:
        crop_w = int(ori_h / tgt_ar)
        x0 = (ori_w - crop_w) // 2
        x1 = x0 + crop_w
        return 0, ori_h, x0, x1


def isotropic_crop_resize(frames: torch.Tensor, size: tuple, is_mask: bool = False):
    """
    frames: (T, C, H, W)
    size: (H, W)
    """
    ori_h, ori_w = frames.shape[2:]
    h, w = size
    y0, y1, x0, x1 = get_crop_bbox(ori_h, ori_w, h, w)
    cropped_frames = frames[:, :, y0:y1, x0:x1]

    # Use NEAREST interpolation for masks (no antialias), bicubic for frames
    if is_mask:
        interpolation_mode = InterpolationMode.NEAREST
        use_antialias = False
    else:
        interpolation_mode = InterpolationMode.BICUBIC
        use_antialias = True

    resized_frames = resize(cropped_frames, size, interpolation_mode, antialias=use_antialias)
    return resized_frames


def get_random_crop_bbox(ori_h, ori_w, crop_max_ratio: float, rnd_state: np.random.RandomState):
    if crop_max_ratio >= 1:
        raise ValueError("crop_max_ratio should be smaller than 1")
    random_ratio = rnd_state.random((4,))
    h_crop_ratio = random_ratio[0] * crop_max_ratio
    w_crop_ratio = random_ratio[1] * crop_max_ratio
    new_h = round(ori_h * (1 - h_crop_ratio))
    new_w = round(ori_w * (1 - w_crop_ratio))
    y0 = round((ori_h - new_h) * random_ratio[2])
    x0 = round((ori_w - new_w) * random_ratio[3])
    return y0, y0 + new_h, x0, x0 + new_w


def random_crop(
    frames: torch.Tensor,
    crop_max_ratio: float,
    rnd_state: np.random.RandomState,
    return_crop_bbox: bool = False,
):
    """
    frames: (T, C, H, W)
    size: (H, W)
    """
    ori_h, ori_w = frames.shape[2:]
    y0, y1, x0, x1 = get_random_crop_bbox(ori_h, ori_w, crop_max_ratio, rnd_state)
    cropped_frames = frames[:, :, y0:y1, x0:x1]
    if not return_crop_bbox:
        return cropped_frames
    else:
        return cropped_frames, (y0, y1, x0, x1)


def read_txt(in_path):
    with open(in_path) as f:
        return f.read()


def center_poses_by_mean_translation(T_w2c):
    R = T_w2c[:, :3, :3]  # (N, 3, 3)
    t = T_w2c[:, :3, 3]  # (N, 3)
    centers = -(np.transpose(R, (0, 2, 1)) @ t[..., None]).squeeze(-1)  # (N, 3)
    mu = np.mean(centers, axis=0)  # (3,)
    Ginv = np.eye(4, dtype=T_w2c.dtype)
    Ginv[:3, 3] = mu
    T_centered = T_w2c @ Ginv[None, ...]

    return T_centered


##### traj pose generation #####
from scipy.interpolate import UnivariateSpline, interp1d  # noqa: E402 - Follows module-level definitions, as upstream.


def txt_interpolation(input_list, n, mode="smooth"):
    x = np.linspace(0, 1, len(input_list))
    if mode == "smooth":
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == "linear":
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


# def sphere2pose(x_up_angle, y_left_angle, r, is_zoom=False):
#     angle_y = np.deg2rad(y_left_angle)
#     sin_value_y = np.sin(angle_y)
#     cos_value_y = np.cos(angle_y)
#     rot_mat_y = np.array(
#         [
#             [cos_value_y, 0, sin_value_y],
#             [0, 1, 0],
#             [-sin_value_y, 0, cos_value_y],
#         ]
#     )
#     angle_x = np.deg2rad(x_up_angle)
#     sin_value_x = np.sin(angle_x)
#     cos_value_x = np.cos(angle_x)
#     rot_mat_x = np.array(
#         [
#             [1, 0, 0],
#             [0, cos_value_x, sin_value_x],
#             [0, -sin_value_x, cos_value_x],
#         ]
#     )

#     R = rot_mat_y @ rot_mat_x
#     T = np.array([-r*cos_value_x*sin_value_y, -r*sin_value_x, r-r*cos_value_x*cos_value_y])

#     if is_zoom:
#         T = np.array([0, 0, r])

#     c2w = np.eye(4)
#     c2w[:3,:3] = R
#     c2w[:3,3] = T

#     return c2w # 4x4

# def generate_traj_txt(x_up_angles, y_left_angles, r, r_zoom, frame):
#     # Initialize a camera.
#     """
#     COLMAP coordinate
#     """

#     if len(x_up_angles) > 3:
#         x_up_angles = txt_interpolation(x_up_angles, frame, mode='smooth')
#         x_up_angles[0] = x_up_angles[0]
#         x_up_angles[-1] = x_up_angles[-1]
#     else:
#         x_up_angles = txt_interpolation(x_up_angles, frame, mode='linear')

#     if len(y_left_angles) > 3:
#         y_left_angles = txt_interpolation(y_left_angles, frame, mode='smooth')
#         y_left_angles[0] = y_left_angles[0]
#         y_left_angles[-1] = y_left_angles[-1]
#     else:
#         y_left_angles = txt_interpolation(y_left_angles, frame, mode='linear')

#     if len(r) > 3:
#         rs = txt_interpolation(r, frame, mode='smooth')
#         rs[0] = r[0]
#         rs[-1] = r[-1]
#     else:
#         rs = txt_interpolation(r, frame, mode='linear')

#     if len(r_zoom) > 3:
#         r_zooms = txt_interpolation(r_zoom, frame, mode='smooth')
#         r_zooms[0] = r_zoom[0]
#         r_zooms[-1] = r_zoom[-1]
#     else:
#         r_zooms = txt_interpolation(r_zoom, frame, mode='linear')

#     c2ws_list = []
#     # if x_up_anlge is 0, and y_left_angle is 0, the set flag to True
#     # is_all_zero = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
#     is_zoom = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
#     is_not_y = all(y == 0 for y in y_left_angles)
#     for x_up_angle, y_left_angle, r, r_zoom in zip(x_up_angles, y_left_angles, rs, r_zooms):
#         if is_not_y:
#             c2w_new = sphere2pose(
#                 np.float32(x_up_angle), np.float32(y_left_angle), np.float32(r_zoom),
#                 is_zoom=is_zoom
#             )
#         else:
#             c2w_new = sphere2pose(
#                 np.float32(x_up_angle), np.float32(y_left_angle), np.float32(r),
#                 is_zoom=is_zoom
#             )
#         c2ws_list.append(c2w_new)
#     c2ws = np.stack(c2ws_list, axis=0) # N, 4, 4
#     return c2ws # Twc


def sphere2pose(x_up_angle, y_left_angle, r, is_zoom=False, is_translation=False):
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

    # Spherical rotation matrix and translation vector
    R = rot_mat_y @ rot_mat_x
    T = np.array([-r * cos_value_x * sin_value_y, -r * sin_value_x, r - r * cos_value_x * cos_value_y])

    # In translation mode, zero out the rotational translation component
    if is_translation:
        T = np.array([0, 0, 0])

    if is_zoom:
        T = np.array([0, 0, r])

    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = T

    return c2w  # 4x4


def generate_traj_txt(x_up_angles, y_left_angles, r, r_zoom, frame, is_translation=False):
    # Initialize a camera.
    """
    COLMAP coordinate
    """

    if len(x_up_angles) > 3:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode="smooth")
        x_up_angles[0] = x_up_angles[0]
        x_up_angles[-1] = x_up_angles[-1]
    else:
        x_up_angles = txt_interpolation(x_up_angles, frame, mode="linear")

    if len(y_left_angles) > 3:
        y_left_angles = txt_interpolation(y_left_angles, frame, mode="smooth")
        y_left_angles[0] = y_left_angles[0]
        y_left_angles[-1] = y_left_angles[-1]
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
    # if x_up_anlge is 0, and y_left_angle is 0, the set flag to True
    # is_all_zero = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
    is_zoom = all(x == 0 for x in x_up_angles) and all(y == 0 for y in y_left_angles)
    is_not_y = all(y == 0 for y in y_left_angles)

    for x_up_angle, y_left_angle, r_val, r_zoom_val in zip(x_up_angles, y_left_angles, rs, r_zooms, strict=False):
        if is_not_y:
            c2w_new = sphere2pose(
                np.float32(x_up_angle),
                np.float32(y_left_angle),
                np.float32(r_zoom_val),
                is_zoom=is_zoom,
                is_translation=is_translation,
            )
        else:
            c2w_new = sphere2pose(
                np.float32(x_up_angle),
                np.float32(y_left_angle),
                np.float32(r_val),
                is_zoom=is_zoom,
                is_translation=is_translation,
            )
        c2ws_list.append(c2w_new)

    c2ws = np.stack(c2ws_list, axis=0)  # N, 4, 4
    return c2ws  # Twc
