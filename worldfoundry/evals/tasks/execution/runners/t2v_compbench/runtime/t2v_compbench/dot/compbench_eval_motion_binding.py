import csv
import json
import math
import os
import os.path as osp

import numpy as np
import scipy
import torch
from matplotlib import colormaps
from torch import nn
from tqdm import tqdm

from worldfoundry.base_models.perception_core.tracking.dot.models import create_model
from worldfoundry.base_models.perception_core.tracking.dot.utils.io import (
    read_frame,
    read_video,
    write_video,
)

# from worldfoundry.base_models.perception_core.tracking.dot.utils.options.demo_options import DemoOptions
from worldfoundry.base_models.perception_core.tracking.dot.utils.options.compbench_demo_options import DemoOptions
from worldfoundry.base_models.perception_core.tracking.dot.utils.torch import get_grid, to_device


class Visualizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.save_mode = args.save_mode
        self.result_path = args.result_path
        self.overlay_factor = args.overlay_factor
        self.spaghetti_radius = args.spaghetti_radius
        self.spaghetti_length = args.spaghetti_length
        self.spaghetti_grid = args.spaghetti_grid
        self.spaghetti_scale = args.spaghetti_scale
        self.spaghetti_every = args.spaghetti_every
        self.spaghetti_dropout = args.spaghetti_dropout

    def forward(self, data, mode):
        if "overlay" in mode:
            video, x_change, y_change = self.plot_overlay(data, mode)
        elif "spaghetti" in mode:
            video = self.plot_spaghetti(data, mode)
        else:
            raise ValueError(f"Unknown mode {mode}")

        return video, x_change, y_change

    def plot_overlay(self, data, mode):
        T, C, H, W = data["video"].shape
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        tracks = data["tracks"]

        if tracks.ndim == 4:
            col = get_rainbow_colors(int(mask.sum())).cuda()
        else:
            col = get_rainbow_colors(tracks.size(1)).cuda()

        video = []
        ##
        x_change_list = []
        y_change_list = []
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            tgt_frame = data["video"][tgt_step]
            tgt_frame = tgt_frame.permute(1, 2, 0)  # [480,856,3]

            # Plot rainbow points
            tgt_pos = tracks[tgt_step, ..., :2]  # [856,480,2]  tracks [16,856,480,3]
            tgt_vis = tracks[tgt_step, ..., 2]  # [856,480]
            if tracks.ndim == 4:
                tgt_pos = tgt_pos[mask]  # mask [856,480]
                tgt_vis = tgt_vis[mask]
            ##
            rainbow, alpha, x_change, y_change = draw(tgt_pos, tgt_vis, col, H, W)  # col [410880,3]
            x_change_list.append(x_change)
            y_change_list.append(y_change)

            # Plot rainbow points with white stripes in occluded regions
            if "stripes" in mode:
                rainbow_occ, alpha_occ = draw(tgt_pos, 1 - tgt_vis, col, H, W)
                stripes = torch.arange(H).view(-1, 1) + torch.arange(W).view(1, -1)
                stripes = stripes % 9 < 3
                rainbow_occ[stripes] = 1.0
                rainbow = alpha * rainbow + (1 - alpha) * rainbow_occ
                alpha = alpha + (1 - alpha) * alpha_occ

            # Overlay rainbow points over target frame
            tgt_frame = self.overlay_factor * alpha * rainbow + (1 - self.overlay_factor * alpha) * tgt_frame

            # Convert from H W C to C H W
            tgt_frame = tgt_frame.permute(2, 0, 1)
            video.append(tgt_frame)
        video = torch.stack(video)
        return video, x_change_list, y_change_list

    def plot_spaghetti(self, data, mode):
        bg_color = 1.0
        T, C, H, W = data["video"].shape
        G, S, R, L = self.spaghetti_grid, self.spaghetti_scale, self.spaghetti_radius, self.spaghetti_length
        D = self.spaghetti_dropout

        # Extract a grid of tracks
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        mask = mask[G // 2 : -G // 2 + 1 : G, G // 2 : -G // 2 + 1 : G]
        tracks = data["tracks"]
        if tracks.ndim == 4:
            tracks = tracks[:, G // 2 : -G // 2 + 1 : G, G // 2 : -G // 2 + 1 : G]
            tracks = tracks[:, mask]
        elif D > 0:
            N = tracks.size(1)
            assert D < 1
            samples = np.sort(np.random.choice(N, int((1 - D) * N), replace=False))
            tracks = tracks[:, samples]
        col = get_rainbow_colors(tracks.size(1)).cuda()

        # Densify tracks over temporal axis
        tracks = spline_interpolation(tracks, length=L)

        video = []
        cur_frame = None
        cur_alpha = None
        grid = get_grid(H, W).cuda()
        grid[..., 0] *= W - 1
        grid[..., 1] *= H - 1
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            for delta in range(L):
                # Plot rainbow points
                tgt_pos = tracks[tgt_step * L + delta, :, :2]
                tgt_vis = torch.ones_like(tgt_pos[..., 0])
                tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                tgt_col = col.clone()
                rainbow, alpha = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W), radius=R)
                rainbow, alpha = rainbow.cpu(), alpha.cpu()

                # Overlay rainbow points over previous points / frames
                if cur_frame is None:
                    cur_frame = rainbow
                    cur_alpha = alpha
                else:
                    cur_frame = alpha * rainbow + (1 - alpha) * cur_frame
                    cur_alpha = 1 - (1 - cur_alpha) * (1 - alpha)

                plot_first = "first" in mode and tgt_step == 0 and delta == 0
                plot_last = "last" in mode and delta == 0
                plot_every = "every" in mode and delta == 0 and tgt_step % self.spaghetti_every == 0
                if delta == 0:
                    if plot_first or plot_last or plot_every:
                        # Plot target frame
                        tgt_col = data["video"][tgt_step].permute(1, 2, 0).reshape(-1, 3)
                        tgt_pos = grid.view(-1, 2)
                        tgt_vis = torch.ones_like(tgt_pos[..., 0])
                        tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                        tgt_frame, alpha_frame = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W))
                        tgt_frame, alpha_frame = tgt_frame.cpu(), alpha_frame.cpu()

                        # Overlay target frame over previous points / frames
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * cur_frame
                        alpha_frame = 1 - (1 - cur_alpha) * (1 - alpha_frame)

                        # Add last points on top
                        tgt_frame = alpha * rainbow + (1 - alpha) * tgt_frame
                        alpha_frame = 1 - (1 - alpha_frame) * (1 - alpha)

                        # Set background color
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * torch.ones_like(tgt_frame) * bg_color

                        if plot_first or plot_every:
                            cur_frame = tgt_frame
                            cur_alpha = alpha_frame
                    else:
                        tgt_frame = cur_alpha * cur_frame + (1 - cur_alpha) * torch.ones_like(cur_frame) * bg_color

                    # Convert from H W C to C H W
                    tgt_frame = tgt_frame.permute(2, 0, 1)

                    # Translate everything to make the target frame look static
                    if "static" in mode:
                        end_pos = project(torch.tensor([[0, 0]]), T * L, T * L, H, W)[0]
                        cur_pos = project(torch.tensor([[0, 0]]), tgt_step * L + delta, T * L, H, W)[0]
                        delta_pos = S * (end_pos - cur_pos)
                        tgt_frame = translation(tgt_frame, delta_pos[0], delta_pos[1], bg_color)
                    video.append(tgt_frame)
        video = torch.stack(video)
        return video


def translation(frame, dx, dy, pad_value):
    C, H, W = frame.shape
    grid = get_grid(H, W, device=frame.device)
    grid[..., 0] = grid[..., 0] - (dx / (W - 1))
    grid[..., 1] = grid[..., 1] - (dy / (H - 1))
    frame = frame - pad_value
    frame = torch.nn.functional.grid_sample(frame[None], grid[None] * 2 - 1, mode="bilinear", align_corners=True)[0]
    frame = frame + pad_value
    return frame


def spline_interpolation(x, length=10):
    if length != 1:
        T, N, C = x.shape
        x = x.view(T, -1).cpu().numpy()
        original_time = np.arange(T)
        cs = scipy.interpolate.CubicSpline(original_time, x)
        new_time = np.linspace(original_time[0], original_time[-1], T * length)
        x = torch.from_numpy(cs(new_time)).view(-1, N, C).float().cuda()
    return x


def get_rainbow_colors(size):
    col_map = colormaps["jet"]
    col_range = np.array(range(size)) / (size - 1)
    col = torch.from_numpy(col_map(col_range)[..., :3]).float()
    col = col.view(-1, 3)
    return col


def draw(pos, vis, col, height, width, radius=1):
    H, W = height, width
    frame = torch.zeros(H * W, 4, device=pos.device)
    pos = pos[vis.bool()]

    if pos.shape[0] != 0:
        y_change = sum(pos[:, 1]) / pos.shape[0]
        x_change = sum(pos[:, 0]) / pos.shape[0]
    else:
        y_change = torch.tensor(-10000)
        x_change = torch.tensor(-10000)

    col = col[vis.bool()]
    if radius > 1:
        pos, col = get_radius_neighbors(pos, col, radius)
    else:
        pos, col = get_cardinal_neighbors(pos, col)
    inbound = (pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1)
    pos = pos[inbound]
    col = col[inbound]
    pos = pos.round().long()
    idx = pos[:, 1] * W + pos[:, 0]
    idx = idx.view(-1, 1).expand(-1, 4)
    frame.scatter_add_(0, idx, col)
    frame = frame.view(H, W, 4)
    frame, alpha = frame[..., :3], frame[..., 3]
    nonzero = alpha > 0
    frame[nonzero] /= alpha[nonzero][..., None]
    alpha = nonzero[..., None].float()
    return frame, alpha, x_change.cpu().item(), y_change.cpu().item()


def get_cardinal_neighbors(pos, col, eps=0.01):
    pos_nw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor()], dim=-1)
    pos_sw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor() + 1], dim=-1)
    pos_ne = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor()], dim=-1)
    pos_se = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor() + 1], dim=-1)
    w_n = pos[:, 1].floor() + 1 - pos[:, 1] + eps
    w_s = pos[:, 1] - pos[:, 1].floor() + eps
    w_w = pos[:, 0].floor() + 1 - pos[:, 0] + eps
    w_e = pos[:, 0] - pos[:, 0].floor() + eps
    w_nw = (w_n * w_w)[:, None]
    w_sw = (w_s * w_w)[:, None]
    w_ne = (w_n * w_e)[:, None]
    w_se = (w_s * w_e)[:, None]
    col_nw = torch.cat([w_nw * col, w_nw], dim=-1)
    col_sw = torch.cat([w_sw * col, w_sw], dim=-1)
    col_ne = torch.cat([w_ne * col, w_ne], dim=-1)
    col_se = torch.cat([w_se * col, w_se], dim=-1)
    pos = torch.cat([pos_nw, pos_sw, pos_ne, pos_se], dim=0)
    col = torch.cat([col_nw, col_sw, col_ne, col_se], dim=0)
    return pos, col


def get_radius_neighbors(pos, col, radius):
    R = math.ceil(radius)
    center = torch.stack([pos[:, 0].round(), pos[:, 1].round()], dim=-1)
    nn = torch.arange(-R, R + 1)
    nn = torch.stack([nn[None, :].expand(2 * R + 1, -1), nn[:, None].expand(-1, 2 * R + 1)], dim=-1)
    nn = nn.view(-1, 2).cuda()
    in_radius = nn[:, 0] ** 2 + nn[:, 1] ** 2 <= radius**2
    nn = nn[in_radius]
    w = 1 - nn.pow(2).sum(-1).sqrt() / radius + 0.01
    w = w[None].expand(pos.size(0), -1).reshape(-1)
    pos = (center.view(-1, 1, 2) + nn.view(1, -1, 2)).view(-1, 2)
    col = col.view(-1, 1, 3).repeat(1, nn.size(0), 1)
    col = col.view(-1, 3)
    col = torch.cat([col * w[:, None], w[:, None]], dim=-1)
    return pos, col


def project(pos, t, time_steps, heigh, width):
    T, H, W = time_steps, heigh, width
    pos = torch.stack([pos[..., 0] / (W - 1), pos[..., 1] / (H - 1)], dim=-1)
    pos = pos - 0.5
    pos = pos * 0.25
    t = 1 - torch.ones_like(pos[..., :1]) * t / (T - 1)
    pos = torch.cat([pos, t], dim=-1)
    M = torch.tensor([[0.8, 0, 0.5], [-0.2, 1.0, 0.1], [0.0, 0.0, 0.0]])
    pos = pos @ M.t().to(pos.device)
    pos = pos[..., :2]
    pos[..., 0] += 0.25
    pos[..., 1] += 0.45
    pos[..., 0] *= W - 1
    pos[..., 1] *= H - 1
    return pos


def combine_fore_back(foreground, background, output_csv):
    back_x = []
    back_y = []
    with open(background) as file2:
        reader2 = csv.reader(file2)
        background_lines = list(reader2)
        for i in range(len(background_lines) - 1):
            back_x.append(background_lines[i + 1][-2])
            back_y.append(background_lines[i + 1][-1])

    with open(output_csv, "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["id", "prompt", "object_1", "d_1", "object_2", "d_2", "obj_net_left", "obj_net_up"])

        with open(foreground) as file1:
            reader1 = csv.reader(file1)
            foreground_lines = list(reader1)

            for j in range(len(foreground_lines) - 1):
                if foreground_lines[j + 1][-2] != "" and foreground_lines[j + 1][-1] != "":
                    fore_x = foreground_lines[j + 1][-2]
                    fore_y = foreground_lines[j + 1][-1]
                    num = math.ceil((j + 1) / 2) - 1
                    try:
                        obj_net_left = float(back_x[num]) - float(fore_x)
                        obj_net_up = float(back_y[num]) - float(fore_y)
                    except BaseException:  # back_x[num] or fore_x is ''
                        obj_net_left = 0.0
                        obj_net_up = 0.0
                    row = [*foreground_lines[j + 1][:6], obj_net_left, obj_net_up]
                    writer.writerow(row)
                    output_file.flush()

                elif foreground_lines[j + 1][-2] == "" and foreground_lines[j + 1][-1] == "":
                    row = [*foreground_lines[j + 1][:6], "", ""]
                    writer.writerow(row)
                    output_file.flush()
                else:
                    print("no x_change or no y_change")


def object_score(obj1_net_left, left_thresh, obj1_net_up, up_thresh, d_1):
    W = 856
    H = 480
    correct_direction = False
    score_tmp = 0
    obj1_net_left = float(obj1_net_left)
    obj1_net_up = float(obj1_net_up)
    if d_1 == "left":
        if obj1_net_left > left_thresh:
            correct_direction = True
            score_tmp = abs(obj1_net_left) / W
    elif d_1 == "right":
        if obj1_net_left < -left_thresh:
            correct_direction = True
            score_tmp = abs(obj1_net_left) / W
    elif d_1 == "up":
        if obj1_net_up > up_thresh:
            correct_direction = True
            score_tmp = abs(obj1_net_up) / H
    elif d_1 == "down":
        if obj1_net_up < -up_thresh:
            correct_direction = True
            score_tmp = abs(obj1_net_up) / H
    else:
        print("direction not in [left, right, up, down]")
    return correct_direction, score_tmp


def cal_score(output_csv, score_csv):
    # mid point y:240, x:428  height = 480, width = 856
    left_thresh = 0
    up_thresh = 0
    id = []
    score = []

    with open(score_csv, "w") as score_file:
        writer = csv.writer(score_file)
        writer.writerow(["id", "object_1", "d_1", "object_2", "d_2", "Score"])

        with open(output_csv) as file1:
            reader1 = csv.reader(file1)
            lines = list(reader1)
            vid_num = (len(lines) - 1) // 2
            for i in range(vid_num):
                id = lines[i * 2 + 1][0]
                d_1 = lines[i * 2 + 1][3]
                d_2 = lines[i * 2 + 2][5]
                obj1 = lines[i * 2 + 1][2]
                obj2 = lines[i * 2 + 2][4]
                obj1_net_left = lines[i * 2 + 1][6]
                obj1_net_up = lines[i * 2 + 1][7]
                obj2_net_left = lines[i * 2 + 2][6]
                obj2_net_up = lines[i * 2 + 2][7]
                correct_direction = False
                score_tmp = 0

                if d_1 != "" and d_2 == "":  # only 1 object
                    if obj1_net_left != "":  # 1 object detected
                        correct_direction, score_tmp = object_score(
                            obj1_net_left, left_thresh, obj1_net_up, up_thresh, d_1
                        )
                    elif obj1_net_left == "":  # 1 object not detected
                        score_tmp = -1
                    else:
                        print("only 1 object detected, error in obj1_net_left")

                elif d_1 != "" and d_2 != "":  # 2 objects
                    if obj1_net_left != "":  # 1st object detected
                        correct_direction_1, score_tmp_1 = object_score(
                            obj1_net_left, left_thresh, obj1_net_up, up_thresh, d_1
                        )
                    elif obj1_net_left == "":  # 1st object not detected
                        correct_direction_1 = False
                        score_tmp_1 = -1
                    else:
                        print("2 objects detected, error in obj1_net_left")

                    if obj2_net_left != "":  # 2nd object detected
                        correct_direction_2, score_tmp_2 = object_score(
                            obj2_net_left, left_thresh, obj2_net_up, up_thresh, d_2
                        )
                    elif obj2_net_left == "":  # 2nd object not detected
                        correct_direction_2 = False
                        score_tmp_2 = -1
                    else:
                        print("2 objects detected, error in obj2_net_left")

                    if correct_direction_1 and correct_direction_2:
                        score_tmp = (score_tmp_1 + score_tmp_2) * 0.5
                    elif correct_direction_1 and not correct_direction_2:
                        score_tmp = 0.5 * score_tmp_1
                    elif not correct_direction_1 and correct_direction_2:
                        score_tmp = 0.5 * score_tmp_2
                    elif not correct_direction_1 and not correct_direction_2:
                        if score_tmp_1 < 0 and score_tmp_2 < 0:  # noqa: SIM108 - Commented branches; 2 object neither detected
                            score_tmp = -1
                        else:
                            score_tmp = 0  # 1 or 2 among 2 objects detected, wrong direction/static
                    else:
                        print("error in direction judgement")

                else:
                    print("wrong record in d_1 or d_2")

                score.append(score_tmp)
                writer.writerow([id, obj1, d_1, obj2, d_2, score_tmp])


def model_score(csv_path):
    new_csv = []
    with open(csv_path) as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0  # neither detected: -1, detected: motion score 0~1, total scale: -1 ~ 1
        cnt = 0
        score_pos = 0
        cnt_pos = 0

        new_header = lines[0]
        new_header[-1] = "Score_before_adjust"
        new_header.append("Score")
        new_csv.append(new_header)

        for line in lines[1:]:
            score_tmp = float(line[-1])

            if score_tmp < 0:
                score_tmp = 0
            elif score_tmp >= 0:
                score_pos += score_tmp
                score_tmp = (score_tmp * 0.8) + 0.2
                cnt_pos += 1

            new_line = line
            # The above code is a Python comment. Comments in Python start with a hash symbol (#) and
            # are used to provide explanations or notes within the code. In this case, the comment is
            # indicating that the code is creating a new line.
            new_line.append(score_tmp)
            new_csv.append(new_line)

            score += score_tmp

            cnt += 1

        score = score / cnt
        score_pos = score_pos / cnt_pos
        print("score: ", score)

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(new_csv)
        writer.writerow(["score: ", score])


def background(args):
    output_dir = os.path.join(args.output_dir, args.t2v_model)
    output_path = args.output_path
    video_folder = args.video_path
    mask_folder = os.path.join(args.mask_folder, args.t2v_model)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    model = create_model(args).cuda()
    visualizer = Visualizer(args).cuda()
    resolution = (args.height, args.width)

    csv_path = f"{output_path}/{args.t2v_model}_background.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            csv_writer.writerow(
                [
                    "id",
                    "prompt",
                    "object_1",
                    "d_1",
                    "object_2",
                    "d_2",
                    "mask_name",
                    "xy_json",
                    "change_in_x",
                    "change_in_y",
                ]
            )

        videos = os.listdir(video_folder)
        videos.sort(key=lambda x: int(x.split(".")[0]))

        evaluated = max(line_count - 1, 0)

        for i in range(evaluated, len(videos)):
            video_name = videos[i]
            ind = int(video_name[0:4]) - 1
            vid = video_name

            save_prefix = osp.join(output_dir, vid.split(".")[0])
            os.makedirs(save_prefix, exist_ok=True)

            args.result_path = save_prefix
            tracks_path = osp.join(args.result_path, "tracks.pth")

            this_prompt = prompts[ind]["prompt"]
            object_1 = prompts[ind]["object_1"]  # A is on the left of B
            object_2 = prompts[ind]["object_2"]
            d_1 = prompts[ind]["d_1"]
            d_2 = prompts[ind]["d_2"]
            directions = ["left", "right", "up", "down", ""]
            if d_1 not in directions[:4] or d_2 not in directions:
                print(d_1, d_2, " direction not included!!!, index: ", vid)
                break

            video = read_video(osp.join(video_folder, vid), resolution=resolution).cuda()  # , time_steps=20

            if not osp.exists(tracks_path) or args.recompute_tracks:
                with torch.no_grad():
                    pred = model({"video": video[None]}, mode=args.inference_mode, **vars(args))
                tracks = pred["tracks"][0]
                if args.save_tracks:
                    torch.save(tracks.cpu(), tracks_path)
            else:
                tracks = torch.load(tracks_path)

            masks = os.listdir(osp.join(mask_folder, vid.split(".")[0]))
            real_masks = []
            for file_name in masks:
                if file_name.split(".")[0] == "mask_background" and file_name.split(".")[-1] == "jpg":
                    real_masks.append(file_name)
            print(real_masks)

            if len(real_masks) != 0:
                for mask_name in real_masks:
                    mask_path = osp.join(mask_folder, vid.split(".")[0], mask_name)
                    if any(["mask" in mode] for mode in args.visualization_modes) and osp.exists(mask_path):
                        mask = read_frame(mask_path, resolution=resolution)[0] > 0.5
                    else:
                        mask = torch.ones(args.height, args.width).bool()

                    data = {"video": video, "tracks": tracks, "mask": mask}

                    data = to_device(data, "cuda")

                    if data["tracks"].ndim == 4 and args.rainbow_mode == "left_right":
                        data["mask"] = data["mask"].permute(1, 0)
                        data["tracks"] = data["tracks"].permute(0, 2, 1, 3)
                    elif data["tracks"].ndim == 3:
                        points = data["tracks"][0]
                        x, y = points[..., 0].long(), points[..., 1].long()
                        x, y = x - x.min(), y - y.min()
                        idx = y + x * y.max() if args.rainbow_mode == "left_right" else x + y * x.max()
                        order = idx.argsort(dim=0)
                        data["tracks"] = data["tracks"][:, order]

                    for mode in args.visualization_modes:
                        video, x_change_list, y_change_list = visualizer(data, mode=mode)

                    for cnt in range(len(x_change_list)):
                        last_x = x_change_list[-(cnt + 1)]
                        last_y = y_change_list[-(cnt + 1)]
                        if (last_x == -10000 and last_y != -10000) or (last_y == -10000 and last_x != -10000):
                            print("NO WAY")
                            break
                        if last_x != -10000 or last_y != -10000:
                            print("last? ", -(cnt + 1))
                            break
                    change_in_x = last_x - x_change_list[0]
                    change_in_y = last_y - y_change_list[0]

                    xy_json = {"x_change_list": x_change_list, "y_change_list": y_change_list}

                    save_path = osp.join(save_prefix, mask_name.split(".")[0] + ".mp4")
                    write_video(video, save_path)
                    csv_writer.writerow(
                        [
                            vid.split(".")[0],
                            this_prompt,
                            object_1,
                            d_1,
                            object_2,
                            d_2,
                            mask_name,
                            xy_json,
                            change_in_x,
                            change_in_y,
                        ]
                    )
                    csvfile.flush()
            else:
                csv_writer.writerow([vid.split(".")[0], this_prompt, object_1, d_1, object_2, d_2, "", "", "", ""])
                csvfile.flush()

    background_csv = f"{output_path}/{args.t2v_model}_background.csv"
    return background_csv


def foreground(args):
    output_dir = os.path.join(args.output_dir, args.t2v_model)
    output_path = args.output_path
    video_folder = args.video_path
    mask_folder = os.path.join(args.mask_folder, args.t2v_model)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    model = create_model(args).cuda()
    visualizer = Visualizer(args).cuda()
    resolution = (args.height, args.width)

    csv_path = f"{output_path}/{args.t2v_model}_foreground.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            csv_writer.writerow(
                [
                    "id",
                    "prompt",
                    "object_1",
                    "d_1",
                    "object_2",
                    "d_2",
                    "mask_name",
                    "xy_json",
                    "change_in_x",
                    "change_in_y",
                ]
            )

        videos = os.listdir(video_folder)
        videos.sort(key=lambda x: int(x.split(".")[0]))

        evaluated = max(line_count - 1, 0)

        for i in range(evaluated, len(videos)):
            video_name = videos[i]
            ind = int(video_name[0:4]) - 1
            vid = video_name

            this_prompt = prompts[ind]["prompt"]
            object_1 = prompts[ind]["object_1"]  # A is on the left of B
            object_2 = prompts[ind]["object_2"]
            d_1 = prompts[ind]["d_1"]
            d_2 = prompts[ind]["d_2"]
            directions = ["left", "right", "up", "down", ""]
            if d_1 not in directions[:4] or d_2 not in directions:
                print(d_1, d_2, " direction not included!!!, index: ", vid)
                break

            masks = os.listdir(osp.join(mask_folder, vid.split(".")[0]))
            real_masks = []
            for file_name in masks:
                if file_name.split(".")[-1] == "jpg" and file_name.split("_")[1] == "foreground":
                    real_masks.append(file_name)

            print(real_masks)
            if len(real_masks) != 0:
                for mask_name in real_masks:
                    save_prefix = osp.join(output_dir, vid.split(".")[0])
                    os.makedirs(save_prefix, exist_ok=True)

                    args.result_path = save_prefix
                    tracks_path = osp.join(args.result_path, "tracks.pth")

                    video = read_video(osp.join(video_folder, vid), resolution=resolution).cuda()  # , time_steps=20

                    if not osp.exists(tracks_path) or args.recompute_tracks:
                        with torch.no_grad():
                            pred = model({"video": video[None]}, mode=args.inference_mode, **vars(args))
                        tracks = pred["tracks"][0]
                        if args.save_tracks:
                            torch.save(tracks.cpu(), tracks_path)
                    else:
                        tracks = torch.load(tracks_path)

                    mask_path = osp.join(mask_folder, vid.split(".")[0], mask_name)
                    if any(["mask" in mode] for mode in args.visualization_modes) and osp.exists(mask_path):
                        mask = read_frame(mask_path, resolution=resolution)[0] > 0.5
                    else:
                        mask = torch.ones(args.height, args.width).bool()

                    data = {"video": video, "tracks": tracks, "mask": mask}

                    data = to_device(data, "cuda")

                    if data["tracks"].ndim == 4 and args.rainbow_mode == "left_right":
                        data["mask"] = data["mask"].permute(1, 0)
                        data["tracks"] = data["tracks"].permute(0, 2, 1, 3)
                    elif data["tracks"].ndim == 3:
                        points = data["tracks"][0]
                        x, y = points[..., 0].long(), points[..., 1].long()
                        x, y = x - x.min(), y - y.min()
                        idx = y + x * y.max() if args.rainbow_mode == "left_right" else x + y * x.max()
                        order = idx.argsort(dim=0)
                        data["tracks"] = data["tracks"][:, order]

                    for mode in args.visualization_modes:
                        video, x_change_list, y_change_list = visualizer(data, mode=mode)

                    for cnt in range(len(x_change_list)):
                        last_x = x_change_list[-(cnt + 1)]
                        last_y = y_change_list[-(cnt + 1)]
                        if (last_x == -10000 and last_y != -10000) or (last_y == -10000 and last_x != -10000):
                            print("NO WAY")
                            break
                        if last_x != -10000 or last_y != -10000:
                            print("last? ", -(cnt + 1))
                            break
                    change_in_x = last_x - x_change_list[0]
                    change_in_y = last_y - y_change_list[0]

                    xy_json = {"x_change_list": x_change_list, "y_change_list": y_change_list}

                    save_path = osp.join(save_prefix, mask_name.split(".")[0] + ".mp4")
                    write_video(video, save_path)
                    csv_writer.writerow(
                        [
                            vid.split(".")[0],
                            this_prompt,
                            object_1,
                            d_1,
                            object_2,
                            d_2,
                            mask_name,
                            xy_json,
                            change_in_x,
                            change_in_y,
                        ]
                    )
                    csvfile.flush()

                if len(real_masks) == 1:
                    csv_writer.writerow(
                        [vid.split(".")[0], this_prompt, object_1, d_1, object_2, d_2, "", "", "", ""]
                    )  # the 2nd object
                    csvfile.flush()

            else:
                csv_writer.writerow([vid.split(".")[0], this_prompt, object_1, d_1, object_2, d_2, "", "", "", ""])
                csvfile.flush()
                csv_writer.writerow([vid.split(".")[0], this_prompt, object_1, d_1, object_2, d_2, "", "", "", ""])
                csvfile.flush()

    foreground_csv = f"{output_path}/{args.t2v_model}_foreground.csv"
    return foreground_csv


if __name__ == "__main__":
    args = DemoOptions().parse_args()

    foreground_csv = foreground(args)
    background_csv = background(args)

    output_csv = os.path.join(args.output_path, f"{args.t2v_model}_motion_back_fore.csv")
    combine_fore_back(foreground_csv, background_csv, output_csv)

    # score of each video is recorded in score_csv
    score_csv = os.path.join(args.output_path, f"{args.t2v_model}_motion_score.csv")
    cal_score(output_csv, score_csv)

    # final model score printed out and recorded in the last line of score_csv
    model_score(score_csv)

    print(foreground_csv)
    print(background_csv)
    print(score_csv)
    print("Done.")
