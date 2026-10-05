import argparse
import os
import cv2
import numpy as np
import torch
import time
import logging
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.base_models.perception_core.optical_flow.raft import (
    InputPadder,
    RAFT,
    checkpoint_path,
)


def _evalcrafter_assets_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_EVALCRAFTER_ASSETS_ROOT", bundled_benchmark_asset("evalcrafter"))).resolve()


def tensor2img(img_t: torch.Tensor) -> np.ndarray:
    img = img_t[0].detach().cpu().numpy()
    return np.transpose(img, (1, 2, 0))


def img2tensor(img: np.ndarray, device: torch.device | str) -> torch.Tensor:
    img_t = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    return torch.from_numpy(img_t.astype(np.float32)).to(device)


def _flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp BCHW tensor ``x`` with pixel-space optical flow BCHW(2)."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).expand(b, h, w, 2)
    flow_grid = base_grid + flow.permute(0, 2, 3, 1)
    flow_grid[..., 0] = 2.0 * flow_grid[..., 0] / max(w - 1, 1) - 1.0
    flow_grid[..., 1] = 2.0 * flow_grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, flow_grid, mode="bilinear", padding_mode="border", align_corners=True)


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2


def compute_flow_gradients(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = flow.shape[:2]
    flow_x_du = np.zeros((h, w))
    flow_x_dv = np.zeros((h, w))
    flow_y_du = np.zeros((h, w))
    flow_y_dv = np.zeros((h, w))
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]
    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow: np.ndarray, bw_flow: np.ndarray, img: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        fw_flow_t = img2tensor(fw_flow, img.device)
        bw_flow_t = img2tensor(bw_flow, img.device)
        fw_flow_w = tensor2img(_flow_warp(fw_flow_t, bw_flow_t))
        warp_img = _flow_warp(img, bw_flow_t)

    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[np.logical_or(mask1, mask2)] = 1
    return occlusion, warp_img


def calculate_flow_score(video_path, model):
    
    # Create an output directory
    # output_dir = "output"
    # os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)

    cap.release()

    optical_flows = []

    with torch.no_grad():
        for i in range(len(frames) - 1):
            image1 = frames[i]
            image2 = frames[i + 1]

            image1 = torch.tensor(image1).permute(2,0,1).float().unsqueeze(0).to(device)
            image2 = torch.tensor(image2).permute(2,0,1).float().unsqueeze(0).to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # Compute the magnitude of optical flow vectors
            flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
            # Calculate the mean optical flow value for the current pair of frames
            mean_optical_flow = flow_magnitude.mean().item()
            optical_flows.append(mean_optical_flow)

    # Calculate the average optical flow for the entire video
    mean_optical_flow_video = np.mean(optical_flows)
    # mean_optical_flow_video = np.mean(optical_flows)
    print(f"Mean optical flow for the video: {mean_optical_flow_video}")

    return mean_optical_flow_video

def calculate_motion_ac_score(video_path, amp, model):
    
    # Create an output directoryﬁ
    # output_dir = "output"
    # os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)

    cap.release()

    optical_flows = []

    with torch.no_grad():
        for i in range(len(frames) - 1):
            image1 = frames[i]
            image2 = frames[i + 1]

            image1 = torch.tensor(image1).permute(2,0,1).float().unsqueeze(0).to(device)
            image2 = torch.tensor(image2).permute(2,0,1).float().unsqueeze(0).to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            # Compute the magnitude of optical flow vectors
            flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
            # Calculate the mean optical flow value for the current pair of frames
            mean_optical_flow = flow_magnitude.mean().item()
            optical_flows.append(mean_optical_flow)

    # Calculate the average optical flow for the entire video
    mean_optical_flow_video = np.mean(optical_flows)
    print(f"Mean optical flow for the video: {mean_optical_flow_video}")
    if np.abs(mean_optical_flow_video) > 5:
        amp_pred = 'large'
    else:
        amp_pred = 'slow'

    if amp_pred == amp: # may use a distance to 3?
        amp_recognition_score = 1
    else:
        amp_recognition_score = 0 

    return amp_recognition_score

# Adapted from https://github.com/phoenix104104/fast_blind_video_consistency
def compute_video_warping_error(video_path, model):

    cap = cv2.VideoCapture(video_path)
    frames = []
    warping_error = 0
    err = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)
    
    # Num = 5
    Num = len(frames)
    tensor_frames = torch.stack([torch.from_numpy(frame) for frame in frames])
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    with torch.no_grad():
        for i in range(Num - 1):
            frame1 = extracted_frames[i]
            frame2 = extracted_frames[i + 1]

            # Calculate optical flow using Farneback method
            img1 = frame1.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            img2 = frame2.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            # img1 = torch.tensor(img2tensor(frame1)).float().to(device)
            # img2 = torch.tensor(img2tensor(frame2)).float().to(device)

            # Downsample the images by a factor of 2
            img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            ### compute fw flow
            
            _, fw_flow = model(img1, img2, iters=20, test_mode=True) # with optical flow model: RAFT
            fw_flow = tensor2img(fw_flow)
            ### compute bw flow
            _, bw_flow = model(img2, img1, iters=20, test_mode=True) # with optical flow model: RAFT
            bw_flow = tensor2img(bw_flow)

            ### compute occlusion
            fw_occ, warp_img2 = detect_occlusion(bw_flow, fw_flow, img2)
            fw_occ = torch.tensor(fw_occ).float().to(device)

            ### load flow
            flow = fw_flow

            ### load occlusion mask
            occ_mask = fw_occ
            noc_mask = 1 - occ_mask
            diff = (warp_img2- img1) * noc_mask
            diff_squared = diff ** 2

            
            # Calculate the sum and mean
            N = torch.sum(noc_mask)
            if N == 0:
                N = diff_squared.numel()
            err += torch.sum(diff_squared) / N

    warping_error = err / (len(extracted_frames) - 1)

    return warping_error


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    parser.add_argument('--model', type=str, default=str(checkpoint_path()), help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric

    assets_root = _evalcrafter_assets_root()
    dir_prompts = assets_root / "prompts"
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [str(dir_prompts / (os.path.splitext(os.path.basename(x))[0] + '.txt')) for x in video_paths]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model = CLIPModel.from_pretrained("../../checkpoints/clip-vit-base-patch32").to(device)
    # clip_tokenizer = AutoTokenizer.from_pretrained("../../checkpoints/clip-vit-base-patch32")
    
    import json
    # Load the JSON data from the file
    with open(assets_root / "metadata.json", "r") as infile:
        data = json.load(infile)
    # Extract the dictionaries
    face_vid = {}
    text_vid = {}
    color_vid = {}
    count_vid = {}
    amp_vid = {}
    action_vid = {}
    for item_key, item_value in data.items():
        attributes = item_value["attributes"]
        face = attributes.get("face", "")
        text = attributes.get("text", "")
        color = attributes.get("color", "")
        count = attributes.get("count", "")
        amp = attributes.get("amp", "")
        action = attributes.get("action", "")
        if face:
            face_vid[item_key] = face
        if text:
            text_vid[item_key] = text
        if color:
            color_vid[item_key] = color
        if count:
            count_vid[item_key] = count
        if amp:
            amp_vid[item_key] = amp
        if action:
            action_vid[item_key] = action

    if metric == 'motion_ac_score':
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()
        model.args.mixed_precision = False

    elif metric == 'flow_score':
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()
        model.args.mixed_precision = False
    
    elif metric == 'warping_error':
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()
        model.args.mixed_precision = False


    # Calculate SD scores for all video-text pairs
    scores = []

    test_num = 10
    test_num = len(video_paths)
    count = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt_path = prompt_paths[i]
        if count == test_num:
            break
        else:
            text = read_text_file(prompt_path)
            if metric == 'motion_ac_score':
                # get the videos' basenames list action_vid  for recognition
                basename = os.path.basename(video_path)[:4]
                if  basename in amp_vid.keys():
                    score = calculate_motion_ac_score(video_path, amp_vid[os.path.basename(video_path)[:4]], model)
                else:
                    score = None
            elif metric == 'flow_score':
                # get the videos' basenames list action_vid  for recognition
                # basename = os.path.basename(video_path)[:4]
                score = calculate_flow_score(video_path, model)

            elif metric == 'warping_error':
                # get the videos' basenames list action_vid  for recognition
                basename = os.path.basename(video_path)[:4]
                score = compute_video_warping_error(video_path, model)
            if score is not None:
                scores.append(score)
                count+=1
                average_score = sum(scores) / len(scores)
                logger.info(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score} ")
                # wandb.log({
                #     f"Current {metric}": score,
                #     f"Average {metric}": average_score,
                # })
            
    # Calculate the average SD score across all video-text pairs
    average_score = sum(scores) / len(scores)
    logger.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")
