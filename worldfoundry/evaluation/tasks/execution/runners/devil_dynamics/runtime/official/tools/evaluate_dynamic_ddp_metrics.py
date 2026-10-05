import os
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
import pdb
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop


import numpy as np
import argparse
import torch.nn as nn
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore")
import timm
import time
import tqdm
import pandas as pd
from metrics_utils.cal_ssim import cal_ssim_dist, cal_ssim_dist_chunk
from metrics_utils.standard_video_dataset import StandardVidoDataset, standard_collate_fn
from worldfoundry.base_models.perception_core.optical_flow.raft.raft import RAFT
from worldfoundry.base_models.perception_core.optical_flow.raft.utils.utils import InputPadder
from viclip import get_viclip, retrieve_text, _frame_from_video, frames2tensor, get_vid_feat
from PIL import Image
import subprocess
from sklearn.linear_model import LinearRegression

import random

from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones as dinov2_backbones


RUNTIME_ROOT = Path(__file__).resolve().parents[1]


def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _load_torch_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload is not a state dict")
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in payload.items()
        if isinstance(key, str)
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference with a pretrained DINO model on DiDemo dataset.")
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save extracted features.')
    parser.add_argument('--dynamic_score_save_name', type=str, default='dynamics_merged_results.xlsx')
    parser.add_argument('--regress_model_weight_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_REGRESSION_CKPT', 'WORLDFOUNDRY_DEVIL_LINEAR_REGRESSION_CKPT', 'WORLDFOUNDRY_DEVIL_LINEAR_REGRESS_MODEL'))
    parser.add_argument('--raft_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_RAFT_CKPT', 'WORLDFOUNDRY_RAFT_THINGS_CKPT'))
    parser.add_argument('--clip_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_CLIP_VIT_L14_PT', 'WORLDFOUNDRY_VBENCH_CLIP_VIT_L14_PT'))
    parser.add_argument('--viclip_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_VICLIP_CKPT', 'WORLDFOUNDRY_VBENCH_VICLIP_CKPT'))
    parser.add_argument('--dinov2_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_DINOV2_VITL14_CKPT', 'WORLDFOUNDRY_WBENCH_MEGASAM_DINOV2_CKPT'))
    parser.add_argument('--timm_dino_model_name', type=str, default=os.environ.get('WORLDFOUNDRY_DEVIL_TIMM_DINO_MODEL_NAME', 'vit_small_patch16_224_dino'))
    parser.add_argument('--timm_dino_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_TIMM_DINO_CKPT'))
    parser.add_argument('--temporal_info_script', type=str, default=str(RUNTIME_ROOT / 'cal_temporal_info.sh'))
    
    return parser.parse_args()


def cal_p99_1(data):
    # 计算1%和99%的百分位数
    p1 = np.percentile(data, 1)
    p99 = np.percentile(data, 99)

    # 计算动态范围
    dynamic_range_percentile = p99 - p1
    return dynamic_range_percentile


class RegressionModel:
    def __init__(self, model_path):
        # Load models from a .pth file
        self.models = _load_torch_payload(model_path)
    
    def predict(self, input_inter_frame, input_inter_segment, input_video_level):
        # Predict using the loaded models and clamp the values
        prediction_inter_frame = np.clip(self.models['inter_frame'].predict(input_inter_frame), 0, 1)
        prediction_inter_segment = np.clip(self.models['inter_segment'].predict(input_inter_segment), 0, 1)
        prediction_video_level = np.clip(self.models['video_level'].predict(input_video_level), 0, 1)
        # Calculate the mean of predictions for each entry
        return np.mean([prediction_inter_frame, prediction_inter_segment, prediction_video_level], axis=0)


def json_to_xlsx(json_file, xlsx_file):
    # 读取 JSON 数据
    df = pd.read_json(json_file)  # 假设 JSON 文件每行一个 JSON 对象
    # 将 DataFrame 保存到 XLSX 文件
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)  # 不保存行索引

def save_dicts_to_excel(dicts, filename):
    # 获取所有字典中的所有键
    df_list = pd.DataFrame(dicts)

    # Save the DataFrame to an Excel file
    df_list.to_excel(filename, index=False)

    print(f"Dictionaries saved to '{filename}'.")


def load_and_merge(pth_folder, save_folder, dynamic_score_save_name):
    all_datas = []
    for file in os.listdir(pth_folder):
        if file.endswith(".pth"):
            path = os.path.join(pth_folder, file)
            all_datas.extend(torch.load(path, map_location='cpu'))
    save_dicts_to_excel(all_datas, os.path.join(save_folder, dynamic_score_save_name))


def cal_info_variance(model, video_data, video_lengths):
    features = model(video_data)
    # 处理提取的特征,例如保存到文件或进一步处理
    
    features = F.normalize(features, dim=-1, p=2)
    features = features.split(video_lengths)
    features_mean = [feat.mean(axis=0, keepdim=True) for feat in features]
    # features_normed = features - features_mean
    distances = [(1 - F.cosine_similarity(feat_mean[:, None], feat[None], dim=-1)).mean(1).item() for feat, feat_mean in zip(features, features_mean)]

    return distances, features

# def cal_clip_info()

def build_raft(model_path, rank):
    args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    model = RAFT(args_new)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = dict()
    for key in state_dict:
        new_state_dict[key.replace('module.', '')] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model = nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])
    model.eval()
    return model

def build_dino(model_name, rank, args):
    model = timm.create_model(model_name, pretrained=args.timm_dino_model_path is None)
    if args.timm_dino_model_path:
        model.load_state_dict(_state_dict(_load_torch_payload(args.timm_dino_model_path)), strict=False)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    return model

def build_dinov2(model_name, rank, args):
    if model_name != 'dinov2_vitl14':
        raise ValueError(f"unsupported DEVIL DINOv2 model: {model_name}")
    if not args.dinov2_model_path:
        raise ValueError("DINOv2 ViT-L/14 checkpoint is required; set WORLDFOUNDRY_DEVIL_DINOV2_VITL14_CKPT")
    model = dinov2_backbones.dinov2_vitl14(pretrained=False)
    model.load_state_dict(_state_dict(_load_torch_payload(args.dinov2_model_path)), strict=False)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    return model

def build_viclip(model_name, rank, args):
    model_cfgs = {
        'viclip-l-internvid-10m-flt': {
            'size': 'l',
            'pretrained': args.viclip_model_path,
        },
    }
    cfg = model_cfgs[model_name]
    viclip = get_viclip(cfg['size'], cfg['pretrained'])['viclip']
    viclip = viclip.to(rank)
    viclip = nn.parallel.DistributedDataParallel(viclip, device_ids=[rank])
    viclip.eval()
    return viclip


def cal_flow_strength(model, video_data, video_lengths, video_names):
    results = []
    for i, frames in enumerate(video_data):
        static_score = []
        for image1, image2 in zip(frames[:-1], frames[1:]):
            padder = InputPadder(image1[None].shape)
            image1, image2 = padder.pad(image1[None], image2[None])
            _, flow_up = model(image1, image2, iters=50, test_mode=True)
            max_rad = get_score(image1, flow_up)
            static_score.append(max_rad)
        static_score = sum(static_score) / len(static_score)
        results.append(static_score)
    return results


def cal_flow_strength_batch(model, video_data, video_lengths, video_names, max_batch=20):
    results = []
    for i, frames in enumerate(video_data):
        static_score = []
        padder = InputPadder(frames.shape)
        image1s, image2s = padder.pad(frames[:-1], frames[1:])
        iter_n = int(math.ceil(image1s.shape[0] / max_batch))
        flow_ups = []
        for bidx in range(iter_n):
            image1 = image1s[bidx * max_batch: (bidx + 1) * max_batch]
            image2 = image2s[bidx * max_batch: (bidx + 1) * max_batch]
            _, flow_up = model(image1.contiguous(), image2.contiguous(), iters=20, test_mode=True)
            flow_ups.append(flow_up)

        flow_ups = torch.cat(flow_ups, dim=0)
        u_ = flow_ups[:, 0]
        v_ = flow_ups[:, 1]
        rad = torch.sqrt(torch.square(u_) + torch.square(v_))
        rad_flat = rad.flatten(1, 2)
        static_score = rad_flat.mean().item()
        results.append(static_score)
    return results

def get_score(img, flo, only_top=False):
    flo = flo[0].permute(1,2,0).cpu().numpy()

    u = flo[:,:,0]
    v = flo[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    
    h, w = rad.shape
    rad_flat = rad.flatten()

    if only_top:
        cut_index = int(h * w * 0.05)
        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])
    else:
        max_rad = np.mean(abs(np.sort(-rad_flat)))

    return max_rad.item()

def build_clip(model_name, rank, args):
    model_clip, process_clip = clip.load(args.clip_model_path, device=rank)
    model_clip = nn.parallel.DistributedDataParallel(model_clip, device_ids=[rank])
    model_clip.eval()
    return model_clip, process_clip


def cal_viclip_segment_dist(model, video_data, video_lengths, block_lengths_ratio=[0.25]):
    distances = []
    for video in video_data.split(video_lengths):
        dist_obj = 0
        for r in block_lengths_ratio:
            segments = get_segments(video, r)
            segments = to_same_size(segments)
            try:
                seg_feats = get_vid_feat(segments, model.module)
            except BaseException as e:
                print(e)
                pdb.set_trace()
            dist_obj += cal_segment_dist(seg_feats).item()
        distances.append(1 - dist_obj / len(block_lengths_ratio))
    return distances


def cal_dino_segment_dist(model, video_data, video_lengths):
    features = model.module.get_intermediate_layers(video_data, n=1)[0]
    features = F.normalize(features, dim=-1, p=2)
    features = features.split(video_lengths)
    distances = []
    for feat in features:
        acf = [calc_acf(feat, k) for k in range(feat.shape[0] // 8, feat.shape[0])]
        acf = sum(acf) / len(acf)
        distances.append(acf)
    return distances


def cal_dino_segment_dist_chunk(model, video_data, video_lengths, max_frames=100):
    distances = []

    # Iterate through each segment defined by video_lengths
    start = 0
    for length in video_lengths:
        segment_end = start + length
        all_features = []

        for i in range(start, segment_end, max_frames):
            end = min(i + max_frames, segment_end)
            batch = video_data[i:end]
            features = model.module.get_intermediate_layers(batch, n=1)[0]
            features = F.normalize(features, dim=-1, p=2)
            all_features.append(features)

        # Concatenate all features for the current video segment
        all_features = torch.cat(all_features, dim=0)
        
        # Calculate ACF based on all features of the segment
        acfs = [calc_acf(all_features, k) for k in range(all_features.shape[0] // 8, all_features.shape[0])]
        if acfs:
            segment_acf = sum(acfs) / len(acfs)
            distances.append(segment_acf)

        start = segment_end
    
    return distances


def get_segments(video, block_ratio):
    total_frames = video.shape[0]
    segment_length = int(total_frames * block_ratio)

    num_segments = int((total_frames + segment_length - 1) // segment_length)
    
    segments = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, total_frames)
        segment = video[start:end]
        
        if segment.size(0) < segment_length:
            repeats = segment_length // segment.size(0) + 1
            segment = torch.repeat_interleave(segment, repeats=repeats, dim=0)[:segment_length]
        
        segments.append(segment)
    
    return torch.stack(segments)

def to_same_size(segments, target_size=8):
    batch_size, segment_length, *rest_dims = segments.size()

    if segment_length > target_size:
        # 下采样：线性插值
        indices = torch.linspace(0, segment_length - 1, steps=target_size).long()
        resized_segments = segments[:, indices]
    else:
        # 上采样：通过重复最后一帧填充
        repeats_needed = target_size - segment_length
        last_frame = segments[:, -1:]  # 取最后一帧
        repeated_last_frames = last_frame.repeat(1, repeats_needed, *[1] * len(rest_dims))
        resized_segments = torch.cat((segments, repeated_last_frames), dim=1)

    return resized_segments

def cal_segment_dist(features):
    features = F.normalize(features, dim=-1, p=2)
    sims = features @ features.T
    # 获取上三角掩码矩阵
    mask = torch.triu(torch.ones_like(sims, dtype=torch.bool))

    # 使用掩码选择上三角元素
    sims = torch.masked_select(sims, mask)
    return sims.mean()

def calc_acf(features, k):
    acf = features[:-k] * features[k:]
    acf = acf.sum(-1).mean()
    return (1-acf).abs().item()

def cal_frechet(block_stats):
    # 计算块之间的 Frechet 距离
    distances = []
    for i in range(len(block_stats)-1):
        dist = frechet_distance(*block_stats[i], *block_stats[i+1])
        distances.append(dist.item())
    return sum(distances) / len(distances)
    
def cal_frechet_distance(model, video_data, video_lengths, block_lengths_ratio, features=None):
    if features is None:
        features = model(video_data)
        # 处理提取的特征,例如保存到文件或进一步处理
        features = F.normalize(features, dim=-1, p=2)
        features = features.split(video_lengths)
    distances = []
    for i, feat in enumerate(features):
        block_stats = process_blocks(feat, [int(r * video_lengths[i]) for r in block_lengths_ratio])
        distance = []
        for stats in block_stats:
            distance.append(cal_frechet(stats))
        distances.append(sum(distance) / len(distance))
    return distances

def process_blocks(features, block_lengths):
    # 计算块的数量]
    results = []
    for block_length in block_lengths:
        num_blocks = len(features) // block_length
        block_stats = []
        for i in range(num_blocks):
            block_start = i * block_length
            block_end = block_start + block_length
            block_features = features[block_start:block_end]
            mu, sigma = calculate_statistics(block_features)
            block_stats.append((mu, sigma))
        results.append(block_stats)
    return results

def calculate_covariance_matrix(features):
    """计算特征向量的协方差矩阵。
    参数:
    features (torch.Tensor): 形状为 (n_samples, n_features) 的张量。
    
    返回:
    torch.Tensor: 协方差矩阵，形状为 (n_features, n_features)。
    """
    # 中心化特征向量
    mean_centered = features - features.mean(dim=0)

    # 计算协方差矩阵
    cov_matrix = mean_centered.T @ mean_centered / (features.size(0) - 1)
    
    return cov_matrix


def calculate_statistics(features):
    mu = torch.mean(features, axis=0)
    sigma = calculate_covariance_matrix(features)
    return mu, sigma

def matrix_sqrt(matrix):
    """使用SVD计算矩阵的平方根，返回平方根矩阵"""
    u, s, v = torch.svd(matrix)
    sqrt_diag = torch.diag(torch.sqrt(s))
    return u @ sqrt_diag @ v.t()


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    term1 = diff.dot(diff)
    term2 = torch.trace(sigma1 + sigma2 - 2 * matrix_sqrt(sigma1 @ sigma2))
    
    return term1 + term2

def cal_temporal_entropy(video_paths, output_folder, temporal_info_script):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for video_path in video_paths:
        args = [video_path, output_folder]
        if output_folder == '/' or output_folder.startswith('/ '):
            print(f'***************output_folder: {output_folder}***************')
            exit(0)
        result = subprocess.run(['bash', temporal_info_script] + args, stdout=subprocess.PIPE, text=True, check=True)
        results.append(float(result.stdout.strip()))
        # print(f'{output_folder} done.')
        shutil.rmtree(output_folder)
    return results


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除文件夹
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def cal_dynamics_scores(args):
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    args.rank =  rank
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    output_file = f'{args.save_dir}/.tmp/results_{rank}.pth'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clear_folder(os.path.dirname(output_file))
    with open(output_file, 'w') as file:
        file.write('')  # 清空文件内容

    # 加载预训练的模型
    model_dino = build_dino(args.timm_dino_model_name, rank, args)
    model_dino_v2 = build_dinov2('dinov2_vitl14', rank, args)
    model_clip, process_clip = build_clip('ViT-L/14', rank, args)
    model_viclip = build_viclip('viclip-l-internvid-10m-flt', rank, args)
    process_clip.transforms =  process_clip.transforms[:2] + process_clip.transforms[-1:]
    process_clip.transforms[0] = Resize((336, 336))
    model_flow = build_raft(args.raft_model_path, rank)

    transform = Compose([
        Resize((224, 224)),  # 调整图像大小以匹配模型
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_no_resize = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = StandardVidoDataset(
        video_dir=args.video_dir,
        transform=transform,
        transform_no_resize=transform_no_resize,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=5, collate_fn=standard_collate_fn)

    if rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), position=0, leave=True)
    results = []
    with torch.no_grad():
        for frames in dataloader:
            org_videos = frames['org_videos']

            video_data = frames['videos'].to(rank)
            video_no_resize = [f.to(rank) for f in frames['videos_no_resize']]
            video_names = frames['video_names']
            video_lengths = frames['video_lengths']
            video_paths = frames['video_paths']
            # # Inter-frame level metrics
            flow_strength = cal_flow_strength_batch(model_flow, video_no_resize, video_lengths, video_names)
            ssim_dists, phash_dists = cal_ssim_dist(video_no_resize, org_videos, video_names)
            dino_segment_dist = cal_dino_segment_dist(model_dino_v2, video_data, video_lengths, )
            viclip_segment_dist = cal_viclip_segment_dist(model_viclip, video_data, video_lengths, [0.25])

            # # Video level metrics
            temporal_info = cal_temporal_entropy(video_paths, f'{args.save_dir}/.tmp-rank-{rank}', args.temporal_info_script)
            dino_var_dist, dino_features = cal_info_variance(model_dino, video_data, video_lengths)

            res = [{
                "video_name": video_names[i],
                "ssim": ssim_dists[i],
                "phash": phash_dists[i],
                "flow": flow_strength[i],
                "dino_segm_dist": dino_segment_dist[i],
                "viclip_segm_dist": viclip_segment_dist[i],
                "info_dino": dino_var_dist[i],
                "temporal_entropy": temporal_info[i],
            } for i in range(len(video_lengths))]
            results.extend(res)
            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    torch.save(results, output_file)

    dist.barrier()

    if rank == 0:
        load_and_merge(f'{args.save_dir}/.tmp', args.save_dir, args.dynamic_score_save_name)
        shutil.rmtree(f'{args.save_dir}/.tmp')

    dist.destroy_process_group()


def get_prompt_dynamic_grades(Video_names, col_name='video_name'):

    results = np.zeros(len(Video_names), dtype=int)

    keywords = ['static', 'low', 'medium', 'very_high', 'high']
    
    values = [0, 1, 2, 4, 3,]
    for i, name in enumerate(Video_names[col_name]):
        for keyword, value in zip(keywords, values):
            if keyword in name.lower():
                results[i] = value
                break 

    return results


def cal_controllability_winning_rate(prompt_dynamics, video_dynamics):

    video_dynamics = np.array(video_dynamics)
    prompt_dynamics = np.array(prompt_dynamics)
    x_degree_other = dict()
    y_degree_other = dict()
    for y in np.unique(prompt_dynamics):
        idx = prompt_dynamics == y
        x_degree_other.update({y: video_dynamics[~idx]})
        y_degree_other.update({y: prompt_dynamics[~idx]})

    winning_rates = []
    for x, y in zip(video_dynamics, prompt_dynamics):
        x_other = x_degree_other[y]
        y_other = y_degree_other[y]
        win_rate = ((x - x_other) * (y - y_other) > 0).mean()
        winning_rates.append(win_rate)
    return sum(winning_rates) / len(winning_rates)


def cal_dynamics_range(dynamics_scores):
    return cal_p99_1(dynamics_scores)

def cal_dynamics_controllability(video_names, dynamics_scores):
    prompt_dynamics = get_prompt_dynamic_grades(video_names)
    return cal_controllability_winning_rate(prompt_dynamics, dynamics_scores)


def get_overall_dynamic_scores(dynamics_scores, regress_models):
    inter_frame = np.stack([dynamics_scores['flow'], dynamics_scores['ssim'], dynamics_scores['phash']], axis=1)
    inter_segme = np.stack([dynamics_scores['dino_segm_dist'], dynamics_scores['viclip_segm_dist']], axis=1)
    video_level = np.stack([dynamics_scores['info_dino'], dynamics_scores['temporal_entropy']], axis=1)
    return regress_models.predict(inter_frame, inter_segme, video_level)


def cal_dynamics_metircs(args):
    regress_models = RegressionModel(args.regress_model_weight_path)
    dynamic_scores_path = os.path.join(args.save_dir, args.dynamic_score_save_name)
    dynamics_scores = pd.read_excel(dynamic_scores_path)
    overall_dynamics_scores = get_overall_dynamic_scores(dynamics_scores, regress_models)
    dynamics_scores['Overall_dynamics_scores'] = overall_dynamics_scores
    dynamics_range = cal_dynamics_range(overall_dynamics_scores)
    dynamics_controllability = cal_dynamics_controllability(dynamics_scores, overall_dynamics_scores)
    dynamics_scores.to_excel(dynamic_scores_path, index=False)

    return dynamics_scores, dynamics_range, dynamics_controllability

if __name__ == '__main__':
    args = parse_args()
    set_seed()
    cal_dynamics_scores(args)
    if args.rank == 0:
        dynamics_scores, dynamics_range, dynamics_controllability = cal_dynamics_metircs(args)
        print(f'Method:{os.path.basename(args.video_dir)}, Dyanmics Range: {dynamics_range}, Dynamics Controllability: {dynamics_controllability}')
