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

import numpy as np
import argparse
import torch.nn as nn

import timm
import tqdm
import pandas as pd
import sys
from transformers import ViTModel

from metrics_utils.standard_video_dataset import StandardVidoDataset, standard_collate_fn
from quality_metrics import background_consistency, subject_consistency, motion_smoothness, MotionSmoothness, calculate_naturalness_score

import random

from worldfoundry.base_models.capabilities import vbench_asset_path


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


def _default_vbench_asset(asset_id):
    try:
        return str(vbench_asset_path(asset_id))
    except Exception:
        return None


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference with a pretrained DINO model on DiDemo dataset.")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--video_dir', type=str,
                        help='Directory containing video files.')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save extracted features.')
    parser.add_argument('--quality_save_name', type=str, default='Quality-Results.xlsx')
    parser.add_argument('--raft_model_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_RAFT_CKPT', 'WORLDFOUNDRY_RAFT_THINGS_CKPT'))
    parser.add_argument('--amt_config_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_AMT_CONFIG', 'WORLDFOUNDRY_VBENCH_AMT_S_CONFIG'))
    parser.add_argument('--amt_ckpt_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_AMT_CKPT', 'WORLDFOUNDRY_VBENCH_AMT_S_CKPT'))
    parser.add_argument('--clip_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_CLIP_VIT_B32_PT', 'WORLDFOUNDRY_VBENCH_CLIP_VIT_B32_PT'))
    parser.add_argument('--dino_source_dir', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_DINO_SOURCE_DIR'))
    parser.add_argument('--dino_checkpoint_path', type=str, default=_first_env('WORLDFOUNDRY_DEVIL_DINO_VITB16_CKPT', 'WORLDFOUNDRY_VBENCH_DINO_VITB16_CKPT'))
    parser.add_argument('--naturalness_path', type=str, default=None)
    parser.add_argument('--gemini_api_key', type=str, default=None)
    return parser.parse_args()


def build_dino_vitb16(args, rank):
    source = args.dino_source_dir or _default_vbench_asset("vbench_dino_source")
    checkpoint = args.dino_checkpoint_path or _default_vbench_asset("vbench_dino_vitb16_checkpoint")
    if source and checkpoint:
        model = torch.hub.load(str(source), "dino_vitb16", source="local", pretrained=False)
        model.load_state_dict(_state_dict(_load_torch_payload(checkpoint)), strict=False)
        return model.to(rank)
    from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import (
        load_dino_vitb16_feature_model,
    )

    return load_dino_vitb16_feature_model(device=rank)


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


def load_and_merge(pth_folder, save_folder, save_name='merged_results.xlsx'):
    all_datas = []
    save_path =  os.path.join(save_folder, save_name)
    for file in os.listdir(pth_folder):
        if file.endswith(".pth"):
            path = os.path.join(pth_folder, file)
            all_datas.extend(torch.load(path, map_location='cpu'))
    
    save_dicts_to_excel(all_datas, save_path)
    return save_path


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


def find_mp4_files(directory):
    mp4_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory, followlinks=True):
        for file in files:
            # 检查文件扩展名是否为.mp4
            if file.endswith(".mp4") and not file.startswith('._'):
                # 将完整的文件路径添加到列表中
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def merge_excel_files(file1, file2, output_file):
    # 加载Excel文件
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    # 根据 'video_name' 对数据框进行排序
    df1 = df1.sort_values(by='video_name').drop_duplicates().reset_index(drop=True)
    df2 = df2.sort_values(by='video_name').drop_duplicates().reset_index(drop=True)
    # 检查排序后的 'video_name' 列是否相同
    if not df1['video_name'].equals(df2['video_name']):
        raise ValueError("The 'video_name' columns do not match.")
    intersection = pd.Series(list(set(df1['video_name']).intersection(set(df2['video_name']))))
    df1 = df1[df1['video_name'].isin(intersection)]
    df2 = df2[df2['video_name'].isin(intersection)]

    # 根据 'video_name' 合并数据框
    merged_df = pd.merge(df1, df2, on='video_name')
    
    # 保存合并后的数据框到新的Excel文件
    merged_df.to_excel(output_file, index=False)
    print(f"Merged file saved as {output_file}")


def main_dist(args):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    output_file = f'{args.save_dir}/.tmp/results_{rank}.pth'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clear_folder(os.path.dirname(output_file))
    
    video_list = find_mp4_files(args.video_dir)
    with open(output_file, 'w') as file:
        file.write('')  # 清空文件内容
    if rank == 0:
        progress_bar = tqdm.tqdm(total=len(video_list), position=0, leave=True)

    clip_model, preprocess =  clip.load(args.clip_path, rank)
    dino_model = build_dino_vitb16(args, rank)
    # dino_model = ViTModel.from_pretrained('facebook/dino-vitb16').to(rank)

    motion_smooth_config = {'config': args.amt_config_path, 
            'ckpt': args.amt_ckpt_path}
    results = []
    with torch.no_grad():
        _, bg_consistency = background_consistency(clip_model, preprocess, video_list, rank)
        _, sj_consistency = subject_consistency(dino_model, video_list, rank)
        _, smoothness = motion_smoothness(motion_smooth_config, video_list, rank)
        for video_path in video_list:
            video_name = os.path.basename(video_path)
            results.append({
                'video_name': video_name,
                'bg_consistency': bg_consistency[video_name],
                'subject_consistency': sj_consistency[video_name],
                'motion_smoothness': smoothness[video_name]
            })
        if rank == 0:
            progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    torch.save(results, output_file)
    dist.barrier()
    if rank == 0:
        load_and_merge(f'{args.save_dir}/.tmp', args.save_dir)
        shutil.rmtree(f'{args.save_dir}/.tmp')
    dist.destroy_process_group()



def calculate_quality_scores(args):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output_file = f'{args.save_dir}/.tmp/results_{rank}.pth'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clear_folder(os.path.dirname(output_file))
    video_list = find_mp4_files(args.video_dir)
    with open(output_file, 'w') as file:
        file.write('')  # 清空文件内容

    clip_model, preprocess =  clip.load(args.clip_path, rank)
    dino_model = build_dino_vitb16(args, rank)
    motion_smooth_config = {'config': args.amt_config_path, 
            'ckpt': args.amt_ckpt_path}
    motion = MotionSmoothness(motion_smooth_config, rank)

    clip_model = nn.parallel.DistributedDataParallel(clip_model, device_ids=[rank])
    dino_model = nn.parallel.DistributedDataParallel(dino_model, device_ids=[rank])
    motion.model=nn.parallel.DistributedDataParallel(motion.model, device_ids=[rank])
    results = []

    dataset = StandardVidoDataset(
        video_dir=args.video_dir,
        return_video=False
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=5, collate_fn=standard_collate_fn)
    if rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), position=0, leave=True)

    results = []
    with torch.no_grad():
        for video_data in dataloader:
            video_path = video_data['video_paths'][0]
            bg_consistency = background_consistency(clip_model.module, preprocess, video_path, rank)
            sj_consistency = subject_consistency(dino_model, video_path, rank)
            smoothness = motion_smoothness(motion_smooth_config, video_path, motion, rank)
        
            video_name = os.path.basename(video_path)
            results.append({
                'video_name': video_name,
                'background_consistency': bg_consistency[video_name],
                'subject_consistency': sj_consistency[video_name],
                'motion_smoothness': smoothness[video_name]
            })

            if rank == 0:
                progress_bar.update(1)

    if rank == 0:
        progress_bar.close()

    torch.save(results, output_file)
    dist.barrier()
    
    if rank == 0:
        save_path = load_and_merge(f'{args.save_dir}/.tmp', args.save_dir, 'merged_results.xlsx')
        shutil.rmtree(f'{args.save_dir}/.tmp')
    else:
        save_path = os.path.join(args.save_dir, 'merged_results.xlsx')

    dist.barrier()
    dist.destroy_process_group()
    return save_path

if __name__ == '__main__':
    args = parse_args()
    set_seed()
    save_path_1 = calculate_quality_scores(args)
    if args.naturalness_path:
        save_path_nat = args.naturalness_path
    else:
        save_path_nat = calculate_naturalness_score(args.video_dir, args.save_dir, api_key=args.gemini_api_key)
    merge_excel_files(save_path_1, save_path_nat, os.path.join(args.save_dir, args.quality_save_name))
