import torch
import os
from worldfoundry.core.distributed.collectives.evaluation import dist_init, print0
try:
    from .WorldArena import WorldArenaBenchmark
except ImportError:
    from WorldArena import WorldArenaBenchmark
from datetime import datetime
import argparse
import json
import yaml

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='WorldArenaBenchmark', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )

    parser.add_argument(
        "--config_path",
        "--config",
        dest="config_path",
        type=str,
        default="",
        help="Path to config YAML that defines per-dimension settings",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def main():
    args = parse_args()

    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")


    config = load_config(args.config_path)

    # base paths
    save_path_default = config['save_path']
    # optional dedicated output for action_following runs
    save_path_action = config.get('save_path_action_following') or config.get('save_path_action_floowing', save_path_default)

    # choose output dir: if only action_following is requested, use the dedicated path
    save_path = save_path_action if args.dimension == ['action_following'] else save_path_default

    data_base = config['data']['val_base']
    gt_path = config['data']['gt_path']

    # action_following specific data roots (fallback to default if not provided)
    data_action = config.get('data_action_following', {})
    data_base_action = data_action.get('val_base', data_base)
    gt_path_action = data_action.get('gt_path', gt_path)
    #data_name = config['model_name']
    data_name = os.path.basename(data_base).replace("_dataset", "")


    kwargs = {}

    #dimension
    for dim in args.dimension:
        if dim == 'semantic_alignment':
            kwargs[f"{dim}_caption_model_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('caption', None)
            kwargs[f"{dim}_clip_model_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('CLIP', None)
            kwargs[f"{dim}_bleus_model_ckpt"] = config.get(dim, {}).get('BLEUs', None) 

            key_caption = f"{dim}_caption_model_ckpt"
            key_clip = f"{dim}_clip_model_ckpt"
            key_bleus = f"{dim}_bleus_model_ckpt"

            print(f"{dim}: caption = {kwargs[key_caption]}, CLIP = {kwargs[key_clip]}, BLEUs = {kwargs[key_bleus]}")
        elif dim == 'aesthetic_quality':
            kwargs[f"{dim}_clip_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('clip', None)
            kwargs[f"{dim}_head_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('aesthetic_head', None)
            print(f"{dim}: clip = {kwargs[f'{dim}_clip_ckpt']}, head = {kwargs[f'{dim}_head_ckpt']}")

        elif dim == 'background_consistency':
            kwargs[f"{dim}_clip_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('clip', None)
            kwargs[f"{dim}_raft_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('raft', None)
            print(f"{dim}: clip = {kwargs[f'{dim}_clip_ckpt']}, raft = {kwargs[f'{dim}_raft_ckpt']}")

        elif dim == 'dynamic_degree':
            kwargs[f"{dim}_raft_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('raft', None)
            print(f"{dim}: raft = {kwargs[f'{dim}_raft_ckpt']}")

        elif dim == 'flow_score':
            kwargs[f"{dim}_raft_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('raft', None)
            print(f"{dim}: raft = {kwargs[f'{dim}_raft_ckpt']}")

        elif dim == 'photometric_smoothness':
            kwargs[f"{dim}_cfg_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('cfg', None)
            kwargs[f"{dim}_model_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('model', None)
            print(f"{dim}: cfg = {kwargs[f'{dim}_cfg_ckpt']}, model = {kwargs[f'{dim}_model_ckpt']}")

        elif dim == 'motion_smoothness':
            kwargs[f"{dim}_model_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('model', None)
            print(f"{dim}: model = {kwargs[f'{dim}_model_ckpt']}")

        elif dim == 'subject_consistency':
            kwargs[f"{dim}_repo_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('repo', None)
            kwargs[f"{dim}_weight_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('weight', None)
            kwargs[f"{dim}_model_name"] = config.get('ckpt',{}).get(dim, {}).get('model', 'dino_vitb16')
            kwargs[f"{dim}_raft_ckpt"] = config.get('ckpt',{}).get(dim, {}).get('raft', None)
            print(f"{dim}: repo = {kwargs[f'{dim}_repo_ckpt']}, weight = {kwargs[f'{dim}_weight_ckpt']}, raft = {kwargs[f'{dim}_raft_ckpt']}")

        elif dim in ['imaging_quality', 'image_quality']:
            # Support both legacy `imaging_quality` and updated `image_quality` naming
            ckpt_key = dim if dim in config.get('ckpt', {}) else 'imaging_quality'
            kwargs[f"{dim}_musiq_ckpt"] = config.get('ckpt',{}).get(ckpt_key, {}).get('musiq', None)
            print(f"{dim}: musiq = {kwargs[f'{dim}_musiq_ckpt']}")

        elif dim in ['psnr', 'ssim']:
            pass
            
        else:           
            kwargs[f"{dim}_model_ckpt"] = config.get('ckpt',{}).get(dim, None)

            key_model_ckpt = f"{dim}_model_ckpt"
            print(f"{dim}: ckpt = {kwargs[key_model_ckpt]}")

    print0(f'start evaluation')
    my_benchmark = WorldArenaBenchmark(device, save_path)

    my_benchmark.evaluate(
        data_base = data_base,
        data_base_action = data_base_action,
        data_name = data_name,
        dimension_list = args.dimension,
        local=True,
        gt_path=gt_path,
        gt_path_action=gt_path_action,
        overwrite=args.overwrite,
        **kwargs
    )
    print0('done')


if __name__ == "__main__":
    main()
