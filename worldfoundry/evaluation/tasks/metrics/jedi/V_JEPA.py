# Code copied and modified from https://github.com/facebookresearch/jepa/
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import yaml, os

import torch
import torch.nn as nn
from vjepa.models import vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant, vit_gigantic
from vjepa.models.attentive_pooler import AttentiveClassifier

from .V_JEPA_utils import *

import logging
logger = logging.getLogger(__name__)


class VJEPA:
    def __init__(self,
        model_dir=None,
        config_fname=None,
        normalize=((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),            
        finetuned=True):

        model_dir = model_dir if model_dir is not None else os.getcwd()
        self.model_dir = model_dir

        if not os.path.exists(f'{model_dir}/vith16.pth.tar'):
            download_vjepa(save_path=model_dir, finetuned_probe=False)
        if finetuned and not os.path.exists(f'{model_dir}/ssv2-probe.pth.tar'):
            download_vjepa(save_path=model_dir, finetuned_probe=True)
        
        self.encoder, self.classifier = get_default_vjepa(config_fname=config_fname, model_dir=model_dir, finetuned=finetuned)
        import vjepa.datasets.utils.video.transforms as video_transforms
        self.transforms = video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        self.finetuned = finetuned
    
    @torch.no_grad()
    def get_feats(self, videos):
        videos = videos.permute(0, 2, 1, 3, 4)
        videos = torch.stack([self.transforms(videos[i]) for i in range(videos.shape[0])])
        encoded_videos = self.encoder([[videos]])[0]
        if self.finetuned:
            return self.classifier.pooler(encoded_videos).squeeze(1).cpu().numpy()
        else:
            return encoded_videos.mean(dim=1).cpu().numpy()

def download_vjepa(save_path, finetuned_probe=False):
    fname = 'ssv2-probe.pth.tar' if finetuned_probe else 'vith16.pth.tar'
    target = os.path.join(save_path, fname)
    if os.path.exists(target):
        return target
    raise FileNotFoundError(
        f"Missing V-JEPA checkpoint {fname} under {save_path}. "
        "Download via WorldFoundry capability jedi_vjepa_vith16 / jedi_vjepa_ssv2_probe "
        "or set WORLDFOUNDRY_JEDI_MODEL_DIR."
    )


VIT_NAME_LOOKUP = {
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_large': vit_large,
    'vit_huge': vit_huge,
    'vit_giant': vit_giant,
    'vit_gigantic': vit_gigantic,
}

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = VIT_NAME_LOOKUP[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder',
    fientune=True
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except KeyError:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

def load_checkpoint(
    device,
    r_path,
    classifier,
):
    checkpoint = torch.load(r_path, map_location=device)
    epoch = checkpoint['epoch']
    pretrained_dict = {key.removeprefix('module.'): value for key, value in checkpoint['classifier'].items()}
    msg = classifier.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')
    return classifier

def get_default_vjepa(
    config_fname=None,
    model_dir=None,
    finetuned=True,
):
    
    model_dir = model_dir if model_dir is not None else os.getcwd()

    # Load config
    args_eval = None
    if config_fname:
        with open(config_fname, 'r') as y_file:
            args_eval = yaml.load(y_file, Loader=yaml.FullLoader)
    else:
        from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

        bundled_config = bundled_benchmark_asset("jedi", "vith16_ssv2_16x2x3.yaml")
        if not bundled_config.is_file():
            raise FileNotFoundError(
                "Bundled JEDi config missing at worldfoundry/data/benchmarks/assets/jedi/vith16_ssv2_16x2x3.yaml"
            )
        with open(bundled_config, 'r') as y_file:
            args_eval = yaml.load(y_file, Loader=yaml.FullLoader)
    assert args_eval is not None

    logger.info('loaded params...')
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    eval_tag = args_eval.get('tag', None)
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=f'{model_dir}/vith16.pth.tar',
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    # -- init classifier
    classifier =  None
    if finetuned:
        classifier = AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=encoder.num_heads,
            depth=1,
            num_classes=num_classes,
        ).to(device)
        classifier = load_checkpoint(
            device=device,
            r_path=f'{model_dir}/ssv2-probe.pth.tar',
            classifier=classifier
        )
        classifier.eval()
    return encoder, classifier
