import os
import json
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from vbench.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm


def background_consistency(clip_model, preprocess, video_path, device, read_frame=False):
    sim = 0.0
    cnt = 0
    video_results = dict()
    image_transform = clip_transform(224)
    video_sim = 0.0
    if read_frame:
        video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
        tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
        images = []
        for tmp_path in tmp_paths:
            images.append(preprocess(Image.open(tmp_path)))
        images = torch.stack(images)
    else:
        images = load_video(video_path)
        images = image_transform(images)
    images = images.to(device)
    image_features = clip_model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1, p=2)
    for i in range(len(image_features)):
        image_feature = image_features[i].unsqueeze(0)
        if i == 0:
            first_image_feature = image_feature
        else:
            sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
            sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
            cur_sim = (sim_pre + sim_fir) / 2
            video_sim += cur_sim
            cnt += 1
        former_image_feature = image_feature
    sim_per_image = video_sim / (len(image_features) - 1)
    video_results.update({os.path.basename(video_path): sim_per_image})

    return video_results
