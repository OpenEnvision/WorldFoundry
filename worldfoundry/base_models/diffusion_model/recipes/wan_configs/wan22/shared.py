# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Fields shared by every official Wan 2.2 EasyDict identity.

Same UMT5 / bf16 / 1000-step / 16 fps / Chinese neg-prompt defaults
as Wan 2.1, plus ``frame_num = 81``.  Per-identity files update this
object then set VAE (Wan2.1 vs Wan2.2), DiT width, and sampler
shift / CFG / MoE boundary.

Changing a field here changes T2V A14B, I2V A14B, TI2V 5B, S2V, and
Animate at once.
"""

import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.frame_num = 81
