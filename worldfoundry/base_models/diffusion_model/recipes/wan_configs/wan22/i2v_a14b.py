# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Official Wan 2.2 I2V A14B dual-expert EasyDict.

Same 40×5120 DiT / Wan2.1 VAE as Wan 2.1 I2V, plus MoE-style
``low_noise_model`` / ``high_noise_model`` checkpoints and a 0.900
sigma boundary.  CFG is ``(3.5, 3.5)`` for the two experts; 40
sample steps, shift 5.

LingBot-World-V2 deep-copies this object and tightens the sampler
(see :mod:`..lingbot_world_v2`).
"""

import torch
from easydict import EasyDict

from .shared import wan_shared_cfg

#------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
i2v_A14B.sample_shift = 5.0
i2v_A14B.sample_steps = 40
i2v_A14B.boundary = 0.900
i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
