"""Wan 2.2 / Matrix-Game-3 action-recipe architecture data.

Pins the UMT5 + Wan2.2 VAE38 + 40-layer 5.1k-wide DiT used by
Matrix-Game-3.0 world-action recipes.  Shared fields (text length,
dtype, train timesteps) live on ``wan_shared_cfg``; the
``matrix_game3`` EasyDict adds 48-channel VAE stride ``(4, 16, 16)``
and English negative prompts.

``WAN_CONFIGS`` is the runner lookup; ``MAX_AREA_CONFIGS`` caps the
704×1280 pixel budget.
"""

from __future__ import annotations


import torch

from easydict import EasyDict

wan_shared_cfg = EasyDict()
wan_shared_cfg.t5_model = "umt5_xxl"
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512
wan_shared_cfg.param_dtype = torch.bfloat16
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
wan_shared_cfg.frame_num = 481

matrix_game3 = EasyDict(__name__="Config: Matrix Game 3.0")
matrix_game3.update(wan_shared_cfg)
matrix_game3.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
matrix_game3.t5_tokenizer = "google/umt5-xxl"
matrix_game3.vae_checkpoint = "Wan2.2_VAE.pth"
matrix_game3.vae_stride = (4, 16, 16)
matrix_game3.patch_size = (1, 2, 2)
matrix_game3.in_dim = 48
matrix_game3.out_dim = 48
matrix_game3.dim = 5120
matrix_game3.ffn_dim = 13824
matrix_game3.freq_dim = 256
matrix_game3.num_heads = 40
matrix_game3.num_layers = 40
matrix_game3.window_size = (-1, -1)
matrix_game3.qk_norm = True
matrix_game3.cross_attn_norm = True
matrix_game3.eps = 1e-6
matrix_game3.sample_shift = 5.0
matrix_game3.num_inference_steps = 50
matrix_game3.sample_guide_scale = 5.0
matrix_game3.sample_neg_prompt = (
    "Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork, painting, "
    "still image, overall grayness, worst quality, low quality, JPEG compression residue, ugly, "
    "mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "malformed limbs, fused fingers, still image, cluttered background, three legs, crowded "
    "background, walking backwards"
)

WAN_CONFIGS = {
    "matrix_game3": matrix_game3,
}

MAX_AREA_CONFIGS = {
    "704*1280": 704 * 1280,
}

__all__ = ["MAX_AREA_CONFIGS", "WAN_CONFIGS", "matrix_game3", "wan_shared_cfg"]
