"""LingBot-World-V2 inference overlay on Wan 2.2 I2V A14B.

Deep-copies the canonical :data:`~.wan22.i2v_a14b.i2v_A14B` dual-expert
graph and overrides causal-fast sampling: 4 steps, shift 10, boundary
0.947, 361 frames, and a Chinese negative prompt tuned for driving
scenes.  ``fast_checkpoint`` points at the distilled transformer
weights.

Supported pixel sizes match the I2V A14B set (720p / 480p landscape
and portrait).  Architecture width stays identical to A14B.
"""

from copy import deepcopy

from .wan22.i2v_a14b import i2v_A14B

LINGBOT_WORLD_V2_CONFIG = deepcopy(i2v_A14B)
LINGBOT_WORLD_V2_CONFIG.__name__ = "Config: LingBot-World-V2 I2V A14B causal-fast"
LINGBOT_WORLD_V2_CONFIG.fast_checkpoint = "transformers"
LINGBOT_WORLD_V2_CONFIG.sample_shift = 10.0
LINGBOT_WORLD_V2_CONFIG.boundary = 0.947
LINGBOT_WORLD_V2_CONFIG.sample_steps = 4
LINGBOT_WORLD_V2_CONFIG.frame_num = 361
LINGBOT_WORLD_V2_CONFIG.sample_guide_scale = (5.0, 5.0)
LINGBOT_WORLD_V2_CONFIG.sample_neg_prompt = (
    "画面突变，镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，"
    "液化，不合逻辑的结构，卡顿，PPT幻灯片感，过暗，欠曝，低对比度，过度锐化，"
    "3D渲染感，人物，行人，游客，汽车，电线"
)

LINGBOT_WORLD_V2_1P3B_CONFIG = deepcopy(LINGBOT_WORLD_V2_CONFIG)
LINGBOT_WORLD_V2_1P3B_CONFIG.__name__ = "Config: LingBot-World-V2 I2V 1.3B causal-fast"
LINGBOT_WORLD_V2_1P3B_CONFIG.update(
    dim=1536, ffn_dim=8960, num_heads=12, num_layers=30,
    fast_checkpoint=".", sample_shift=5.0, reference_noise_chunks=20, frame_num=317,
)

LINGBOT_WORLD_V2_CONFIGS = {
    "lingbot-world-v2": LINGBOT_WORLD_V2_CONFIG,
    "lingbot-world-v2-1.3b": LINGBOT_WORLD_V2_1P3B_CONFIG,
}

SUPPORTED_SIZES = frozenset({"720*1280", "1280*720", "480*832", "832*480"})

__all__ = ["LINGBOT_WORLD_V2_CONFIG", "LINGBOT_WORLD_V2_1P3B_CONFIG", "LINGBOT_WORLD_V2_CONFIGS", "SUPPORTED_SIZES"]
