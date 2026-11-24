from __future__ import annotations

WAH_PROMPT_TRIGGER = "wah."

WAH_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)

# Public Warp-as-History recipe.
WAH_NUM_FRAMES = 33
WAH_NUM_LATENT_FRAMES_PER_CHUNK = 9
WAH_HISTORY_SIZES = (16, 2, 1)
WAH_PREV_CHUNK_HISTORY_SIZES = WAH_HISTORY_SIZES
WAH_PYRAMID_NUM_STAGES = 3
WAH_PYRAMID_STEPS = (2, 2, 2)
WAH_VISIBLE_TOKEN_THRESHOLD = 0.1
WAH_INVISIBLE_FILL_MODES = frozenset({"mean_first_frame", "black"})

LORA_DISABLED_VALUES = frozenset({"", "0", "false", "no", "none", "null", "off"})
