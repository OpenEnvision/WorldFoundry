"""Shared public defaults for the GEN3C pipeline and runtime facade."""

DEFAULT_GEN3C_PROMPT = (
    "A cinematic view of a futuristic science-fiction city with coherent structures, realistic lighting, "
    "and smooth natural camera motion."
)

DEFAULT_GEN3C_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

__all__ = ["DEFAULT_GEN3C_NEGATIVE_PROMPT", "DEFAULT_GEN3C_PROMPT"]
