import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline


DATA_PATH = os.environ.get(
    "GEN3C_DATA_PATH",
    "./worldfoundry/data/test_cases/gen3c/image.png",
)
MODEL_PATH = os.environ.get("GEN3C_MODEL_PATH", "gen3c")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR")
MOGE_PRETRAINED = os.environ.get("GEN3C_MOGE_PRETRAINED")
DEVICE = os.environ.get("GEN3C_DEVICE", "cuda")
OUTPUT_DIR = os.environ.get("GEN3C_OUTPUT_DIR", "./gen3c_output")
SCENE_NAME = os.environ.get("GEN3C_SCENE_NAME", "gen3c_demo")
PROMPT = os.environ.get(
    "GEN3C_PROMPT",
    "",
)
TRAJECTORY = os.environ.get("GEN3C_TRAJECTORY", "left")
FPS = int(os.environ.get("GEN3C_FPS", "24"))


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


pipeline = Gen3CPipeline.from_pretrained(
    model_path=MODEL_PATH,
    required_components={
        "checkpoint_dir": CHECKPOINT_DIR,
        "moge_pretrained": MOGE_PRETRAINED,
    },
    device=DEVICE,
    num_steps=int(os.environ.get("GEN3C_NUM_STEPS", "35")),
    num_video_frames=int(os.environ.get("GEN3C_NUM_VIDEO_FRAMES", "121")),
    fps=FPS,
    height=int(os.environ.get("GEN3C_HEIGHT", "704")),
    width=int(os.environ.get("GEN3C_WIDTH", "1280")),
    seed=int(os.environ.get("GEN3C_SEED", "1")),
    guidance=float(os.environ.get("GEN3C_GUIDANCE", "1")),
    foreground_masking=_env_bool("GEN3C_FOREGROUND_MASKING", True),
    offload_diffusion_transformer=_env_bool("GEN3C_OFFLOAD_DIFFUSION_TRANSFORMER", False),
    offload_tokenizer=_env_bool("GEN3C_OFFLOAD_TOKENIZER", False),
    offload_text_encoder_model=_env_bool("GEN3C_OFFLOAD_TEXT_ENCODER_MODEL", False),
    offload_prompt_upsampler=_env_bool("GEN3C_OFFLOAD_PROMPT_UPSAMPLER", False),
    offload_guardrail_models=_env_bool("GEN3C_OFFLOAD_GUARDRAIL_MODELS", False),
    disable_guardrail=_env_bool("GEN3C_DISABLE_GUARDRAIL", True),
    disable_prompt_encoder=_env_bool("GEN3C_DISABLE_PROMPT_ENCODER", False),
    num_gpus=int(os.environ.get("GEN3C_NUM_GPUS", "1")),
)

image = Image.open(DATA_PATH).convert("RGB")
result = pipeline(
    images=image,
    trajectory=TRAJECTORY,
    prompt=PROMPT,
    scene_name=SCENE_NAME,
    output_dir=OUTPUT_DIR,
    return_dict=True,
    fps=FPS,
)

output_path = result.get("generated_video_path")
if output_path is None:
    output_path = str(Path(OUTPUT_DIR) / f"{SCENE_NAME}.mp4")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, [np.asarray(frame) for frame in result["frames"]], fps=FPS)

print(f"GEN3C video saved to: {output_path}")
