import os
from pathlib import Path

import imageio
from PIL import Image

from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline


REPO_ROOT = os.getenv(
    "LYRA2_REPO_ROOT",
    str(Path(__file__).resolve().parents[2] / "thirdparty" / "Lyra-2"),
)
CHECKPOINT_DIR = os.getenv("LYRA2_CHECKPOINT_DIR", os.path.join(REPO_ROOT, "checkpoints/model"))
NEGATIVE_PROMPT_PATH = os.getenv(
    "LYRA2_NEGATIVE_PROMPT_PATH",
    os.path.join(REPO_ROOT, "checkpoints/text_encoder/negative_prompt.pt"),
)
DA3_MODEL_PATH = os.getenv(
    "LYRA2_DA3_MODEL_PATH",
    os.path.join(REPO_ROOT, "checkpoints/recon/model.pt"),
)

image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pipeline = Lyra2Pipeline.from_pretrained(
    model_path=REPO_ROOT,
    required_components={
        "checkpoint_dir": CHECKPOINT_DIR,
        "negative_prompt_path": NEGATIVE_PROMPT_PATH,
        "da3_model_path_custom": DA3_MODEL_PATH,
    },
    device="cuda",
)

pipeline.stream(
    images=input_image,
    prompt="A narrow medieval lane with warm lanterns and detailed storefronts.",
    interactions=["forward", "camera_l"],
    fps=16,
    reset_memory=True,
)
pipeline.stream(
    prompt="Continue exploring the lane and turn slightly to inspect the right side.",
    interactions=["forward", "right"],
    fps=16,
)

imageio.mimsave("lyra2_stream_demo.mp4", pipeline.memory_module.all_frames, fps=16)
