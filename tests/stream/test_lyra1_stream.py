import os
from pathlib import Path

import imageio
from PIL import Image

from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline


REPO_ROOT = os.getenv(
    "LYRA1_REPO_ROOT",
    str(Path(__file__).resolve().parents[2] / "thirdparty" / "Lyra-1"),
)
CACHE_ROOT = os.getenv(
    "LYRA1_CACHE_ROOT",
    str(Path(__file__).resolve().parents[2] / "cache" / "hfd" / "Lyra"),
)
CHECKPOINT_DIR = os.getenv("LYRA1_CHECKPOINT_DIR", CACHE_ROOT)
STATIC_CKPT_PATH = os.getenv(
    "LYRA1_STATIC_CKPT_PATH",
    os.path.join(CACHE_ROOT, "lyra_static.pt"),
)

image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pipeline = Lyra1Pipeline.from_pretrained(
    model_path=REPO_ROOT,
    required_components={
        "checkpoint_dir": CHECKPOINT_DIR,
        "static_ckpt_path": STATIC_CKPT_PATH,
        "default_mode": "static",
    },
    device="cuda",
)

pipeline.stream(
    images=input_image,
    prompt="An old alley with stone pavement and a visible route ahead.",
    interactions=["forward", "camera_l"],
    mode="static",
    reset_memory=True,
)
pipeline.stream(
    prompt="Continue moving while shifting attention to the right.",
    interactions=["forward", "right"],
    mode="static",
)

imageio.mimsave("lyra1_stream_demo.mp4", pipeline.memory_module.all_frames, fps=24)
