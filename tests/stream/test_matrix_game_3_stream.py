import os
from pathlib import Path

import imageio
from PIL import Image

from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline


REPO_ROOT = os.getenv(
    "MATRIX_GAME3_REPO_ROOT",
    str(Path(__file__).resolve().parents[2] / "thirdparty" / "Matrix-Game-3"),
)
CHECKPOINT_DIR = os.getenv("MATRIX_GAME3_CHECKPOINT_DIR", REPO_ROOT)

image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pipeline = MatrixGame3Pipeline.from_pretrained(
    model_path=REPO_ROOT,
    required_components={
        "checkpoint_dir": CHECKPOINT_DIR,
        "fa_version": "0",
        "vae_type": "wan",
    },
    device="cuda",
)

pipeline.stream(
    images=input_image,
    prompt="Walk through the open street and turn slightly to inspect the right side.",
    interactions=["forward", "camera_r", "forward"],
    num_iterations=1,
    fps=17,
    reset_memory=True,
)
pipeline.stream(
    images=None,
    prompt="Continue moving ahead while adjusting the camera back to center.",
    interactions=["forward", "camera_l", "forward_right"],
    num_iterations=1,
    fps=17,
)

imageio.mimsave("matrix_game_3_stream_demo.mp4", pipeline.memory_module.all_frames, fps=17)
