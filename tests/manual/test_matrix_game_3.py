import os
from pathlib import Path

if __name__ != "__main__" and os.getenv("WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    import pytest

    pytest.skip("Matrix-Game-3 demo inference is opt-in; set WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS=1.", allow_module_level=True)

import imageio
from PIL import Image

from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline


def _rank() -> int:
    return int(os.getenv("RANK", "0"))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIR = PROJECT_ROOT / "worldfoundry" / "data" / "test_cases" / "matrix-game-3" / "001"
CHECKPOINT_DIR = os.getenv(
    "MATRIX_GAME3_CHECKPOINT_DIR",
    str(PROJECT_ROOT / "cache" / "hfd" / "Skywork--Matrix-Game-3.0"),
)
OUTPUT_DIR = Path(os.getenv("MATRIX_GAME3_OUTPUT_DIR", str(PROJECT_ROOT / "tmp")))
OUTPUT_PATH = Path(os.getenv("MATRIX_GAME3_OUTPUT_PATH", str(OUTPUT_DIR / "matrix_game_3_demo.mp4")))

image_path = Path(
    os.getenv(
        "MATRIX_GAME3_IMAGE_PATH",
        str(DEFAULT_CASE_DIR / "image.png"),
    )
)
default_prompt = os.getenv(
    "MATRIX_GAME3_DEFAULT_PROMPT",
    "A colorful, animated cityscape with a gas station and various buildings.",
)
input_image = Image.open(image_path).convert("RGB")
interactions = [
    item.strip()
    for item in os.getenv("MATRIX_GAME3_INTERACTIONS", "forward,forward_right,camera_r,forward").split(",")
    if item.strip()
]


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

pipeline = MatrixGame3Pipeline.from_pretrained(
    model_path=CHECKPOINT_DIR,
    required_components={
        "checkpoint_dir": CHECKPOINT_DIR,
        "vae_type": os.getenv("MATRIX_GAME3_VAE_TYPE", "mg_lightvae"),
        "lightvae_pruning_rate": float(os.getenv("MATRIX_GAME3_LIGHTVAE_PRUNING_RATE", "0.5")),
        "use_int8": _env_bool("MATRIX_GAME3_USE_INT8", True),
        "compile_vae": _env_bool("MATRIX_GAME3_COMPILE_VAE", True),
        "num_inference_steps": int(os.getenv("MATRIX_GAME3_NUM_INFERENCE_STEPS", "3")),
        "fa_version": os.getenv("MATRIX_GAME3_FA_VERSION", "3"),
        "ulysses_size": int(os.getenv("MATRIX_GAME3_ULYSSES_SIZE", "1")),
        "t5_fsdp": _env_bool("MATRIX_GAME3_T5_FSDP", False),
        "dit_fsdp": _env_bool("MATRIX_GAME3_DIT_FSDP", False),
        "use_async_vae": _env_bool("MATRIX_GAME3_USE_ASYNC_VAE", False),
        "async_vae_warmup_iters": int(os.getenv("MATRIX_GAME3_ASYNC_VAE_WARMUP_ITERS", "0")),
    },
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    prompt=os.getenv(
        "MATRIX_GAME3_PROMPT",
        default_prompt,
    ),
    interactions=interactions,
    num_iterations=int(os.getenv("MATRIX_GAME3_NUM_ITERATIONS", "12")),
    size=os.getenv("MATRIX_GAME3_SIZE", "704*1280"),
    fps=int(os.getenv("MATRIX_GAME3_FPS", "17")),
    output_dir=str(OUTPUT_DIR),
    save_name=OUTPUT_PATH.stem,
    visualize_ops=_env_bool("MATRIX_GAME3_VISUALIZE_OPS", True),
    show_progress=_env_bool("MATRIX_GAME3_SHOW_PROGRESS", True),
    seed=int(os.getenv("MATRIX_GAME3_SEED", "42")),
)
if _rank() == 0:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if output_video is not None and not OUTPUT_PATH.is_file():
        imageio.mimsave(OUTPUT_PATH, output_video, fps=int(os.getenv("MATRIX_GAME3_FPS", "17")))
    print(f"Matrix-Game-3 video saved to: {OUTPUT_PATH}")
else:
    print(f"Matrix-Game-3 rank {_rank()} completed without writing output.")
