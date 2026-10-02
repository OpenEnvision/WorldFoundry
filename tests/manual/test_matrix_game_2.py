import os

if __name__ != "__main__" and os.getenv("WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    import pytest

    pytest.skip("Matrix-Game-2 demo inference is opt-in; set WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS=1.", allow_module_level=True)

from diffusers.utils import export_to_video
from PIL import Image
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline


image_path = os.environ.get(
    "MATRIX_GAME2_IMAGE_PATH",
    "./worldfoundry/data/test_cases/matrix-game-2/universal/0000.png",
)
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = os.environ.get("MATRIX_GAME2_MODEL_PATH", "Skywork/Matrix-Game-2.0")
pipeline = MatrixGame2Pipeline.from_pretrained(
    model_path=pretrained_model_path,
    mode=os.environ.get("MATRIX_GAME2_MODE", "universal"),
    device="cuda"
)

output_video = pipeline(
    images=input_image,
    num_frames=int(os.environ.get("MATRIX_GAME2_NUM_FRAMES", "150")),
    seed=int(os.environ["MATRIX_GAME2_SEED"]) if os.environ.get("MATRIX_GAME2_SEED") else None,
    interactions=[
        item.strip()
        for item in os.environ.get(
            "MATRIX_GAME2_INTERACTIONS",
            "forward,left,right,forward_left,forward_right,camera_l,camera_r",
        ).split(",")
        if item.strip()
    ],
)

output_path = os.environ.get("MATRIX_GAME2_OUTPUT_PATH", "matrix_game_2_demo.mp4")
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
export_to_video(output_video, output_path, fps=int(os.environ.get("MATRIX_GAME2_FPS", "12")))
print(f"Matrix-Game-2 video saved to: {output_path}")
