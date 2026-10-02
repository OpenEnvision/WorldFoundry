import os

if __name__ != "__main__" and os.getenv("WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    import pytest

    pytest.skip(
        "HunyuanWorld-Voyager demo inference is opt-in; set WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS=1.",
        allow_module_level=True,
    )

# from diffusers.utils import export_to_video
import imageio
from PIL import Image
from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_world_voyager import HunyuanWorldVoyagerPipeline
from worldfoundry.representations.point_clouds_generation.hunyuan_world.hunyuan_world_voyager_representation import (
    DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO,
)


image_path = os.getenv(
    "HUNYUAN_WORLD_VOYAGER_IMAGE_PATH",
    "./worldfoundry/data/test_cases/hunyuan_world_voyager/case1/ref_image.png",
)
moge_model_path = DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO
hunyuan_world_voyager_model_path = os.getenv(
    "HUNYUAN_WORLD_VOYAGER_MODEL_PATH",
    "tencent/HunyuanWorld-Voyager",
)

input_image = Image.open(image_path).convert('RGB')
test_prompt = "An old-fashioned European village with thatched roofs on the houses."

pipeline = HunyuanWorldVoyagerPipeline.from_pretrained(
    model_path=hunyuan_world_voyager_model_path,
    required_components = {"represent_model_path": moge_model_path},
    save_representation_video=True
)

print("Testing interaction sequence...")
interaction_sequence = ["forward", "left", "camera_r"] # can also be a single interaction
output_video = pipeline(images=input_image, interactions=interaction_sequence,
                        prompt=test_prompt, num_frames=37)
imageio.mimsave(os.getenv("HUNYUAN_WORLD_VOYAGER_OUTPUT_PATH", "hunyuan_world_voyager_sequence.mp4"), output_video, fps=12)
