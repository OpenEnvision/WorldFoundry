from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline


image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

prompt = "A person walks forward through a bright suburban neighborhood."
interactions = ["forward", "camera_left", "forward", "camera_right"]

pretrained_model_path = "MeiGen-AI/Infinite-World"

pipeline = InfiniteWorldPipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    prompt=prompt,
    interactions=interactions,
    num_chunks=1,
)
export_to_video(output_video, "./infinite_world_demo.mp4", fps=30)
