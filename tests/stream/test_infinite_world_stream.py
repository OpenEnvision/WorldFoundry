from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline


image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

prompt = "A person keeps moving through a quiet residential street."

pretrained_model_path = "MeiGen-AI/Infinite-World"

pipeline = InfiniteWorldPipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
)

pipeline.stream(
    images=input_image,
    prompt=prompt,
    interactions=["forward", "camera_left"],
    num_chunks=1,
    reset_memory=True,
)

output_video = pipeline.stream(
    images=None,
    prompt=prompt,
    interactions=["forward", "camera_right"],
    num_chunks=1,
)

export_to_video(
    pipeline.memory_module.all_frames if pipeline.memory_module.all_frames is not None else output_video,
    "./infinite_world_stream_demo.mp4",
    fps=30,
)
