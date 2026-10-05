from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.pipelines.neoverse.pipeline_neoverse import NeoVersePipeline


image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

prompt = "A smooth first-person navigation sequence inside a coherent 3D scene."

pipeline = NeoVersePipeline.from_pretrained(
    model_path="Yuppie1204/NeoVerse",
    device="cuda",
)

pipeline.stream(
    images=input_image,
    prompt=prompt,
    interactions=["forward", "camera_l"],
    reset_memory=True,
)

output_video = pipeline.stream(
    images=None,
    prompt=prompt,
    interactions=["forward", "camera_r"],
)

export_to_video(
    pipeline.memory_module.all_frames if pipeline.memory_module.all_frames else output_video,
    "./neoverse_stream_demo.mp4",
    fps=16,
)
