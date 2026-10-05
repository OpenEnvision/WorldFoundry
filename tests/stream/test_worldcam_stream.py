from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.pipelines.worldcam.pipeline_worldcam import WorldCamPipeline


image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

prompt = "A player moves through a realistic first-person game environment."
pretrained_model_path = "worldcam/worldcam"
wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B"

pipeline = WorldCamPipeline.from_pretrained(
    model_path=pretrained_model_path,
    required_components={"wan_model_path": wan_model_path},
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
    "./worldcam_stream_demo.mp4",
    fps=12,
)
