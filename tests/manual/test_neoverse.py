from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.pipelines.neoverse.pipeline_neoverse import NeoVersePipeline


image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

prompt = "A stable first-person camera explores a realistic indoor scene."
interactions = ["forward", "camera_l", "forward", "right"]

pipeline = NeoVersePipeline.from_pretrained(
    model_path="Yuppie1204/NeoVerse",
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    prompt=prompt,
    interactions=interactions,
)
export_to_video(output_video, "./neoverse_demo.mp4", fps=16)
