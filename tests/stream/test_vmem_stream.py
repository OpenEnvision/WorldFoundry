import imageio.v2 as imageio
import os
from PIL import Image

from worldfoundry.pipelines.vmem.pipeline_vmem import VMemPipeline


image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pipeline = VMemPipeline.from_pretrained(
    model_path="liguang0115/vmem",
    required_components={
        "surfel_model_path": "liguang0115/cut3r",
        "runtime_root": os.getenv("VMEM_RUNTIME_ROOT", "thirdparty/vmem"),
    },
    device="cuda",
)

pipeline.stream(
    images=input_image,
    interactions=["forward", "camera_l"],
    reset_memory=True,
)

output_video = pipeline.stream(
    images=None,
    interactions=["forward", "camera_r"],
)

imageio.mimsave("./vmem_stream_demo.mp4", output_video, fps=13)
