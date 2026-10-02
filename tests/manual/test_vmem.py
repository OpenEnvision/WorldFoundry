import imageio.v2 as imageio
import os
from PIL import Image

if __name__ != "__main__" and os.getenv("WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    import pytest

    pytest.skip("VMem demo inference is opt-in; set WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS=1.", allow_module_level=True)

from worldfoundry.pipelines.vmem.pipeline_vmem import VMemPipeline


image_path = os.getenv("VMEM_IMAGE_PATH", "./worldfoundry/data/test_cases/studio_demo/00/image.jpg")
input_image = Image.open(image_path).convert("RGB")

interactions = ["forward", "camera_l", "forward", "camera_r"]

pipeline = VMemPipeline.from_pretrained(
    model_path="liguang0115/vmem",
    required_components={
        "surfel_model_path": "liguang0115/cut3r",
        "runtime_root": os.getenv(
            "VMEM_RUNTIME_ROOT",
            "./worldfoundry/synthesis/visual_generation/vmem/vmem_runtime",
        ),
    },
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    interactions=interactions,
)

imageio.mimsave("./vmem_demo.mp4", output_video, fps=13)
