import os

import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v3 import (
    DEFAULT_DEPTH_ANYTHING3_SMALL_REPO,
)
from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v3 import (
    DepthAnything3Pipeline,
)

DATA_PATH = os.environ.get(
    "DEPTH_ANYTHING3_DATA_PATH",
    "./worldfoundry/data/test_cases/test_image_case1/ref_image.png",
)
MODEL_PATH = os.environ.get("DEPTH_ANYTHING3_MODEL_PATH", DEFAULT_DEPTH_ANYTHING3_SMALL_REPO)
DEVICE = os.environ.get(
    "DEPTH_ANYTHING3_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
OUTPUT_DIR = os.environ.get("DEPTH_ANYTHING3_OUTPUT_DIR", "./depth_anything_v3_output")


pipeline = DepthAnything3Pipeline.from_pretrained(
    pretrained_model_path=MODEL_PATH,
    device=DEVICE,
)

result = pipeline(
    input_data=DATA_PATH,
    output_dir=OUTPUT_DIR,
    export_format="depth_vis",
)

print("Depth shape:", result["depth"].shape)
print("Metric depth:", result["is_metric"])
print("Point cloud points:", 0 if result["point_cloud"] is None else result["point_cloud"]["points"].shape[0])
