import os

import torch

from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v2 import (
    DepthAnything2Pipeline,
)


DATA_TYPE = os.environ.get("DEPTH_ANYTHING2_DATA_TYPE", "image")
DATA_PATH = os.environ.get(
    "DEPTH_ANYTHING2_DATA_PATH",
    "./worldfoundry/data/test_cases/test_image_case1/ref_image.png",
)
MODEL_PATH = os.environ.get("DEPTH_ANYTHING2_MODEL_PATH")
ENCODER = os.environ.get("DEPTH_ANYTHING2_ENCODER", "vitl")
DEVICE = os.environ.get(
    "DEPTH_ANYTHING2_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
INPUT_SIZE = int(os.environ.get("DEPTH_ANYTHING2_INPUT_SIZE", "518"))
OUTPUT_DIR = os.environ.get("DEPTH_ANYTHING2_OUTPUT_DIR", "./depth_anything_v2_output")
GRAYSCALE = os.environ.get("DEPTH_ANYTHING2_GRAYSCALE", "").lower() in {"1", "true", "yes"}


pipeline = DepthAnything2Pipeline.from_pretrained(
    pretrained_model_path=MODEL_PATH,
    encoder=ENCODER,
    device=DEVICE,
    data_type=DATA_TYPE,
    default_input_size=INPUT_SIZE,
)

results = pipeline(
    DATA_PATH,
    grayscale=GRAYSCALE,
)

saved_files = results.save(OUTPUT_DIR)
print("Saved files:", saved_files)
