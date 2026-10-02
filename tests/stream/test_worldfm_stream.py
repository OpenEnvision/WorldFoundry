import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from worldfoundry.pipelines.worldfm.pipeline_worldfm import WorldFMPipeline


def _make_pose(tx=0.0, ty=0.0, tz=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


image_path = "./data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

K = np.array(
    [
        [722.91626, 0.0, input_image.width / 2.0],
        [0.0, 722.91626, input_image.height / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

required_components = {
    "hw_path": os.environ.get("WORLDFM_HY3DWORLD_PATH"),
    "moge_path": os.environ.get("WORLDFM_MOGE_PATH"),
    "moge_pretrained": os.environ.get("WORLDFM_MOGE_PRETRAINED"),
    "realesrgan_path": os.environ.get("WORLDFM_REALESRGAN_PATH"),
    "zim_path": os.environ.get("WORLDFM_ZIM_PATH"),
}
required_components = {key: value for key, value in required_components.items() if value}

pipeline = WorldFMPipeline.from_pretrained(
    model_path=os.environ.get("WORLDFM_MODEL_PATH", "inspatio/worldfm"),
    required_components=required_components,
    device="cuda",
)

first_turn = {
    "K": K,
    "interactions": [_make_pose(0.0, 0.0, 0.0), _make_pose(0.04, 0.0, 0.0)],
    "scene_name": "worldfm_stream_demo",
    "output_dir": "./worldfm_stream_output",
    "reset_memory": True,
}

panorama_path = os.environ.get("WORLDFM_PANORAMA_PATH")
if panorama_path:
    first_turn["panorama_path"] = panorama_path
else:
    first_turn["images"] = input_image

pipeline.stream(**first_turn)

output_frames = pipeline.stream(
    interactions=[_make_pose(0.08, 0.0, 0.0), _make_pose(0.12, 0.0, 0.0)],
    output_dir="./worldfm_stream_output",
)

imageio.mimsave(
    "./worldfm_stream_demo.mp4",
    pipeline.memory_module.all_frames if pipeline.memory_module.all_frames else output_frames,
    fps=30,
)
