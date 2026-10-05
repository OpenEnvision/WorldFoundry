
import pytest

# This test module imports worldfoundry code that requires the optional
# "imageio" dependency at import time; skip when it is unavailable.
pytest.importorskip("imageio")
import os

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.pipelines.worldfm.pipeline_worldfm import WorldFMPipeline


def _make_pose(tx=0.0, ty=0.0, tz=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


def test_worldfm_pipeline_gpu_integration(tmp_path):
    if os.environ.get("WORLDFM_RUN_INTEGRATION") != "1":
        pytest.skip("Set WORLDFM_RUN_INTEGRATION=1 with local WorldFM assets to run this GPU integration test.")

    device = os.environ.get("WORLDFM_DEVICE", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("WorldFM GPU integration test requested but CUDA is unavailable.")

    image_path = worldfoundry_data_path("test_cases", "test_image_case1", "ref_image.png")
    input_image = Image.open(image_path).convert("RGB")
    K = np.array(
        [
            [722.91626, 0.0, input_image.width / 2.0],
            [0.0, 722.91626, input_image.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target_poses = [
        _make_pose(0.0, 0.0, 0.0),
        _make_pose(0.05, 0.0, 0.0),
        _make_pose(0.10, 0.0, 0.0),
    ]
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
        device=device,
    )
    call_kwargs = {
        "K": K,
        "interactions": target_poses,
        "scene_name": "worldfm_demo",
        "output_dir": str(tmp_path / "worldfm_output"),
        "return_dict": True,
    }
    panorama_path = os.environ.get("WORLDFM_PANORAMA_PATH")
    if panorama_path:
        call_kwargs["panorama_path"] = panorama_path
    else:
        call_kwargs["images"] = input_image

    result = pipeline(**call_kwargs)
    if result["generated_video_path"] is None:
        fallback_video = tmp_path / "worldfm_demo.mp4"
        imageio.mimsave(fallback_video, result["frames"], fps=30)
        assert fallback_video.is_file()
    else:
        assert os.path.isfile(result["generated_video_path"])
