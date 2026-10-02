
import pytest

# This test module imports worldfoundry code that requires the optional
# "imageio" dependency at import time; skip when it is unavailable.
pytest.importorskip("imageio")
import os
from pathlib import Path

import imageio
import pytest
from PIL import Image

from worldfoundry.evaluation.utils import worldfoundry_data_path


def test_lyra1_gpu_integration(tmp_path):
    if os.environ.get("LYRA1_RUN_INTEGRATION") != "1":
        pytest.skip("Set LYRA1_RUN_INTEGRATION=1 with local Lyra-1 checkpoints to run this GPU integration test.")

    device = os.environ.get("LYRA1_DEVICE", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Lyra-1 GPU integration test requested but CUDA is unavailable.")

    from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline

    repo_root = os.environ.get(
        "LYRA1_REPO_ROOT",
        str(Path(__file__).resolve().parents[1] / "thirdparty" / "Lyra-1"),
    )
    cache_root = os.environ.get(
        "LYRA1_CACHE_ROOT",
        str(Path(__file__).resolve().parents[1] / "cache" / "hfd" / "Lyra"),
    )
    checkpoint_dir = os.environ.get("LYRA1_CHECKPOINT_DIR", cache_root)
    static_ckpt_path = os.environ.get("LYRA1_STATIC_CKPT_PATH", os.path.join(cache_root, "lyra_static.pt"))
    input_image = Image.open(worldfoundry_data_path("test_cases", "test_image_case1", "ref_image.png")).convert("RGB")
    pipeline = Lyra1Pipeline.from_pretrained(
        model_path=repo_root,
        required_components={
            "checkpoint_dir": checkpoint_dir,
            "static_ckpt_path": static_ckpt_path,
            "default_mode": "static",
        },
        device=device,
    )
    output_video = pipeline(
        images=input_image,
        prompt="A narrow old-town street with warm storefront lights and a path extending ahead.",
        interactions=["forward", "camera_l", "forward", "right"],
        mode="static",
    )
    target = tmp_path / "lyra1_demo.mp4"
    imageio.mimsave(target, output_video, fps=24)
    assert target.is_file()
