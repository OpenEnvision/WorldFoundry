
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


def test_lyra2_gpu_integration(tmp_path):
    if os.environ.get("LYRA2_RUN_INTEGRATION") != "1":
        pytest.skip("Set LYRA2_RUN_INTEGRATION=1 with local Lyra-2 checkpoints to run this GPU integration test.")

    device = os.environ.get("LYRA2_DEVICE", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Lyra-2 GPU integration test requested but CUDA is unavailable.")

    from worldfoundry.pipelines.lyra.pipeline_lyra2 import Lyra2Pipeline

    repo_root = os.environ.get(
        "LYRA2_REPO_ROOT",
        str(Path(__file__).resolve().parents[1] / "thirdparty" / "Lyra-2"),
    )
    checkpoint_dir = os.environ.get("LYRA2_CHECKPOINT_DIR", os.path.join(repo_root, "checkpoints/model"))
    negative_prompt_path = os.environ.get(
        "LYRA2_NEGATIVE_PROMPT_PATH",
        os.path.join(repo_root, "checkpoints/text_encoder/negative_prompt.pt"),
    )
    da3_model_path = os.environ.get(
        "LYRA2_DA3_MODEL_PATH",
        os.path.join(repo_root, "checkpoints/recon/model.pt"),
    )
    input_image = Image.open(worldfoundry_data_path("test_cases", "test_image_case1", "ref_image.png")).convert("RGB")
    pipeline = Lyra2Pipeline.from_pretrained(
        model_path=repo_root,
        required_components={
            "checkpoint_dir": checkpoint_dir,
            "negative_prompt_path": negative_prompt_path,
            "da3_model_path_custom": da3_model_path,
        },
        device=device,
    )
    output_video = pipeline(
        images=input_image,
        prompt="A cozy village street with timber houses, warm lights, and a walkable alley ahead.",
        interactions=["forward", "camera_l", "forward", "right"],
        fps=16,
    )
    target = tmp_path / "lyra2_demo.mp4"
    imageio.mimsave(target, output_video, fps=16)
    assert target.is_file()
