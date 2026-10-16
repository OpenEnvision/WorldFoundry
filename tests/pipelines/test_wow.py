import os
from pathlib import Path

import pytest
import torch


def test_wow_imports_do_not_require_runtime_dependencies():
    from worldfoundry.pipelines.wow.pipeline_wow import WoWArgs, WoWPipeline
    from worldfoundry.synthesis.visual_generation.wow.wow_synthesis import WoWSynthesis

    assert WoWArgs.__name__ == "WoWArgs"
    assert WoWPipeline.__name__ == "WoWPipeline"
    assert WoWSynthesis.__name__ == "WoWSynthesis"


@pytest.mark.gpu
def test_wow_gpu_integration_generates_video(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("GPU runtime is required for WoW integration.")

    model_dir = Path(os.environ.get("WORLDFOUNDRY_WOW_MODEL_DIR", "WoW-world-model/WoW-1-Wan-1.3B-2M"))
    if not model_dir.is_dir():
        pytest.skip(f"WoW checkpoint directory is not staged: {model_dir}")

    input_path = Path(
        os.environ.get(
            "WORLDFOUNDRY_WOW_INPUT_IMAGE",
            "worldfoundry/data/test_cases/test_vla_image_case1/init_frame.png",
        )
    )
    if not input_path.is_file():
        pytest.skip(f"WoW input image is not staged: {input_path}")

    from worldfoundry.base_models.diffusion_model.diffsynth import save_video
    from worldfoundry.pipelines.wow.pipeline_wow import WoWArgs, WoWPipeline

    args = WoWArgs(
        gpu=int(os.environ.get("WORLDFOUNDRY_WOW_GPU", "0")),
        steps=int(os.environ.get("WORLDFOUNDRY_WOW_STEPS", "4")),
        seed=int(os.environ.get("WORLDFOUNDRY_WOW_SEED", "42")),
        num_frames=int(os.environ.get("WORLDFOUNDRY_WOW_FRAMES", "17")),
        no_tiled=os.environ.get("WORLDFOUNDRY_WOW_TILED", "1") == "0",
        enable_vram_management=True,
        no_vram_management=False,
        persistent_param_gb=int(os.environ.get("WORLDFOUNDRY_WOW_PERSISTENT_PARAM_GB", "70")),
    )
    pipeline = WoWPipeline.from_pretrained(
        synthesis_model_path=str(model_dir),
        synthesis_args=args,
        device=f"cuda:{args.gpu}",
    )
    output_video = pipeline(
        input_path=input_path,
        text_prompt=os.environ.get("WORLDFOUNDRY_WOW_PROMPT", "Put the screw driver into the drawer."),
        args=args,
    )

    output_path = tmp_path / "wow_output.mp4"
    save_video(output_video, str(output_path), fps=15, quality=5)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
