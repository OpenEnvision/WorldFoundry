import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile_manifest
from worldfoundry.pipelines.helixworld import HelixWorldPipeline

MODEL_DATA = Path(__file__).resolve().parents[2] / "worldfoundry" / "data" / "models"


def test_helixworld_catalog_recipe_and_public_pipeline_resolve():
    registry = load_model_zoo_registry()
    entry = registry.get("helixworld-preview")
    assert entry.model_id == "helixworld"
    assert entry.integration_status == "integrated"
    spec = build_pipeline_runner_spec(WorldModelConfig(model_id="helixworld", runner="worldfoundry.pipeline"))
    assert spec.pipeline_target == "worldfoundry.pipelines.helixworld:HelixWorldPipeline"
    profile = load_runtime_profile_manifest(MODEL_DATA / "runtime" / "profiles" / "helixworld.yaml")
    assert profile.execution["defaults"]["num_frames"] == 121
    recipe = default_native_diffusion_registry().resolve("NoizAI/HelixWorld-preview")
    assert recipe.execution.strategy == "joint-chunked"
    assert recipe.checkpoints["model"].files == ("weights/model.safetensors",)
    assert HelixWorldPipeline._checkpoint_overrides("/weights/model.safetensors", {"text_encoder_path": "/weights/gemma"}) == {
        "model": "/weights/model.safetensors", "gemma": "/weights/gemma", "tokenizer": "/weights/gemma",
    }



@pytest.mark.parametrize("audio_samples", [12000, 4000])
def test_helixworld_exports_fhwc_float_video_with_audio(tmp_path, audio_samples):
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        pytest.skip("FFmpeg and ffprobe are required for the MP4 export contract")
    frames = torch.zeros(6, 32, 32, 3)
    frames[..., 1] = .75
    audio = torch.full((2, audio_samples), .1)
    requests = []

    def generate(request):
        requests.append(request)
        return DiffusionOutput(sample=frames, latents=torch.zeros(1),
                               artifacts={"audio": audio, "audio_sampling_rate": 24000})

    pipeline = HelixWorldPipeline.__new__(HelixWorldPipeline)
    pipeline.model_id = "helixworld"
    pipeline.generation_type = "i2v"
    pipeline.process = lambda **kwargs: {"prompt": kwargs["prompt"], "images": kwargs["images"]}
    pipeline.native_pipeline = generate
    path = tmp_path / "helixworld.mp4"
    result = pipeline(prompt="Forest.", images="first.png", audio_prompt="Birds.", av_prompt="Birds in trees.",
                      action_plan="W", fps=12, output_path=path, return_dict=True)
    assert requests[0].inputs["actions"] == "W"
    assert requests[0].inputs["frame_rate"] == 12
    assert requests[0].sampling.num_inference_steps == 4
    assert result["video"].shape == (1, 3, 6, 32, 32)
    assert result["audio"] is audio and result["audio_sampling_rate"] == 24000
    assert result["artifact_path"] == str(path)
    info = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)]))
    assert {stream["codec_type"] for stream in info["streams"]} == {"audio", "video"}
    video_stream = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    assert int(video_stream["nb_frames"]) == 6
    rgb = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    assert 180 <= rgb[1] <= 200 and rgb[0] < 5 and rgb[2] < 5
    pipeline(prompt="Forest.", images="first.png", frame_rate=20, return_dict=True)
    assert requests[-1].inputs["frame_rate"] == requests[-1].inputs["fps"] == 20
