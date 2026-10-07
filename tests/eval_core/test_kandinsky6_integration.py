from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model.recipes.kandinsky6 import CHECKPOINT_NAMES
from worldfoundry.base_models.diffusion_model.recipes.kandinsky6_config import _CONFIGS, load_config
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.operators.kandinsky6_operator import Kandinsky6Operator
from worldfoundry.pipelines.kandinsky6 import Kandinsky6Pipeline


@pytest.mark.parametrize("name", CHECKPOINT_NAMES)
def test_recipe_and_official_configuration(name):
    recipe = default_native_diffusion_registry().resolve("kandinsky6-" + name)
    cfg = load_config(_CONFIGS / "checkpoints" / (name + ".yaml"))
    assert recipe.execution.strategy == "kandinsky6-joint"
    assert cfg.dit.is_multimodal
    assert cfg.dit.in_audio_dim == cfg.dit.out_audio_dim
    assert cfg.generation.sample_frames == 121
    assert cfg.piflow.enabled == ("distill" in name)
    if cfg.piflow.enabled:
        assert cfg.dit.out_visual_dim == 16
        assert cfg.generation.guidance_weight == 1.0


def test_operator_normalizes_first_frame_and_rejects_unsupported_media(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    request = Kandinsky6Operator().prepare("A walking cat", images=[image], seed=42)
    assert request.inputs["image"] == str(image.resolve())
    assert request.sampling.seed == 42
    assert request.num_frames == 121
    with pytest.raises(ValueError, match="input video"):
        Kandinsky6Operator().prepare("Cat", video="clip.mp4")
    with pytest.raises(ValueError, match="4.n.1"):
        Kandinsky6Operator().prepare("Cat", num_frames=120)


def test_local_checkpoint_preflight_fails_before_model_loading(tmp_path):
    with pytest.raises(FileNotFoundError):
        Kandinsky6Pipeline.from_pretrained(tmp_path, device="cpu", offload_mode="none")


def test_native_pipeline_writes_muxed_artifact_and_report(tmp_path):
    from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.mux import mux_video_audio

    class Native:
        model_id = "kandinsky6-lite"

        def __call__(self, request):
            assert request.sampling.num_inference_steps == 50
            assert request.sampling.guidance_scale == 5.0
            frames = torch.zeros((3, 5, 16, 16), dtype=torch.uint8)
            audio = np.zeros(44100 // 4, dtype=np.int16)
            mux_video_audio(frames, audio, request.inputs["save_path"])
            return SimpleNamespace(artifacts={"audio": [audio]}, metadata={"seed": request.sampling.seed})

    result = Kandinsky6Pipeline(Native()).__call__("A cat", output_path=tmp_path / "output.json", return_dict=True)
    import av

    with av.open(result["artifact_path"]) as clip:
        assert len(clip.streams.video) == 1
        assert len(clip.streams.audio) == 1
        assert len(list(clip.decode(video=0))) == 5
    assert Path(result["metadata_path"]).is_file()
    assert result["has_audio"] is True


@pytest.mark.parametrize("distilled", [False, True])
def test_joint_sampling_with_small_native_transformer(distilled):
    from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.dit import DiffusionTransformer3D
    from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.piflow_dit import (
        PiFlowDiffusionTransformer3D,
    )
    from worldfoundry.base_models.diffusion_model.runners.kandinsky6_sampling import Kandinsky6Sampler
    from worldfoundry.base_models.diffusion_model.schedulers.kandinsky6.sampler_config import PiFlowConfig

    class Encoder:
        def encode(self, texts):
            assert texts in (["Cat"], ["Bad"])
            return (
                {
                    "text_embeds": torch.zeros(2, 8, dtype=torch.bfloat16),
                    "pooled_embed": torch.zeros(1, 4, dtype=torch.bfloat16),
                },
                torch.tensor([0, 2], dtype=torch.int32),
                torch.ones(2, dtype=torch.bool),
            )

    class Video(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def decode(self, x):
            return SimpleNamespace(sample=x[:, :3].float())

    class Audio:
        scaling_factor = 1.0

        def wrapped_decode(self, x):
            return x.float()

    class Vocoder:
        def __call__(self, x):
            return x[:, :1].mean(dim=1)

    common = dict(
        in_visual_dim=4,
        out_visual_dim=4,
        in_text_dim=8,
        in_text_dim2=4,
        time_dim=12,
        model_dim=12,
        ff_dim=24,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(2, 2, 2),
        in_audio_dim=4,
        is_multimodal=True,
        text_token_padding=True,
        attention_engine="sdpa",
        visual_cond=False,
    )
    model = (
        PiFlowDiffusionTransformer3D(n_grid=2, out_audio_dim=4, **common)
        if distilled
        else DiffusionTransformer3D(**common)
    )
    sampler = Kandinsky6Sampler(
        model.bfloat16().eval(),
        Encoder(),
        Video(),
        torch.device("cpu"),
        audio_vae=Audio(),
        vocoder=Vocoder(),
        piflow_conf=PiFlowConfig(enabled=distilled, dx_num_grid_points=2, num_policy_substeps=2),
    )
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
    from worldfoundry.base_models.diffusion_model.runners.kandinsky6 import Kandinsky6Runner

    runner = Kandinsky6Runner(model_id="kandinsky6-lite", sampler=sampler, components={})
    output = runner.run(
        DiffusionRequest(
            prompt="Cat",
            negative_prompt="Bad",
            height=16,
            width=16,
            num_frames=5,
            sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1.0 if distilled else 2.0, seed=42),
        )
    )
    assert output.sample.shape == (1, 3, 2, 2, 2)
    assert len(output.artifacts["audio"]) == 1
    assert output.artifacts["audio"][0].dtype == np.int16


def test_spatial_vae_tiles_cover_partial_bottom_and_right_edges():
    from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.hunyuan_vae import (
        AutoencoderKLHunyuanVideo,
    )

    vae = AutoencoderKLHunyuanVideo.__new__(AutoencoderKLHunyuanVideo)
    torch.nn.Module.__init__(vae)
    vae.spatial_compression_ratio = 1
    vae.tile_sample_min_height = 4
    vae.tile_sample_min_width = 4
    vae.tile_sample_stride_height = 3
    vae.tile_sample_stride_width = 3
    vae.post_quant_conv = torch.nn.Identity()
    vae.decoder = torch.nn.Identity()
    latents = torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 8, 8)
    decoded = vae.tiled_decode(latents).sample
    torch.testing.assert_close(decoded, latents)


def test_checkpoint_config_binds_local_snapshot_without_mutating_defaults(tmp_path):
    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.loaders.kandinsky6 import component_config
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy

    weights = tmp_path / "transformer/diffusion_pytorch_model.safetensors"
    weights.parent.mkdir()
    weights.touch()
    ctx = ComponentBuildContext(
        model_id="kandinsky6-lite",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device=torch.device("cpu")),
        checkpoints={
            "weights": CheckpointSpec(source=tmp_path, files=("transformer/diffusion_pytorch_model.safetensors",))
        },
        recipe_options={"checkpoint_name": "lite"},
    )
    cfg, device = component_config(ctx)
    assert cfg.paths.dit == str(weights)
    assert cfg.audio_vae.tod_vae_ckpt == str(tmp_path / "audio_vae")
    assert (
        load_config(_CONFIGS / "checkpoints/lite.yaml").paths.dit == "transformer/diffusion_pytorch_model.safetensors"
    )


def test_standard_evaluation_invocation(tmp_path):
    from worldfoundry.evaluation.api.generation import GenerationRequest
    from worldfoundry.evaluation.models.pipelines.invocation import build_pipeline_invocation, invoke_pipeline

    class Native:
        model_id = "kandinsky6-pro-distill"

        def __call__(self, request):
            Path(request.inputs["save_path"]).write_bytes(b"video")
            return SimpleNamespace(artifacts={"audio": None}, metadata={})

    invocation = build_pipeline_invocation(
        request=GenerationRequest(sample_id="cat", task_name="text-to-video", inputs={"prompt": "A cat"}),
        output_dir=tmp_path,
        artifact_filename="generated.mp4",
    )
    result = invoke_pipeline(Kandinsky6Pipeline(Native()), invocation)
    assert result["status"] == "success"
    assert Path(result["artifact_path"]).is_file()


def test_meta_transformer_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import save_file

    from worldfoundry.base_models.diffusion_model.loaders.kandinsky6 import create_bare_dit
    from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.dit import DiffusionTransformer3D
    from worldfoundry.base_models.diffusion_model.recipes.kandinsky6_config import PipelineConfig

    settings = dict(
        in_visual_dim=4,
        out_visual_dim=4,
        in_text_dim=8,
        in_text_dim2=4,
        time_dim=12,
        model_dim=12,
        ff_dim=24,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(2, 2, 2),
        in_audio_dim=4,
        is_multimodal=True,
    )
    original = DiffusionTransformer3D(**settings, attention_engine="sdpa").bfloat16()
    path = tmp_path / "transformer.safetensors"
    save_file({name: value.clone() for name, value in original.state_dict().items()}, str(path))
    cfg = PipelineConfig.model_validate(
        {"dit": settings, "paths": {"dit": str(path), "vae": "unused", "qwen": "unused", "clip": "unused"}}
    )
    loaded = create_bare_dit(cfg, "cpu", "sdpa")
    assert all(not buffer.is_meta for buffer in loaded.buffers())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
