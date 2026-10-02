"""Stage 6 tests for the bespoke NativeMiniMaxH3Pipeline orchestration."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.processor import VAEProcessor
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
)
from worldfoundry.pipelines.minimax import pipeline_minimax_h3 as minimax_pipeline
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import (
    NativeMiniMaxH3Pipeline,
    _resolve_component_devices,
    _resolve_canvas,
)


def test_resolve_canvas_landscape_and_portrait() -> None:
    h, w = _resolve_canvas(768, "16:9")
    assert h % (16 * 2) == 0 and w % (16 * 2) == 0
    assert w > h  # landscape
    h2, w2 = _resolve_canvas(768, "9:16")
    assert h2 > w2  # portrait
    hs, ws = _resolve_canvas(512, "1:1")
    assert hs == ws


def test_resolve_canvas_rejects_unknown_ratio() -> None:
    with pytest.raises(ValueError):
        _resolve_canvas(768, "5:7")


def test_component_placement_avoids_single_card_weight_overcommit() -> None:
    assert _resolve_component_devices("cuda", cuda_count=4) == {
        "transformer": "cuda:0",
        "text_encoder": "cuda:1",
        "video_vae": "cuda:2",
        "audio_vae": "cuda:3",
    }
    assert _resolve_component_devices("cuda", cuda_count=1) == {
        "transformer": "cuda:0",
        "text_encoder": "cpu",
        "video_vae": "cpu",
        "audio_vae": "cpu",
    }
    assert _resolve_component_devices("cuda:2", cuda_count=4)["text_encoder"] == "cpu"


def test_from_pretrained_routes_each_component_to_its_device(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    loaded = {}
    roots = {}
    partition = tmp_path / "FL2VA"
    partition.mkdir()
    (partition / "model_index.json").write_text("{}")

    def loader(name):
        def load(_cls, _root, *, device):
            loaded[name] = device
            roots[name] = _root
            return object()
        return classmethod(load)

    for name in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        monkeypatch.setattr(NativeMiniMaxH3Pipeline, f"_load_{name}", loader(name))
    monkeypatch.setattr(
        NativeMiniMaxH3Pipeline, "_load_tokenizer", classmethod(lambda _cls, _root: object())
    )

    pipe = NativeMiniMaxH3Pipeline.from_pretrained(model_path=str(tmp_path), device="cuda")

    assert loaded == pipe.component_devices
    assert pipe.device == "cuda:0"
    assert set(roots.values()) == {partition}


def test_public_prompt_encoding_passes_one_dimensional_ids() -> None:
    class Encoder:
        def encode_ids(self, input_ids):
            assert input_ids.ndim == 1
            return torch.zeros(len(input_ids), 5120, dtype=torch.bfloat16)

    class Tokenizer:
        def __call__(self, _prompt, *, return_tensors, add_special_tokens):
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    pipe = NativeMiniMaxH3Pipeline(text_encoder=Encoder(), tokenizer=Tokenizer(), device="cpu")

    assert pipe._encode_prompt("a cube").shape == (3, 5120)


def test_video_decode_reverses_official_pixel_normalization() -> None:
    processor = VAEProcessor(
        vae_ratio=16, vae_ratio_t=4, clip_length=17,
        frame_overlap=0, token_overlap=0, tokens_chunk_size=4,
        isolated_last_frame=False, latent_patch_size=1, crop_mode="center",
        use_3d_conv=True,
    )

    class VideoVAE:
        sglang_config = SimpleNamespace(
            arch_config=SimpleNamespace(latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
        )

        def __init__(self):
            self.processor = processor

        def decode_base(self, _latent):
            return torch.full((1, 3, 2, 2, 2), 1.5)

    pipe = NativeMiniMaxH3Pipeline(video_vae=VideoVAE(), device="cpu")
    video = pipe._decode_video(torch.zeros(1, 24, 1, 1, 1))

    assert video.shape == (1, 3, 2, 2, 2)
    assert torch.all(video >= 0) and torch.all(video <= 1)
    assert torch.allclose(video[0, :, 0, 0, 0], torch.tensor([0.8285, 0.792, 0.7435]))


def test_mp4_mux_preserves_aligned_video_and_pads_short_audio(tmp_path) -> None:
    ffmpeg = minimax_pipeline._resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip('ffmpeg is unavailable')
    import imageio

    # A four-second audio target aligns up to 107 video frames at 24 fps.
    video = torch.linspace(0, 1, 107).view(1, 1, 107, 1, 1).expand(1, 3, 107, 32, 32)
    samples = torch.arange(128000, dtype=torch.float32) / 32000
    tone = 0.2 * torch.sin(2 * torch.pi * 440 * samples)
    audio = tone.view(1, 1, -1).expand(2, 1, -1)
    output = tmp_path / 'aligned.mp4'

    NativeMiniMaxH3Pipeline._write_mp4_with_audio(
        video=video, audio=audio, output_path=output, fps=24
    )

    reader = imageio.get_reader(str(output))
    try:
        assert reader.count_frames() == 107
    finally:
        reader.close()
    tail = subprocess.run(
        [ffmpeg, '-v', 'error', '-ss', '4.2', '-i', str(output),
         '-map', '0:a:0', '-t', '0.15', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'],
        check=True, capture_output=True,
    ).stdout
    tail_samples = np.frombuffer(tail, dtype='<f4')
    assert tail_samples.size > 0
    assert float(np.max(np.abs(tail_samples))) < 0.001


def test_initial_video_noise_is_seeded_in_raw_latent_layout(monkeypatch) -> None:
    captured = {}

    def capture_loop(**kwargs):
        captured['video'] = kwargs['initial_video_rows']
        captured['audio'] = kwargs['initial_audio_rows']
        return captured['video'], captured['audio']

    monkeypatch.setattr(minimax_pipeline, 'minimax_h3_denoise_loop', capture_loop)
    pipe = NativeMiniMaxH3Pipeline(transformer=object(), device='cpu')
    plan = pipe._resolve_plan(task='t2va', short_edge=32, aspect_ratio='1:1', duration_seconds=4.0)
    result = pipe.generate(
        prompt_embeds=torch.zeros(3, 5120, dtype=torch.bfloat16),
        task='t2va', short_edge=32, aspect_ratio='1:1', duration_seconds=4.0,
        num_inference_steps=2, seed=42, return_latent_rows=True,
    )
    raw = torch.randn(
        1, 24, plan['latent_t'], plan['latent_h'], plan['latent_w'],
        generator=torch.Generator().manual_seed(42), dtype=torch.float32,
    )
    expected = minimax_pipeline.minimax_h3_patchify_video_latent(
        raw, patch_size=(1, 2, 2)
    )
    assert torch.equal(result['video_rows'], expected)
    assert torch.equal(
        result['audio_rows'],
        torch.randn(2 * plan['audio_t'], 32,
                    generator=torch.Generator().manual_seed(42), dtype=torch.float32),
    )


def _tiny_pipeline() -> tuple[NativeMiniMaxH3Pipeline, MiniMaxH3DiTArchConfig]:
    cfg = MiniMaxH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=64,
        time_embed_hidden_size=256,
        time_embed_dim=128,
        adaln_out_features=18 * 256,
        final_adaln_out_features=2 * 256,
        rope_inv_freq_len=16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    # Zero-init the AdaLN projections (as in the real checkpoint) so a
    # random-weight smoke model stays numerically stable in bf16.
    with torch.no_grad():
        for block in model.blocks:
            block.adaln_proj.linear.weight.zero_()
            block.adaln_proj.linear.bias.zero_()
        model.final_layer.adaln_proj.linear.weight.zero_()
        model.final_layer.adaln_proj.linear.bias.zero_()
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.mul_(0.1)
    pipe = NativeMiniMaxH3Pipeline(transformer=model, device=device)
    return pipe, cfg


def test_plan_resolution_dims() -> None:
    pipe, _ = _tiny_pipeline()
    plan = pipe._resolve_plan(task="t2va", short_edge=64, aspect_ratio="1:1", duration_seconds=5.0)
    assert plan["height"] == plan["width"]
    assert plan["latent_h"] == plan["height"] // 16
    assert plan["latent_t"] >= 2
    assert plan["audio_t"] == 200  # 5.0s * 40Hz


def test_generate_t2va_latent_rows_end_to_end() -> None:
    pipe, cfg = _tiny_pipeline()
    device = torch.device(pipe.device)
    # Small canvas so the packed sequence stays tiny.
    prompt_embeds = torch.randn(6, cfg.text_dim, dtype=torch.bfloat16, device=device)
    result = pipe.generate(
        prompt_embeds=prompt_embeds,
        task="t2va",
        short_edge=32,
        aspect_ratio="1:1",
        duration_seconds=4.0,
        num_inference_steps=3,
        seed=0,
        return_latent_rows=True,
    )
    assert result["task"] == "t2va"
    assert result["video_rows"].shape[-1] == 96
    assert result["audio_rows"].shape[-1] == 32
    assert torch.isfinite(result["video_rows"]).all()


def test_generate_t2va_decodes_to_latents() -> None:
    pipe, cfg = _tiny_pipeline()
    device = torch.device(pipe.device)
    prompt_embeds = torch.randn(6, cfg.text_dim, dtype=torch.bfloat16, device=device)
    result = pipe.generate(
        prompt_embeds=prompt_embeds,
        task="t2va",
        short_edge=32,
        aspect_ratio="1:1",
        duration_seconds=4.0,
        num_inference_steps=3,
        seed=0,
    )
    plan = result["plan"]
    vid = result["video_latent"]
    # [B, C, T, H, W]
    assert vid.shape[1] == 24
    assert vid.shape[2] == plan["latent_t"]
    assert vid.shape[3] == plan["latent_h"]
    assert vid.shape[4] == plan["latent_w"]
    aud = result["audio_latent"]
    assert aud.shape[0] == 2  # stereo channels-as-batch
    assert aud.shape[1] == 32
