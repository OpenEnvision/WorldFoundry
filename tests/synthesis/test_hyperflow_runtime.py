# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Public launch, reference presentation and sampling contracts for in-tree HyperFlow."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from worldfoundry.base_models.diffusion_model.models.denoisers.hyperflow import HyperFlowMetadata
from worldfoundry.base_models.diffusion_model.schedulers.hyperflow import DEFAULT_SIGMAS_8STEP, HyperFlowSchedule
from worldfoundry.pipelines.minimax._minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
from worldfoundry.synthesis.visual_generation.hyperflow import media as media_module
from worldfoundry.synthesis.visual_generation.hyperflow import pipeline as implementation
from worldfoundry.synthesis.visual_generation.hyperflow import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.hyperflow.conditioning import prepare_presentation, sample_video_frames
from worldfoundry.synthesis.visual_generation.hyperflow.media import PreparedMedia, ReferenceMedia, prepare_media
from worldfoundry.synthesis.visual_generation.hyperflow.runtime import HyperFlowRuntime
from worldfoundry.synthesis.visual_generation.hyperflow.sampling import denoise, prepare_latents

pytestmark = pytest.mark.unit


def _image(tmp_path: Path, name: str = "image.png") -> Path:
    path = tmp_path / name
    Image.fromarray(np.arange(17 * 29 * 3, dtype=np.uint8).reshape(17, 29, 3)).save(path)
    return path


@pytest.mark.parametrize("mode", ["t2va", "first", "last", "first_last", "ref2va"])
def test_public_plan_runs_the_in_tree_worker(tmp_path: Path, mode: str) -> None:
    image = _image(tmp_path)
    inputs = {
        "t2va": {},
        "first": {"image": image},
        "last": {"last_image": image},
        "first_last": {"images": [image, image]},
        "ref2va": {"references": [image]},
    }[mode]
    runtime = HyperFlowRuntime(model_path=tmp_path, weights_path=tmp_path, python_executable=sys.executable)
    plan = runtime.build_plan(prompt="A fox", output_path=tmp_path / "out.mp4", num_frames=120, **inputs)
    assert plan["command"][:3] == [
        str(Path(sys.executable).absolute()),
        "-m",
        "worldfoundry.synthesis.visual_generation.hyperflow.generate",
    ]
    assert plan["workflow"] == (mode if mode in ("t2va", "ref2va") else "fl2va")
    assert plan["num_frames"] == 124
    assert "PYTHONPATH" not in plan["env"]
    assert "hyperflow_h3" not in " ".join(plan["command"])


def test_parallel_plan_preserves_options_and_loopback_rendezvous(tmp_path: Path) -> None:
    runtime = HyperFlowRuntime(gpus=4, device="cuda:2")
    plan = runtime.build_plan(
        prompt="A fox",
        output_path=tmp_path / "out.mp4",
        height=384,
        width=640,
        attention_backend="fa3",
        sol_attn=True,
        offload=False,
        memory_reserve_margin="22GB",
    )
    command = plan["command"]
    assert command[1:3] == ["-m", "torch.distributed.run"]
    assert command[command.index("--nproc_per_node") + 1] == "4"
    assert "--master_addr=127.0.0.1" in command
    assert command[command.index("--attention-backend") + 1] == "fa3"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--memory-reserve-margin") + 1] == "22GB"
    assert "--no-offload" in command and "--sol-attn" in command


@pytest.mark.parametrize(
    "options",
    [
        {"height": 384},
        {"num_inference_steps": 7},
        {"fps": 30},
        {"num_frames": 346},
        {"width": 641, "height": 384},
    ],
)
def test_invalid_public_sampling_options_fail_before_loading(tmp_path: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HyperFlowRuntime().build_plan(prompt="A fox", output_path=tmp_path / "out.mp4", **options)


def test_ordered_audio_reference_requires_visual_context(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"local audio")
    runtime = HyperFlowRuntime()
    with pytest.raises(ValueError, match="image or video"):
        runtime.build_plan(prompt="A fox", output_path=tmp_path / "out.mp4", references=[audio])
    image = _image(tmp_path)
    plan = runtime.build_plan(prompt="A fox", output_path=tmp_path / "out.mp4", references=[audio, image])
    command = plan["command"]
    assert [command[index + 1] for index, flag in enumerate(command) if flag == "--ref"] == [
        f"audio:{audio}",
        f"image:{image}",
    ]


def test_preflight_probes_package_not_external_hyperflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def probe(python: Path, modules: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        calls.append(modules)
        return {"ok": False, "error": "test environment"}

    monkeypatch.setattr(runtime_module, "python_module_probe", probe)
    result = HyperFlowRuntime(model_path=tmp_path).preflight("ref2va")
    assert result["status"] == "blocked"
    assert calls[0][0] == "worldfoundry.synthesis.visual_generation.hyperflow.generate"
    assert "peft" not in calls[0] and "hyperflow_h3" not in calls[0]
    assert any("transformer_ref" in path for path in result["missing_or_incomplete_files"])


def test_runtime_and_help_are_import_light() -> None:
    code = """
import builtins, runpy, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'diffusers', 'transformers'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from worldfoundry.synthesis.visual_generation.hyperflow.runtime import HyperFlowRuntime
sys.argv = ['generate', '--help']
runpy.run_module('worldfoundry.synthesis.visual_generation.hyperflow.generate', run_name='__main__')
"""
    checked = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stderr
    assert "--sol-attn" in checked.stdout


def test_public_pipeline_keeps_loading_and_artifact_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldfoundry.pipelines.hyperflow.pipeline_hyperflow import HyperFlowPipeline

    pipeline = HyperFlowPipeline.from_pretrained(
        model_path=tmp_path,
        weights_path=tmp_path,
        python_executable=sys.executable,
        device="cuda",
        gpus=1,
    )
    recorded = []

    def run(plan: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        recorded.append((plan, timeout_seconds))
        return {"video": plan["output_path"], "status": "succeeded"}

    monkeypatch.setattr(pipeline.runtime, "run", run)
    result = pipeline(prompt="A fox", output_dir=tmp_path, return_dict=True, timeout_seconds=123)
    assert result["video"] == str(tmp_path / "hyperflow.mp4")
    assert recorded[0][1] == 123
    assert recorded[0][0]["workflow"] == "t2va"


def test_keyframe_geometry_stretches_anchor_and_cover_crops_follower(tmp_path: Path) -> None:
    first, last = _image(tmp_path), _image(tmp_path, "last.png")
    prepared = prepare_media(image=first, last_image=last, height=32, width=64)
    assert prepared.keyframe_indices == (0, -1)
    assert prepared.workflow == "fl2va"
    source = Image.open(first).convert("RGB")
    expected_first = source.resize((64, 32), Image.Resampling.LANCZOS)
    expected_last = source.resize((64, 38), Image.Resampling.LANCZOS).crop((0, 3, 64, 35))
    assert np.array_equal(np.asarray(prepared.references[0].image), np.asarray(expected_first))
    assert np.array_equal(np.asarray(prepared.references[1].image), np.asarray(expected_last))
    assert prepare_media(last_image=last, height=32, width=64).keyframe_indices == (-1,)


def test_reference_audio_truncates_at_source_rate_before_resampling() -> None:
    from torchaudio.transforms import Resample

    waveform = torch.linspace(-1, 1, 90000)[None]
    actual = media_module.normalize_audio(waveform, 16000, 124)
    source = waveform[:, : int(124 / 24 * 16000)].expand(2, -1).contiguous()
    expected = Resample(16000, 32000)(source)
    assert torch.equal(actual, expected)
    assert actual.shape[0] == 2
    assert torch.equal(actual[0], actual[1])


def test_reference_video_resampling_keeps_half_up_frame_slots() -> None:
    frames = np.broadcast_to(np.arange(5, dtype=np.uint8)[:, None, None, None], (5, 768, 768, 3))
    actual = media_module.normalize_video(frames, 48, 124)
    assert actual[:, 0, 0, 0].tolist() == [0, 2, 4]
    assert actual.shape == (3, 768, 768, 3)


def test_video_sampling_uses_2fps_and_half_even_labels() -> None:
    frames = np.arange(26, dtype=np.uint8)[:, None, None, None].repeat(3, axis=3)
    sampled, times = sample_video_frames(frames, 2)
    assert sampled[:, 0, 0, 0].tolist() == [0, 12, 24]
    assert times == [0.25, 1.0]
    assert f"{times[0]:.1f}" == "0.2"
    with pytest.raises(ValueError, match="13 frames"):
        sample_video_frames(frames[:12], 2)


class _Tokenizer:
    def __call__(self, value: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [ord(char) for char in value]}

    def convert_tokens_to_ids(self, value: str) -> int:
        return {"<|vision_start|>": 256, "<|vision_end|>": 257, "<|image_pad|>": 258, "<|video_pad|>": 259}[value]


class _ImageProcessor:
    merge_size = 2

    def __call__(self, *, images: list[Any], **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)), "pixel_values": torch.zeros(len(images), 8)}


class _VideoProcessor:
    temporal_patch_size = 2

    def __call__(self, *, videos: list[np.ndarray], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["do_sample_frames"] is False
        return {
            "video_grid_thw": torch.tensor([[(len(video) + 1) // 2, 2, 2] for video in videos]),
            "pixel_values_videos": torch.zeros(1, 8),
        }


def test_reference_presentation_keeps_audio_video_image_order() -> None:
    processor = SimpleNamespace(image_processor=_ImageProcessor(), video_processor=_VideoProcessor())
    references = [
        ReferenceMedia("video", frames=np.zeros((26, 2, 2, 3), dtype=np.uint8), audio=torch.zeros(2, 9)),
        ReferenceMedia("image", image=Image.new("RGB", (2, 2))),
        ReferenceMedia("audio", audio=torch.zeros(2, 9)),
    ]
    ids, tags, vision = prepare_presentation(_Tokenizer(), processor, "Prompt", references)
    text = "".join(chr(value) for value in ids if value < 256)
    assert text == "<Audio 1>: <Video 1>: <0.2 seconds><1.0 seconds><Picture 1>: <Audio 2>: Prompt"
    assert tags.tolist() == [0 if value >= 256 else 1 for value in ids]
    assert set(vision) == {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}


def test_noise_order_and_reference_rows_are_reproducible() -> None:
    media = PreparedMedia("fl2va", 32, 32, 124, [], (0,))
    condition = torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2)
    first = prepare_latents(
        media,
        torch.ones(3, dtype=torch.long),
        [condition],
        [],
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(42)
    condition_noise = torch.randn(condition.shape, generator=generator)
    video_noise = torch.randn((1, 24, 37, 2, 2), generator=generator)
    audio_noise = torch.randn((414, 32), generator=generator)
    time = condition.new_tensor(0.999)
    expected = minimax_h3_patchify_video_latent(time * condition + (1 - time) * condition_noise, patch_size=(1, 2, 2))
    assert torch.equal(first.video[:1], expected)
    assert torch.equal(first.video[1:], minimax_h3_patchify_video_latent(video_noise, patch_size=(1, 2, 2)))
    assert torch.equal(first.audio, audio_noise)
    repeated = prepare_latents(
        media,
        torch.ones(3, dtype=torch.long),
        [condition],
        [],
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    assert torch.equal(first.video, repeated.video)
    assert torch.equal(first.audio, repeated.audio)


class _Velocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[int] = []

    def forward(
        self, *, hidden_states: torch.Tensor, audio_hidden_states: torch.Tensor, step: int, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.steps.append(step)
        return torch.ones_like(hidden_states), torch.ones_like(audio_hidden_states)


def test_eight_step_loop_never_updates_conditioning_rows() -> None:
    refs = [ReferenceMedia("image"), ReferenceMedia("audio", audio=torch.ones(2, 20))]
    media = PreparedMedia("ref2va", 32, 32, 124, refs)
    latents = prepare_latents(
        media,
        torch.ones(3, dtype=torch.long),
        [torch.ones(1, 24, 1, 2, 2)],
        [torch.ones(4, 32)],
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    video_condition = latents.video[:1].clone()
    audio_condition = latents.audio[:4].clone()
    model = _Velocity()
    video, audio = denoise(model, latents, torch.ones(1, 3, 48), HyperFlowSchedule())
    assert model.steps == list(range(8))
    assert video.shape == (1, 24, 37, 2, 2)
    assert audio.shape == (2, 32, 207)
    assert torch.equal(latents.video[:1], video_condition)
    assert torch.equal(latents.audio[:4], audio_condition)


class _Codec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.register_buffer("statistics", torch.ones(2))

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


def test_component_codec_calls_fire_the_offload_forward_hook() -> None:
    component = implementation.HyperFlowComponent(_Codec())
    calls = []
    component.register_forward_pre_hook(lambda module, args: calls.append("forward"))
    assert component.device == torch.device("cpu")
    assert component.get_memory_footprint() == 24
    assert torch.equal(component(torch.ones(4), method="decode"), torch.ones(4))
    assert calls == ["forward"]


def test_offload_margin_is_armed_after_every_component_is_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class Manager:
        def add(self, name: str, component: nn.Module) -> None:
            events.append(("add", name))

        def enable_auto_cpu_offload(self, **kwargs: Any) -> None:
            events.append(("enable", kwargs))

    original = builtins.__import__

    def patched(name: str, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(ComponentsManager=Manager) if name == "diffusers" else original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched)
    components = {
        "vae": implementation.HyperFlowComponent(_Codec()),
        "transformer": implementation.HyperFlowComponent(_Codec()),
    }
    implementation._place_components(components, device=torch.device("cpu"), offload=True, margin="24GB")
    assert events == [
        ("add", "vae"),
        ("add", "transformer"),
        ("enable", {"device": torch.device("cpu"), "memory_reserve_margin": "24GB"}),
    ]
    assert (
        implementation._place_components(components, device=torch.device("cpu"), offload=False, margin="24GB") is None
    )


class _ReferenceCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixels: list[torch.Tensor] = []

    def _moments(self, pixels: torch.Tensor, frames: int) -> torch.Tensor:
        self.pixels.append(pixels)
        shape = (1, 24, frames, 2, 2)
        return torch.cat([torch.full(shape, 0.1234567), torch.zeros(shape)], dim=1)

    def _adaptive_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._moments(pixels, 1)

    def encode_temporal(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._moments(pixels, 7)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.shape == (2, 1, 1600)
        return torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)


def test_reference_encoding_preserves_posterior_seed_rounding_and_audio_row_order() -> None:
    codec = _ReferenceCodec()
    component = implementation.HyperFlowComponent(codec)
    components = implementation.HyperFlowComponents(
        transformer=component,
        vae=component,
        audio_vae=component,
        text_encoder=None,
        metadata=HyperFlowMetadata(0.25, 256, 256, DEFAULT_SIGMAS_8STEP, 12, 3),
        video_config={"latents_mean": [0.25] * 24, "latents_std": [1.5] * 24},
        audio_config={"latents_mean": [1.0, 2.0], "latents_std": [2.0, 4.0]},
        manager=None,
        conditioner_manager=None,
    )
    references = [
        ReferenceMedia("image", image=Image.new("RGB", (32, 32), (0, 128, 255))),
        ReferenceMedia("video", frames=np.zeros((26, 32, 32, 3), dtype=np.uint8)),
        ReferenceMedia("audio", audio=torch.ones(2, 1600)),
    ]
    rng_state = torch.random.get_rng_state()
    video, audio = implementation.encode_references(components, references, torch.device("cpu"))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert [pixels.shape[2] for pixels in codec.pixels] == [1, 22]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    assert torch.equal(codec.pixels[0][0, :, 0, 0, 0], (torch.tensor([0, 128, 255]).float() / 255 - mean) / std)
    for actual, frames in zip(video, (1, 7), strict=True):
        noise = torch.randn((1, 24, frames, 2, 2), generator=torch.Generator().manual_seed(42))
        expected = ((noise + 0.1234567).half().float() - 0.25) / 1.5
        assert torch.equal(actual, expected)
    encoded_audio = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).transpose(1, 2)
    expected_audio = (encoded_audio - torch.tensor([1.0, 2.0])) / torch.tensor([2.0, 4.0])
    assert torch.equal(audio[0], expected_audio.reshape(-1, 2))


def test_local_mux_retains_video_frames_and_stereo_audio(tmp_path: Path) -> None:
    import av

    output = tmp_path / "out.mp4"
    media_module.write_video_audio(torch.zeros(3, 32, 32, 3, dtype=torch.uint8), torch.zeros(2, 4800), output)
    with av.open(str(output)) as container:
        assert container.streams.video[0].average_rate == 24
        assert len(list(container.decode(video=0))) == 3
        assert container.streams.audio[0].codec_context.sample_rate == 32000
        assert container.streams.audio[0].codec_context.layout.name == "stereo"


def test_generate_main_normalizes_unindexed_cuda_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.synthesis.visual_generation.hyperflow import generate

    set_devices: list[torch.device] = []
    infer_devices: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: set_devices.append(torch.device(dev)))
    monkeypatch.setattr(media_module, "prepare_media", lambda **kwargs: "prepared")
    monkeypatch.setattr(
        implementation,
        "run_inference",
        lambda *args, **kwargs: (infer_devices.append(kwargs["device"]) or ("v", "a")),
    )
    monkeypatch.setattr(media_module, "write_video_audio", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.attention.resolve_attention_backend",
        lambda backend, gpus=1: backend,
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--model",
            str(tmp_path / "model"),
            "--weights",
            str(tmp_path / "weights.safetensors"),
            "--prompt",
            "A fox",
            "--output",
            str(tmp_path / "out.mp4"),
            "--device",
            "cuda",
        ],
    )

    assert generate.main() == 0
    assert set_devices == [torch.device("cuda", 0)]
    assert infer_devices == [torch.device("cuda", 0)]
