
import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from worldfoundry.synthesis.visual_generation.dualcamctrl.runtime import DualCamCtrlRuntime
from worldfoundry.base_models.diffusion_model.runners.staged import StagedDiffusionPipeline
from worldfoundry.core.media.codecs.video import coerce_video_frames


def test_dualcamctrl_resolves_shared_wan_files_from_checkpoint_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    base_root = tmp_path / "base"
    t2v_root = tmp_path / "Wan-AI" / "Wan2.1-T2V-1.3B"
    i2v_root = tmp_path / "Wan-AI" / "Wan2.1-I2V-14B-480P"
    tokenizer_root = t2v_root / "google" / "umt5-xxl"
    for directory in (base_root, t2v_root, i2v_root, tokenizer_root):
        directory.mkdir(parents=True, exist_ok=True)
    expected = (
        base_root / "diffusion_pytorch_model.safetensors",
        t2v_root / "models_t5_umt5-xxl-enc-bf16.pth",
        t2v_root / "Wan2.1_VAE.pth",
        i2v_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    for path in expected:
        path.touch()

    runtime = DualCamCtrlRuntime(load_model=False, base_model_path=base_root, local_model_path=tmp_path)

    assert tuple(Path(config.path) for config in runtime._base_model_configs()) == expected
    assert Path(runtime._model_config(runtime.tokenizer_repo, "google/*").path) == tokenizer_root


def test_dualcamctrl_resolves_control_checkpoint_from_checkpoint_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    checkpoint = tmp_path / "FayeHongfeiZhang" / "DualCamCtrl" / "checkpoints" / "dualcamctrl_diffusion_transformer.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    runtime = DualCamCtrlRuntime(load_model=False, local_model_path=tmp_path, allow_download=False)

    assert runtime._resolve_checkpoint_path() == str(checkpoint)


def test_dualcamctrl_resolves_flat_hfd_mirror_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    base_root = tmp_path / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    checkpoint = base_root / "diffusion_pytorch_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    runtime = DualCamCtrlRuntime(load_model=False, local_model_path=tmp_path, allow_download=False)
    config = runtime._model_config(
        runtime.base_model_repo,
        "diffusion_pytorch_model*.safetensors",
    )

    assert Path(config.path) == checkpoint


def test_staged_vae_frames_are_directly_accepted_by_video_writer() -> None:
    pipeline = object.__new__(StagedDiffusionPipeline)
    decoded = np.zeros((1, 3, 2, 4, 6), dtype=np.float32)
    frames = pipeline.vae_output_to_video(torch.from_numpy(decoded))

    assert all(isinstance(frame, Image.Image) for frame in frames)
    assert coerce_video_frames(frames).shape == (2, 4, 6, 3)
