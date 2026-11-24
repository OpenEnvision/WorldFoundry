from __future__ import annotations

import os
import sys
import textwrap

import pytest

# This test module imports worldfoundry code that requires the optional
# "imageio" dependency at import time; skip when it is unavailable.
pytest.importorskip("imageio")

import subprocess
from pathlib import Path

import numpy as np

from worldfoundry.operators.kairos_operator import KairosOperator
from worldfoundry.pipelines.kairos.pipeline_kairos import KairosPipeline
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.synthesis.visual_generation.kairos import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.kairos.runtime import KairosRuntime
from worldfoundry.zoo.catalog.registry import discover_model_registry


def test_kairos_dit_import_and_pytorch_attention_fallback(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["NUMEXPR_NUM_THREADS"] = "64"
    code = textwrap.dedent(
        """
        import math
        import torch
        import torch.nn.functional as F
        from worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.dits import kairos_dit as module

        norm = module.FusedRMSNorm(8, eps=1e-6)
        assert norm(torch.ones(2, 8)).shape == (2, 8)
        module.FLASH_ATTN_2_AVAILABLE = False
        module.FLASH_ATTN_3_AVAILABLE = False
        module.SAGE_ATTN_AVAILABLE = False
        torch.manual_seed(7)
        q = torch.randn(2, 5, 3, 4)
        k = torch.randn(2, 7, 3, 4)
        v = torch.randn(2, 7, 3, 4)
        qh = q.permute(0, 2, 1, 3)
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        output = module.flash_attention(q, k, v, num_heads=3)
        reference = F.scaled_dot_product_attention(qh, kh, vh).permute(0, 2, 1, 3)
        torch.testing.assert_close(output, reference)

        offset = k.shape[1] - q.shape[1]
        q_positions = torch.arange(q.shape[1]) + offset
        k_positions = torch.arange(k.shape[1])
        window = (2, 1)
        window_mask = (
            (k_positions.unsqueeze(0) >= (q_positions - window[0]).unsqueeze(1))
            & (k_positions.unsqueeze(0) <= (q_positions + window[1]).unsqueeze(1))
        )
        output, lse = module.flash_attention(
            q, k, v, num_heads=3, window_size=window, return_attn_probs=True
        )
        reference = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=window_mask
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(output, reference)
        logits = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        reference_lse = torch.logsumexp(logits.masked_fill(~window_mask, -torch.inf), dim=-1)
        torch.testing.assert_close(lse, reference_lse)

        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        output = module.flash_attention(q, k, v, num_heads=3, causal=True)
        reference = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=causal_mask
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(output, reference)

        user_mask = torch.ones(2, 1, 5, 7, dtype=torch.bool)
        user_mask[..., 3] = False
        output = module.flash_attention(
            q, k, v, num_heads=3, attn_mask=user_mask, window_size=window
        )
        reference = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=user_mask & window_mask
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(output, reference)
        """
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        cwd=tmp_path,
    )


def test_kairos_workspace_defaults_use_available_official_components(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    model_root = checkpoint_root / "kairos-agi--kairos-sensenova-4B-480P-pretrained"
    text_encoder_root = checkpoint_root / "Qwen--Qwen2.5-VL-7B-Instruct"
    vae_path = checkpoint_root / "Wan-AI--Wan2.1-T2V-14B" / "Wan2.1_VAE.pth"
    dit_path = model_root / "kairos-common-4B-480P.safetensors"
    for path in (dit_path, vae_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    text_encoder_root.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))

    entry = studio_catalog.find_entry("kairos-sensenova")

    assert entry.default_model_ref == str(model_root)
    assert entry.default_load_kwargs == {
        "models_root": str(model_root),
        "variant": "pretrained",
        "pretrained_dit": str(dit_path),
        "text_encoder_path": str(text_encoder_root),
        "vae_path": str(vae_path),
    }
    assert entry.default_call_kwargs["nproc_per_node"] == 1


def test_kairos_operator_accepts_prompt_and_image() -> None:
    operator = KairosOperator()
    operator.get_interaction(9)

    assert operator.process_interaction()["seed"] == 9
    assert operator.process_prompt("A waterfall.")["prompt"] == "A waterfall."
    assert operator.process_perception(images="/tmp/input.png")["images"] == "/tmp/input.png"


def test_kairos_pipeline_forwards_runtime_options() -> None:
    pipe = KairosPipeline.from_pretrained(
        model_path={
            "models_root": "/tmp/models",
            "config_path": "/tmp/config.py",
            "variant": "720p",
        },
        device="cuda:1",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.models_root == "/tmp/models"
    assert runtime.config_path == "/tmp/config.py"
    assert runtime.variant == "720p"
    assert runtime.device == "cuda:1"


def test_kairos_runtime_builds_torchrun_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    models_root = tmp_path / "models"
    config = tmp_path / "config.py"
    config.write_text(
        "KAIROS_MODEL_DIR = 'models'\npipeline = {'pipeline_args': {'vae_path': 'old', 'text_encoder_path': 'old'}}\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/operator/package-path")

    class FakePopen:
        def __init__(self, command, cwd, env, stdout=None, stderr=None, text=None, bufsize=None):
            del stdout, stderr, text, bufsize
            input_file = Path(command[command.index("--input_file") + 1])
            config_file = Path(command[command.index("--config_file") + 1])
            payload = json.loads(input_file.read_text())
            calls.append((command, cwd, env, payload, json.loads(config_file.read_text())))
            target = Path(payload["output_dir"])
            target.mkdir(parents=True, exist_ok=True)
            (target / "output.mp4").write_bytes(b"video")
            self.stdout = iter(())

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(runtime_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runtime_module, "_load_video_frames", lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8))
    model = KairosRuntime(models_root=models_root, config_path=config, python_executable="/tmp/python")
    result = model.predict(prompt="demo", output_path=tmp_path / "result.mp4", num_frames=2)

    assert len(calls) == 1
    command, cwd, env, payload, config_data = calls[0]
    assert env["NUMEXPR_MAX_THREADS"] == "64"
    assert env["NUMEXPR_NUM_THREADS"] == "64"
    assert env["PYTHONPATH"] == "/operator/package-path"
    assert "WORLDFOUNDRY_THIRDPARTY_ROOT" not in env
    assert command[:3] == ["/tmp/python", "-m", "torch.distributed.run"]
    assert command[command.index("--nproc_per_node") + 1] == "1"
    assert command[command.index("--master_port") + 1] == "29556"
    assert command[command.index("--config_file") + 1].endswith("kairos_config.json")
    assert command[8:10] == ["-m", "worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.examples.inference"]
    assert cwd == str(Path(runtime_module.__file__).parent / "kairos_runtime")
    assert config_data["KAIROS_MODEL_DIR"] == str(models_root)
    assert "runtime_root" not in result["runtime_plan"]
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_model_registry_contains_kairos() -> None:
    registry = discover_model_registry()
    model = registry.get("kairos-sensenova")

    assert model.has_loader is True
    assert model.has_infer is True
    assert model.pipeline_target == "worldfoundry.pipelines.kairos.pipeline_kairos:KairosPipeline"
