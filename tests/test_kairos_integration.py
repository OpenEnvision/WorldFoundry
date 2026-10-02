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

from worldfoundry.evaluation.models import discover_model_registry
from worldfoundry.operators.kairos_operator import KairosOperator
from worldfoundry.pipelines.kairos.pipeline_kairos import KairosPipeline
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.synthesis.visual_generation.kairos import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.kairos.runtime import KairosRuntime


def test_kairos_dit_import_and_pytorch_attention_fallback() -> None:
    runtime_root = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/kairos/kairos_runtime"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["NUMEXPR_NUM_THREADS"] = "64"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(runtime_root), env.get("PYTHONPATH", "")) if part
    )
    code = textwrap.dedent(
        """
        import math
        import torch
        import torch.nn.functional as F
        from kairos.modules.dits import kairos_dit as module

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
        cwd=runtime_root,
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
        "runtime_root": "",
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
            "runtime_root": "/tmp/kairos",
            "models_root": "/tmp/models",
            "config_path": "/tmp/config.py",
            "variant": "720p",
        },
        device="cuda:1",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.runtime_root == "/tmp/kairos"
    assert runtime.models_root == "/tmp/models"
    assert runtime.config_path == "/tmp/config.py"
    assert runtime.variant == "720p"
    assert runtime.device == "cuda:1"


def test_kairos_runtime_builds_torchrun_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "kairos-sensenova"
    models_root = tmp_path / "models"
    config = runtime_root / "kairos" / "configs" / "kairos_4b_config_DMD.py"
    (runtime_root / "examples").mkdir(parents=True)
    (runtime_root / "kairos" / "third_party").mkdir(parents=True)
    config.parent.mkdir(parents=True)
    (runtime_root / "examples" / "inference.py").write_text("print('stub')\n", encoding="utf-8")
    (runtime_root / "kairos" / "third_party" / "manage_libs.py").write_text("print('libs')\n", encoding="utf-8")
    config.write_text(
        "KAIROS_MODEL_DIR = 'models'\n"
        "pipeline = {'pipeline_args': {'vae_path': 'old', 'text_encoder_path': 'old'}}\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(command, check, cwd, env, stdout=None, stderr=None, text=None):
        del check, env, stdout, stderr, text
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="")

    class FakePopen:
        def __init__(self, command, cwd, env, stdout=None, stderr=None, text=None, bufsize=None):
            del env, stdout, stderr, text, bufsize
            calls.append((command, cwd))
            output_dir = Path(command[command.index("--input_file") + 1]).read_text(encoding="utf-8")
            marker = '"output_dir": "'
            target = output_dir.split(marker, 1)[1].split('"', 1)[0]
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "output.mp4").write_bytes(b"video")
            self.stdout = iter(())

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        runtime_module,
        "_load_video_frames",
        lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    model = KairosRuntime(
        runtime_root=runtime_root,
        models_root=models_root,
        config_path=config,
        python_executable="/tmp/python",
        defaults={"run_manage_libs": True},
    )
    subprocess_env = model._subprocess_env({})
    assert subprocess_env["NUMEXPR_MAX_THREADS"] == "64"
    assert subprocess_env["NUMEXPR_NUM_THREADS"] == "64"
    result = model.predict(
        prompt="demo",
        output_path=tmp_path / "result.mp4",
        num_frames=2,
        run_manage_libs=True,
    )

    assert calls[0][0][:2] == ["/tmp/python", str(runtime_root / "kairos" / "third_party" / "manage_libs.py")]
    command = calls[1][0]
    assert command[:3] == ["/tmp/python", "-m", "torch.distributed.run"]
    assert command[command.index("--nproc-per-node") + 1] == "1"
    assert command[command.index("--config_file") + 1].endswith("kairos_config.py")
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_model_registry_contains_kairos() -> None:
    registry = discover_model_registry()
    model = registry.get("kairos-sensenova")

    assert model.has_loader is True
    assert model.has_infer is True
    assert model.pipeline_target == "worldfoundry.pipelines.kairos.pipeline_kairos:KairosPipeline"
