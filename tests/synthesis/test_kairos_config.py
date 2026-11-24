# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.core.io.paths import package_data_path
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.synthesis.visual_generation.kairos.config import load_kairos_config
from worldfoundry.synthesis.visual_generation.kairos.runtime import KairosRuntime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "variant,pipeline_type",
    [
        ("720p", "KairosEmbodiedPipeline"),
        ("robot-480p-distilled", "KairosEmbodiedPipeline_DMD"),
    ],
)
def test_kairos_config_preserves_recipe_and_checkpoint_overrides(
    tmp_path: Path, variant: str, pipeline_type: str
) -> None:
    runtime = KairosRuntime(models_root=tmp_path / "weights", variant=variant)
    original = load_kairos_config(runtime.config_path)
    paths = {
        "pretrained_dit": "/checkpoints/dit.safetensors",
        "text_encoder_path": "/checkpoints/qwen-awq",
        "vae_path": "/checkpoints/vae.pth",
        "prompt_rewriter_path": "/checkpoints/rewriter",
    }
    resolved = load_kairos_config(runtime._write_config_json(tmp_path, paths))
    assert resolved["KAIROS_MODEL_DIR"] == str(tmp_path / "weights")
    assert resolved["pipeline"]["pipeline_type"] == pipeline_type
    assert resolved["pipeline"]["pretrained_dit"] == paths["pretrained_dit"]
    assert resolved["pipeline"]["pipeline_args"]["text_encoder_path"] == paths["text_encoder_path"]
    assert resolved["pipeline"]["pipeline_args"]["vae_path"] == paths["vae_path"]
    assert resolved["prompt_rewriter_path"] == paths["prompt_rewriter_path"]
    assert resolved["pipeline"]["pipeline_args"]["dit_config"] == original["pipeline"]["pipeline_args"]["dit_config"]
    assert resolved["pipeline"]["pipeline_args"]["vram_management_enabled"] is False
    restored = load_kairos_config(runtime._write_config_json(tmp_path, {}))
    assert restored["pipeline"]["pretrained_dit"].startswith(str(tmp_path / "weights"))
    assert load_kairos_config(runtime.config_path) == original


@pytest.mark.parametrize("source", ["import os\n", "x = dict(a=1)\n", "x = print('not config data')\n"])
def test_kairos_config_rejects_executable_statements(tmp_path: Path, source: str) -> None:
    config = tmp_path / "config.py"
    config.write_text(source)
    with pytest.raises(ValueError):
        load_kairos_config(config)


def test_kairos_config_reads_names_and_fstrings_without_imports(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text(
        'root = "/weights"\nvae = f"{root}/vae.pth"\npipeline = {"vae_path": vae, "steps": [4, 3, 2, 1]}\n'
    )
    assert load_kairos_config(config) == {
        "root": "/weights",
        "vae": "/weights/vae.pth",
        "pipeline": {"vae_path": "/weights/vae.pth", "steps": [4, 3, 2, 1]},
    }


@pytest.mark.parametrize("launcher", [None, "/opt/kairos/bin/torchrun"])
def test_kairos_command_preserves_explicit_launcher_and_rendezvous(launcher: str | None) -> None:
    runtime = KairosRuntime(python_executable="/opt/kairos/bin/python", torchrun_executable=launcher)
    prefix = [launcher] if launcher else ["/opt/kairos/bin/python", "-m", "torch.distributed.run"]
    assert runtime._command(Path("input.json"), Path("config.json"), {"nproc_per_node": 4, "master_port": 31129}) == [
        *prefix,
        "--nnodes=1",
        "--master_port",
        "31129",
        "--nproc_per_node",
        "4",
        "-m",
        "worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.examples.inference",
        "--input_file",
        "input.json",
        "--config_file",
        "config.json",
    ]


def test_kairos_declares_the_pinned_kernel_at_environment_setup() -> None:
    import yaml

    path = package_data_path("models", "runtime", "environments", "video", "kairos-sensenova.yaml")
    data = yaml.safe_load(path.read_text())
    dependency = (
        "sageattention @ git+https://github.com/kairos-agi/kairos-sensenova.git@"
        "40d3fcb518c3ff894501bf4b080ccaa589363adc#subdirectory=kairos/third_party/SageAttention"
    )
    assert dependency not in data["pip_packages"]
    assert any(dependency in command and "--no-build-isolation" in command for command in data["setup_commands"])
    entry = studio_catalog.find_entry("kairos-sensenova")
    assert "runtime_root" not in entry.default_load_kwargs
    assert "run_manage_libs" not in entry.call_params
    assert "master_port" in entry.call_params
    assert "run_manage_libs" not in KairosRuntime().defaults
