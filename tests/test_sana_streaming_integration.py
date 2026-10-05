from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.base_models.diffusion_model.models.networks.sana.ops import fused_gdn
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_v2v_attn_blocks import (
    _bidirectional_triton_gdn_is_supported,
)
from worldfoundry.base_models.diffusion_model.recipes.sana import sana_recipe
from worldfoundry.cli.model_run import load_model_run_schema
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.runtime.inference_catalog import (
    SANA_STREAMING_DEMO_PROMPT,
    get_model_inference_spec,
)
from worldfoundry.studio.inference import catalog as studio_catalog


@pytest.mark.parametrize(
    "model_id",
    (
        "sana-streaming-2b-720p",
        "sana-streaming-bidirectional-2b-720p",
    ),
)
def test_sana_streaming_typed_input_path_materializes_as_video(model_id: str) -> None:
    schema = load_model_run_schema(model_id)
    input_field = next(field for field in schema.fields if field.option == "--pipeline.input-path")

    assert input_field.label == "Source Video"
    assert input_field.input_key == "video"


def test_image_conditioned_typed_input_path_remains_an_image() -> None:
    schema = load_model_run_schema("ati-wan21-14b")
    input_field = next(field for field in schema.fields if field.option == "--pipeline.input-path")

    assert input_field.input_key == "image"


def test_fused_gdn_accepts_torch_build_without_smem_device_property(monkeypatch) -> None:
    monkeypatch.setattr(fused_gdn.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        fused_gdn.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(),
    )

    config = fused_gdn._get_kernel_config()

    assert config["STATE_FP32"] is True
    assert config["num_warps"] == 8


@pytest.mark.parametrize(
    "model_id",
    (
        "sana-streaming-2b-720p",
        "sana-streaming-bidirectional-2b-720p",
    ),
)
def test_sana_streaming_runtime_profile_emits_mp4(model_id: str) -> None:
    profile = load_runtime_profile(model_id, check_conda_env_exists=False)

    assert profile.artifact_kind == "generated_video"
    assert profile.artifact_filename.endswith(".mp4")


@pytest.mark.parametrize(
    "model_id",
    (
        "sana-streaming-2b-720p",
        "sana-streaming-bidirectional-2b-720p",
    ),
)
def test_sana_streaming_demo_prompt_matches_the_default_source_video(model_id: str) -> None:
    spec = get_model_inference_spec(model_id)
    assert spec is not None
    prompt_field = next(field for field in spec.tasks[0].inputs if field.field_id == "prompt")

    assert prompt_field.default == SANA_STREAMING_DEMO_PROMPT
    assert studio_catalog.CURATED_OVERRIDES[model_id]["default_prompt"] == SANA_STREAMING_DEMO_PROMPT


def test_bidirectional_streaming_uses_the_profile_declared_diffusers_vae() -> None:
    recipe = sana_recipe("sana-streaming-bidirectional-2b-720p")

    assert recipe.checkpoints["codec"].files == (
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
    )
    assert "build_diffusers_ltx2_tensor_video_codec" in {
        component.factory.__name__ for component in recipe.components
    }


def test_sana_streaming_inference_contract_honors_hfd_root(tmp_path: Path) -> None:
    hfd_root = tmp_path / "hfd"
    env = dict(os.environ)
    env["WORLDFOUNDRY_HFD_ROOT"] = str(hfd_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from worldfoundry.runtime.inference_catalog import get_model_inference_spec; "
                "spec = get_model_inference_spec('sana-streaming-bidirectional-2b-720p'); "
                "print(json.dumps(spec.variants[0].checkpoint_map()))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoints = json.loads(result.stdout)

    assert checkpoints == {
        "primary": str(
            hfd_root
            / "Efficient-Large-Model--SANA-Streaming_bidirectional"
            / "dit"
            / "sana_bidirectional_short.pth"
        ),
        "vae": str(hfd_root / "Lightricks--LTX-2"),
        "text_encoder": str(hfd_root / "Efficient-Large-Model--gemma-2-2b-it"),
    }


def test_bidirectional_fused_gdn_rejects_triton_31_llvm_backend() -> None:
    assert _bidirectional_triton_gdn_is_supported("3.1.0") is False
    assert _bidirectional_triton_gdn_is_supported("3.2.0") is True
