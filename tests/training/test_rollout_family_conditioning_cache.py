from __future__ import annotations

import argparse
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import Conditioning
from worldfoundry.cli.training_commands import register_training_subparser
from worldfoundry.cli.training_commands.common import training_family
from worldfoundry.core.io.integrity import canonical_json
from worldfoundry.training.data.hunyuan_video import HunyuanVideoTextFeatureEncoder
from worldfoundry.training.data.ltx import LTXTextFeatureEncoder
from worldfoundry.training.data.ltx.rollout_cache import (
    LTX_ROLLOUT_CONDITIONING_LAYOUTS,
    materialize_ltx_rollout_conditioning_cache,
)
from worldfoundry.training.data.rollout_cache import (
    RolloutConditioningDataset,
    prepare_rollout_conditioning_cache,
)
from worldfoundry.training.data.rollout_manifest import RolloutPromptDataset, RolloutPromptRecord
from worldfoundry.training.data.wan.encoding import WanTextFeatureEncoder
from worldfoundry.training.data.wan22 import WAN22_T2V_A14B_REPOSITORY, wan22_text_checkpoints
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES


def _prompts(tmp_path) -> RolloutPromptDataset:
    prompt = "a small robot walking"
    record = RolloutPromptRecord(
        prompt_id="prompt",
        prompt=prompt,
        safety_audit=PromptSafetyAudit(
            prompt=prompt,
            unsafe_probabilities={name: 0.01 for name in SHIELDGEMMA_PROMPT_POLICIES},
            threshold=0.5,
        ),
        generation={"height": 64, "width": 96, "num_frames": 9},
    )
    path = tmp_path / "prompts.jsonl"
    path.write_text(canonical_json(record.to_dict()) + "\n", encoding="utf-8")
    return RolloutPromptDataset.from_file(path)


class _LTXConditioner:
    def __init__(self) -> None:
        self.inputs: dict[str, object] = {}

    def encode(self, request, *, device, dtype):
        self.inputs = dict(request.inputs)
        return Conditioning(
            positive={
                "video_context": torch.arange(12, device=device, dtype=dtype).reshape(1, 3, 4),
                "audio_context": torch.ones(1, 2, 4, device=device, dtype=dtype),
                "context_mask": torch.ones(1, 3, device=device, dtype=torch.int64),
            }
        )


@pytest.mark.parametrize(
    ("model_recipe", "expected_keys"),
    (
        (
            "hunyuanvideo-t2v",
            {"text_states", "text_mask", "text_states_2"},
        ),
        (
            "hunyuanvideo-1.5-t2v",
            {"text_states", "text_mask", "byt5_text_states", "byt5_text_mask"},
        ),
    ),
)
def test_hunyuan_conditioner_cache_contains_adapter_inputs(
    tmp_path,
    model_recipe: str,
    expected_keys: set[str],
) -> None:
    class Conditioner:
        def encode(self, request, *, device, dtype):
            values: dict[str, object] = {
                "text_states": torch.zeros(1, 3, 8, device=device, dtype=dtype),
                "text_mask": torch.ones(1, 3, device=device, dtype=torch.int64),
            }
            if model_recipe == "hunyuanvideo-t2v":
                values["text_states_2"] = torch.zeros(1, 6, device=device, dtype=dtype)
            else:
                values.update(
                    {
                        "text_states_2": None,
                        "byt5_text_states": torch.zeros(1, 2, 5, device=device, dtype=dtype),
                        "byt5_text_mask": torch.ones(1, 2, device=device, dtype=torch.int64),
                    }
                )
            return Conditioning(positive=values)

    prompts = _prompts(tmp_path)
    encoder = HunyuanVideoTextFeatureEncoder(
        Conditioner(),
        model_recipe=model_recipe,
        device="cpu",
        dtype=torch.float32,
    )
    cache = tmp_path / "cache"
    prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache,
        encoder=encoder,
        model_recipe=model_recipe,
        conditioner={"component": "conditioner"},
        tokenizer={"component": "tokenizer"},
        tensor_layouts=encoder.tensor_layouts,
    )

    sample = RolloutConditioningDataset(prompts, cache)[0]
    assert set(sample.conditioning) == expected_keys


def test_ltx_conditioner_cache_keeps_joint_av_context_and_forwards_fps(tmp_path) -> None:
    prompts = _prompts(tmp_path)
    conditioner = _LTXConditioner()
    encoder = LTXTextFeatureEncoder(conditioner, device="cpu", dtype=torch.float32)
    cache = tmp_path / "cache"
    prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache,
        encoder=encoder,
        model_recipe="ltx-2.3-i2v",
        conditioner={"component": "conditioner"},
        tokenizer={"component": "tokenizer"},
        encoder_options={"fps": 24.0},
        tensor_layouts=LTX_ROLLOUT_CONDITIONING_LAYOUTS,
    )

    sample = RolloutConditioningDataset(prompts, cache)[0]
    assert set(sample.conditioning) == {"video_context", "audio_context", "context_mask"}
    assert tuple(sample.conditioning["audio_context"].shape) == (2, 4)
    assert conditioner.inputs == {"fps": 24.0, "frame_rate": 24.0}


@pytest.mark.parametrize(
    ("recipe_name", "checkpoint_file"),
    (
        ("ltx_2_video_flow_grpo.yaml", "ltx-2-19b-dev.safetensors"),
        ("ltx_2p3_video_flow_grpo.yaml", "ltx-2.3-22b-dev.safetensors"),
    ),
)
def test_ltx_rollout_cache_uses_the_policy_dev_checkpoint_for_projection(
    monkeypatch,
    tmp_path,
    recipe_name: str,
    checkpoint_file: str,
) -> None:
    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training" / recipe_name)
    _prompts(tmp_path)
    captured: dict[str, object] = {}
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)

    def build_components(self, native_recipe, **kwargs):
        del self, native_recipe
        captured["model"] = kwargs["checkpoint_overrides"]["model"]
        return {conditioner_key: _LTXConditioner()}

    monkeypatch.setattr(NativeDiffusionAssembler, "build_components", build_components)
    result = materialize_ltx_rollout_conditioning_cache(
        recipe,
        manifest_path=tmp_path / "prompts.jsonl",
        cache_dir=tmp_path / "conditioning",
        device="cpu",
    )

    assert captured["model"].files == (checkpoint_file,)
    assert result.index.conditioner["projection"]["files"] == [checkpoint_file]
    cached = RolloutConditioningDataset(
        RolloutPromptDataset.from_file(tmp_path / "prompts.jsonl"),
        tmp_path / "conditioning",
    )[0]
    assert "audio_context" in cached.conditioning


def test_wan22_conditioner_cache_contains_only_context(tmp_path) -> None:
    class Conditioner:
        def __init__(self) -> None:
            self.text_encoder = nn.Linear(1, 1, bias=False)

        def encode(self, request, *, device, dtype):
            return Conditioning(positive={"context": torch.zeros(1, 3, 4, device=device, dtype=dtype)})

    prompts = _prompts(tmp_path)
    encoder = WanTextFeatureEncoder(Conditioner(), text_length=3, context_features=4)
    cache = tmp_path / "cache"
    prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache,
        encoder=encoder,
        model_recipe="wan2.2-t2v-a14b",
        conditioner={"component": "umt5"},
        tokenizer={"component": "umt5-tokenizer"},
    )

    sample = RolloutConditioningDataset(prompts, cache)[0]
    assert set(sample.conditioning) == {"context"}
    assert tuple(sample.conditioning["context"].shape) == (3, 4)


def test_wan22_text_assets_come_from_the_official_a14b_repository() -> None:
    checkpoints = wan22_text_checkpoints()
    assert checkpoints.text_encoder.repo_id == WAN22_T2V_A14B_REPOSITORY
    assert checkpoints.tokenizer.repo_id == WAN22_T2V_A14B_REPOSITORY
    assert checkpoints.text_encoder.files == ("models_t5_umt5-xxl-enc-bf16.pth",)
    assert "google/umt5-xxl/tokenizer.json" in checkpoints.tokenizer.files


@pytest.mark.parametrize(
    ("recipe_name", "module_name", "function_name", "family"),
    (
        (
            "ltx_2_video_flow_grpo.yaml",
            "worldfoundry.training.data.ltx.rollout_cache",
            "materialize_ltx_rollout_conditioning_cache",
            "ltx",
        ),
        (
            "hunyuan_video_1p5_flow_grpo.yaml",
            "worldfoundry.training.data.hunyuan_video.rollout_cache",
            "materialize_hunyuan_video_rollout_conditioning_cache",
            "hunyuan-video",
        ),
        (
            "wan22_t2v_a14b_flow_grpo.yaml",
            "worldfoundry.training.data.wan22.rollout_cache",
            "materialize_wan22_rollout_conditioning_cache",
            "wan22",
        ),
    ),
)
def test_train_cache_routes_rollout_conditioner_by_model_family(
    monkeypatch,
    tmp_path,
    capsys,
    recipe_name: str,
    module_name: str,
    function_name: str,
    family: str,
) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train-cache",
            "--recipe",
            str(root / "configs/post_training" / recipe_name),
            "--base-dir",
            str(root),
            "--manifest",
            str(tmp_path / "prompts.jsonl"),
            "--cache",
            str(tmp_path / "conditioning"),
            "--device",
            "cpu",
        ]
    )
    called: dict[str, object] = {}

    def materialize(recipe, **kwargs):
        called["model_recipe"] = recipe.model.recipe
        called.update(kwargs)
        return SimpleNamespace(
            index=SimpleNamespace(to_dict=lambda: {"entries": []}),
            entries=(),
            unconditional_conditioning=None,
        )

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, function_name, materialize)

    assert args.func(args) == 0
    assert training_family(str(called["model_recipe"])) == family
    assert called["manifest_path"] == (tmp_path / "prompts.jsonl").resolve()
    assert json.loads(capsys.readouterr().out)["prompt_count"] == 0
