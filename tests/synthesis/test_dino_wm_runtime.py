# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.synthesis.visual_generation.dino_wm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit
PACKAGE = "worldfoundry.synthesis.visual_generation.dino_wm"


def test_planning_exports_computed_actions_without_changing_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np
    import torch

    tree = ast.parse((runtime.RUNTIME_DIR / "official_plan.py").read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "perform_planning"
    )
    namespace: dict[str, Any] = {"np": np, "json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "official_plan.py", "exec"), namespace)
    actions = torch.tensor([[[0.2, -0.3], [0.4, 0.1]]], requires_grad=True)
    original = actions.detach().clone()
    observation = object()
    goal = object()

    def plan(**kwargs: Any) -> tuple[torch.Tensor, None]:
        assert kwargs == {"obs_0": observation, "obs_g": goal, "actions": None}
        return actions, None

    def evaluate(value: torch.Tensor, length: None, **kwargs: Any) -> tuple[dict[str, Any], None, None, None]:
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert length is None
        assert kwargs == {"save_video": True, "filename": "output_final"}
        return {"score": np.float32(0.5)}, None, None, None

    workspace = SimpleNamespace(
        debug_dset_init=False,
        planner=SimpleNamespace(plan=plan),
        obs_0=observation,
        obs_g=goal,
        evaluator=SimpleNamespace(eval_actions=evaluate),
        wandb_run=SimpleNamespace(log=lambda _: None),
        log_filename=str(tmp_path / "planning.log"),
    )
    monkeypatch.chdir(tmp_path)
    result = namespace["perform_planning"](workspace)
    assert result == {"final_eval/score": 0.5}
    np.testing.assert_array_equal(np.load(tmp_path / "planned_actions.npy", allow_pickle=False), original.numpy())
    torch.testing.assert_close(actions, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "relative_path",
    [
        "encoder/dino.yaml",
        "encoder/dino_cls.yaml",
        "encoder/identity.yaml",
        "encoder/r3m.yaml",
        "encoder/resnet.yaml",
        "decoder/transposed_conv.yaml",
        "decoder/vqvae.yaml",
        "action_encoder/proprio.yaml",
        "action_encoder/repeat.yaml",
        "proprio_encoder/proprio.yaml",
        "proprio_encoder/repeat.yaml",
        "predictor/vit.yaml",
    ],
)
def test_packaged_model_config_targets_resolve_to_owned_classes(relative_path: str) -> None:
    config = OmegaConf.load(runtime.DEFAULT_CONFIG.parent / relative_path)

    assert config._target_.startswith(f"{PACKAGE}.models.")
    target = get_obj_from_str(config._target_)
    assert callable(target)
    assert target.__module__.startswith(f"{PACKAGE}.models.")


def test_r3m_config_cleanup_resolves_the_owned_class() -> None:
    from worldfoundry.synthesis.visual_generation.dino_wm.models.encoder.r3m import R3M, cleanup_config

    config = OmegaConf.create({"agent": {"_target_": "r3m.R3M", "size": 18, "langweight": 1.0}})
    cleaned = cleanup_config(config)

    assert get_obj_from_str(cleaned._target_) is R3M
    assert cleaned.langweight == 0
    assert config.agent._target_ == "r3m.R3M"


@pytest.mark.parametrize("relative_path", ["env/__init__.py", "env/pointmaze/__init__.py"])
def test_gym_registration_targets_resolve_inside_the_env_package(relative_path: str) -> None:
    tree = ast.parse((runtime.RUNTIME_DIR / relative_path).read_text())
    targets = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register"
        for keyword in node.keywords
        if keyword.arg == "entry_point" and isinstance(keyword.value, ast.Constant)
    ]
    assert targets
    repo = Path(__file__).resolve().parents[2]
    for target in targets:
        assert isinstance(target, str)
        assert target.startswith(f"{PACKAGE}.env.")
        module, name = target.split(":")
        source = repo.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            source = repo.joinpath(*module.split("."), "__init__.py")
        definitions = ast.parse(source.read_text()).body
        assert any(
            (isinstance(node, ast.ClassDef) and node.name == name)
            or (isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == name for alias in node.names))
            for node in definitions
        )


@pytest.mark.parametrize("feature_key", ["x_norm_patchtokens", "x_norm_clstoken"])
def test_encoder_uses_canonical_factory_with_eager_attention(monkeypatch: pytest.MonkeyPatch, feature_key: str) -> None:
    import torch

    from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones
    from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.attention import Attention
    from worldfoundry.base_models.perception_core.general_perception.dinov2.models.vision_transformer import (
        DinoVisionTransformer,
    )
    from worldfoundry.synthesis.visual_generation.dino_wm.models.dino import DinoV2Encoder

    model = DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2, block_chunks=0
    ).eval()
    monkeypatch.setattr(backbones, "dinov2_vits14", lambda: model)

    def reject_remote(*args: object, **kwargs: object) -> None:
        raise AssertionError("external code loader called")

    monkeypatch.setattr(torch.hub, "load", reject_remote)
    encoder = DinoV2Encoder("dinov2_vits14", feature_key).eval()
    image = torch.randn(1, 3, 28, 42)
    expected = model.forward_features(image)[feature_key]
    if feature_key == "x_norm_clstoken":
        expected = expected.unsqueeze(1)

    assert encoder.base_model is model
    assert encoder.emb_dim == 32
    assert encoder.patch_size == 14
    assert all(not module.fused_attn for module in model.modules() if isinstance(module, Attention))
    torch.testing.assert_close(encoder(image), expected, rtol=0, atol=0)


def test_checkpoint_maps_released_encoder_classes_without_global_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    from functools import partial
    from types import ModuleType

    import torch

    from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones
    from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.attention import MemEffAttention
    from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.block import NestedTensorBlock
    from worldfoundry.base_models.perception_core.general_perception.dinov2.models.vision_transformer import (
        DinoVisionTransformer,
    )
    from worldfoundry.core.nn.blocks.layers import LayerScale, Mlp, PatchEmbed
    from worldfoundry.synthesis.visual_generation.dino_wm.checkpoint import load_checkpoint
    from worldfoundry.synthesis.visual_generation.dino_wm.models.dino import DinoV2Encoder
    from worldfoundry.synthesis.visual_generation.dino_wm.models.proprio import ProprioceptiveEmbedding

    model = DinoVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        init_values=1.0,
        block_chunks=0,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
    )
    monkeypatch.setattr(backbones, "dinov2_vits14", lambda: model)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").eval()
    action_encoder = ProprioceptiveEmbedding(in_chans=2, emb_dim=4).eval()
    image = torch.randn(1, 3, 28, 42)
    actions = torch.randn(1, 2, 2)
    expected = encoder(image)
    expected_actions = action_encoder(actions)
    expected_state = encoder.state_dict()
    for module in model.modules():
        if isinstance(module, Mlp):
            del module.drop1, module.drop2
        if isinstance(module, MemEffAttention):
            del module.fused_attn
    names = {
        DinoV2Encoder: "models.dino",
        ProprioceptiveEmbedding: "models.proprio",
        DinoVisionTransformer: "dinov2.models.vision_transformer",
        MemEffAttention: "dinov2.layers.attention",
        NestedTensorBlock: "dinov2.layers.block",
        Mlp: "dinov2.layers.mlp",
        PatchEmbed: "dinov2.layers.patch_embed",
        LayerScale: "dinov2.layers.layer_scale",
    }
    path = tmp_path / "model_latest.pth"
    with monkeypatch.context() as writer:
        for cls, name in names.items():
            parts = name.split(".")
            for count in range(1, len(parts) + 1):
                module_name = ".".join(parts[:count])
                if module_name not in sys.modules:
                    writer.setitem(sys.modules, module_name, ModuleType(module_name))
            writer.setattr(cls, "__module__", name)
            writer.setattr(sys.modules[name], cls.__name__, cls, raising=False)
        torch.save({"encoder": encoder, "action_encoder": action_encoder, "epoch": 65}, path)
    namespaces = {name: sys.modules.get(name) for name in names.values()}
    loaded = load_checkpoint(path, map_location="cpu")

    assert loaded["epoch"] == 65
    assert type(loaded["encoder"]) is DinoV2Encoder
    assert type(loaded["encoder"].base_model) is DinoVisionTransformer
    assert loaded["encoder"].state_dict().keys() == expected_state.keys()
    for key, value in expected_state.items():
        torch.testing.assert_close(loaded["encoder"].state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded["encoder"](image), expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded["action_encoder"](actions), expected_actions, rtol=0, atol=0)
    assert {name: sys.modules.get(name) for name in names.values()} == namespaces


@pytest.mark.parametrize("container", [False, True])
def test_checkpoint_loads_direct_and_wrapped_decoders(tmp_path: Path, container: bool) -> None:
    import torch

    from worldfoundry.synthesis.visual_generation.dino_wm.checkpoint import load_checkpoint
    from worldfoundry.synthesis.visual_generation.dino_wm.models.vqvae import VQVAE

    decoder = VQVAE(channel=16, n_res_block=1, n_res_channel=4, emb_dim=8, n_embed=16, quantize=False).eval()
    checkpoint = tmp_path / "decoder.pth"
    torch.save({"decoder": decoder} if container else decoder, checkpoint)
    loaded = load_checkpoint(checkpoint)
    restored = loaded["decoder"] if container else loaded
    value = torch.randn(1, 8, 2, 2)
    torch.testing.assert_close(restored.decode(value), decoder.decode(value), rtol=0, atol=0)


def test_model_config_maps_released_targets_without_changing_values() -> None:
    from worldfoundry.synthesis.visual_generation.dino_wm.checkpoint import MODEL_MODULES, model_config

    config = OmegaConf.create(
        {
            "model": {"_target_": "models.visual_world_model.VWorldModel", "image_size": 224},
            "encoder": {"_target_": "models.dino.DinoV2Encoder", "name": "dinov2_vits14"},
            "predictor": {"_target_": "models.vit.ViTPredictor", "depth": 6},
            "decoder": {"_target_": "models.vqvae.VQVAE", "quantize": False},
            "proprio_encoder": {"_target_": "models.proprio.ProprioceptiveEmbedding", "emb_dim": 10},
            "action_encoder": {"_target_": "models.proprio.ProprioceptiveEmbedding", "emb_dim": 10},
            "num_action_repeat": 1,
        }
    )
    resolved = model_config(config)
    assert resolved.num_action_repeat == 1
    for key in ("model", "encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"):
        original = config[key]._target_
        module, _, name = original.rpartition(".")
        assert resolved[key]._target_ == f"{MODEL_MODULES[module]}.{name}"
        assert get_obj_from_str(resolved[key]._target_).__name__ == name
        assert config[key]._target_ == original
        assert {k: v for k, v in resolved[key].items() if k != "_target_"} == {
            k: v for k, v in config[key].items() if k != "_target_"
        }


def test_build_command_launches_package_with_unchanged_planner_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATASET_DIR", str(tmp_path / "trajectories"))
    options = {
        "ckpt_base_path": str(tmp_path / "checkpoints"),
        "model_name": "wall_single",
        "model_epoch": "65",
        "seed": 7,
        "n_evals": 2,
        "goal_source": "dset",
        "goal_H": 5,
    }
    command = runtime.build_command(
        {
            "python": "/env/bin/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_dir": str(tmp_path / "out"),
            "options": options,
        }
    )
    assert command[:3] == ["/env/bin/python", "-m", f"{PACKAGE}.infer"]
    assert command[3:] == [
        "--config",
        str(runtime.DEFAULT_CONFIG),
        "--ckpt-base-path",
        options["ckpt_base_path"],
        "--model-name",
        "wall_single",
        "--model-epoch",
        "65",
        "--output-dir",
        str(tmp_path / "out"),
        "--dataset-dir",
        str(tmp_path / "trajectories"),
        "--seed",
        "7",
        "--n-evals",
        "2",
        "--goal-source",
        "dset",
        "--goal-h",
        "5",
    ]
