# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""LingBot versions load the same task-token alignment graph and joint metadata."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.lingbot.modeling import TaskTokenDepthHead
from worldfoundry.base_models.llm_mllm_core.action.lingbot.preprocessing import (
    FeatureInfo,
    prepare_action,
    prepare_state,
)

pytestmark = pytest.mark.unit


def test_lingbot_task_token_head_strictly_reloads_checkpoint_layout() -> None:
    config = {
        "dim_head": 4,
        "dim_out": 12,
        "num_layers": 2,
        "num_heads": 4,
        "num_backbone_tokens": 3,
        "ff_mult": 2,
    }
    head = TaskTokenDepthHead(config, llm_hidden_size=16).eval()
    reloaded = TaskTokenDepthHead(config, llm_hidden_size=16).eval()
    state = head.state_dict()
    assert state["projector.proj_in1.weight"].shape == (16, 16)
    assert state["projector.layers.0.0.to_kv.weight"].shape == (32, 16)
    reloaded.load_state_dict(state, strict=True)
    features, queries = torch.randn(2, 7, 16), torch.randn(2, 3, 16)
    torch.testing.assert_close(head(features, queries), reloaded(features, queries), atol=0, rtol=0)


def test_lingbot_versions_use_canonical_joint_metadata() -> None:
    from worldfoundry.synthesis.action_generation.lingbot_vla.preprocessing.features import FeatureInfo as V1FeatureInfo
    from worldfoundry.synthesis.action_generation.lingbot_vla_v2.preprocessing.features import (
        FeatureInfo as V2FeatureInfo,
    )

    assert V1FeatureInfo is V2FeatureInfo is FeatureInfo
    metadata = FeatureInfo()
    metadata.update_info(
        SimpleNamespace(joints=[{"arm": 6}, "{'gripper': 1}", {"unused": 0}], cameras=["front", "wrist"])
    )
    assert metadata.joints == ["arm", "gripper"]
    assert metadata.images == ["observation.images.front", "observation.images.wrist"]
    assert metadata.joints_max_dim == {"arm": 6, "gripper": 1}


def test_lingbot_joint_padding_preserves_state_action_order() -> None:
    observation = {"state": torch.arange(5), "action": torch.arange(15).reshape(3, 5)}
    state = prepare_state(observation, 8)
    action = prepare_action(observation, 8)
    assert state.tolist() == [0, 1, 2, 3, 4, 0, 0, 0]
    torch.testing.assert_close(action[:, :5], observation["action"], atol=0, rtol=0)
    assert torch.count_nonzero(action[:, 5:]) == 0


@pytest.mark.parametrize("implementation", ["eager", "fused"])
def test_lingbot_v2_moe_keeps_expert_widths_and_storage(implementation: str) -> None:
    from transformers import PretrainedConfig

    from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP
    from worldfoundry.synthesis.action_generation.lingbot_vla_v2.modeling.qwen2_expert import (
        Qwen2FusedExperts,
        Qwen2TokenMoeBlock,
    )

    config = PretrainedConfig(
        hidden_size=8,
        intermediate_size=24,
        hidden_act="silu",
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        _moe_implementation=implementation,
    )
    block = Qwen2TokenMoeBlock(config).eval()
    assert type(block.shared_expert) is QwenMLP
    assert block.shared_expert.gate_proj.weight.shape == (16, 8)
    if implementation == "eager":
        assert isinstance(block.experts, torch.nn.ModuleList)
        assert all(type(expert) is QwenMLP for expert in block.experts)
        assert block.experts[0].gate_proj.weight.shape == (12, 8)
    else:
        assert type(block.experts) is Qwen2FusedExperts
        assert block.experts.gate_proj.shape == (3, 12, 8)
    reloaded = Qwen2TokenMoeBlock(config).eval()
    reloaded.load_state_dict(block.state_dict(), strict=True)
    inputs = torch.randn(2, 5, 8)
    for actual, expected in zip(reloaded(inputs), block(inputs), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_lingbot_logging_stays_within_its_family() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
import os

os.environ["VEOMNI_VERBOSITY"] = "WARNING"
parents = (logging.getLogger(), logging.getLogger("worldfoundry"))
before = [(logger.level, logger.propagate, tuple(logger.handlers)) for logger in parents]

from worldfoundry.synthesis.action_generation.lingbot_vla_v2.preprocessing import logging as lingbot_logging

family_logger = lingbot_logging.get_logger()
after = [(logger.level, logger.propagate, tuple(logger.handlers)) for logger in parents]
assert after == before, "LingBot changed the application logger configuration"
assert family_logger.name == "worldfoundry.synthesis.action_generation.lingbot_vla_v2"
model_logger = lingbot_logging.get_logger(f"{family_logger.name}.modeling.policy")
assert model_logger.getEffectiveLevel() == logging.WARNING
lingbot_logging.set_verbosity_info()
assert model_logger.getEffectiveLevel() == logging.INFO
assert [(logger.level, logger.propagate, tuple(logger.handlers)) for logger in parents] == before
""",
        ],
        check=True,
        timeout=60,
    )
