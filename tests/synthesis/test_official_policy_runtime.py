# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from worldfoundry.synthesis.action_generation.official_policy.runtime import (
    OfficialPolicyRuntime,
    build_runtime_config,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [None, False, 0, "false", "off"])
def test_plan_reports_the_enforced_in_tree_loading_policy(value: bool | int | str | None) -> None:
    options = {"trust_remote_code": value, "attn_implementation": "eager"}
    config = build_runtime_config(
        model_id="test-policy",
        profile_checkpoints=(),
        defaults={"backend": "custom_from_pretrained", "require_checkpoint": False},
        options=options,
        device="cpu",
    )
    plan = OfficialPolicyRuntime(config).plan_payload(
        instruction="move",
        image=None,
        observation={"state": [1.0, 2.0]},
        action_context=(),
        output_path="/out/action.json",
    )
    assert config.trust_remote_code is False
    assert plan["trust_remote_code"] is False
    assert plan["runtime_options"]["trust_remote_code"] == value
    assert plan["runtime_options"]["attn_implementation"] == "eager"
    assert plan["inputs"]["observation"] == {"state": [1.0, 2.0]}
    assert plan["missing_assets"] == []


@pytest.mark.parametrize("value", [True, 1, "true"])
def test_runtime_config_retains_remote_code_rejection(value: bool | int | str) -> None:
    with pytest.raises(ValueError, match="remote checkpoint code, which is forbidden"):
        build_runtime_config(
            model_id="test-policy",
            profile_checkpoints=(),
            defaults={},
            options={"trust_remote_code": value},
            device="cpu",
        )
