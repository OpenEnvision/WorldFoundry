from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import json
from pathlib import Path
from types import SimpleNamespace

from torch import nn

from worldfoundry.training.checkpoint import (
    SYNCHRONOUS_DCP_STAGING,
    TrainingCheckpointArtifact,
)
from worldfoundry.training.engine import WanDMDTrainingRun, WanFlowPolicyTrainingRun
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.tuning import FullModelArtifact


class _RoleBundle:
    def __init__(self, model: nn.Module, *, role: str) -> None:
        adapter = SimpleNamespace(trainable_module=model)
        self.student = adapter
        self.policy = adapter
        self.student_peft = None
        self.policy_peft = None
        self.role = role

    def runtime_identity(self) -> dict[str, object]:
        return {self.role: {"owner": "worldfoundry-native"}}


class _Checkpointer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.saved = 0

    def save(self, state: object, *, asynchronous: bool) -> TrainingCheckpointArtifact:
        assert state is not None
        assert asynchronous is False
        self.saved += 1
        return TrainingCheckpointArtifact(
            path=self.root / "step-00000004",
            global_step=4,
            staging_strategy=SYNCHRONOUS_DCP_STAGING,
            optional_state_presence={
                "lr_scheduler": False,
                "ema": False,
                "grad_scaler": False,
                "algorithm_state": False,
            },
            identity={"algorithm": "flow-policy"},
            file_size_bytes={"payload": 1},
        )


def _dmd_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "dmd-export", "output_dir": "unused"},
            "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "data.jsonl"},
            "algorithm": {
                "type": "dmd",
                "student_timesteps": [1000, 757, 522],
                "student_sigmas": [1.0, 0.757, 0.522],
                "real_score_checkpoint": "teacher",
                "fake_score_checkpoint": "fake",
            },
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-5},
            "fake_score_optimizer": {"type": "adamw", "learning_rate": 1.0e-5},
            "export": {
                "format": "safetensors",
                "options": {"max_shard_size_bytes": 64},
            },
        }
    )


def _flow_recipe(*, export_format: str) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "flow-export", "output_dir": "unused"},
            "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "data": {"manifest": "prompts.jsonl"},
            "algorithm": {
                "type": "flow-grpo",
                "sigmas": [1.0, 0.5, 0.0],
                "sde_step_indices": [0, 1],
                "reward_weights": {
                    "video_quality": 1.0,
                    "motion_quality": 1.0,
                    "text_alignment": 1.0,
                },
                "reward_model": {"type": "videoalign"},
            },
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-5},
            "export": {"format": export_format},
        }
    )


def test_dmd_full_export_dispatches_to_idempotent_safetensors(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    session = SimpleNamespace(engine=SimpleNamespace(global_step=3))
    run = WanDMDTrainingRun(
        recipe=_dmd_recipe(),
        session=session,
        checkpoint_state=object(),
        checkpointer=SimpleNamespace(),
        roles=_RoleBundle(model, role="student"),
        output_dir=output,
        data_identity={"samples": ["data"]},
        resume_artifact=None,
        distributed_context=None,
    )
    run._summary = object()

    first = run.export_student()
    second = run.export_student()

    assert isinstance(first, FullModelArtifact)
    assert second == first
    assert first.tensor_count == len(model.state_dict())
    assert first.file_size_bytes
    events = [json.loads(line) for line in run.metrics_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["format"] == "safetensors"
    assert events[0]["role"] == "student"


def test_flow_full_export_dispatches_to_configured_dcp(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    checkpointer = _Checkpointer(output / "checkpoints")
    session = SimpleNamespace(
        engine=SimpleNamespace(global_step=4),
        wait_for_checkpoints=lambda: None,
    )
    run = WanFlowPolicyTrainingRun(
        recipe=_flow_recipe(export_format="distributed-checkpoint"),
        session=session,
        dataloader=SimpleNamespace(),
        checkpoint_state=object(),
        checkpointer=checkpointer,
        roles=_RoleBundle(nn.Linear(2, 2), role="policy"),
        reward_adapter=SimpleNamespace(),
        output_dir=output,
        data_identity={"samples": ["data"]},
        reward_identity={"model": "reward"},
        resume_artifact=None,
        distributed_context=None,
    )
    run._summary = object()

    artifact = run.export_policy()

    assert isinstance(artifact, TrainingCheckpointArtifact)
    assert artifact.global_step == 4
    assert checkpointer.saved == 1
    event = json.loads(run.metrics_path.read_text())
    assert event["format"] == "distributed-checkpoint"
    assert event["role"] == "policy"


def test_wan_algorithms_do_not_import_private_sibling_helpers() -> None:
    root = Path(__file__).resolve().parents[2] / "worldfoundry/training/engine/wan"
    dmd_source = (root / "dmd.py").read_text(encoding="utf-8")
    flow_source = (root / "flow_policy.py").read_text(encoding="utf-8")

    assert "from .sft import _" not in dmd_source
    assert "from .dmd import _" not in flow_source
    assert "from worldfoundry.training.post_training import" not in dmd_source
    assert "from worldfoundry.training.post_training import" not in flow_source
