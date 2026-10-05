from __future__ import annotations

import ast
from pathlib import Path

import worldfoundry.training.checkpoint as checkpoint
from worldfoundry.training.checkpoint import artifacts, checkpointer, errors, staging, state

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_PACKAGE = _ROOT / "worldfoundry/training/checkpoint"


def test_monolithic_dcp_module_does_not_return() -> None:
    assert not (_CHECKPOINT_PACKAGE / "dcp.py").exists()


def test_checkpoint_facade_exports_canonical_leaf_objects() -> None:
    expected = {
        "IMMUTABLE_DTENSOR_ASYNC_STAGING": artifacts.IMMUTABLE_DTENSOR_ASYNC_STAGING,
        "SYNCHRONOUS_DCP_STAGING": artifacts.SYNCHRONOUS_DCP_STAGING,
        "TrainingCheckpointArtifact": artifacts.TrainingCheckpointArtifact,
        "TRAINING_CHECKPOINT_COMMIT_SCHEMA": checkpointer.TRAINING_CHECKPOINT_COMMIT_SCHEMA,
        "TRAINING_CHECKPOINT_MANIFEST_SCHEMA": checkpointer.TRAINING_CHECKPOINT_MANIFEST_SCHEMA,
        "TRAINING_CHECKPOINT_POINTER_SCHEMA": checkpointer.TRAINING_CHECKPOINT_POINTER_SCHEMA,
        "TrainingCheckpointer": checkpointer.TrainingCheckpointer,
        "IncompleteTrainingCheckpointError": errors.IncompleteTrainingCheckpointError,
        "TrainingCheckpointCompatibilityError": errors.TrainingCheckpointCompatibilityError,
        "TrainingCheckpointError": errors.TrainingCheckpointError,
        "PendingTrainingCheckpoint": staging.PendingTrainingCheckpoint,
        "TRAINING_PROGRESS_SCHEMA": state.TRAINING_PROGRESS_SCHEMA,
        "TRAINING_RUNTIME_STATE_SCHEMA": state.TRAINING_RUNTIME_STATE_SCHEMA,
        "TrainingProgress": state.TrainingProgress,
        "TrainingState": state.TrainingState,
    }
    for name, canonical in expected.items():
        assert getattr(checkpoint, name) is canonical


def test_checkpoint_leaf_dependencies_only_point_toward_lower_layers() -> None:
    allowed_siblings = {
        "errors.py": set(),
        "artifacts.py": set(),
        "state.py": {"artifacts", "errors"},
        "staging.py": {"artifacts"},
        "checkpointer.py": {"artifacts", "errors", "staging", "state"},
    }
    violations: list[str] = []
    for filename, allowed in allowed_siblings.items():
        path = _CHECKPOINT_PACKAGE / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
                continue
            if node.module not in allowed:
                violations.append(f"{filename}:{node.lineno}: {node.module}")
    assert violations == []
