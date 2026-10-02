from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.training.post_training import resolve_role_checkpoint


def _default() -> CheckpointSpec:
    return CheckpointSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
        files=("diffusion_pytorch_model.safetensors",),
        file_size_bytes={"diffusion_pytorch_model.safetensors": 123},
    )


def test_role_checkpoint_resolves_default_and_explicit_native_identity() -> None:
    default = _default()
    implicit = resolve_role_checkpoint(
        role="real_score",
        reference="default",
        native_default=default,
    )
    explicit = resolve_role_checkpoint(
        role="real-score",
        reference=("Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"),
        native_default=default,
    )

    assert implicit.checkpoint is default
    assert explicit.checkpoint is default
    assert implicit.source_kind == explicit.source_kind == "native-default"
    assert implicit.role == explicit.role == "real-score"


def test_role_checkpoint_requires_explicit_hub_reference_or_local_override(
    tmp_path: Path,
) -> None:
    default = _default()
    with pytest.raises(ValueError, match="REPO@REVISION"):
        resolve_role_checkpoint(
            role="teacher",
            reference="Wan-AI/Wan2.1-T2V-1.3B",
            native_default=default,
        )
    with pytest.raises(ValueError, match="REPO@REVISION"):
        resolve_role_checkpoint(
            role="teacher",
            reference=str(tmp_path),
            native_default=default,
        )
    local = CheckpointSpec(source=tmp_path, files=("weights.safetensors",))
    resolved = resolve_role_checkpoint(
        role="teacher",
        reference="default",
        native_default=default,
        local_override=local,
    )
    assert resolved.checkpoint is local


def test_role_checkpoint_preserves_layout_for_another_revision() -> None:
    resolved = resolve_role_checkpoint(
        role="fake-score",
        reference="example/derived-wan@1111111111111111111111111111111111111111",
        native_default=_default(),
    )

    assert resolved.source_kind == "pinned-hub"
    assert resolved.checkpoint.repo_id == "example/derived-wan"
    assert resolved.checkpoint.revision == "1" * 40
    assert resolved.checkpoint.files == ("diffusion_pytorch_model.safetensors",)
    assert not resolved.checkpoint.file_size_bytes


def test_explicit_native_reference_accepts_matching_local_mirror(
    tmp_path: Path,
) -> None:
    default = _default()
    mirror = CheckpointSpec(
        source=tmp_path,
        files=default.files,
        file_size_bytes=default.file_size_bytes,
    )
    reference = "Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"

    resolved = resolve_role_checkpoint(
        role="real-score",
        reference=reference,
        native_default=default,
        local_override=mirror,
    )

    assert resolved.source_kind == "local"
    assert resolved.requested_reference == reference
    assert resolved.checkpoint is mirror

    wrong_size = CheckpointSpec(
        source=tmp_path,
        files=default.files,
        file_size_bytes={default.files[0]: 999},
    )
    with pytest.raises(ValueError, match="files differ"):
        resolve_role_checkpoint(
            role="real-score",
            reference=reference,
            native_default=default,
            local_override=wrong_size,
        )
    with pytest.raises(ValueError, match="repository and revision"):
        resolve_role_checkpoint(
            role="real-score",
            reference="example/derived@1111111111111111111111111111111111111111",
            native_default=default,
            local_override=mirror,
        )
