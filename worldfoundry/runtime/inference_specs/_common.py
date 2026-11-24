# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared field, artifact, fixture, and checkpoint-path helpers for curated inference specs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.inference import InferenceArtifactSpec, InferenceFieldSpec
from worldfoundry.core.io.paths import checkpoint_root_candidates, hfd_root_path, project_root


def _first_existing_path(*candidates: str | Path) -> str:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return str(Path(candidates[0]).expanduser()) if candidates else ""


def _local_hfd_checkpoint_path(repo_dir_name: str, *parts: str) -> str:
    """Resolve a flat-HFD export while respecting an explicit HFD root.

    Shared Workspace installations commonly keep flat ``owner--repo`` exports
    directly below the adjacent plural ``ckpts`` directory.  The generic HFD
    default, however, is ``ckpt/hfd`` when a singular ``ckpt`` directory also
    exists.  Discover the conventional checkpoint-root sibling only when the
    caller did not explicitly select an HFD root.
    """

    configured = hfd_root_path(repo_dir_name, *parts)
    if os.environ.get("WORLDFOUNDRY_HFD_ROOT"):
        return str(configured)
    return _first_existing_path(
        configured,
        *checkpoint_root_candidates(repo_dir_name, *parts),
    )


def _read_text_file_or_default(path: str | Path, default: str) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return default
    return value or default


_TEST_CASES_ROOT = Path(__file__).resolve().parents[2] / "data" / "test_cases"


_RUNTIME_CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "data" / "models" / "runtime" / "configs"


_PROJECT_ROOT = project_root(__file__)


_WORKSPACE_ROOT = _PROJECT_ROOT.parent


GENERIC_IMAGE_FIXTURE = str(_TEST_CASES_ROOT / "studio_demo" / "00" / "image.jpg")


def _field(
    field_id: str,
    label: str,
    *,
    kind: str = "string",
    target: str = "call_kwargs",
    required: bool = False,
    default: Any = None,
    choices: Sequence[Any] = (),
    description: str = "",
) -> InferenceFieldSpec:
    return InferenceFieldSpec(
        field_id=field_id,
        label=label,
        kind=kind,
        target=target,
        required=required,
        default=default,
        choices=tuple(str(choice) for choice in choices),
        description=description,
    )


def _artifact(
    artifact_id: str,
    kind: str,
    *,
    required: bool = False,
    preview: bool = False,
    description: str = "",
) -> InferenceArtifactSpec:
    return InferenceArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        required=required,
        preview=preview,
        description=description,
    )
