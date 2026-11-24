# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline discovery reads literal class metadata without importing models."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from worldfoundry.studio.inference import catalog

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ('MODEL_ID = "sample"', "sample"),
        ('MODEL_ID: ClassVar[str] = "sample"', "sample"),
        ('OTHER = MODEL_ID = "sample"', "sample"),
        ('MODEL_ID: str\nMODEL_ID = "sample"', "sample"),
        ("MODEL_ID: ClassVar[str]", None),
        ('OTHER: ClassVar[str] = "sample"', None),
        ("MODEL_ID: ClassVar[str] = 1", None),
    ],
)
def test_class_string_constant(declaration: str, expected: str | None) -> None:
    node = ast.parse("class SamplePipeline:\n    " + declaration.replace("\n", "\n    ")).body[0]
    assert isinstance(node, ast.ClassDef)
    assert catalog.discovery._class_string_constant(node, "MODEL_ID") == expected


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("ABSTRACT_PIPELINE = True", True),
        ("ABSTRACT_PIPELINE: ClassVar[bool] = True", True),
        ("ABSTRACT_PIPELINE: ClassVar[bool] = False", False),
        ("ABSTRACT_PIPELINE: bool\nABSTRACT_PIPELINE = False", False),
        ("ABSTRACT_PIPELINE: ClassVar[bool]", None),
        ("OTHER: ClassVar[bool] = True", None),
        ("ABSTRACT_PIPELINE: ClassVar[bool] = 1", None),
    ],
)
def test_class_bool_constant(declaration: str, expected: bool | None) -> None:
    node = ast.parse("class SamplePipeline:\n    " + declaration.replace("\n", "\n    ")).body[0]
    assert isinstance(node, ast.ClassDef)
    assert catalog.discovery._class_bool_constant(node, "ABSTRACT_PIPELINE") is expected


def test_discovery_preserves_annotated_ids_and_abstract_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "pipelines" / "sample" / "pipeline_sample.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class AbstractPipeline:\n"
        "    ABSTRACT_PIPELINE: ClassVar[bool] = True\n"
        "\n"
        "class SamplePipeline(AbstractPipeline):\n"
        "    ABSTRACT_PIPELINE: ClassVar[bool] = False\n"
        '    MODEL_ID: ClassVar[str] = "declared-model-id"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog.discovery, "_pipeline_file_candidates", lambda: (source,))
    monkeypatch.setattr(catalog.discovery, "_source_root", lambda: tmp_path)

    entries = catalog.discovery._discover_ast_pipelines.__wrapped__()

    assert [(entry.model_id, entry.class_name) for entry in entries] == [("declared-model-id", "SamplePipeline")]
