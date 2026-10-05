"""Behavior tests for worldfoundry.core.io.manifests (moved in SA-10).

The implementation was moved verbatim from ``worldfoundry.evaluation.utils``;
these tests pin the load semantics and the error-message contract (paths in
YAML parse errors, EF-18) at the new canonical location.
"""

from __future__ import annotations

import pytest
import yaml

from worldfoundry.core.io.manifests import (
    MANIFEST_SUFFIXES,
    load_manifest,
    load_manifest_collection,
    manifest_paths,
)


def test_load_manifest_reads_yaml_mapping(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("id: demo\nvalues:\n  - 1\n  - 2\n", encoding="utf-8")
    assert load_manifest(path) == {"id": "demo", "values": [1, 2]}


def test_load_manifest_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected .yaml or .yml"):
        load_manifest(path)


def test_load_manifest_parse_error_includes_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match=str(path)):
        load_manifest(path)


def test_manifest_paths_lists_yaml_files_sorted(tmp_path):
    (tmp_path / "b.yaml").write_text("b: 1\n", encoding="utf-8")
    (tmp_path / "a.yml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("nope", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.yaml").write_text("c: 1\n", encoding="utf-8")

    paths = manifest_paths(tmp_path)
    assert list(paths) == sorted(paths)
    assert {p.name for p in paths} == {"a.yml", "b.yaml", "c.yaml"}


def test_manifest_paths_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        manifest_paths(tmp_path / "missing")


def test_manifest_paths_file_input_raises(tmp_path):
    path = tmp_path / "single.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        manifest_paths(path)


def test_load_manifest_collection_merges_directory_items(tmp_path):
    (tmp_path / "_manifest.yaml").write_text("schema: demo/1\n", encoding="utf-8")
    (tmp_path / "one.yaml").write_text("items:\n  - id: a\n", encoding="utf-8")
    (tmp_path / "two.yaml").write_text("items:\n  - id: b\n  - id: c\n", encoding="utf-8")

    payload = load_manifest_collection(tmp_path, item_key="items")
    assert payload["schema"] == "demo/1"
    assert sorted(item["id"] for item in payload["items"]) == ["a", "b", "c"]


def test_load_manifest_collection_single_file(tmp_path):
    path = tmp_path / "collection.yaml"
    path.write_text("items:\n  - id: only\n", encoding="utf-8")
    payload = load_manifest_collection(path, item_key="items")
    assert payload == {"items": [{"id": "only"}]}


def test_load_manifest_collection_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest_collection(tmp_path / "missing", item_key="items")


def test_manifest_suffixes_contract():
    assert MANIFEST_SUFFIXES == (".yaml", ".yml")
