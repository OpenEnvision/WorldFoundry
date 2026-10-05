from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.inference import catalog as studio_catalog


def _stage_videocrafter_checkpoints(checkpoint_root: Path) -> dict[str, Path]:
    expected_paths = {
        "videocrafter1_i2v": checkpoint_root / "VideoCrafter--Image2Video-512" / "model.ckpt",
        "videocrafter1_t2v": checkpoint_root / "VideoCrafter--Text2Video-1024" / "model.ckpt",
        "videocrafter2_t2v": checkpoint_root / "VideoCrafter--VideoCrafter2" / "model.ckpt",
    }
    for path in expected_paths.values():
        path.parent.mkdir(parents=True)
        path.touch()
    return expected_paths


def _assert_videocrafter_defaults(entries: dict[str, studio_catalog.CatalogEntry], expected_paths: dict[str, Path]) -> None:
    assert set(entries) == set(expected_paths)
    for model_id, path in expected_paths.items():
        assert entries[model_id].default_model_ref == str(path)
        assert entries[model_id].default_call_kwargs["num_frames"] == 16
        assert entries[model_id].default_call_kwargs["num_inference_steps"] == 50
        assert entries[model_id].default_call_kwargs["seed"] == 123
    assert entries["videocrafter1_i2v"].default_call_kwargs["height"] == 320
    assert entries["videocrafter1_i2v"].default_call_kwargs["width"] == 512
    assert entries["videocrafter1_t2v"].default_call_kwargs["height"] == 576
    assert entries["videocrafter1_t2v"].default_call_kwargs["width"] == 1024
    assert entries["videocrafter2_t2v"].default_call_kwargs["height"] == 320
    assert entries["videocrafter2_t2v"].default_call_kwargs["width"] == 512


def test_videocrafter_workspace_defaults_use_complete_hfd_exports(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    expected_paths = _stage_videocrafter_checkpoints(checkpoint_root)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))
    monkeypatch.setattr(studio_catalog, "_discover_ast_pipelines", lambda: ())

    entries = {
        entry.model_id: entry
        for entry in studio_catalog._canonical_runtime_entries()
        if entry.model_id in expected_paths
    }

    _assert_videocrafter_defaults(entries, expected_paths)


def test_videocrafter_exact_workspace_ids_do_not_bypass_curated_defaults(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    expected_paths = _stage_videocrafter_checkpoints(checkpoint_root)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))
    studio_catalog._discover_ast_pipelines.cache_clear()
    studio_catalog._discover_catalog_infos.cache_clear()
    monkeypatch.setattr(
        studio_catalog,
        "discover_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("exact Workspace ids must not scan the full catalog")),
    )

    entries = {model_id: studio_catalog.find_entry(model_id) for model_id in expected_paths}

    _assert_videocrafter_defaults(entries, expected_paths)
    assert all(entry.model_id == model_id for model_id, entry in entries.items())
