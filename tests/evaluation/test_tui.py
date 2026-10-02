"""Headless interaction and command-matrix tests for the Textual TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Collapsible, Input, Select

from worldfoundry.cli.tui_app import (
    WorldFoundryTui,
    _format_file_artifact,
    _human_bytes,
    _looks_like_media_file,
    _looks_like_remote_id,
    _looks_like_video_file,
)
from worldfoundry.cli.tui_discovery import (
    build_model_benchmark_command,
    build_model_infer_command,
    infer_variant_options,
    is_infer_model_row,
    load_tui_catalog,
)


@pytest.mark.unit
def test_tui_path_and_artifact_helpers(tmp_path: Path) -> None:
    artifact = tmp_path / "video.mp4"
    artifact.write_bytes(b"1234")

    assert _looks_like_remote_id("org/model")
    assert not _looks_like_remote_id(str(artifact))
    assert _looks_like_media_file(str(artifact))
    assert _looks_like_video_file(str(artifact))
    assert _human_bytes(1536) == "1.5KiB"
    assert _format_file_artifact(artifact, root=tmp_path) == "video.mp4 (4B)"


@pytest.mark.unit
def test_tui_mounts_and_switches_modes(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = WorldFoundryTui(
            initial_model_id="helios",
            initial_benchmark_id="vbench",
            output_dir=tmp_path,
        )
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            assert app._brand_logo_width() >= 40
            assert app.action == "infer"
            assert app.query_one("#output-dir", Input).value.endswith("/infer")

            app.query_one("#action", Select).value = "eval"
            await pilot.pause()
            assert app.action == "eval"
            assert app.query_one("#output-dir", Input).value.endswith("__vbench")
            assert "vbench" in app._command_text()

            app.query_one("#eval-intent", Select).value = "score-artifacts"
            app.query_one("#eval-artifact-dir", Input).value = str(tmp_path)
            await pilot.pause()
            command = app._command_text()
            assert " score " in f" {command} "
            assert "--artifacts" in command
            assert "--model" not in command
            assert app.query_one("#models-pane").display is False

            app.query_one("#action", Select).value = "ui"
            await pilot.pause()
            assert app.action == "ui"
            assert "worldfoundry.studio.cli" in app._command_text()

            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.screen.has_class("-narrow")
            assert app.query_one("#brand-collapse", Collapsible).collapsed
            assert app.query_one("#right-column").region.x <= 2

    asyncio.run(exercise())


@pytest.mark.unit
def test_every_tui_catalog_entry_builds_a_command(tmp_path: Path) -> None:
    catalog = load_tui_catalog()

    for row in catalog.models:
        if is_infer_model_row(row):
            options = infer_variant_options(row.model_id) or (("default", None),)
            for _label, variant in options:
                assert build_model_infer_command(
                    model_id=row.model_id,
                    ckpt_type=variant,
                    output_dir=tmp_path / "infer",
                )
        assert build_model_benchmark_command(
            model_id=row.model_id,
            benchmark_id="vbench",
            output_dir=tmp_path / "eval",
        )

    for row in catalog.benchmarks:
        assert build_model_benchmark_command(
            model_id="act",
            benchmark_id=row.benchmark_id,
            output_dir=tmp_path / "eval",
        )


@pytest.mark.unit
def test_tui_groups_legacy_and_studio_model_aliases() -> None:
    catalog = load_tui_catalog()
    infer_rows = {row.model_id: row for row in catalog.models if is_infer_model_row(row)}

    assert "flashworld" in infer_rows
    assert "flash-world" not in infer_rows
    assert "cosmos-predict-2.5" in infer_rows
    assert "cosmos-predict2.5" not in infer_rows
    assert "fantasyworld" in infer_rows
    assert "fantasy-world" not in infer_rows
    assert "longvie-2" in infer_rows
    assert "longvie2" not in infer_rows

    assert "lyra" in infer_rows
    assert "lyra-1" not in infer_rows
    assert "lyra-2" not in infer_rows
    assert infer_variant_options("lyra") == (("Lyra 2", "lyra2"), ("Lyra 1", "lyra1"))

    # Catalog entries that are also registered family variants stay in the
    # variant selector instead of becoming a second model row.
    assert "hy-world-2.0" not in infer_rows
    assert "vggt-omega" not in infer_rows


@pytest.mark.unit
def test_legacy_infer_alias_uses_real_workspace_contract(tmp_path: Path) -> None:
    env_root = tmp_path / "envs"
    command = build_model_infer_command(
        model_id="flashworld",
        ckpt_type="flash-world",
        output_dir=tmp_path / "out",
        conda_envs_root=env_root,
        gpu="2",
    )

    assert command[0] == "env"
    assert f"WORLDFOUNDRY_CONDA_ENVS_ROOT={env_root}" in command
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert "worldfoundry.studio.workspace_job" in command
    assert command[command.index("--model-id") + 1] == "flashworld"
    assert command[command.index("--variant-id") + 1] == "default"
    assert "run_infer.sh" not in command
