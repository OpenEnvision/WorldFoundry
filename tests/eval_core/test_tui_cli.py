from __future__ import annotations

import asyncio
import importlib
import json
import sys

import pytest

from worldfoundry import cli
from worldfoundry.cli.tui import main as tui_main
from worldfoundry.cli.tui_discovery import (
    INFER_MODEL_VARIANTS,
    build_model_benchmark_command,
    build_model_infer_command,
    build_suite_command,
    format_shell_command,
    infer_control_specs,
    infer_model_variant_ids,
    is_infer_model_row,
    load_tui_catalog,
)


def _assert_worldfoundry_cli_prefix(command: tuple[str, ...]) -> None:
    assert command[:3] == (sys.executable, "-m", "worldfoundry.cli")


def _parse_worldfoundry_cli_command(command: tuple[str, ...]):
    _assert_worldfoundry_cli_prefix(command)
    return cli._build_parser().parse_args(list(command[3:]))


def test_tui_catalog_loads_model_and_benchmark_rows() -> None:
    catalog = load_tui_catalog()

    assert any(row.model_id == "openvla" for row in catalog.models)
    assert any(row.benchmark_id == "libero" for row in catalog.benchmarks)
    summary = catalog.to_dict()["summary"]
    assert summary["models"] == len(catalog.models)
    assert summary["benchmarks"] == len(catalog.benchmarks)
    # TUI training support (training_targets / training_stages) was removed
    # from tui_discovery together with worldfoundry.training.visual_generation.
    assert summary["runtime_profiles"] == len(catalog.runtime_profiles)
    assert "leaderboard_ready_benchmarks" in summary
    assert any(row.profile_id == "openvla" for row in catalog.runtime_profiles)
    assert not [row.model_id for row in catalog.models if row.integration_status == "infer"]
    robotwin = next(row for row in catalog.benchmarks if row.benchmark_id == "robotwin")
    assert robotwin.maturity
    assert robotwin.leaderboard_valid is False
    assert robotwin.to_dict()["official_benchmark_verified"] is False
    assert catalog.conda_status["env_specs"] >= 1


def test_model_benchmark_command_uses_unified_run_surface() -> None:
    command = build_model_benchmark_command(
        model_id="openvla",
        benchmark_id="libero",
        output_dir="runs/tui/openvla__libero",
        mode="contract",
        model_variant="openvla-7b",
        json_output=True,
    )

    _assert_worldfoundry_cli_prefix(command)
    assert command[3] == "run"
    assert "--model" in command
    assert "--benchmark" in command
    assert "--model-variant" in command
    assert "--json" in command
    assert "openvla" in format_shell_command(command)
    parsed = _parse_worldfoundry_cli_command(command)
    assert parsed.command == "run"
    assert parsed.model_ids == ["openvla"]
    assert parsed.benchmark_ids == ["libero"]
    assert parsed.benchmark_mode == "contract"


def test_tui_infer_catalog_uses_official_script_models_by_default() -> None:
    catalog = load_tui_catalog()
    depth_row = next(row for row in catalog.models if row.model_id == "depth-anything-v2")

    assert is_infer_model_row(depth_row)
    assert depth_row.runner_kind == "infer_script"
    assert any("scripts/inference/run_infer.sh" in note for note in depth_row.notes)
    # Infer commands were consolidated onto the Studio workspace_job pipeline;
    # run_infer.sh is now a thin exec shim over the same entry point
    # (tui_discovery.exec_run_infer_sh), so the built command targets
    # workspace_job directly and GPU selection rides on CUDA_VISIBLE_DEVICES.
    command = build_model_infer_command(
        model_id="depth-anything-v2",
        output_dir="runs/tui/infer",
        input_path="/tmp/input.png",
    )
    command_text = format_shell_command(command)
    assert "-m worldfoundry.studio.workspace_job infer" in command_text
    assert "--model-id depth-anything-v2" in command_text
    assert "--input-path /tmp/input.png" in command_text
    assert "CUDA_VISIBLE_DEVICES" not in command_text

    explicit_gpu_command = build_model_infer_command(
        model_id="depth-anything-v2",
        output_dir="runs/tui/infer",
        input_path="/tmp/input.png",
        gpu="0",
    )
    assert "CUDA_VISIBLE_DEVICES=0" in format_shell_command(explicit_gpu_command)


def test_tui_infer_control_schema_is_variant_specific() -> None:
    depth_specs = {spec.field_id: spec for spec in infer_control_specs("depth-anything-v2", "depth-anything-v2-small")}
    flash_specs = {spec.field_id: spec for spec in infer_control_specs("flash-world", "flash-world")}
    lingbot_specs = {spec.field_id: spec for spec in infer_control_specs("lingbot-world", "base-act-preview")}
    recam_specs = {spec.field_id: spec for spec in infer_control_specs("recammaster", "recammaster")}
    neoverse_specs = {spec.field_id: spec for spec in infer_control_specs("neoverse", "neoverse")}
    vggt_specs = {spec.field_id: spec for spec in infer_control_specs("vggt", "vggt-omega")}
    cut3r_specs = {spec.field_id: spec for spec in infer_control_specs("cut3r", "cut3r")}
    wan22_specs = {spec.field_id: spec for spec in infer_control_specs("wan2.2", "wan2.2-ti2v-5b")}
    hy2_specs = {spec.field_id: spec for spec in infer_control_specs("hunyuan", "hy-world-2.0")}
    lyra2_specs = {spec.field_id: spec for spec in infer_control_specs("lyra", "lyra2")}
    worldcam_specs = {spec.field_id: spec for spec in infer_control_specs("worldcam", "worldcam")}
    dynamicrafter_specs = {spec.field_id: spec for spec in infer_control_specs("dynamicrafter", "dynamicrafter-512-i2v")}

    assert set(depth_specs) == {"input"}
    assert depth_specs["input"].label == "Input image"
    assert {"input", "input_dir", "output_formats"}.issubset(flash_specs)
    assert flash_specs["input_dir"].label == "JSON config"
    assert lingbot_specs["input_dir"].label == "Action/example directory"
    assert lingbot_specs["mode"].label == "Control mode"
    assert "size" not in lingbot_specs
    assert recam_specs["video"].label == "Input video"
    assert recam_specs["trajectory"].placeholder == "100,100,0,0,30"
    assert {"trajectory_file", "resize_mode", "disable_lora", "vis_rendering"}.issubset(neoverse_specs)
    assert {"input", "input_dir", "task"}.issubset(vggt_specs)
    assert {"input", "input_dir", "task"}.issubset(cut3r_specs)
    assert wan22_specs["mode"].placeholder == "ti2v-5B"
    assert {"input", "input_dir", "task", "output_path"} == set(hy2_specs)
    assert {"size", "steps", "guidance_scale"}.issubset(lyra2_specs)
    assert "frames" not in lyra2_specs
    assert worldcam_specs["input"].label == "Conditioning video"
    assert "frames" not in worldcam_specs
    assert not {"negative_prompt", "guidance_scale", "dtype", "max_sequence_length"} & set(dynamicrafter_specs)


def test_tui_keeps_studio_runtime_for_non_script_infer_models() -> None:
    catalog = load_tui_catalog()
    rows = {row.model_id: row for row in catalog.models}

    # Studio catalog entries are folded into their owning TUI rows:
    # ``hunyuan-worldplay`` belongs to the ``hunyuan`` script-infer family and
    # ``multiworld-ittakestwo`` owns its own studio-runtime row.  The zoo rows
    # (``hy-worldplay``/``multiworld``) keep their catalog-derived runner kind
    # (``runner_entry_kind``) instead of being rewritten to ``studio_runtime``.
    ittakestwo = rows["multiworld-ittakestwo"]
    assert ittakestwo.runner_kind == "studio_runtime"
    assert ittakestwo.integration_status == "integrated"
    # runner_status vocabulary moved from "ready" to evidence-based values
    # (_inference_runner_status): verified / partially_verified /
    # contract_ready. The exact value depends on recorded inference
    # evidence, so assert membership rather than pin one value.
    assert ittakestwo.runner_status in {"verified", "partially_verified", "contract_ready"}
    assert is_infer_model_row(ittakestwo)

    hunyuan = rows["hunyuan"]
    assert hunyuan.integration_status == "integrated"
    assert is_infer_model_row(hunyuan)

    # hy-worldplay is catalog-integrated with an in-tree runner; the TUI row
    # reflects the model-zoo entry rather than the studio merge.
    worldplay = rows["hy-worldplay"]
    assert worldplay.integration_status == "integrated"
    assert worldplay.runner_kind == "runnable_runner"
    assert worldplay.runner_status in {"verified", "partially_verified", "contract_ready"}

    # Infer commands for the legacy model ids still resolve to the Studio
    # runtime via alias routing (workspace_job infer), so non-script models
    # keep a working studio-runtime inference path.
    for model_id in ("hy-worldplay", "multiworld", "multiworld-ittakestwo"):
        command = build_model_infer_command(
            model_id=model_id,
            output_dir="runs/tui/infer",
        )
        assert "worldfoundry.studio.workspace_job infer" in format_shell_command(command)


def test_tui_script_infer_variants_all_build_official_commands() -> None:
    for model_id, variants in INFER_MODEL_VARIANTS.items():
        resolved = infer_model_variant_ids(model_id)
        # infer_variant_options drops declared variants that no longer map to
        # a Studio runtime target (obsolete IDs are not forwarded to
        # workspace_job), so the resolved set is a subset of the declaration.
        assert set(resolved) <= set(variants)
        for variant_id in resolved:
            command = build_model_infer_command(
                model_id=model_id,
                ckpt_type=variant_id,
                output_dir="runs/tui/infer",
                ckpt_root="/ckpt",
                conda_envs_root="/conda/envs",
            )
            command_text = format_shell_command(command)

            assert "-m worldfoundry.studio.workspace_job infer" in command_text
            assert "--model-id" in command_text
            assert "--variant-id" in command_text
            assert "WORLDFOUNDRY_CKPT_DIR=/ckpt" in command_text
            assert "WORLDFOUNDRY_CONDA_ENVS_ROOT=/conda/envs" in command_text
            assert "CUDA_VISIBLE_DEVICES" not in command_text


def test_suite_command_defaults_to_plan_only() -> None:
    command = build_suite_command(
        output_dir="runs/tui/suite",
        model_ids=("openvla",),
        benchmark_ids=("libero",),
    )

    _assert_worldfoundry_cli_prefix(command)
    assert command[3] == "run"
    assert "--plan-only" in command
    assert command.count("--model") == 1
    assert command.count("--benchmark") == 1
    parsed = _parse_worldfoundry_cli_command(command)
    assert parsed.command == "run"
    assert parsed.plan_only is True
    assert parsed.model_ids == ["openvla"]
    assert parsed.benchmark_ids == ["libero"]




def test_tui_cli_catalog_json_does_not_import_textual(capsys) -> None:
    assert tui_main(["--catalog-json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["models"] >= 1
    assert payload["summary"]["benchmarks"] >= 1
    assert payload["summary"]["runtime_profiles"] >= 1
    assert "conda_status" in payload


def test_cli_tui_print_command(capsys) -> None:
    assert (
        cli.main(
            [
                "tui",
                "--model-id",
                "openvla",
                "--benchmark-id",
                "libero",
                "--print-command",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert "eval:" in output
    assert "-m worldfoundry.cli run" in output
    assert "--model openvla" in output
    assert "--benchmark libero" in output


def test_tui_fallback_lists_catalog_runtime_and_conda(capsys) -> None:
    assert (
        tui_main(
            [
                "--fallback",
                "--model-id",
                "openvla",
                "--benchmark-id",
                "robotwin",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "WorldFoundry TUI fallback" in output
    assert "Runtime Profiles" in output
    assert "Conda" in output
    # "Training Targets" section removed together with TUI training support.
    assert "Suite plan command" in output
    # Section renamed from "Benchmark contract command" alongside the
    # zoo benchmark-run CLI consolidation.
    assert "Benchmark run command" in output
    assert "-m worldfoundry.cli zoo benchmark-run" in output
    assert "--benchmark-id robotwin" in output
    assert "Runtime preflight command" in output
    assert "-m worldfoundry.cli preflight runtime" in output
    assert "--plan-only" in output


def test_textual_app_headless_generates_action_commands_when_available() -> None:
    pytest.importorskip("textual")
    from worldfoundry.cli.tui_app import WorldFoundryTui

    async def run_app() -> dict[str, str]:
        app = WorldFoundryTui(
            initial_model_id="open-magvit2",
            initial_benchmark_id="libero",
            output_dir="runs/tui-headless",
            prompt="A small official demo inference sample.",
            steps="1",
        )
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            commands: dict[str, str] = {}
            for action in ("infer", "eval", "ui"):
                app.action = action
                if action == "infer":
                    app.selected_model_id = "open-magvit2"
                elif action == "eval":
                    app.selected_model_id = "openvla"
                    app.selected_benchmark_id = "libero"
                app._sync_ckpt_type_for_selected_model()
                app._refresh_ckpt_type_select()
                app._sync_action_layout()
                app._populate_tables()
                app._update_command()
                commands[action] = app._command_text()
                await pilot.pause()
            return commands

    commands = asyncio.run(run_app())

    assert "worldfoundry.studio.workspace_job infer" in commands["infer"]
    assert "--model-id" in commands["infer"]
    assert "-m worldfoundry.cli run" in commands["eval"]
    assert "--model openvla" in commands["eval"]
    assert "worldfoundry.studio.cli" in commands["ui"]


def test_textual_app_headless_button_interactions_when_available(tmp_path) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Collapsible

    from worldfoundry.cli.tui_app import WorldFoundryTui

    async def wait_until(predicate, timeout: float = 5.0) -> bool:
        end = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < end:
            if predicate():
                return True
            await asyncio.sleep(0.05)
        return predicate()

    async def run_app() -> dict[str, bool]:
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        (output_dir / "sample.txt").write_text("ok\n", encoding="utf-8")
        app = WorldFoundryTui(
            initial_model_id="open-magvit2",
            initial_benchmark_id="libero",
            output_dir=output_dir,
            prompt="A small official demo inference sample.",
            steps="1",
        )
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            results: dict[str, bool] = {}

            async def click_visible_button(selector: str) -> None:
                app.query_one(selector, Button).scroll_visible(animate=False)
                await pilot.pause()
                assert await pilot.click(selector, offset=(1, 1))

            await pilot.click("#toggle-command", offset=(1, 1))
            await pilot.pause()
            results["command_preview_open"] = not app.query_one("#command-preview", Collapsible).collapsed
            results["buttons_remain_visible"] = app.query_one("#run-controls").region.y >= 0
            for selector in ("#copy", "#preflight", "#artifacts", "#gpu-status", "#refresh"):
                await click_visible_button(selector)
                await pilot.pause()
                results[f"{selector}_clicked"] = True

            app._command = lambda: (
                sys.executable,
                "-c",
                "import time; print('button-test-started', flush=True); time.sleep(30)",
            )
            await click_visible_button("#run")
            results["run_started"] = await wait_until(lambda: app.running_process is not None)
            results["run_disabled_while_running"] = app.query_one("#run", Button).disabled
            results["stop_enabled_while_running"] = not app.query_one("#stop", Button).disabled
            await click_visible_button("#stop")
            results["stop_cleared_process"] = await wait_until(
                lambda: app.running_task is None and app.running_process is None,
                timeout=8.0,
            )
            results["run_enabled_after_stop"] = not app.query_one("#run", Button).disabled
            results["stop_disabled_after_stop"] = app.query_one("#stop", Button).disabled

            await pilot.press("ctrl+r")
            results["shortcut_run_started"] = await wait_until(lambda: app.running_process is not None)
            await pilot.press("ctrl+s")
            results["shortcut_stop_cleared_process"] = await wait_until(
                lambda: app.running_task is None and app.running_process is None,
                timeout=8.0,
            )
            return results

    results = asyncio.run(run_app())

    assert all(results.values()), results


def test_textual_app_headless_selects_and_tables_when_available() -> None:
    pytest.importorskip("textual")
    from textual.widgets import DataTable, Input, Label, Select

    from worldfoundry.cli.tui_app import WorldFoundryTui

    async def run_app() -> dict[str, bool]:
        app = WorldFoundryTui(
            initial_model_id="open-magvit2",
            initial_benchmark_id="libero",
            output_dir="runs/tui-select-headless",
        )
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            results: dict[str, bool] = {}
            action = app.query_one("#action", Select)
            for value in ("eval", "ui", "infer"):
                action.value = value
                await pilot.pause()
                results[f"action_{value}"] = app.action == value

            app.selected_model_id = "open-magvit2"
            app._sync_ckpt_type_for_selected_model()
            app._refresh_ckpt_type_select()
            resolved = infer_model_variant_ids("open-magvit2")
            assert resolved
            app.query_one("#ckpt-type", Select).value = resolved[0]
            await pilot.pause()
            results["ckpt_select"] = app.ckpt_type == resolved[0]

            app.selected_model_id = "depth-anything-v2"
            app._sync_ckpt_type_for_selected_model()
            app._sync_infer_control_layout()
            results["depth_input_label"] = str(app.query_one("#input-path-label", Label).render()) == "Input image"
            app.selected_model_id = "flash-world"
            app._sync_ckpt_type_for_selected_model()
            app._sync_infer_control_layout()
            results["flash_input_dir_label"] = str(app.query_one("#input-dir-label", Label).render()) == "JSON config"
            app.selected_model_id = "lingbot-world"
            app._sync_ckpt_type_for_selected_model()
            app._sync_infer_control_layout()
            results["lingbot_input_dir_label"] = (
                str(app.query_one("#input-dir-label", Label).render()) == "Action/example directory"
            )

            filter_input = app.query_one("#filter", Input)
            filter_input.value = "openvla"
            await pilot.pause()
            results["filter_updates"] = filter_input.value == "openvla"
            filter_input.value = ""
            action.value = "eval"
            await pilot.pause()
            models = app.query_one("#models-table", DataTable)
            benchmarks = app.query_one("#benchmarks-table", DataTable)
            results["tables_have_rows"] = models.row_count > 0 and benchmarks.row_count > 0
            models.focus()
            models.move_cursor(row=0)
            await pilot.press("enter")
            benchmarks.focus()
            benchmarks.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            results["selection_still_valid"] = bool(app.selected_model_id and app.selected_benchmark_id)
            return results

    results = asyncio.run(run_app())

    assert all(results.values()), results


def test_tui_cli_writes_suite_plan(tmp_path, capsys) -> None:
    assert (
        tui_main(
            [
                "--model-id",
                "openvla",
                "--benchmark-id",
                "robotwin",
                "--output-dir",
                str(tmp_path / "tui"),
                "--write-suite-plan",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "suite-plan: status=planned" in output
    manifest_path = tmp_path / "tui" / "suite-plan" / "suite_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 1
    assert payload["cells"][0]["model_id"] == "openvla"
    assert payload["cells"][0]["benchmark_id"] == "robotwin"
    assert payload["cells"][0]["status"] == "planned"
    assert payload["cells"][0]["compatibility"] == "compatible"
    assert payload["cells"][0]["output_artifact"] == "action_trace"


def test_robotics_framework_imports_use_embodied_owner() -> None:
    embodied = importlib.import_module("worldfoundry.evaluation.tasks.embodied")
    normalizer = importlib.import_module("worldfoundry.evaluation.tasks.embodied.normalizer")

    assert hasattr(embodied, "EmbodiedGenerationSpec")
    assert hasattr(embodied, "run_vla_va_wam")
    assert hasattr(normalizer, "normalize_results")
