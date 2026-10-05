from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_module():
    script = REPO_ROOT / "scripts" / "embodied" / "prepare_official_assets.py"
    spec = importlib.util.spec_from_file_location("prepare_official_assets_git_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def test_prepare_git_item_fetches_pinned_revision_and_repairs_existing_checkout(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source = tmp_path / "source"
    source.mkdir()
    _git("init", cwd=source)
    _git("config", "user.name", "WorldFoundry Test", cwd=source)
    _git("config", "user.email", "worldfoundry@example.invalid", cwd=source)
    _git("config", "uploadpack.allowReachableSHA1InWant", "true", cwd=source)

    tracked = source / "payload.txt"
    tracked.write_text("first\n", encoding="utf-8")
    _git("add", "payload.txt", cwd=source)
    _git("commit", "-m", "first", cwd=source)
    first_revision = _git("rev-parse", "HEAD", cwd=source)
    tracked.write_text("second\n", encoding="utf-8")
    _git("commit", "-am", "second", cwd=source)
    second_revision = _git("rev-parse", "HEAD", cwd=source)

    target = tmp_path / "checkout"
    args = Namespace(skip_existing=True, plan_only=False, timeout_seconds=20)
    env = dict(os.environ)
    first_item = module.PrepareItem(
        category="benchmark",
        kind="git_repo",
        owner_id="example",
        asset_id="example",
        local_path=target,
        source=source.resolve().as_uri(),
        revision=first_revision,
    )

    first_row = module.prepare_git_item(first_item, args, env, tmp_path / "logs")
    assert first_row.get("returncode") == 0, first_row
    assert first_row.get("fetch_returncode") == 0, first_row
    assert first_row.get("checkout_returncode") == 0, first_row
    assert first_row.get("revision_matches") is True, first_row
    assert first_row["status"] == "ready", first_row
    assert first_row["fetch_command"][-1] == first_revision
    assert "--detach" in first_row["checkout_command"]
    assert _git("rev-parse", "HEAD", cwd=target) == first_revision
    detached = subprocess.run(
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        cwd=target,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert detached.returncode == 1

    second_item = module.PrepareItem(
        category="benchmark",
        kind="git_repo",
        owner_id="example",
        asset_id="example",
        local_path=target,
        source=source.resolve().as_uri(),
        revision=second_revision,
    )
    second_row = module.prepare_git_item(second_item, args, env, tmp_path / "logs")

    assert second_row["ready_before"] is True
    assert second_row["revision_matches_before"] is False
    assert second_row["status"] == "ready"
    assert _git("rev-parse", "HEAD", cwd=target) == second_revision


def test_git_plan_exposes_fetch_and_detached_checkout(tmp_path: Path) -> None:
    module = _load_prepare_module()
    item = module.PrepareItem(
        category="benchmark",
        kind="git_repo",
        owner_id="example",
        asset_id="example",
        local_path=tmp_path / "checkout",
        source="https://example.invalid/repo.git",
        revision="abc1234",
    )
    row = module.prepare_git_item(
        item,
        Namespace(skip_existing=True, plan_only=True, timeout_seconds=20),
        {},
        tmp_path / "logs",
    )

    assert row["command"][:3] == ["git", "clone", "--depth"]
    assert row["fetch_command"][-4:] == ["--depth", "1", "origin", "abc1234"]
    assert row["checkout_command"][-3:] == ["checkout", "--detach", "FETCH_HEAD"]


def test_real_embodied_discovery_inherits_catalog_source_and_dataset_pins(tmp_path: Path) -> None:
    module = _load_prepare_module()
    env = module.target_env(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "hf-models",
        tmp_path / "hf-datasets",
    )
    items = module.discover_benchmark_items(env, {"calvin", "robotwin"})
    by_key = {(item.owner_id, item.kind, item.asset_id): item for item in items}

    assert by_key[("calvin", "git_repo", "official_repo")].revision == (
        "fa03f01f19c65920e18cf37398a9ce859274af76"
    )
    assert by_key[("robotwin", "git_repo", "official_repo")].revision == (
        "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
    )
    assert by_key[("robotwin", "hf_dataset", "TianxingChen/RoboTwin2.0")].revision == (
        "9dc9299c163db059931898a9f0852098a61155a1"
    )
    assert by_key[("robotwin", "hf_dataset", "lerobot/robotwin_unified")].revision == (
        "1287871839fae2296bc27b88a5457c3e1eba8e1f"
    )


def test_generated_environment_connects_prepared_embodied_assets(tmp_path: Path) -> None:
    module = _load_prepare_module()
    env = module.target_env(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "hf-models",
        tmp_path / "hf-datasets",
    )
    benchmark_env = module.benchmark_environment_defaults(env, {"calvin", "robocerebra", "robotwin"})
    env.update(benchmark_env)

    assert benchmark_env["WORLDFOUNDRY_CALVIN_ROOT"] == str(
        tmp_path / "data" / "repos" / "mees--calvin"
    )
    assert benchmark_env["WORLDFOUNDRY_CALVIN_DATASET_ROOT"] == str(
        tmp_path / "data" / "datasets" / "calvin"
    )
    assert benchmark_env["WORLDFOUNDRY_CALVIN_SPLIT"] == "validation"
    assert benchmark_env["WORLDFOUNDRY_ROBOTWIN_ROOT"] == str(
        tmp_path / "data" / "repos" / "RoboTwin-Platform--RoboTwin"
    )
    assert benchmark_env["WORLDFOUNDRY_ROBOCEREBRA_ROOT"] == str(
        tmp_path / "data" / "repos" / "buaa-colalab--RoboCerebra"
    )

    output = tmp_path / "embodied_env.sh"
    module.write_env_file(output, env, benchmark_keys=benchmark_env)
    contents = output.read_text(encoding="utf-8")
    for key in (
        "WORLDFOUNDRY_CALVIN_ROOT",
        "WORLDFOUNDRY_CALVIN_DATASET_ROOT",
        "WORLDFOUNDRY_CALVIN_SPLIT",
        "WORLDFOUNDRY_ROBOCEREBRA_ROOT",
        "WORLDFOUNDRY_ROBOTWIN_ROOT",
    ):
        assert f"export {key}=" in contents


def test_generated_environment_quotes_values_for_posix_shell(tmp_path: Path) -> None:
    module = _load_prepare_module()
    env = module.target_env(
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "hf-models",
        tmp_path / "hf-datasets",
    )
    injection_marker = tmp_path / "must-not-exist"
    data_value = f"{tmp_path}/data with spaces/$(touch {injection_marker})/$HOME"
    model_value = f"{tmp_path}/model'quote`uname`"
    env["WORLDFOUNDRY_DATA_DIR"] = data_value
    env["WORLDFOUNDRY_MODEL_DIR"] = model_value

    output = tmp_path / "embodied env.sh"
    module.write_env_file(output, env)
    completed = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; printf "%s\\n%s\\n" "$WORLDFOUNDRY_DATA_DIR" "$WORLDFOUNDRY_MODEL_DIR"',
            "sh",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert completed.stdout.splitlines() == [data_value, model_value]
    assert not injection_marker.exists()


def test_pinned_hf_asset_requires_a_matching_worldfoundry_revision_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    target = tmp_path / "dataset"
    target.mkdir()
    (target / "sample.json").write_text("{}\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(module, "tool_path", lambda _name: "/usr/bin/hf")

    def fake_run(command, *, env, log_path, timeout):
        calls.append(command)
        return 0, 0.01

    monkeypatch.setattr(module, "run_command", fake_run)
    args = Namespace(
        hf_tool="hf",
        hf_workers=2,
        skip_existing=True,
        plan_only=False,
        timeout_seconds=20,
    )
    item = module.PrepareItem(
        "benchmark",
        "hf_dataset",
        "robotwin",
        "example/robotwin",
        target,
        source="example/robotwin",
        revision="revision-a",
    )

    first = module.prepare_hf_item(item, args, {}, tmp_path / "logs")
    assert first["ready_before"] is True
    assert first["revision_matches_before"] is False
    assert first["status"] == "ready"
    assert len(calls) == 1
    assert module.hf_revision_marker_matches(item, "dataset") is True

    second = module.prepare_hf_item(item, args, {}, tmp_path / "logs")
    assert second["revision_matches_before"] is True
    assert second["status"] == "ready"
    assert len(calls) == 1

    changed = module.PrepareItem(
        "benchmark",
        "hf_dataset",
        "robotwin",
        "example/robotwin",
        target,
        source="example/robotwin",
        revision="revision-b",
    )
    third = module.prepare_hf_item(changed, args, {}, tmp_path / "logs")
    assert third["revision_matches_before"] is False
    assert third["status"] == "ready"
    assert len(calls) == 2
    assert module.hf_revision_marker_matches(changed, "dataset") is True


def test_existing_pinned_checkout_blocks_tracked_and_untracked_changes(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source = tmp_path / "source"
    source.mkdir()
    _git("init", cwd=source)
    _git("config", "user.name", "WorldFoundry Test", cwd=source)
    _git("config", "user.email", "worldfoundry@example.invalid", cwd=source)
    tracked = source / "payload.txt"
    tracked.write_text("first\n", encoding="utf-8")
    _git("add", "payload.txt", cwd=source)
    _git("commit", "-m", "first", cwd=source)
    first_revision = _git("rev-parse", "HEAD", cwd=source)
    tracked.write_text("second\n", encoding="utf-8")
    _git("commit", "-am", "second", cwd=source)
    second_revision = _git("rev-parse", "HEAD", cwd=source)

    target = tmp_path / "checkout"
    _git("clone", source.resolve().as_uri(), str(target), cwd=tmp_path)
    args = Namespace(skip_existing=True, plan_only=False, timeout_seconds=20)
    env = dict(os.environ)

    current_item = module.PrepareItem(
        "benchmark",
        "git_repo",
        "example",
        "example",
        target,
        source=source.resolve().as_uri(),
        revision=second_revision,
    )
    (target / "payload.txt").write_text("local tracked edit\n", encoding="utf-8")
    tracked_row = module.prepare_git_item(current_item, args, env, tmp_path / "logs")
    assert tracked_row["status"] == "blocked_dirty_checkout"
    assert tracked_row["ready"] is False

    _git("checkout", "--", "payload.txt", cwd=target)
    _git("checkout", "--detach", first_revision, cwd=target)
    (target / "local-note.txt").write_text("untracked\n", encoding="utf-8")
    wrong_revision_row = module.prepare_git_item(current_item, args, env, tmp_path / "logs")
    assert wrong_revision_row["status"] == "blocked_dirty_checkout"
    assert wrong_revision_row["revision_matches_before"] is False
    assert _git("rev-parse", "HEAD", cwd=target) == first_revision
