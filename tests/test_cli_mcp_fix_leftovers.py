"""Regression tests for leftover CLI/MCP repairs (CM-08 / CM-29 remainder / CM-33).

CPU-only; no GPU, network, or optional third-party packages required.
Does not rename flags or change success-path exit codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── CM-08: CliUsageError routes to exit 2 ────────────────────────


def test_evaluate_usage_raises_cli_usage_error() -> None:
    from worldfoundry.cli.main import _validate_evaluate_usage
    from worldfoundry.cli.utils import CliUsageError

    with pytest.raises(CliUsageError, match="--samples-path requires --embodied-spec"):
        _validate_evaluate_usage(
            argparse.Namespace(
                embodied_spec=None,
                samples_path=Path("/tmp/samples.json"),
                requests_path=None,
                task_type=None,
                benchmark_name=None,
                data_path=None,
            )
        )


def test_evaluate_mutex_flags_raise_cli_usage_error() -> None:
    from worldfoundry.cli.main import _validate_evaluate_usage
    from worldfoundry.cli.utils import CliUsageError

    with pytest.raises(CliUsageError, match="cannot be combined"):
        _validate_evaluate_usage(
            argparse.Namespace(
                embodied_spec="{}",
                samples_path=None,
                requests_path=None,
                task_type="video",
                benchmark_name=None,
                data_path=None,
            )
        )


def test_score_usage_error_exits_two_with_json(capsys: pytest.CaptureFixture[str]) -> None:
    from worldfoundry.cli.main import main

    returncode = main(["score", "--benchmark", "does-not-exist-xyz", "--plan-only", "--json"])
    assert returncode == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["type"] == "usage"
    assert "artifacts" in payload["error"]["message"]
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_help_success_exit_code_unchanged() -> None:
    from worldfoundry.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code in {0, None}


# ── CM-29 remainder: DEFAULT_CONTEXT setter ──────────────────────


def test_set_default_context_shares_job_store() -> None:
    from worldfoundry.mcp.tools.context import (
        DEFAULT_CONTEXT,
        MCPToolContext,
        get_default_context,
        set_default_context,
    )
    from worldfoundry.mcp.tools.server_info import server_info_payload

    original = MCPToolContext(
        output_root=DEFAULT_CONTEXT.output_root,
        model_manifest_dir=DEFAULT_CONTEXT.model_manifest_dir,
        benchmark_manifest_dir=DEFAULT_CONTEXT.benchmark_manifest_dir,
        job_store=DEFAULT_CONTEXT.job_store,
    )
    fresh = MCPToolContext(output_root=Path("/tmp/wf-mcp-cm29-root").resolve())
    try:
        assert set_default_context(fresh) is fresh
        assert get_default_context() is fresh
        assert DEFAULT_CONTEXT.output_root == fresh.output_root
        assert DEFAULT_CONTEXT.job_store is fresh.job_store
        payload = server_info_payload()
        assert payload["output_root"] == str(fresh.output_root)
    finally:
        set_default_context(original)


def test_create_mcp_server_uses_absolute_default_root() -> None:
    from worldfoundry.mcp import server as server_mod
    from worldfoundry.mcp.tools.context import resolve_mcp_output_root

    source = Path(server_mod.__file__).read_text(encoding="utf-8")
    assert "resolve_mcp_output_root" in source
    assert 'Path(os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", "runs/mcp"))' not in source
    assert resolve_mcp_output_root().is_absolute()


# ── CM-33: branding follows the invoked entry ────────────────────


def test_cli_prog_name_follows_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.cli.utils import cli_prog_name

    monkeypatch.setattr(sys, "argv", ["/usr/bin/worldfoundry", "--help"])
    assert cli_prog_name() == "worldfoundry"
    monkeypatch.setattr(sys, "argv", ["/usr/bin/worldfoundry-eval", "--help"])
    assert cli_prog_name() == "worldfoundry-eval"
    monkeypatch.setattr(sys, "argv", [str(REPO_ROOT / "worldfoundry" / "cli" / "__main__.py")])
    assert cli_prog_name() == "worldfoundry-eval"


def test_banner_uses_worldfoundry_when_invoked_as_worldfoundry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from worldfoundry.cli.main import main

    monkeypatch.setattr(sys, "argv", ["/usr/bin/worldfoundry"])
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "worldfoundry zoo benchmarks" in output
    assert "worldfoundry-eval zoo benchmarks" not in output


def test_help_prog_follows_worldfoundry_entry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from worldfoundry.cli.main import main

    monkeypatch.setattr(sys, "argv", ["/usr/bin/worldfoundry"])
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code in {0, None}
    assert "usage: worldfoundry" in capsys.readouterr().out


# ── CM-24 remainder: unused helper gone; query matcher shared ────


def test_unused_hfd_helper_removed() -> None:
    import worldfoundry.cli.zoo as zoo

    assert not hasattr(zoo, "_default_hfd_dataset_root")


def test_shared_query_matcher() -> None:
    from worldfoundry.mcp.tools.query import matches_query

    assert matches_query("Wan", "Wan2.1", "other")
    assert matches_query("wan*", "Wan2.1")
    assert not matches_query("missing", "Wan2.1")
