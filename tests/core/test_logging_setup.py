from __future__ import annotations

import json
import logging
import os
import sys
from argparse import Namespace

import pytest

from worldfoundry.core.observability import logging_setup
from worldfoundry.core.observability.logging_setup import (
    clear_log_context,
    configure_logging,
    get_logger,
    is_configured,
    log_context,
)

_LOG_LEVEL_ENV = "WORLDFOUNDRY_LOG_LEVEL"
_LOG_FILE_ENV = "WORLDFOUNDRY_LOG_FILE"
_LOG_JSON_ENV = "WORLDFOUNDRY_LOG_JSON"
_SP_CONFIGURE_ENV = "TRAINER_CONFIGURE_LOGGING"

_TRACKED_ENV = (
    _LOG_LEVEL_ENV,
    _LOG_FILE_ENV,
    _LOG_JSON_ENV,
    _SP_CONFIGURE_ENV,
    "WORLDFOUNDRY_LOG_CONTEXT",
    "WORLDFOUNDRY_RUN_ID",
)


def test_loguru_json_callable_returns_escaped_template() -> None:
    rendered = logging_setup._loguru_json_format(
        {
            "time": None,
            "level": "INFO",
            "name": "test",
            "message": "hello",
            "extra": {"benchmark_id": "demo"},
            "exception": None,
            "process": None,
        }
    ).format_map({})

    assert json.loads(rendered)["benchmark_id"] == "demo"


@pytest.mark.parametrize(
    "token",
    (
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "github_pat_abcdefghijklmnopqrstuvwxyz123456",
        # Synthetic redaction sample: separate the prefix from the dummy payload.
        "-".join(("xoxb", "1234567890", "abcdefghijklmnop")),
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signaturepadpadpad",
    ),
)
def test_redact_text_covers_known_secret_token_formats(token: str) -> None:
    redacted = logging_setup._redact_text(f"credential={token}")

    assert token not in redacted
    assert "[REDACTED]" in redacted


@pytest.fixture
def isolated_logging():
    """Snapshot and restore global logging + env state around each test.

    ``configure_logging`` mutates the stdlib root logger, the module-level
    "configured" flag, and several environment variables; without restoration
    one test's configuration leaks into the next and into the rest of the
    session.
    """
    import worldfoundry.core.observability.logging_setup as _ls

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_configured = _ls._CONFIGURED
    saved_context = _ls.get_log_context()
    saved_env = {k: os.environ.get(k) for k in _TRACKED_ENV}
    # The distributed ``log`` facade is a process-global singleton that
    # ``configure_logging`` reparents onto the root logger; snapshot it so a
    # test cannot leave it half-reconfigured for the rest of the session. The
    # import is guarded so stdlib-only environments (no torch) still run these
    # tests.
    saved_dist = None
    try:
        from worldfoundry.core.distributed.runtime.logging import distributed_logger as _dl

        saved_dist = (_dl, list(_dl.handlers), _dl.propagate, _dl.level)
    except Exception:
        saved_dist = None
    try:
        yield
    finally:
        root.handlers = list(saved_handlers)
        root.setLevel(saved_level)
        _ls._CONFIGURED = saved_configured
        _ls.clear_log_context()
        if saved_context:
            _ls.bind_log_context(**saved_context)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_dist is not None:
            dl, handlers, propagate, level = saved_dist
            dl.handlers = list(handlers)
            dl.propagate = propagate
            dl.setLevel(level)


def _flush() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_unified_pipeline_emits_stdlib_records(isolated_logging, capsys):
    configure_logging(level="DEBUG", force=True)

    logging.getLogger("stdlib.src").info("from-stdlib")

    err = capsys.readouterr().err
    assert "from-stdlib" in err
    # The stdlib logger name is rendered in the unified prefix.
    assert "[stdlib.src]" in err


def test_facade_logs_flow_through_pipeline(isolated_logging, capsys):
    """The loguru-compatible ``log`` facade shares the configured pipeline.

    Requires ``torch`` (the distributed facade module imports it); skipped in
    minimal environments.
    """
    pytest.importorskip("torch")
    from worldfoundry.core.distributed.runtime.logging import log

    configure_logging(level="DEBUG", force=True)
    log.info("facade-says-hi")

    assert "facade-says-hi" in capsys.readouterr().err


def test_file_sink_writes_and_idempotent(isolated_logging, tmp_path):
    log_file = tmp_path / "wf.log"
    configure_logging(level="INFO", log_file=str(log_file), force=True)

    logging.getLogger("t").info("once")
    _flush()
    assert "once" in log_file.read_text()

    # A second call without force=True is a no-op: no second file is created
    # and no handler/sink is duplicated.
    configure_logging(level="INFO", log_file=str(tmp_path / "wf2.log"))
    assert not (tmp_path / "wf2.log").exists()

    logging.getLogger("t").info("once")
    _flush()
    # Two emits -> two lines, never duplicated per emit.
    assert log_file.read_text().count("once") == 2


def test_level_filtering(isolated_logging, capsys):
    configure_logging(level="WARNING", force=True)

    logging.getLogger("t").info("should-not-appear")
    logging.getLogger("t").warning("should-appear")

    err = capsys.readouterr().err
    assert "should-not-appear" not in err
    assert "should-appear" in err


def test_json_file_sink(isolated_logging, tmp_path):
    log_file = tmp_path / "wf.jsonl"
    configure_logging(level="DEBUG", log_file=str(log_file), json=True, force=True)

    logging.getLogger("t").info("hello-json")
    _flush()

    raw = log_file.read_text()
    lines = [line for line in raw.splitlines() if line.strip()]
    assert lines, "expected at least one json line in the file sink"
    payload = json.loads(lines[0])
    assert set(payload) == {
        "schema_version",
        "timestamp",
        "level",
        "logger",
        "event",
        "message",
        "run_id",
        "job_id",
        "benchmark_id",
        "model_id",
        "phase",
        "sample_id",
        "rank",
        "pid",
        "exception",
        "fields",
    }
    assert payload["schema_version"] == "worldfoundry.log.v1"
    assert payload["message"] == "hello-json"
    assert payload["logger"] == "t"
    assert payload["exception"] is None


def test_event_context_redaction_and_exception_logging(isolated_logging, tmp_path, capsys):
    """Events carry run context, never leak secrets, and keep stdout clean."""
    log_file = tmp_path / "events.jsonl"
    configure_logging(level="DEBUG", log_file=str(log_file), json=True, force=True)

    with log_context(run_id="run-123", benchmark_id="vbench", phase="score"):
        logger = get_logger("worldfoundry.test.events")
        logger.event(
            "INFO",
            "benchmark.started",
            "Starting benchmark with api_key=sk-secretvalue123",
            model_id="demo-model",
            api_key="sk-secretvalue123",
        )
        try:
            raise RuntimeError("token=top-secret-value")
        except RuntimeError:
            logger.event(
                "ERROR",
                "benchmark.failed",
                "Benchmark failed",
                exc_info=True,
                sample_id="sample-7",
            )
    _flush()

    payloads = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    assert len(payloads) == 2
    started, failed = payloads
    assert started["event"] == "benchmark.started"
    assert started["run_id"] == "run-123"
    assert started["benchmark_id"] == "vbench"
    assert started["model_id"] == "demo-model"
    assert started["phase"] == "score"
    assert started["fields"]["api_key"] == "[REDACTED]"
    assert failed["event"] == "benchmark.failed"
    assert failed["sample_id"] == "sample-7"
    assert "RuntimeError" in failed["exception"]

    raw = log_file.read_text()
    captured = capsys.readouterr()
    stderr = captured.err
    assert "secretvalue123" not in raw
    assert "top-secret-value" not in raw
    assert "secretvalue123" not in stderr
    assert "top-secret-value" not in stderr
    assert captured.out == ""
    clear_log_context()


def test_distributed_logger_reparented(isolated_logging, capsys):
    """``print_rank_0`` / ``print_per_rank`` flow through the unified pipeline
    rather than the singleton's own handler. Requires ``torch``."""
    pytest.importorskip("torch")
    from worldfoundry.core.distributed.runtime.logging import distributed_logger, print_rank_0

    configure_logging(level="DEBUG", force=True)

    assert distributed_logger.propagate is True
    assert distributed_logger.handlers == []

    # distributed is not initialized in the test process, so rank-0 branch
    # emits on this rank and reaches root (-> stderr).
    print_rank_0("rank0-says-hi")
    _flush()
    assert "rank0-says-hi" in capsys.readouterr().err


def test_env_defaults(isolated_logging, tmp_path, monkeypatch):
    log_file = tmp_path / "env.log"
    monkeypatch.setenv(_LOG_LEVEL_ENV, "WARNING")
    monkeypatch.setenv(_LOG_FILE_ENV, str(log_file))

    configure_logging(force=True)  # no explicit args -> env drives it

    logging.getLogger("t").info("filtered-out")
    logging.getLogger("t").warning("env-passes")
    _flush()

    text = log_file.read_text()
    assert "filtered-out" not in text
    assert "env-passes" in text


def test_sp_neutralization(isolated_logging):
    configure_logging(force=True)
    assert os.environ.get(_SP_CONFIGURE_ENV) == "0"


def test_bad_level_raises():
    with pytest.raises(ValueError):
        configure_logging(level="VERBOSE", force=True)


def test_is_configured_flag(isolated_logging):
    import worldfoundry.core.observability.logging_setup as _ls

    # Earlier tests in the session may have configured logging (e.g. via the
    # CLI); reset the process-global flag so this test checks the transition
    # rather than session history.  ``isolated_logging`` restores the flag.
    _ls._CONFIGURED = False
    assert is_configured() is False
    configure_logging(force=True)
    assert is_configured() is True


def test_replace_string_logging_filter(isolated_logging):
    """Regression test for the ``return`` bug in ``ReplaceStringLoggingFilter``.

    The filter must *keep* the record (return truthy) and replace its message
    in place. Requires ``numpy`` (print_utils imports it)."""
    pytest.importorskip("numpy")
    from worldfoundry.core.observability.formatting import ReplaceStringLoggingFilter

    flt = ReplaceStringLoggingFilter(["*"], lambda msg: "REPLACED")

    matched = logging.LogRecord("n", logging.INFO, __file__, 1, "anything", None, None)
    assert flt.filter(matched) is True
    assert matched.msg == "REPLACED"

    other = logging.LogRecord("n", logging.INFO, __file__, 1, "other", None, None)
    # Even a non-matching record must be kept (not silently dropped).
    assert flt.filter(other) is True


def test_print_to_file_keeps_stdout_and_stderr_separate(tmp_path):
    """``PrintToFile`` must honour distinct output and error destinations."""
    pytest.importorskip("numpy")
    from worldfoundry.core.observability.formatting import PrintToFile

    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    with PrintToFile(str(stdout_path), str(stderr_path)):
        print("standard output")
        print("standard error", file=sys.stderr)

    assert stdout_path.read_text() == "standard output\n"
    assert stderr_path.read_text() == "standard error\n"


def test_cli_output_run_gets_isolated_lifecycle_log(isolated_logging, tmp_path, monkeypatch):
    """Output-owning CLI commands receive one generated run ID and JSONL file."""
    pytest.importorskip("yaml")
    from worldfoundry.cli.main import _prepare_cli_run_observability, _write_cli_run_lifecycle

    monkeypatch.delenv(_LOG_FILE_ENV, raising=False)
    args = Namespace(command="run", output_dir=tmp_path, run_id=None, benchmark_id="vbench", model_id="demo-model")
    state = _prepare_cli_run_observability(args, explicit_log_file=None)
    assert state is not None
    event_path, run_id, _ = state
    assert args.run_id == run_id
    assert event_path == tmp_path / "logs" / run_id / "events.jsonl"

    _write_cli_run_lifecycle(
        state,
        event="run.finished",
        level="INFO",
        message="WorldFoundry command finished",
        exit_code=0,
    )
    rows = [json.loads(line) for line in event_path.read_text().splitlines() if line]
    assert [row["event"] for row in rows] == ["run.started", "run.finished"]
    assert all(row["run_id"] == run_id for row in rows)


def test_cli_logging_flag_pre_scan():
    """``_extract_logging_flags`` strips the global flags wherever they appear
    and returns them. Requires the CLI import chain (``yaml``)."""
    pytest.importorskip("yaml")
    from worldfoundry.cli.main import _extract_logging_flags

    # The pre-scan grew a ``verbose`` slot (``-v``/``--verbose``) and now
    # returns a 5-tuple: (level, log_file, log_json, verbose, rest).
    level, log_file, log_json, verbose, rest = _extract_logging_flags(
        ["--log-level", "DEBUG", "run", "--log-file=/tmp/x.log", "--log-json"]
    )
    assert level == "DEBUG"
    assert log_file == "/tmp/x.log"
    assert log_json is True
    assert verbose is False
    assert rest == ["run"]

    level, log_file, log_json, verbose, rest = _extract_logging_flags(
        ["-v", "zoo", "benchmark-run", "--verbose"]
    )
    assert level is None and log_file is None and log_json is None
    assert verbose is True
    assert rest == ["zoo", "benchmark-run"]


@pytest.mark.parametrize("json_output", [False, True])
def test_direct_loguru_exception_redacted_in_console_and_file(isolated_logging, tmp_path, capsys, json_output):
    from loguru import logger

    output = tmp_path / "exceptions.log"
    configure_logging(log_file=str(output), json=json_output, force=True)
    secret = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    try:
        raise ValueError("value={braces} <red>markup</red> " + secret)
    except ValueError:
        logger.exception("direct exception {literal}")
    logger.complete()
    console = capsys.readouterr().err
    file_output = output.read_text()
    file_text = json.loads(file_output)["exception"] if json_output else file_output
    for text in (console, file_text):
        assert secret not in text
        assert "[REDACTED]" in text
        assert "ValueError" in text
        assert "{braces}" in text
        assert "<red>markup</red>" in text
    if json_output:
        record = json.loads(file_output.strip())
        assert "_worldfoundry_exception_text" not in record
        assert "_worldfoundry_exception_text" not in record.get("extra", {})
