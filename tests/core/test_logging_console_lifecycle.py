# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Console logging follows stderr ownership across in-process CLI calls."""

from __future__ import annotations

import io
import sys

import pytest

from worldfoundry.core.observability import logging_setup

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("loguru", [False, True])
def test_console_releases_replaced_capture_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], loguru: bool
) -> None:
    monkeypatch.setattr(logging_setup, "_LOGURU_AVAILABLE", loguru)
    previous = io.StringIO()
    with monkeypatch.context() as capture:
        capture.setattr(sys, "stderr", previous)
        logging_setup.configure_logging(force=True, colorize=False)
    previous.close()

    logging_setup.get_logger(__name__).error("active console stream")
    output = capsys.readouterr().err
    assert "active console stream" in output
    assert "Logging error" not in output
    assert "Traceback" not in output
