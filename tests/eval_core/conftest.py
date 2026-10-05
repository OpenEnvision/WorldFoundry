from __future__ import annotations

from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    suite_root = Path(__file__).resolve().parent
    for item in items:
        if item.path.is_relative_to(suite_root):
            item.add_marker(pytest.mark.fast_eval_core)
