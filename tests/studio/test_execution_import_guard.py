from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from worldfoundry.core.utils.python.import_guard import third_party_lazy_import_guard
from worldfoundry.studio.inference.execution import StudioManager


def test_pipeline_class_import_waits_for_third_party_lazy_import_guard() -> None:
    started = Event()
    entry = SimpleNamespace(module_path="json", class_name="JSONDecoder")

    def import_pipeline_class():
        started.set()
        return StudioManager.import_pipeline_class(object(), entry)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with third_party_lazy_import_guard():
            future = executor.submit(import_pipeline_class)
            assert started.wait(timeout=1.0)
            assert not future.done()
        assert future.result(timeout=1.0) is json.JSONDecoder
