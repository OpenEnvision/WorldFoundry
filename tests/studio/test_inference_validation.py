
import pytest

# This test module imports worldfoundry code that requires the optional
# "colorspacious" dependency at import time; skip when it is unavailable.
pytest.importorskip("colorspacious")
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.serving.workspace import (
    JobCreateRequest,
    _resolve_inference_contract,
    _validate_explicit_kwargs,
)


def test_task_default_call_kwargs_can_be_passed_explicitly() -> None:
    entry = find_entry("db-cogact")
    payload = JobCreateRequest(
        model_id=entry.model_id,
        call_kwargs={"num_steps": 10, "cfg_scale": 1.5, "seed": 42},
    )
    _, task, *_ = _resolve_inference_contract(entry, payload)

    _validate_explicit_kwargs(entry, task, payload)


def test_entry_default_load_kwargs_can_be_passed_explicitly() -> None:
    entry = find_entry("hunyuan-game-craft")
    payload = JobCreateRequest(
        model_id=entry.model_id,
        load_kwargs={"torchrun_nproc_per_node": 1},
    )
    _, task, *_ = _resolve_inference_contract(entry, payload)

    _validate_explicit_kwargs(entry, task, payload)
