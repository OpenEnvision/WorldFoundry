import json
from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.serving.workspace import JobCreateRequest, _inference_run_kwargs


def _resolved_request(model_id: str) -> dict[str, object]:
    _, run_kwargs = _inference_run_kwargs(
        JobCreateRequest(job_type="inference", model_id=model_id, prompt="A wave crashes."),
    )
    return run_kwargs


def test_bernini_renderer_aliases_select_single_rank_variants() -> None:
    assert find_entry("bernini-r-14b").model_id == "bernini"
    assert find_entry("bernini-r-1.3b").model_id == "bernini"

    expected = {
        "bernini-r-14b": "ByteDance--Bernini-R-Diffusers",
        "bernini-r-1.3b": "ByteDance--Bernini-R-1.3B-Diffusers",
    }
    for model_id, model_ref in expected.items():
        request = _resolved_request(model_id)
        assert Path(str(request["model_ref"])).name == model_ref
        assert Path(str(request["model_ref"])).is_dir()
        load_kwargs = json.loads(request["load_kwargs_text"])
        call_kwargs = json.loads(request["call_kwargs_text"])
        assert load_kwargs["model_id"] == model_id
        assert call_kwargs["nproc_per_node"] == 1
        assert call_kwargs["ulysses_size"] == 1


def test_bernini_workspace_default_uses_full_recipe_on_one_memory_safe_rank() -> None:
    request = _resolved_request("bernini")
    call_kwargs = json.loads(request["call_kwargs_text"])

    assert Path(str(request["model_ref"])).name == "ByteDance--Bernini-Diffusers"
    assert Path(str(request["model_ref"])).is_dir()
    assert call_kwargs["nproc_per_node"] == 1
    assert call_kwargs["ulysses_size"] == 1
