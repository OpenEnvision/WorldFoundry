from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.inference.catalog import _solaris_default_load_kwargs, _solaris_default_ref


def test_solaris_workspace_uses_vendored_runtime_and_checkpoint_not_source_url() -> None:
    model_ref = _solaris_default_ref()
    required = _solaris_default_load_kwargs()["required_components"]

    assert "github.com" not in model_ref
    assert Path(required["runtime_root"]).is_dir()
    assert Path(required["runtime_root"]).name == "solaris_runtime"
    assert required["pretrained_model_dir"] == model_ref
    assert Path(required["model_weights_path"]) == Path(model_ref) / "solaris.pt"
    assert Path(required["eval_data_dir"]).name in {
        "datasets",
        "nyu-visionx--solaris-eval-datasets",
    }
