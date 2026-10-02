from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.joyai_echo import (
    JoyAIEchoLongVideoRuntime,
    JoyAIEchoWMRuntime,
)


@pytest.mark.parametrize("model_id", ("joyai-echo-longvideo", "joyai-echo-wm"))
def test_joyai_echo_workspace_declares_every_external_runtime_component(model_id: str) -> None:
    entry = find_entry(model_id)
    options = entry.default_load_kwargs

    assert Path(entry.default_model_ref).is_dir()
    assert Path(options["source_root"]).is_dir()
    assert Path(entry.default_input_path).is_file()
    assert "gemma-3-12b-it" in Path(options["gemma_path"]).name
    assert options["python_executable"].endswith(f"/{model_id}/bin/python")
    assert entry.default_call_kwargs["execute"] is True
    assert entry.default_call_kwargs["num_frames"] == 241


def test_joyai_echo_cpu_preflight_reports_only_unstaged_gemma_and_environments() -> None:
    longvideo = find_entry("joyai-echo-longvideo").default_load_kwargs
    wm = find_entry("joyai-echo-wm").default_load_kwargs
    reports = (
        JoyAIEchoLongVideoRuntime(device="cuda:0", **longvideo).preflight(),
        JoyAIEchoWMRuntime(device="cuda:0", **wm).preflight(),
    )

    for report in reports:
        assert report["status"] == "blocked"
        assert report["missing_runtime_files"] == [report["python_executable"]]
        assert report["missing_checkpoint_files"] == [str(Path(report["gemma_path"]) / "config.json")]
        assert report["blocked_reasons"] == []
