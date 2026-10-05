from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.irasim.irasim_synthesis import (
    IRASimSynthesis,
)


def test_irasim_refuses_to_write_plan_as_video(tmp_path: Path) -> None:
    runtime = IRASimSynthesis(device="cpu")

    with pytest.raises(RuntimeError, match="inference plan"):
        runtime.predict(output_path=tmp_path / "result.mp4")

    assert not (tmp_path / "result.mp4").exists()


def test_irasim_plan_only_uses_json_artifact(tmp_path: Path) -> None:
    runtime = IRASimSynthesis(device="cpu")

    result = runtime.predict(output_path=tmp_path / "result.mp4", plan_only=True)

    artifact = Path(result["artifact_path"])
    assert artifact == tmp_path / "result.plan.json"
    assert artifact.is_file()
