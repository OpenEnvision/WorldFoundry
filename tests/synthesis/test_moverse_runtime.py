from __future__ import annotations

from worldfoundry.synthesis.visual_generation.moverse.moverse_synthesis import (
    MoVerseSynthesis,
)


def test_moverse_plan_uses_the_current_shared_wan_runtime(tmp_path) -> None:
    result = MoVerseSynthesis(options={"run_dir": str(tmp_path)}).predict(plan_only=True)

    assert result["status"] == "prepared"
    assert result["metadata"]["missing_requirements"] == []
