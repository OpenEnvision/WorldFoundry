from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dvlt_runtime_logic_lives_under_base_models() -> None:
    runtime = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/dvlt/runtime.py"
    synthesis = REPO_ROOT / "worldfoundry/base_models/three_dimensions/depth/dvlt/dvlt_synthesis.py"

    runtime_text = runtime.read_text(encoding="utf-8")
    synthesis_text = synthesis.read_text(encoding="utf-8")

    assert "class DVLTRuntime" in runtime_text
    assert "from accelerate import Accelerator" in runtime_text
    assert "from dvlt.model.dvlt.model import DVLT" in runtime_text
    assert "worldfoundry.synthesis" not in runtime_text
    assert "BaseSynthesis" not in runtime_text
    assert "DVLTRuntime" in synthesis_text
    assert "return self.runtime.predict" in synthesis_text
    for heavy_marker in [
        "importlib.util",
        "sys.path",
        "Accelerator",
        "from dvlt.model.dvlt.model import DVLT",
        "DVLT(",
        "preprocess_images",
        "pointcloud_to_glb",
    ]:
        assert heavy_marker not in synthesis_text

    from worldfoundry.base_models.three_dimensions.depth.dvlt import DVLTRuntime

    assert DVLTRuntime.__module__.startswith("worldfoundry.base_models.")
